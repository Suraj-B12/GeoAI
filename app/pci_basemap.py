"""
Street-map background for the condition report's route figures (app/pci_report_pdf.py).

Web Mercator raster tiles with street and area names (OpenStreetMap standard tiles, © OpenStreetMap
contributors), fetched once and cached on disk. If they cannot be reached the caller draws the route without a
background and says so on the figure.

    view(lonlats, w_pt, h_pt, pad_pt) -> BaseView | None
        .image_path      JPEG of exactly w_pt x h_pt (for reportlab.graphics.shapes.Image)
        .xy(lon, lat)    -> drawing coordinates (pt, origin bottom-left)
        .pt_per_m        drawing points per metre on the ground (for the scale bar)
        .attribution     text that must be printed on the figure

Environment: PCI_BASEMAP=0 disables backgrounds; PCI_TILE_CACHE moves the cache
(default C:\\GeoAI\\pci\\tile_cache). Network: at most MAX_TILES tiles per figure, 6 s timeout per tile, a
User-Agent naming the project (OSM tile usage policy), cached tiles reused for TILE_MAX_AGE_DAYS.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

PROVIDERS = (
    # CARTO's raster styles answer non-browser clients with "API KEY REQUIRED" placeholder tiles (seen
    # 2026-11-07), so only the OpenStreetMap standard tiles are used (tile usage policy: identifying
    # User-Agent, local cache, low volume).
    {"name": "osm", "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
     "subs": "", "scale": 1, "max_z": 19, "attribution": "Map © OpenStreetMap contributors"},
)
USER_AGENT = "GeoAI-capstone-condition-report/1.0 (student research project, Bengaluru)"
MAX_TILES = 64
TIMEOUT_S = 6.0
TILE_MAX_AGE_DAYS = 30
MIN_PT_PER_PX = 0.55        # map text (~11 css px) stays >= ~6 pt
MIN_Z, MAX_Z = 11, 18
WASH = 0.12                 # blend with white so the coloured route stands out
_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get("PCI_BASEMAP", "1").strip().lower() not in ("0", "false", "no", "off")


def cache_dir() -> Path:
    return Path(os.environ.get("PCI_TILE_CACHE", r"C:\GeoAI\pci\tile_cache"))


def world_px(lon: float, lat: float, z: int) -> tuple[float, float]:
    """Web Mercator 'CSS pixel' coordinates (256 per tile) at zoom z; y grows southwards."""
    n = 256.0 * (2 ** z)
    lat = max(-85.0511, min(85.0511, lat))
    x = (lon + 180.0) / 360.0 * n
    r = math.radians(lat)
    y = (1.0 - math.log(math.tan(r) + 1.0 / math.cos(r)) / math.pi) / 2.0 * n
    return x, y


@dataclass
class BaseView:
    image_path: Optional[str]
    xy: Callable[[float, float], tuple[float, float]]
    pt_per_m: float
    attribution: str
    provider: str
    zoom: int


def _fetch(provider: dict, z: int, x: int, y: int, client) -> Optional[bytes]:
    p = cache_dir() / provider["name"] / str(z) / str(x) / f"{y}.png"
    try:
        if p.exists() and (time.time() - p.stat().st_mtime) < TILE_MAX_AGE_DAYS * 86400 and p.stat().st_size > 0:
            return p.read_bytes()
    except OSError:
        pass
    subs = provider["subs"]
    url = provider["url"].format(s=subs[(x + y) % len(subs)] if subs else "", z=z, x=x, y=y)
    for _ in range(2):
        try:
            r = client.get(url, timeout=TIMEOUT_S)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
                data = r.content
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
                    tmp.write_bytes(data)
                    os.replace(tmp, p)
                except OSError:
                    pass
                return data
        except Exception:
            pass
    if p.exists():                       # stale but usable beats nothing
        try:
            return p.read_bytes()
        except OSError:
            return None
    return None


def _compose(provider: dict, z: int, left: float, top: float, w_px: float, h_px: float) -> Optional[bytes]:
    """JPEG of the CSS-pixel window [left, left+w_px] x [top, top+h_px] at zoom z, or None."""
    import httpx
    from PIL import Image

    sc = provider["scale"]
    t0x, t1x = int(math.floor(left / 256)), int(math.floor((left + w_px) / 256))
    t0y, t1y = int(math.floor(top / 256)), int(math.floor((top + h_px) / 256))
    n = 2 ** z
    tiles = [(tx, ty) for tx in range(t0x, t1x + 1) for ty in range(t0y, t1y + 1)]
    if len(tiles) > MAX_TILES or any(not (0 <= ty < n) for _, ty in tiles):
        return None
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        with ThreadPoolExecutor(max_workers=6) as ex:
            datas = list(ex.map(lambda t: _fetch(provider, z, t[0] % n, t[1], client), tiles))
    if any(d is None for d in datas):
        return None
    tile = 256 * sc
    canvas = Image.new("RGB", ((t1x - t0x + 1) * tile, (t1y - t0y + 1) * tile), "white")
    for (tx, ty), data in zip(tiles, datas):
        try:
            im = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            return None
        if im.size != (tile, tile):
            im = im.resize((tile, tile))
        canvas.paste(im, ((tx - t0x) * tile, (ty - t0y) * tile))
    cx, cy = (left - t0x * 256) * sc, (top - t0y * 256) * sc
    crop = canvas.crop((int(round(cx)), int(round(cy)), int(round(cx + w_px * sc)), int(round(cy + h_px * sc))))
    if WASH > 0:
        crop = Image.blend(crop, Image.new("RGB", crop.size, "white"), WASH)
    out = io.BytesIO()
    crop.save(out, "JPEG", quality=86, optimize=True)
    return out.getvalue()


def view(lonlats: list[tuple[float, float]], w_pt: float, h_pt: float, pad_pt: float) -> Optional[BaseView]:
    """A street-map background that fits all lonlats inside (w_pt - 2 pad) x (h_pt - 2 pad); None if disabled or
    unreachable."""
    if not enabled() or not lonlats:
        return None
    lons = [p[0] for p in lonlats]
    lats = [p[1] for p in lonlats]
    lat_c = (min(lats) + max(lats)) / 2
    avail_w, avail_h = max(10.0, w_pt - 2 * pad_pt), max(10.0, h_pt - 2 * pad_pt)
    for provider in PROVIDERS:
        chosen = None
        for z in range(min(MAX_Z, provider["max_z"]), MIN_Z - 1, -1):
            x0, y1 = world_px(min(lons), min(lats), z)
            x1, y0 = world_px(max(lons), max(lats), z)
            bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
            s = min(avail_w / bw, avail_h / bh)
            if s >= MIN_PT_PER_PX:
                chosen = (z, s, (x0 + x1) / 2, (y0 + y1) / 2)
                break
        if chosen is None:
            return None
        z, s, cxp, cyp = chosen
        w_px, h_px = w_pt / s, h_pt / s
        left, top = cxp - w_px / 2, cyp - h_px / 2
        key = hashlib.sha256(f"{provider['name']}|{z}|{left:.3f}|{top:.3f}|{w_px:.3f}|{h_px:.3f}|{WASH}".encode()
                             ).hexdigest()[:32]
        out = cache_dir() / "renders" / f"{key}.jpg"
        if not out.exists():
            data = _compose(provider, z, left, top, w_px, h_px)
            if data is None:
                continue                                   # next provider
            with _lock:
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_bytes(data)
                os.replace(tmp, out)

        def xy(lon, lat, _z=z, _s=s, _l=left, _t=top, _h=h_pt):
            px, py = world_px(lon, lat, _z)
            return (px - _l) * _s, _h - (py - _t) * _s

        m_per_px = 156543.03392 * math.cos(math.radians(lat_c)) / (2 ** z)
        return BaseView(image_path=str(out), xy=xy, pt_per_m=s / m_per_px,
                        attribution=provider["attribution"], provider=provider["name"], zoom=z)
    return None
