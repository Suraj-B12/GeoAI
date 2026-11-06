"""
API for the expert UI's Video and Metrics tabs and its review log (/expert-api/*).

    GET  /expert-api/meta                         type list, severities, queues, the measured
                                                  reliability of the overlay (paper §6.10)
    GET  /expert-api/video/sessions               drives with answered segments
    GET  /expert-api/video/list                   ?queue=needs_review|auto|random|reviewed
                                                  &session=<drive>&label=&limit=&offset=&me=<page session>
    POST /expert-api/video/review/{segment_id}    save + log an expert review (one transaction;
                                                  409 if another expert holds it or it changed)
    POST /expert-api/claims/next                  hand this page its next items of a review list
    POST /expert-api/claims/grant                 lease specific items (citizen lists, edits)
    POST /expert-api/claims/heartbeat             renew this page's leases (every minute)
    POST /expert-api/claims/release               drop leases (save, skip, page unload)
    GET  /expert-api/claims/active                who is reviewing what right now (names only)
    POST /expert-api/citizen/review/{assessment_id}
                                                  save a citizen review on the server: lease check,
                                                  conditional Supabase write, log (several experts)
    POST /expert-api/citizen/log/{assessment_id}  log a citizen review the page saved to Supabase
                                                  itself (single-reviewer mode, page opened as a file)
    GET  /expert-api/explain/video/{segment_id}?type=Potholes
    GET  /expert-api/explain/citizen/{assessment_id}?type=Potholes
                                                  "where the model looked": Grad-CAM of the
                                                  per-type probe (app/model.explain_type)
    GET  /expert-api/metrics                      model vs expert (app/expert_review.metrics)

Several experts at once: the review lists are handed out as leases (research.review_claims,
migration 007; see app/expert_review.py "Claims"), so no item is shown to two experts at the same
time, and every save is refused with 409 when another expert holds the item or saved it since the
page loaded it.

Open like the rest of the expert page (which writes citizen reviews to Supabase with the anon
key): no X-API-Key, because the page is static and holds no secret. Writes are therefore
narrow: a video save can only set the expert_* columns of an assessment that belongs to a
video segment, with validated values; a citizen log row is accepted only when Supabase already
holds that same review (the page saved it first), so the log cannot be filled with reviews that
were never made. Explanations cost GPU time, so they are cached on disk by image SHA-256 + type
+ variant, limited per client, and at most two wait for the model at once.

The local PostgreSQL is reached through app.video_store.connect() in a worker thread; if it is
down these endpoints answer 503 and nothing else on the server is affected.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import threading
import time
import urllib.parse
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from PIL import Image

from app import expert_review as ER

PROJECT_ROOT = Path(__file__).resolve().parent.parent
router = APIRouter(prefix="/expert-api", tags=["expert"])

CONNECT_TIMEOUT_S = 3
EXPLAIN_PER_MINUTE = 30               # per client address
EXPLAIN_MAX_WAITING = 2               # explanations waiting for the model at once
MAX_IMAGE_BYTES = 25 * 1024 * 1024
_ID = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
_UUID = re.compile(r"^[0-9a-fA-F-]{36}$")
_explain_waiting = 0
_explain_guard = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)
_GRADCAM_SHA = hashlib.sha256((PROJECT_ROOT / "scripts" / "stage2_gradcam.py").read_bytes()).hexdigest()[:12] \
    if (PROJECT_ROOT / "scripts" / "stage2_gradcam.py").exists() else "missing"


class _Unavailable(Exception):
    pass


def _db(fn: Callable) -> Any:
    """fn(conn) on a fresh local-PostgreSQL connection; _Unavailable when it can't connect."""
    import psycopg

    from app import video_store
    try:
        conn = video_store.connect(connect_timeout=CONNECT_TIMEOUT_S)
    except RuntimeError as e:
        raise _Unavailable(str(e)) from None
    except psycopg.OperationalError:
        raise _Unavailable("local PostgreSQL is unreachable") from None
    try:
        return fn(conn)
    except psycopg.errors.UndefinedTable:
        raise _Unavailable("tables missing: run scripts/setup_local_db.py") from None
    except psycopg.OperationalError:
        raise _Unavailable("local PostgreSQL connection failed") from None
    finally:
        conn.close()


async def _run(fn: Callable) -> Any:
    try:
        return await asyncio.to_thread(_db, fn)
    except _Unavailable as e:
        raise HTTPException(status_code=503, detail=f"review database unavailable: {e}")
    except ER.NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ER.ReviewError as e:
        raise HTTPException(status_code=400, detail=str(e))


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="body must be JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    return body


# ============================================================
# Meta
# ============================================================

def _overlay_reliability() -> Optional[dict]:
    """Per-class test hit rates of the overlay vs the location prior (paper §6.10), if present."""
    p = PROJECT_ROOT / "eval_results" / "gradcam_localization_report.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    to_irc = {"Linear crack": ["Longitudinal Cracking", "Transverse Cracking"],
              "Alligator crack": ["Alligator Cracking"], "Pothole": ["Potholes"], "Raveling": ["Ravelling"],
              "Weathering": ["Hungry Surface"], "Patch and utility cut": ["Patching"],
              "Block crack": ["Transverse Cracking"]}
    out = {}
    for cls, row in (d.get("per_class_test") or {}).items():
        for t in to_irc.get(cls, []):
            if t in out:        # Transverse: keep the larger sample (linear cracks)
                continue
            out[t] = {"attain_class": cls, "hit": row["gradcam"]["hit"], "prior": row["prior"]["hit"],
                      "n": row["n_test"]}
    return {"variant": d.get("chosen_variant"), "macro_hit": d["test"]["gradcam"]["point"]
            if isinstance(d.get("test", {}).get("gradcam"), dict) else None, "per_type": out,
            "source": "eval_results/gradcam_localization_report.json (paper §6.10)"}


@router.get("/meta")
async def meta():
    return {"types": list(ER.EXPERT_TYPES), "normal": ER.NORMAL, "severities": list(ER.SEVERITY_RANK),
            "video_queues": list(ER.VIDEO_QUEUES), "heatmap_ratings": list(ER.HEATMAP_RATINGS),
            "overlay_reliability": _overlay_reliability()}


# ============================================================
# Video
# ============================================================

@router.get("/video/sessions")
async def video_sessions():
    return {"items": await _run(ER.video_sessions)}


@router.get("/video/list")
async def video_list(queue: str = "needs_review", session: Optional[str] = None, label: Optional[str] = None,
                     limit: int = 24, offset: int = 0, me: Optional[str] = None):
    """One page of a list. session = the drive filter; me = this page's reviewer session. Each item
    says who holds a live lease on it (claim_name, claim_mine) and carries image_v (the image's
    SHA-256 prefix) for cacheable image URLs. The review lists themselves are handed out through
    /claims/next; this endpoint is for the Reviewed list and the counts."""
    if session is not None and session and not _ID.match(session):
        raise HTTPException(status_code=400, detail="bad session id")
    if label is not None and len(label) > 60:
        raise HTTPException(status_code=400, detail="bad label")
    return await _run_claims(lambda c: ER.list_video_marked(c, queue, me=me or None, drive=session or None,
                                                            label=label or None, limit=limit, offset=offset))


@router.post("/video/review/{segment_id}")
async def video_review(segment_id: str, request: Request):
    """Save + log, with the multi-expert checks: 409 when another session holds a live lease on
    the item, or when it was reviewed since the page loaded it (body.seen_reviewed_at)."""
    if not _ID.match(segment_id):
        raise HTTPException(status_code=400, detail="bad segment id")
    body = await _json_body(request)
    return await _run_claims(lambda c: ER.save_video_review_claimed(c, segment_id, body))


# ============================================================
# Citizen
# ============================================================

_CITIZEN_COLS = ("id,image_url,status,stage1_label,stage1_confidence,distress_types,severity,stage2_confidence,"
                 "needs_expert_review,model_version,processed_at,expert_reviewed,expert_corrected_types,"
                 "expert_corrected_severity,reviewer_name")


async def _citizen_row(request: Request, assessment_id: str) -> dict:
    if not _UUID.match(assessment_id):
        raise HTTPException(status_code=400, detail="bad assessment id")
    client = getattr(request.app.state, "supabase_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="Supabase client not configured on this server")
    from app.supabase_client import is_local_db
    if is_local_db(client):
        row = await client.get_assessment_by_id(assessment_id)
    else:
        resp = await client.get(f"/rest/v1/assessments?id=eq.{assessment_id}&select={_CITIZEN_COLS}&limit=1")
        if not resp.is_success:
            raise HTTPException(status_code=502, detail=f"Supabase answered {resp.status_code}")
        rows = resp.json() or []
        row = rows[0] if rows else None
    if not row:
        raise HTTPException(status_code=404, detail="no such assessment")
    return row


@router.post("/citizen/log/{assessment_id}")
async def citizen_log(assessment_id: str, request: Request):
    body = await _json_body(request)
    row = await _citizen_row(request, assessment_id)
    saved = [ER.canonical(t) for t in (row.get("expert_corrected_types") or [])]
    sent = [ER.canonical(t) for t in (body.get("types") or []) if isinstance(t, str)]
    if not row.get("expert_reviewed") or sorted(saved) != sorted(sent):
        raise HTTPException(status_code=409, detail="Supabase does not hold this review; save it there first")
    return await _run(lambda c: ER.log_citizen_review(c, assessment_id, body, row))


# ============================================================
# "Where the model looked" (Grad-CAM, reference only)
# ============================================================

def _cache_dir() -> Path:
    from app.local_images import image_root
    d = Path(os.environ.get("EXPERT_GRADCAM_CACHE") or (image_root().parent / "gradcam_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _allow(addr: str) -> bool:
    now = time.time()
    q = _hits[addr]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= EXPLAIN_PER_MINUTE:
        return False
    q.append(now)
    return True


def _explain_cached(sha: str, type_key: str, load_image: Callable[[], Image.Image]) -> dict:
    """Disk-cached explain_type (the cache key covers the image, type, variant and code)."""
    from app.model import PavementClassifier, get_classifier
    variant = f"{PavementClassifier.GRADCAM_METHOD}@{PavementClassifier.GRADCAM_LAYER}"
    key = hashlib.sha256(f"{sha}|{type_key}|{variant}|{_GRADCAM_SHA}".encode()).hexdigest()[:32]
    path = _cache_dir() / f"{key}.json"
    try:
        out = json.loads(path.read_text(encoding="utf-8"))
        out["cached"] = True
        return out
    except (OSError, ValueError):
        pass
    img = load_image()
    size = list(img.size)
    try:
        out = get_classifier().explain_type(img, type_key)
    finally:
        img.close()
    out["image_size"] = size
    out["cached"] = False
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out), encoding="utf-8")
    os.replace(tmp, path)
    return out


async def _explain(request: Request, type_key: str, sha: str, load_image: Callable[[], Image.Image]) -> dict:
    global _explain_waiting
    from scripts.stage2_probe import all_probe_types
    if type_key not in {t.key for t in all_probe_types()}:
        raise HTTPException(status_code=400, detail="type must be an IRC:82 type, Patching or Block Cracking")
    if not _allow(request.client.host if request.client else "?"):
        raise HTTPException(status_code=429, detail="too many explanations this minute")
    with _explain_guard:
        if _explain_waiting >= EXPLAIN_MAX_WAITING:
            raise HTTPException(status_code=429, detail="the model is busy with other explanations; try again")
        _explain_waiting += 1
    try:
        out = await asyncio.to_thread(_explain_cached, sha, type_key, load_image)
    except KeyError:
        raise HTTPException(status_code=400, detail="unknown type")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"explanation unavailable: {e}")
    finally:
        with _explain_guard:
            _explain_waiting -= 1
    out["reference_only"] = ("Where the model's evidence for this type is strongest when asked about it "
                             "directly. Not a measured damage box (paper §6.10).")
    return out


@router.get("/explain/video/{segment_id}")
async def explain_video(segment_id: str, request: Request, type_key: str = Query(..., alias="type")):
    from app.gis_routes import _Unavailable as GisUnavailable, resolve_segment_image
    if not _ID.match(segment_id):
        raise HTTPException(status_code=400, detail="bad segment id")
    try:
        found = await asyncio.to_thread(resolve_segment_image, segment_id)
    except GisUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    if found is None:
        raise HTTPException(status_code=404, detail="no stored image for this segment")
    path, _media, sha = found
    if not sha:
        sha = await asyncio.to_thread(lambda: hashlib.sha256(path.read_bytes()).hexdigest())
    return await _explain(request, type_key, sha, lambda: Image.open(path).convert("RGB"))


@router.get("/explain/citizen/{assessment_id}")
async def explain_citizen(assessment_id: str, request: Request, type_key: str = Query(..., alias="type")):
    row = await _citizen_row(request, assessment_id)
    url = row.get("image_url") or ""
    from app.local_images import local_image_path, read_local_image
    if local_image_path(url) is not None:
        data = await asyncio.to_thread(read_local_image, url)
    else:
        u = urllib.parse.urlparse(url)
        if u.scheme != "https" or u.hostname != "res.cloudinary.com":
            raise HTTPException(status_code=400, detail="image is not a Cloudinary or local image")
        client = getattr(request.app.state, "image_client", None)
        import httpx
        try:
            if client is not None:
                resp = await client.get(url)
            else:
                async with httpx.AsyncClient(timeout=30) as c:
                    resp = await c.get(url)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"image download failed: {type(e).__name__}")
        if not resp.is_success or len(resp.content) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=502, detail="image download failed")
        data = resp.content
    sha = hashlib.sha256(data).hexdigest()
    return await _explain(request, type_key, sha, lambda: Image.open(io.BytesIO(data)).convert("RGB"))


# ============================================================
# Metrics
# ============================================================

@router.get("/metrics")
async def metrics():
    return await _run(ER.metrics)


# ============================================================
# Claims: several experts at once (app/expert_review.py, migration 007)
# ============================================================

_warm_pending: set = set()
_warm_lock = threading.Lock()
WARM_WIDTH = 1920                       # the lightbox's preview width (expert_ui imageUrlFor)


def _warm_previews(segment_ids: list) -> None:
    """Render the lightbox-size previews of freshly handed-out segments into the disk cache, so the
    expert's first enlarge is a cache read. Best effort; make_preview limits concurrent decodes."""
    from app.gis_routes import make_preview, resolve_segment_image
    for sid in segment_ids:
        try:
            found = resolve_segment_image(sid)
            if found is not None:
                path, _media, sha = found
                make_preview(path, WARM_WIDTH, sha)
        except Exception:                                  # noqa: BLE001 - warming must never fail a request
            pass
        finally:
            with _warm_lock:
                _warm_pending.discard(sid)


def _warm(segment_ids: list) -> None:
    with _warm_lock:
        todo = [s for s in segment_ids if s not in _warm_pending]
        _warm_pending.update(todo)
    if todo:
        try:
            asyncio.get_running_loop().run_in_executor(None, _warm_previews, todo)
        except RuntimeError:
            with _warm_lock:
                _warm_pending.difference_update(todo)


async def _run_claims(fn: Callable) -> Any:
    """_run plus Conflict -> 409 with {message, held_by | reviewed_by, reviewed_at}."""
    try:
        return await _run(fn)
    except ER.Conflict as e:
        raise HTTPException(status_code=409, detail=e.info)


@router.post("/claims/next")
async def claims_next(request: Request):
    """Hand this browser session its next items of a review list.
    video:   {source: "video", queue, session, reviewer_name, n, filters: {drive, label}}
    citizen: {source: "citizen", queue: "citizen_pending", session, reviewer_name, n,
              candidates: [assessment ids in the page's order]}  (the page lists them from Supabase)"""
    b = await _json_body(request)
    f = b.get("filters") or {}
    if not isinstance(f, dict):
        raise HTTPException(status_code=400, detail="filters must be an object")
    if b.get("source", "video") == "citizen":
        return await _run_claims(lambda c: ER.claims_grant(
            c, source="citizen", session=b.get("session"), reviewer_name=b.get("reviewer_name"),
            queue=b.get("queue") or "citizen_pending", items=b.get("candidates"), n=b.get("n") or 12))
    if b.get("source", "video") != "video":
        raise HTTPException(status_code=400, detail="source must be video or citizen")
    out = await _run_claims(lambda c: ER.claim_next_video(
        c, session=b.get("session"), reviewer_name=b.get("reviewer_name"), queue=b.get("queue"),
        n=b.get("n") or 12, drive=f.get("drive") or None, label=f.get("label") or None))
    _warm([i["segment_id"] for i in out["items"]])
    return out


@router.post("/claims/grant")
async def claims_grant(request: Request):
    """Lease specific items: {source, queue, session, reviewer_name, items: [ids], n?}. Used before
    editing an already-reviewed item; held_by_others names who has the ones not granted."""
    b = await _json_body(request)
    return await _run_claims(lambda c: ER.claims_grant(
        c, source=b.get("source"), session=b.get("session"), reviewer_name=b.get("reviewer_name"),
        queue=b.get("queue"), items=b.get("items"), n=b.get("n")))


@router.post("/claims/heartbeat")
async def claims_heartbeat(request: Request):
    """Renew this session's leases: {session, items?: [{source, item_id}]}; answers held + lost."""
    b = await _json_body(request)
    return await _run_claims(lambda c: ER.claims_heartbeat(c, session=b.get("session"), items=b.get("items")))


@router.post("/claims/release")
async def claims_release(request: Request):
    """Drop leases: {session, items?: [{source, item_id}], source?, skip?, reviewer_name?}. The page
    sends this with navigator.sendBeacon on unload (any content type; the body is JSON)."""
    b = await _json_body(request)
    return await _run_claims(lambda c: ER.claims_release(
        c, session=b.get("session"), items=b.get("items"), source=b.get("source"),
        skip=bool(b.get("skip")), reviewer_name=b.get("reviewer_name")))


@router.get("/claims/active")
async def claims_active(me: Optional[str] = None, source: Optional[str] = None, items: bool = False):
    """Experts with live leases (by name), per list, and with items=true the items other
    sessions hold (me = this page's session, which is left out)."""
    return await _run_claims(lambda c: ER.claims_active(c, me=me or None, source=source or None,
                                                        with_items=items))


_CITIZEN_SAVE_COLS = _CITIZEN_COLS + ",reviewed_at,expert_notes"


def _citizen_patch(review: dict, now_iso: str) -> dict:
    """The fields a citizen review writes (as the page wrote them before the server did)."""
    return {"needs_expert_review": False, "expert_reviewed": True,
            "expert_corrected_types": review["types"], "expert_corrected_severity": review["severity"],
            "expert_notes": review["notes"], "reviewer_name": review["reviewer_name"], "reviewed_at": now_iso}


def _local_citizen_update(conn, assessment_id: str, review: dict, seen: Optional[str]) -> list:
    """DB_BACKEND=postgres: the same conditional update on the local assessments table."""
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
    cols = ", ".join(c.strip() for c in _CITIZEN_SAVE_COLS.split(","))
    with conn.transaction():
        return conn.cursor(row_factory=dict_row).execute(
            "UPDATE assessments SET needs_expert_review = false, expert_reviewed = true, "
            "expert_corrected_types = %s, expert_corrected_severity = %s, expert_notes = %s, reviewer_name = %s, "
            "reviewed_at = now() WHERE id = %s::uuid AND (CASE WHEN %s::timestamptz IS NULL "
            "THEN NOT COALESCE(expert_reviewed, false) ELSE reviewed_at = %s::timestamptz END) "
            f"RETURNING {cols}",
            (Jsonb(review["types"]), review["severity"], review["notes"], review["reviewer_name"], assessment_id,
             seen, seen)).fetchall()


@router.post("/citizen/review/{assessment_id}")
async def citizen_review(assessment_id: str, request: Request):
    """Save a citizen review on the server, the arbiter for several experts:
      1. take (or renew) this session's lease on the item, else 409 naming who holds it;
      2. write it to the assessment only if it is still as the page saw it: unreviewed
         (seen_reviewed_at null) or reviewed at exactly seen_reviewed_at. Exactly one row must
         change, else 409 (someone saved in between) and nothing is written;
      3. append the research log row and release the lease.
    Body: {session, reviewer_name, types, severity, notes, queue, seen_reviewed_at, heatmap, ui}."""
    if not _UUID.match(assessment_id):
        raise HTTPException(status_code=400, detail="bad assessment id")
    body = await _json_body(request)
    try:
        review = ER.clean_review(body, source="citizen", strict_types=False)
        session = ER.clean_session(body.get("session"))
        seen = body.get("seen_reviewed_at")
        if seen is not None and not isinstance(seen, str):
            raise ER.ReviewError("seen_reviewed_at must be an ISO timestamp or null")
        ER.same_instant(seen, seen)                  # parses it (ReviewError if malformed)
    except ER.ReviewError as e:
        raise HTTPException(status_code=400, detail=str(e))
    client = getattr(request.app.state, "supabase_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="Supabase client not configured on this server")
    await _run_claims(lambda c: ER.take_claim(c, source="citizen", item_id=assessment_id, session=session,
                                              reviewer_name=review["reviewer_name"], queue=review["queue"]))
    from app.supabase_client import is_local_db
    if is_local_db(client):
        rows = await _run(lambda c: _local_citizen_update(c, assessment_id, review, seen))
    else:
        from datetime import datetime, timezone
        cond = "expert_reviewed=eq.false" if seen is None else "reviewed_at=eq." + urllib.parse.quote(seen, safe="")
        resp = await client.patch(
            f"/rest/v1/assessments?id=eq.{assessment_id}&{cond}&select={_CITIZEN_SAVE_COLS}",
            json=_citizen_patch(review, datetime.now(timezone.utc).isoformat()),
            headers={"Prefer": "return=representation"})
        if not resp.is_success:
            raise HTTPException(status_code=502, detail=f"Supabase answered {resp.status_code}")
        rows = resp.json() or []
    if len(rows) != 1:
        cur = await _citizen_row(request, assessment_id)          # 404 when the row is gone
        when = cur.get("reviewed_at")
        raise HTTPException(status_code=409, detail={
            "message": f"{cur.get('reviewer_name') or 'Another expert'} saved a review of this item"
                       f"{f' ({when})' if when else ''}; reload to see it before changing it",
            "reviewed_by": cur.get("reviewer_name"), "reviewed_at": str(when) if when else None})
    row = dict(rows[0])
    for k in ("reviewed_at", "processed_at"):
        if row.get(k) is not None and not isinstance(row[k], str):
            row[k] = row[k].isoformat()
    try:
        out = await _run_claims(lambda c: ER.finish_citizen_review(c, assessment_id, body, row))
    except HTTPException as e:
        # The review is saved; only the research log failed. The page can retry /citizen/log.
        return {"assessment_id": assessment_id, "saved": True, "reviewed_at": row.get("reviewed_at"),
                "log_error": e.detail}
    return {**out, "saved": True, "reviewed_at": row.get("reviewed_at")}



def install(app: FastAPI) -> None:
    app.include_router(router)
