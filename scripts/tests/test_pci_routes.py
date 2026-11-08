"""
HTTP tests for app/pci_routes.py on a harness app that mounts ONLY the PCI router (never app.main:
production runs on port 8000 under a watchdog). Data: the scratch database `geoai_test` holding a copy of
the real drive survey (scripts/pci_copy_to_test_db.py, run here if the copy is missing). Never `geoai`.

  1. meta / path / report / report.pdf / report.csv answer for the longest real drive.
  2. Input validation: unknown fields, bad / outside positions, > 8 via points, lane width, chainage without
     direction, text > 120 chars, Kannada (no shaping) and control characters, non-JSON, A far from the survey.
  3. CSV formula-injection guard; rate limit per client (Cf-Connecting-Ip only when the peer is loopback).
  4. Save needs the operator key (401 without, 403 when no key is configured); dedupe by input hash; cap;
     stored JSON / PDF (byte-identical) / CSV / GeoJSON; stale flag after a segment changes; 404s.

Usage:
    venv/Scripts/python.exe scripts/tests/test_pci_routes.py
    venv/Scripts/python.exe -m pytest scripts/tests/test_pci_routes.py -q
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pytest
except ImportError:
    pytest = None

DSN = os.environ.get("DATABASE_URL")
if not DSN:
    try:
        from dotenv import dotenv_values
        DSN = dotenv_values(PROJECT_ROOT / ".env").get("DATABASE_URL")
    except ImportError:
        pass
if not DSN and pytest is not None and __name__ != "__main__":
    pytest.skip("DATABASE_URL is not set: PCI route tests skipped", allow_module_level=True)

from scripts import pci_copy_to_test_db as COPY  # noqa: E402
from scripts import setup_local_db  # noqa: E402

TEST_DSN = setup_local_db.with_database(DSN, "geoai_test")
assert "geoai_test" in TEST_DSN
os.environ["DATABASE_URL"] = TEST_DSN                      # every connection the routes open: scratch DB
os.environ.setdefault("GIS_PREVIEW_CACHE", tempfile.mkdtemp(prefix="pci_preview_"))

import psycopg  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import pci_chainage as PC  # noqa: E402
from app import pci_routes as R  # noqa: E402
from app import security  # noqa: E402

KEY = "pci-test-key"
TITLE = "PCI ROUTE TEST"
_state: dict = {}


def setup():
    if _state:
        return _state
    if not COPY.is_copied(TEST_DSN):
        print("copying the survey into geoai_test ...")
        COPY.copy("geoai_test")
    app = FastAPI()
    app.include_router(R.router)
    client = TestClient(app, client=("127.0.0.1", 50000))
    g = PC.load_graph(force=True)
    by = {}
    for n in g.nodes:
        by.setdefault(n.session_id, []).append(n)
    sid = max(by, key=lambda s: len(by[s]))
    ns = sorted(by[sid], key=lambda n: n.k)
    _state.update(client=client, g=g, a={"lat": ns[0].lat, "lon": ns[0].lon},
                  b={"lat": ns[400].lat, "lon": ns[400].lon}, created=[])
    return _state


def body(**kw):
    s = setup()
    return {"a": s["a"], "b": s["b"], **kw}


def post(path, json=None, headers=None, **kw):
    R.limiter.reset()
    return setup()["client"].post(path, json=json, headers=headers or {}, **kw)


def test_meta_path_report():
    c = setup()["client"]
    R.limiter.reset()
    m = c.get("/gis/pci/meta").json()
    assert m["schema"] == "geoai.pci.v1" and len(m["bbox"]) == 4 and m["limits"]["via_max"] == 8
    r = post("/gis/pci/path", body())
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["length_m"] > 1500 and p["geojson"]["type"] == "FeatureCollection" and p["passes"]
    assert "_path" not in p and "vertices" not in p
    r = post("/gis/pci/report", body(road_name="Test road", category="urban"))
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["schema"] == "geoai.pci.v1" and rep["sections"] and rep["km"] and rep["stretch"]
    assert rep["header"]["road_name"] == "Test road" and rep["report_id"] is None
    assert rep["header"]["table"] == "5.3"
    r = post("/gis/pci/report.pdf", body())
    assert r.status_code == 200 and r.content[:5] == b"%PDF-" and r.headers["content-type"] == "application/pdf"


def test_validation():
    cases = [
        ({"a": {"lat": 1, "lon": 2}, "b": {"lat": 1, "lon": 2}, "zzz": 1}, 400, "unknown field"),
        ({"a": {"lat": "x", "lon": 2}, "b": {"lat": 1, "lon": 2}}, 422, "must be a number"),
        ({"a": {"lat": "1e400", "lon": 2}, "b": {"lat": 1, "lon": 2}}, 422, "finite"),
        ({"a": {"lat": 13.5, "lon": 77.6}, "b": {"lat": 12.9, "lon": 77.6}}, 422, "outside the surveyed area"),
        (body(via=[setup()["a"]] * 9), 422, "at most 8"),
        (body(lane_width_m=1), 422, "2-30"),
        (body(lane_width_m=True), 422, "number"),
        (body(chainage_at_a_km=12.0), 422, "increases towards B"),
        (body(chainage_at_a_km=4000, chainage_increases_towards_b=True), 422, "0-3000"),
        (body(category="village"), 422, "category"),
        (body(road_name="x" * 121), 422, "longer than 120"),
        (body(road_name="ಹೊಸೂರು ರಸ್ತೆ"), 422, "English letters"),
        (body(road_name="a\x07b"), 422, "control"),
        (body(road_name="Hosur\u202eRoad"), 422, "right-to-left"),
        (body(road_name="Hosur\u200bRoad"), 422, "zero-width"),
        (body(road_name="Road \U0001F600"), 422, "English letters"),
        (body(chainage_at_a_km=0.05, chainage_increases_towards_b=False), 422, "negative at B"),
        (body(prepared_by=5), 422, "text"),
    ]
    for js, code, text in cases:
        r = post("/gis/pci/report", js)
        assert r.status_code == code and text in r.text, (js, r.status_code, r.text)
    r = post("/gis/pci/path", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    a = setup()["a"]
    far = {"lat": a["lat"] + 0.0006, "lon": a["lon"] + 0.0006}      # ~90 m away, inside the bbox
    r = post("/gis/pci/path", body(a=far))
    assert r.status_code in (200, 422)
    if r.status_code == 422:
        assert "limit 40 m" in r.text
    r = post("/gis/pci/report", body(b=a))
    assert r.status_code == 422 and "same point" in r.text


def test_csv_formula_guard():
    r = post("/gis/pci/report.csv", body(road_name="=HYPERLINK(\"http://x\")"))
    assert r.status_code == 200, r.text
    text = r.content.decode("utf-8-sig")
    assert "'=HYPERLINK" in text and "\n=HYPERLINK" not in text and ",=HYPERLINK" not in text
    lines = text.splitlines()
    assert lines[0].startswith("# report,") and lines[[i for i, x in enumerate(lines)
                                                          if not x.startswith("#")][0]].startswith(
        "unit,index,chainage_from")
    assert R.csv_cell("-1+2") == "'-1+2" and R.csv_cell("@x") == "'@x" and R.csv_cell(-1.5) == "-1.5"
    assert R.csv_cell("Km 0+000") == "Km 0+000"


def test_csv_irc2023_columns():
    """report.csv (IRC:82-2023): '#' header lines (edition, road class, surface, whole-stretch PCI), then one row
    per 100 m section, per km row and for the stretch; per parameter value / lo / hi / PCI_i / source; PCI lo/hi
    bracket the single PCI; not-assessed parameters are empty with source 'not_assessed'; ends have 7-decimal
    positions; the stretch row carries the headline (summary.pci)."""
    import csv as _csv
    r = post("/gis/pci/report.csv", body(road_name="CSV test road", surface="SDBC", entered={"rut_depth": 4}))
    assert r.status_code == 200, r.text
    text = r.content.decode("utf-8-sig")
    meta = {row[0][2:]: (row[1] if len(row) > 1 else "") for row in _csv.reader(
        [x for x in text.splitlines() if x.startswith("# ")])}
    assert meta["edition"] == "IRC:82-2023" and meta["road_class"] == "urban"
    assert meta["surface"].startswith("SDBC") and meta["road_name (as entered)"] == "CSV test road"
    assert "whole-stretch PCI" in meta and "columns" in meta
    rows = list(_csv.DictReader([x for x in text.splitlines() if not x.startswith("#")]))
    assert tuple(rows[0].keys()) == R.CSV_COLUMNS
    rep = post("/gis/pci/report", body(road_name="CSV test road", surface="SDBC", entered={"rut_depth": 4})).json()
    kinds = [x["unit"] for x in rows]
    assert kinds.count("100m") == len(rep["sections"]) and kinds.count("km") == len(rep["km"])
    assert kinds[-1] == "stretch"
    for x in rows:
        assert len(x["start_lat"].split(".")[1]) == 7 and len(x["end_lon"].split(".")[1]) == 7, x
        if x["pci_class"] == "insufficient data":
            continue
        lo, hi = float(x["pci_lo"]), float(x["pci_hi"])
        assert 0 <= lo <= hi <= 100, x
        if x["pci_single"]:
            assert lo - 1e-6 <= float(x["pci_single"]) <= hi + 1e-6
        assert x["rut_depth_mm"] == "4" and x["rut_depth_source"] == "entered", x
        for p in ("ravelling", "patching"):
            if x[f"{p}_source"] == "not_assessed":
                assert x[f"{p}_pct"] == "" and x[f"{p}_pci_i"] == "", x
        assert x["cracking_source"].startswith("labels") and x["potholes_source"].startswith("labels"), x
    st = rows[-1]
    hp = rep["summary"]["pci"]
    assert abs(float(st["pci_lo"]) - hp["lo"]) < 0.01 and abs(float(st["pci_hi"]) - hp["hi"]) < 0.01


def test_rate_limit_per_client():
    def req(host, headers=None):
        return SimpleNamespace(client=SimpleNamespace(host=host), headers=headers or {})
    assert R.client_key(req("127.0.0.1", {"cf-connecting-ip": "1.2.3.4"})) == "cf:1.2.3.4"
    assert R.client_key(req("127.0.0.1")) == "local"
    assert R.client_key(req("::1", {"cf-connecting-ip": "5.6.7.8"})) == "cf:5.6.7.8"
    assert R.client_key(req("10.0.0.5", {"cf-connecting-ip": "1.2.3.4"})) == "ip:10.0.0.5"   # header not trusted
    c = setup()["client"]
    R.limiter.reset()
    js = body()
    codes = [c.post("/gis/pci/path", json=js, headers={"Cf-Connecting-Ip": "9.9.9.9"}).status_code for _ in range(31)]
    assert codes[:30] == [200] * 30 and codes[30] == 429, codes
    # another tunnel user is not locked out
    assert c.post("/gis/pci/path", json=js, headers={"Cf-Connecting-Ip": "8.8.8.8"}).status_code == 200
    R.limiter.reset()


def test_wrong_keys_are_rate_limited():
    old = set(security.VALID_API_KEYS)
    try:
        security.VALID_API_KEYS.clear()
        security.VALID_API_KEYS.add(KEY)
        R.limiter.reset()
        c = setup()["client"]
        codes = [c.post("/gis/pci/reports", json=body(title=TITLE), headers={"X-API-Key": f"guess{i}",
                                                                         "Cf-Connecting-Ip": "7.7.7.7"}).status_code
                 for i in range(30)]
        first429 = codes.index(429)
        assert codes[:first429] == [401] * first429 and first429 <= 10, codes
        assert set(codes[first429:]) == {429}, codes
        # the right key is also refused while the client is locked out (no oracle); another client is not
        r = c.post("/gis/pci/reports", json=body(title=TITLE), headers={"X-API-Key": KEY, "Cf-Connecting-Ip": "7.7.7.7"})
        assert r.status_code == 429
        r = c.post("/gis/pci/reports", json={"a": 1}, headers={"X-API-Key": "x", "Cf-Connecting-Ip": "6.6.6.6"})
        assert r.status_code == 401
    finally:
        security.VALID_API_KEYS.clear()
        security.VALID_API_KEYS.update(old)
        R.limiter.reset()


def _conn():
    return psycopg.connect(TEST_DSN, autocommit=True)


def test_save_dedupe_cap_stored_and_stale():
    s = setup()
    c = s["client"]
    old = set(security.VALID_API_KEYS)
    try:
        security.VALID_API_KEYS.clear()
        r = post("/gis/pci/reports", body(title=TITLE))
        assert r.status_code == 403, r.text
        security.VALID_API_KEYS.add(KEY)
        assert post("/gis/pci/reports", body(title=TITLE)).status_code == 401
        assert post("/gis/pci/reports", body(title=TITLE), headers={"X-API-Key": "wrong"}).status_code == 401
        r = post("/gis/pci/reports", body(title=TITLE, road_name="=cmd"), headers={"X-API-Key": KEY})
        assert r.status_code == 200, r.text
        first = r.json()
        s["created"].append(first["id"])
        r2 = post("/gis/pci/reports", body(title=TITLE, road_name="=cmd"), headers={"X-API-Key": KEY}).json()
        assert r2["id"] == first["id"] and r2["created"] is False
        assert first["created"] in (True, False)
        rid = first["id"]
        R.limiter.reset()
        lst = c.get("/gis/pci/reports").json()
        mine = next(x for x in lst["reports"] if x["id"] == rid)
        assert mine["headline_text"].startswith("PCI ") and " m" in mine["headline_text"]
        assert set(mine["bands"]) == {"potholes", "cracking"}
        got = c.get(f"/gis/pci/reports/{rid}").json()
        assert got["report"]["report_id"] == rid and got["stale"] is False and got["sample"] is False
        pdf = c.get(f"/gis/pci/reports/{rid}.pdf")
        with _conn() as conn:
            stored = bytes(conn.execute("SELECT pdf FROM research.pci_reports WHERE id = %s", (rid,)).fetchone()[0])
        assert pdf.status_code == 200 and pdf.content == stored
        assert "'=cmd" in c.get(f"/gis/pci/reports/{rid}.csv").content.decode("utf-8-sig")
        gj = c.get(f"/gis/pci/reports/{rid}.geojson").json()
        assert gj["type"] == "FeatureCollection" and any(f["properties"]["kind"] == "section" for f in gj["features"])
        assert c.get("/gis/pci/reports/999999999").status_code == 404
        assert c.get("/gis/pci/reports/abc.pdf").status_code == 404
        assert c.get(f"/gis/pci/reports/{rid}.exe").status_code == 404
        # cap: a full store refuses a NEW input (samples do not count)
        with _conn() as conn:
            n = conn.execute("SELECT count(*) FROM research.pci_reports WHERE NOT sample").fetchone()[0]
        saved_cap = R.MAX_STORED
        R.MAX_STORED = n
        try:
            r = post("/gis/pci/reports", body(title=TITLE, lane_width_m=3.0), headers={"X-API-Key": KEY})
            assert r.status_code == 409 and "full" in r.text, r.text
        finally:
            R.MAX_STORED = saved_cap
        # stale: a segment of the report changes after the snapshot; its updated_at is restored afterwards
        # (REVIEW_backend #12: the shared scratch DB must not drift)
        seg = got["report"]["segment_ids"][5]
        with _conn() as conn:
            before = conn.execute("SELECT updated_at FROM video_segments WHERE segment_id = %s", (seg,)).fetchone()[0]
            conn.execute("UPDATE video_segments SET status = status WHERE segment_id = %s", (seg,))
            try:
                got = c.get(f"/gis/pci/reports/{rid}").json()
                assert got["stale"] is True and got["changed_segments"] == 1, got["changed_segments"]
                assert c.get(f"/gis/pci/reports/{rid}.pdf").content == stored      # frozen
            finally:
                conn.execute("ALTER TABLE video_segments DISABLE TRIGGER trg_video_segments_updated_at")
                try:
                    conn.execute("UPDATE video_segments SET updated_at = %s WHERE segment_id = %s", (before, seg))
                finally:
                    conn.execute("ALTER TABLE video_segments ENABLE TRIGGER trg_video_segments_updated_at")
                after = conn.execute("SELECT updated_at FROM video_segments WHERE segment_id = %s", (seg,)).fetchone()[0]
                assert after == before
    finally:
        security.VALID_API_KEYS.clear()
        security.VALID_API_KEYS.update(old)
        with _conn() as conn:
            conn.execute("DELETE FROM research.pci_reports WHERE title = %s AND NOT sample", (TITLE,))


def test_validation_irc2023_inputs():
    """IRC:82-2023 request fields: surface, entered {parameter: value}, category mdr (critic, 2026-11-08)."""
    cases = [
        (body(surface="XYZ"), 422, "surface must be one of"),
        (body(surface=5), 422, "surface must be one of"),
        (body(entered=[1, 2]), 422, "entered must be"),
        (body(entered={"shoving": 1}), 422, "entered must be"),
        (body(entered={"iri": 0.1}), 422, "entered.iri must be within"),
        (body(entered={"iri": 31}), 422, "entered.iri must be within"),
        (body(entered={"cracking": 100.5}), 422, "entered.cracking must be within"),
        (body(entered={"potholes": -1}), 422, "entered.potholes must be within"),
        (body(entered={"rut_depth": "x"}), 422, "number"),
        (body(entered={"rut_depth": "1e400"}), 422, "finite"),
        (body(entered={"rut_depth": True}), 422, "number"),
    ]
    for js, code, text in cases:
        r = post("/gis/pci/report", js)
        assert r.status_code == code and text in r.text, (js, r.status_code, r.text)
    # blanks are ignored, not errors
    r = post("/gis/pci/report", body(surface="", entered={"iri": None, "cracking": ""}))
    assert r.status_code == 200, r.text
    assert "entered" not in r.json()["input"] and "surface" not in r.json()["input"]
    # all six entered: every rated unit's single PCI = the Appendix-2 equations on the entered values
    from app import pci_irc2023 as P23
    ent = {"cracking": 5, "ravelling": 2, "potholes": 3, "patching": 3.5, "rut_depth": 4, "iri": 6}
    r = post("/gis/pci/report", body(category="urban", surface="BC", entered=ent))
    assert r.status_code == 200, r.text
    rep = r.json()
    rated = [u for u in rep["km"] if u["pci"].get("pci")]
    assert rated
    for u in rated:                       # entered potholes are per km, scaled to the unit's length
        km = (u["to_m"] - u["from_m"]) / 1000.0
        want = P23.pci("urban", {**ent, "potholes": ent["potholes"] * km})["pci"]
        assert abs(u["pci"]["pci"]["single"] - want) < 0.05, (u["pci"]["pci"]["single"], want)
    full = [u for u in rated if not u.get("part_km")]
    if full:
        assert abs(full[0]["pci"]["pci"]["single"] - P23.pci("urban", ent)["pci"]) < 0.05
    # a surface the road class has no roughness equation for (rural + BC) must not crash the report
    r = post("/gis/pci/report", body(category="rural", surface="BC", entered={"iri": 3.0}))
    assert r.status_code in (200, 422), r.text
    if r.status_code == 200:
        km = next(u for u in r.json()["km"] if u["pci"].get("pci"))
        assert "Roughness (IRI)" in km["pci"]["pci"]["missing"]


def test_report_mdr_category():
    """MDR is its own category (2023 Table 5.2 / A2.2 Part A); pothole PCI not computable for MDR -> bracketed."""
    r = post("/gis/pci/report", body(category="mdr", surface="SD"))
    assert r.status_code == 200, r.text[:300]
    rep = r.json()
    assert rep["header"]["category"] == "mdr" and rep["header"]["surface"] == "SD"
    km = next(u for u in rep["km"] if u["pci"].get("pci"))
    assert "Potholes" in km["pci"]["pci"]["missing"]


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    failures = []
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failures.append(t.__name__)
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - len(failures)}/{len(TESTS)} tests passed")
    sys.exit(1 if failures else 0)
