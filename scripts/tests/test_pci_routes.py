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
    assert text.splitlines()[0].startswith("unit,index,chainage_from")
    assert R.csv_cell("-1+2") == "'-1+2" and R.csv_cell("@x") == "'@x" and R.csv_cell(-1.5) == "-1.5"
    assert R.csv_cell("Km 0+000") == "Km 0+000"


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
        assert any(x["id"] == rid for x in lst["reports"])
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
        # stale: a segment of the report changes after the snapshot
        seg = got["report"]["segment_ids"][5]
        with _conn() as conn:
            conn.execute("UPDATE video_segments SET status = status WHERE segment_id = %s", (seg,))
        got = c.get(f"/gis/pci/reports/{rid}").json()
        assert got["stale"] is True and got["changed_segments"] == 1, got["changed_segments"]
        assert c.get(f"/gis/pci/reports/{rid}.pdf").content == stored      # frozen
    finally:
        security.VALID_API_KEYS.clear()
        security.VALID_API_KEYS.update(old)
        with _conn() as conn:
            conn.execute("DELETE FROM research.pci_reports WHERE title = %s AND NOT sample", (TITLE,))


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
