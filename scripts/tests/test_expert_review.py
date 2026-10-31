"""
Tests for the expert review log, the Video tab API and the "where the model looked" endpoint
(app/expert_review.py, app/expert_routes.py, migration 006).

What must hold:
  1. compare() gives the documented verdicts (right / close / wrong, severity steps, first label),
     maps the citizen tab's older type names to IRC:82, and never counts Normal as a type.
  2. Bad input is refused: unknown types on video, Normal mixed with a distress, a distress
     without severity, a rating for an overlay that was not opened, an unknown queue.
  3. Saving a video review writes the expert_* columns of that segment's assessment, leaves the
     model's answer and the gate's decision (status, needs_expert_review) untouched, and appends
     one log row with the model's answer read server-side. An update appends a second row.
  4. Metrics use the latest review per item, but count the item under the queue it was FIRST
     reviewed from (an edit from the Reviewed list must not move it out of the random sample).
  5. Over HTTP: the list/review/metrics endpoints work, a citizen log is refused unless Supabase
     already holds that review, and an explanation is computed once and then served from the
     disk cache (fake classifier, no GPU).

Uses the scratch database `geoai_test` and the fake drive of scripts/tests/test_local_db.py.
The pure tests (1, 2) run without a database.

Usage:
    venv/Scripts/python.exe scripts/tests/test_expert_review.py
    venv/Scripts/python.exe -m pytest scripts/tests/test_expert_review.py -q
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pytest
except ImportError:
    pytest = None

from app import expert_review as ER  # noqa: E402

DSN = os.environ.get("DATABASE_URL")
if not DSN:
    try:
        from dotenv import dotenv_values
        DSN = dotenv_values(PROJECT_ROOT / ".env").get("DATABASE_URL")
        if DSN:
            os.environ["DATABASE_URL"] = DSN
    except ImportError:
        pass

V_DIST = {"stage1_label": "Distress", "distress_types": ["Potholes", "Longitudinal Cracking"], "severity": "Medium"}
V_NORM = {"stage1_label": "Normal", "distress_types": [], "severity": "None"}


def _raises(fn, exc=ER.ReviewError):
    try:
        fn()
    except exc:
        return True
    raise AssertionError(f"{fn} did not raise {exc.__name__}")


# ============================================================
# 1-2. Pure
# ============================================================

def test_compare_verdicts():
    c = ER.compare(V_DIST, ["Potholes", "Longitudinal Cracking"], "Medium")
    assert c["verdict"] == "right" and c["severity"] == "exact" and c["first_label"] == "right"
    c = ER.compare(V_DIST, ["Potholes"], "High")
    assert c["verdict"] == "close" and c["jaccard"] == 0.5 and c["severity"] == "close"
    assert c["severity_diff"] == -1 and c["extra_types"] == ["Longitudinal Cracking"]
    c = ER.compare(V_DIST, ["Alligator Cracking"], "Low")
    assert c["verdict"] == "wrong" and c["first_label"] == "wrong" and c["missed_types"] == ["Alligator Cracking"]
    c = ER.compare({**V_DIST, "severity": "High"}, ["Potholes"], "Low")
    assert c["severity"] == "wrong"
    assert ER.compare(V_DIST, ["Normal"], "None")["verdict"] == "wrong"
    assert ER.compare(V_NORM, ["Potholes"], "Low")["stage1"] == "disagree"
    c = ER.compare(V_NORM, ["Normal"], "None")
    assert c["verdict"] == "right" and "severity" not in c and "first_label" not in c
    # older citizen names map to IRC:82
    assert ER.compare(V_DIST, ["Pothole (D40)", "Longitudinal Crack (D00)"], "Medium")["verdict"] == "right"
    assert ER.canonical("Block Crack (D43)") == "Transverse Cracking"
    assert ER.canonical("potholes") == "Potholes" and ER.canonical("Open Joint (D50)") == "Open Joint (D50)"
    # a Normal answer's stray types are ignored
    assert ER.compare({"stage1_label": "Normal", "distress_types": ["Potholes"]}, ["Normal"], "None")["verdict"] == "right"


def test_clean_review_validation():
    ok = {"reviewer_name": "Asha", "types": ["Potholes"], "severity": "medium", "queue": "random"}
    r = ER.clean_review(ok, source="video", strict_types=True)
    assert r["severity"] == "Medium" and r["types"] == ["Potholes"] and r["heatmap"]["viewed"] is False
    r = ER.clean_review({**ok, "types": ["normal"], "severity": "High"}, source="video", strict_types=True)
    assert r["types"] == ["Normal"] and r["severity"] == "None"
    _raises(lambda: ER.clean_review({**ok, "types": ["Normal", "Potholes"]}, source="video", strict_types=True))
    _raises(lambda: ER.clean_review({**ok, "types": ["Open Joint"]}, source="video", strict_types=True))
    _raises(lambda: ER.clean_review({**ok, "severity": "None"}, source="video", strict_types=True))
    _raises(lambda: ER.clean_review({**ok, "queue": "citizen_pending"}, source="video", strict_types=True))
    _raises(lambda: ER.clean_review({**ok, "reviewer_name": "A"}, source="video", strict_types=True))
    _raises(lambda: ER.clean_review({**ok, "types": []}, source="video", strict_types=True))
    _raises(lambda: ER.clean_review({**ok, "heatmap": {"types": [], "ratings": {"Potholes": "on"}}},
                                    source="video", strict_types=True))
    _raises(lambda: ER.clean_review({**ok, "heatmap": {"types": ["Potholes"], "ratings": {"Potholes": "yes"}}},
                                    source="video", strict_types=True))
    r = ER.clean_review({**ok, "heatmap": {"types": ["Potholes"], "ratings": {"potholes": "on"}},
                         "ui": {"ms_since_first_interaction": 5300.7}}, source="video", strict_types=True)
    assert r["heatmap"] == {"viewed": True, "types": ["Potholes"], "ratings": {"Potholes": "on"}}
    assert r["ui"] == {"ms_since_first_interaction": 5300}
    # citizen: free names allowed (custom types), queue from the citizen tabs
    r = ER.clean_review({**ok, "types": ["Open Joint (D50)"], "queue": "citizen_pending"},
                        source="citizen", strict_types=False)
    assert r["types"] == ["Open Joint (D50)"]


def test_wilson():
    assert ER.wilson(0, 0) is None
    lo, hi = ER.wilson(8, 10)
    assert 0.49 < lo < 0.50 and 0.94 < hi < 0.95


# ============================================================
# 3-5. Database and HTTP (scratch database geoai_test)
# ============================================================

if DSN:
    from scripts.tests.test_local_db import SESSION, Env  # noqa: E402

    class XEnv(Env):
        def __init__(self):
            os.environ.setdefault("DATABASE_URL", DSN)
            super().__init__()
            from app import video_store
            video_store.store_session(self.conn, self.run, self.answers, image_root=self.image_root)
            # every connection the routes open goes to the scratch database
            self.prev_url = os.environ.get("DATABASE_URL")
            os.environ["DATABASE_URL"] = self.dsn
            ER._schema_ready = False

        def close(self):
            os.environ["DATABASE_URL"] = self.prev_url
            super().close()

    def seg(k):
        return f"{SESSION}_seg{k:05d}"

    if pytest is not None:
        @pytest.fixture(scope="module")
        def xenv():
            e = XEnv()
            yield e
            e.close()


def test_video_review_flow(xenv):
    c = xenv.conn
    page = ER.list_video(c, "needs_review")
    assert page["counts"] == {"needs_review": 1, "auto": 2, "random": 3, "reviewed": 0}, page["counts"]
    assert [i["segment_id"] for i in page["items"]] == [seg(5)]
    assert page["items"][0]["image"] == f"/gis/image/{seg(5)}"
    assert ER.list_video(c, "auto", label="Normal")["total"] == 1
    assert ER.list_video(c, "auto", label="Potholes")["total"] == 1
    rnd = [i["segment_id"] for i in ER.list_video(c, "random")["items"]]
    assert sorted(rnd) == sorted([seg(0), seg(1), seg(5)])
    assert rnd == [i["segment_id"] for i in ER.list_video(c, "random")["items"]]     # stable order

    before = c.execute("SELECT status, needs_expert_review, distress_types, severity FROM assessments "
                       "WHERE road_segment_id = %s", (seg(5),)).fetchone()
    out = ER.save_video_review(c, seg(5), {"reviewer_name": "Asha", "types": ["Ravelling"], "severity": "Medium",
                                           "queue": "needs_review", "notes": "clear ravelling",
                                           "heatmap": {"types": ["Ravelling"], "ratings": {"Ravelling": "partly"}},
                                           "ui": {"ms_since_first_interaction": 12000}})
    assert out["action"] == "confirm" and out["comparison"]["verdict"] == "right"
    row = c.execute("SELECT status, needs_expert_review, distress_types, severity, expert_reviewed, "
                    "expert_corrected_types, expert_corrected_severity, expert_notes, reviewer_name, reviewed_at "
                    "FROM assessments WHERE road_segment_id = %s", (seg(5),)).fetchone()
    assert row[:4] == before, "the model's answer and the gate's decision must not change"
    assert row[4] is True and row[5] == ["Ravelling"] and row[6] == "Medium" and row[7] == "clear ravelling"
    assert row[8] == "Asha" and row[9] is not None
    log = c.execute("SELECT vlm, queue, action, heatmap FROM research.expert_review_log").fetchall()
    assert len(log) == 1 and log[0][1] == "needs_review" and log[0][2] == "confirm"
    assert log[0][0]["status"] == "expert_review" and log[0][0]["distress_types"] == ["Ravelling"]
    assert log[0][3]["ratings"] == {"Ravelling": "partly"}

    # update from the Reviewed list: a second log row; the item stays in its first queue
    out = ER.save_video_review(c, seg(5), {"reviewer_name": "Ravi", "types": ["Potholes"], "severity": "High",
                                           "queue": "reviewed"})
    assert out["action"] == "update" and out["comparison"]["verdict"] == "wrong"
    assert c.execute("SELECT COUNT(*) FROM research.expert_review_log").fetchone()[0] == 2

    ER.save_video_review(c, seg(1), {"reviewer_name": "Asha", "types": ["Normal"], "queue": "random"})
    ER.save_video_review(c, seg(0), {"reviewer_name": "Asha", "types": ["Potholes"], "severity": "High",
                                     "queue": "auto"})
    _raises(lambda: ER.save_video_review(c, seg(2), {"reviewer_name": "Asha", "types": ["Normal"],
                                                     "queue": "random"}), ER.NotFound)
    _raises(lambda: ER.save_video_review(c, seg(0), {"reviewer_name": "Asha", "types": ["Normal", "Potholes"],
                                                     "queue": "auto"}))
    assert c.execute("SELECT COUNT(*) FROM research.expert_review_log").fetchone()[0] == 4
    assert ER.list_video(c, "reviewed")["total"] == 3 and ER.list_video(c, "random")["total"] == 0

    m = ER.metrics(c)
    assert m["log_rows"] == 4 and m["reviewed_items"] == 3
    a = m["all"]
    assert a["verdict"]["right"]["k"] == 1 and a["verdict"]["close"]["k"] == 1 and a["verdict"]["wrong"]["k"] == 1
    assert set(m["by_queue"]) == {"needs_review", "random", "auto"}, m["by_queue"].keys()
    assert m["by_queue"]["needs_review"]["verdict"]["wrong"]["k"] == 1        # latest review, first queue
    assert a["gate"]["flagged"]["n"] == 1 and a["gate"]["flagged"]["wrong"]["k"] == 1
    assert a["stage1"]["tn"] == 1 and a["stage1"]["tp"] == 2
    sev = a["both_distress"]["severity"]
    assert sev["exact"]["k"] == 1 and sev["close"]["k"] == 1                 # seg0 High/High, seg5 Medium/High
    assert a["per_type"]["Potholes"]["tp"] == 1 and a["per_type"]["Alligator Cracking"]["fp"] == 1
    assert a["per_type"]["Potholes"]["fn"] == 1                               # seg5: expert Potholes, model Ravelling
    assert m["reviewers"] == {"Ravi": 1, "Asha": 2}


def test_citizen_log(xenv):
    row = {"id": "11111111-2222-3333-4444-555555555555", "status": "expert_review", "stage1_label": "Distress",
           "stage1_confidence": 0.7, "distress_types": ["Potholes"], "severity": "High", "stage2_confidence": 0.6,
           "needs_expert_review": False, "model_version": "x", "processed_at": None}
    out = ER.log_citizen_review(xenv.conn, row["id"], {"reviewer_name": "Asha", "types": ["Pothole (D40)"],
                                                      "severity": "High", "queue": "citizen_pending"}, row)
    assert out["comparison"]["verdict"] == "right"
    vlm = xenv.conn.execute("SELECT vlm FROM research.expert_review_log WHERE source = 'citizen'").fetchone()[0]
    assert vlm["needs_expert_review"] is True          # recovered from status after the page cleared the flag
    m = ER.metrics(xenv.conn)
    assert "citizen" in m["by_source"] and m["by_source"]["citizen"]["n"] == 1


def test_http_endpoints(xenv):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import app.model as model_mod
    from app import expert_routes

    app = FastAPI()
    expert_routes.install(app)
    from app.gis_routes import install as install_gis
    install_gis(app)
    calls = []

    class FakeClf:
        def explain_type(self, image, key):
            calls.append((image.size, key))
            return {"type": key, "p_yes": 0.8, "logodds": 1.4, "grid": [2, 3],
                    "map": [[0, 0.5, 1.0], [0, 0, 0.2]], "peak": [0, 2], "variant": "gxa@8", "time_ms": 1.0}

    prev_get, prev_cache = model_mod.get_classifier, os.environ.get("EXPERT_GRADCAM_CACHE")
    model_mod.get_classifier = lambda *a, **k: FakeClf()
    os.environ["EXPERT_GRADCAM_CACHE"] = tempfile.mkdtemp(prefix="gradcam_cache_test_")
    try:
        cl = TestClient(app)
        assert cl.get("/expert-api/meta").json()["types"][-1] == "Patching"
        d = cl.get("/expert-api/video/list", params={"queue": "reviewed"}).json()
        assert d["total"] == 3
        assert cl.get("/expert-api/video/list", params={"queue": "nope"}).status_code == 400
        assert cl.get("/expert-api/video/sessions").json()["items"][0]["session_id"] == SESSION
        r = cl.post(f"/expert-api/video/review/{seg(0)}", json={"reviewer_name": "Asha", "types": ["Rutting"],
                                                                "severity": "Low", "queue": "reviewed"})
        assert r.status_code == 200 and r.json()["comparison"]["verdict"] == "wrong"
        assert cl.post(f"/expert-api/video/review/{seg(0)}", json={"types": ["Rutting"]}).status_code == 400
        assert cl.post(f"/expert-api/video/review/{seg(2)}", json={"reviewer_name": "Asha", "types": ["Normal"],
                                                                   "queue": "random"}).status_code == 404
        assert cl.post("/expert-api/video/review/bad id!", json={}).status_code in (400, 404)
        assert cl.get("/expert-api/metrics").json()["log_rows"] == 6
        # citizen log without a Supabase client configured on this app
        assert cl.post("/expert-api/citizen/log/11111111-2222-3333-4444-555555555555",
                       json={"types": ["Potholes"]}).status_code == 503
        # explanation: computed once, then from the cache
        e1 = cl.get(f"/expert-api/explain/video/{seg(0)}", params={"type": "Potholes"})
        assert e1.status_code == 200, e1.text
        assert e1.json()["cached"] is False and e1.json()["peak"] == [0, 2] and "reference_only" in e1.json()
        e2 = cl.get(f"/expert-api/explain/video/{seg(0)}", params={"type": "Potholes"}).json()
        assert e2["cached"] is True and len(calls) == 1
        assert cl.get(f"/expert-api/explain/video/{seg(0)}", params={"type": "Open Joint"}).status_code == 400
        assert cl.get(f"/expert-api/explain/video/{seg(2)}", params={"type": "Potholes"}).status_code == 404
    finally:
        model_mod.get_classifier = prev_get
        if prev_cache is None:
            os.environ.pop("EXPERT_GRADCAM_CACHE", None)
        else:
            os.environ["EXPERT_GRADCAM_CACHE"] = prev_cache


# ============================================================
# Standalone runner
# ============================================================

def main() -> int:
    pure = [test_compare_verdicts, test_clean_review_validation, test_wilson]
    db = [test_video_review_flow, test_citizen_log, test_http_endpoints]
    failed = 0
    for t in pure:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    if not DSN:
        print("SKIP database tests: DATABASE_URL is not set")
        return 1 if failed else 0
    env = XEnv()
    try:
        for t in db:
            try:
                t(env)
                print(f"PASS {t.__name__}")
            except Exception:
                failed += 1
                print(f"FAIL {t.__name__}")
                traceback.print_exc()
    finally:
        env.close()
    print("ALL PASSED" if not failed else f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
