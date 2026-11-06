"""
Expert saves land exactly once, whatever goes wrong (app/expert_review.py, app/expert_routes.py,
migration 008). The page queues every save in the browser and retries it until the server confirms
it, so the server must treat a second arrival of the same save (same op_id) as a replay.

What must hold:
  1. Video: the same save twice -> both 200, the second a replay, ONE log row, lease released.
  2. Video: eight copies of one save at the same moment -> all 200, ONE log row, one log id.
  3. An op_id reused for another item -> 400.
  4. Citizen: Supabase unreachable -> 502 and a supabase_error event, nothing logged; the retry saves.
  5. Citizen: the answer is lost and the save is sent again -> a replay (ONE log row), even after
     another expert has since updated the review; a different reviewer with the same content -> 409.
  6. Citizen: Supabase saved but the research log failed -> saved + log_error; the retry logs it once.
  7. _db retries a deadlock / lock timeout on a fresh connection, and a REAL deadlock between two
     connections ends with both units of work done.
  8. Stress: 6 experts x 25 random operations (hand-out, heartbeat, save, resend, skip) on one drive
     at once: no error other than 409, every op_id logged at most once, no item held twice.
  9. /events validates and stores page events; /events/summary counts them and the saves still
     retrying; /leaderboard counts distinct items per expert (case-insensitive name), all and today.

Scratch database geoai_test only (via test_expert_concurrency.CEnv). No GPU.

Usage:
    venv/Scripts/python.exe scripts/tests/test_expert_save_robustness.py
    venv/Scripts/python.exe -m pytest scripts/tests/test_expert_save_robustness.py -q
"""

from __future__ import annotations

import os
import random
import sys
import threading
import traceback
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pytest
except ImportError:
    pytest = None

from scripts.tests import test_expert_concurrency as TC  # noqa: E402

if not TC.DSN and pytest is not None and __name__ != "__main__":
    pytest.skip("DATABASE_URL is not set: local PostgreSQL tests skipped", allow_module_level=True)

from app import expert_review as ER  # noqa: E402

sess, bseg, BIG = TC.sess, TC.bseg, TC.BIG


def op() -> str:
    return uuid.uuid4().hex


if TC.DSN and pytest is not None:
    @pytest.fixture(scope="module")
    def cenv():
        e = TC.CEnv()
        yield e
        e.close()


def _clean_extra(cenv):
    cenv.fresh()
    with cenv.conn.transaction():
        cenv.conn.execute("DELETE FROM research.expert_review_log")
        cenv.conn.execute("DELETE FROM research.expert_events")


def _app(client=None):
    from fastapi import FastAPI
    from app import expert_routes
    app = FastAPI()
    expert_routes.install(app)
    app.state.supabase_client = client
    return app


def _log_rows(cenv, **where):
    k, v = next(iter(where.items()))
    return cenv.conn.execute(f"SELECT id, op_id FROM research.expert_review_log WHERE {k} = %s", (v,)).fetchall()


def _video_body(session, name, op_id, types=("Normal",), queue="needs_review", seen=None):
    return {"session": session, "reviewer_name": name, "types": list(types), "queue": queue,
            "severity": "None" if list(types) == ["Normal"] else "Medium", "seen_reviewed_at": seen, "op_id": op_id}


# ============================================================
# 1-3. Video replays
# ============================================================

def test_video_resend_is_replay(cenv):
    from fastapi.testclient import TestClient
    _clean_extra(cenv)
    cl = TestClient(_app())
    a = sess("rsA")
    d = cl.post("/expert-api/claims/next", json={"source": "video", "queue": "needs_review", "session": a,
                                                  "reviewer_name": "Asha", "n": 2, "filters": {"drive": BIG}}).json()
    item = d["items"][0]["segment_id"]
    o = op()
    r1 = cl.post(f"/expert-api/video/review/{item}", json=_video_body(a, "Asha", o))
    r2 = cl.post(f"/expert-api/video/review/{item}", json=_video_body(a, "Asha", o))
    assert r1.status_code == 200 and r2.status_code == 200, (r1.text, r2.text)
    assert not r1.json().get("replayed") and r2.json()["replayed"] is True
    assert r1.json()["log_id"] == r2.json()["log_id"] and r2.json()["reviewed_at"] == r1.json()["reviewed_at"]
    assert len(_log_rows(cenv, segment_id=item)) == 1
    assert TC._holder(cenv.conn, item) is None
    # a NEW save of the same item (no op_id match) with the stale view is still a conflict
    r3 = cl.post(f"/expert-api/video/review/{item}", json=_video_body(a, "Asha", op()))
    assert r3.status_code == 409
    # 3. an op_id reused for another item
    other = d["items"][1]["segment_id"]
    r4 = cl.post(f"/expert-api/video/review/{other}", json=_video_body(a, "Asha", o))
    assert r4.status_code == 400 and "another item" in r4.text
    assert cl.post(f"/expert-api/video/review/{other}", json=_video_body(a, "Asha", "bad id!")).status_code == 400


def test_video_simultaneous_copies(cenv):
    _clean_extra(cenv)
    from app import expert_routes as R
    a = sess("simA")
    item = R._db(lambda c: ER.claim_next_video(c, session=a, reviewer_name="Asha", queue="needs_review", n=1,
                                               drive=BIG))["items"][0]["segment_id"]
    o = op()
    results, errors = [], []
    barrier = threading.Barrier(8)

    def go():
        try:
            barrier.wait()
            results.append(R._db(lambda c: ER.save_video_review_claimed(c, item, _video_body(a, "Asha", o))))
        except Exception as e:                      # noqa: BLE001
            errors.append(repr(e))

    ts = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors
    assert len(results) == 8 and len({r["log_id"] for r in results}) == 1
    assert sum(1 for r in results if not r.get("replayed")) == 1
    assert len(_log_rows(cenv, segment_id=item)) == 1


# ============================================================
# 4-6. Citizen: Supabase failures, lost answers, log failures
# ============================================================

class FlakySupabase(TC.FakeSupabase):
    """FakeSupabase that can fail the next PATCH (network error) or lose its answer after writing."""

    def __init__(self, row):
        super().__init__(row)
        self.fail_next = 0
        self.lose_next = 0

    async def patch(self, url, json=None, headers=None):
        import httpx
        if self.fail_next:
            self.fail_next -= 1
            raise httpx.ConnectError("simulated: Supabase unreachable")
        resp = await super().patch(url, json=json, headers=headers)
        if self.lose_next:
            self.lose_next -= 1
            raise httpx.ReadTimeout("simulated: answer lost after the write")
        return resp


def _citizen_row():
    return {"id": str(uuid.uuid4()), "image_url": "https://res.cloudinary.com/x/image/upload/a.jpg",
            "status": "expert_review", "stage1_label": "Distress", "stage1_confidence": 0.7,
            "distress_types": ["Potholes"], "severity": "High", "stage2_confidence": 0.6,
            "needs_expert_review": True, "model_version": "m", "processed_at": None, "expert_reviewed": False,
            "expert_corrected_types": None, "expert_corrected_severity": None, "expert_notes": None,
            "reviewer_name": None, "reviewed_at": None}


def _wait_events(cenv, kind, n=1, timeout=5.0):
    import time
    end = time.time() + timeout
    while time.time() < end:
        got = cenv.conn.execute("SELECT COUNT(*) FROM research.expert_events WHERE kind = %s", (kind,)).fetchone()[0]
        if got >= n:
            return got
        time.sleep(0.05)
    return got


def test_citizen_unreachable_lost_answer_and_replay(cenv):
    from fastapi.testclient import TestClient
    _clean_extra(cenv)
    row = _citizen_row()
    aid = row["id"]
    sb = FlakySupabase(row)
    cl = TestClient(_app(sb))
    a = sess("cuA")
    body = {"session": a, "reviewer_name": "Asha", "types": ["Potholes"], "severity": "High",
            "queue": "citizen_pending", "notes": "deep", "seen_reviewed_at": None, "op_id": op()}
    # 4. unreachable: 502, event, nothing logged, nothing written
    sb.fail_next = 1
    r = cl.post(f"/expert-api/citizen/review/{aid}", json=body)
    assert r.status_code == 502 and "unreachable" in r.text
    assert not sb.row["expert_reviewed"] and not _log_rows(cenv, assessment_id=aid)
    assert _wait_events(cenv, "supabase_error") == 1
    # 5. the write lands but the answer is lost: 502 to the page, which sends it again
    sb.lose_next = 1
    r = cl.post(f"/expert-api/citizen/review/{aid}", json=body)
    assert r.status_code == 502 and sb.row["expert_reviewed"] is True
    r = cl.post(f"/expert-api/citizen/review/{aid}", json=body)
    assert r.status_code == 200 and r.json()["saved"] is True and r.json()["log_id"], r.text
    assert len(_log_rows(cenv, assessment_id=aid)) == 1
    # sent once more (the 200 was lost too): op_id replay, still one row
    r = cl.post(f"/expert-api/citizen/review/{aid}", json=body)
    assert r.status_code == 200 and r.json()["replayed"] is True
    assert len(_log_rows(cenv, assessment_id=aid)) == 1
    # another expert updates it afterwards; Asha's queued resend is still a replay, not a conflict
    sb.row.update({"reviewer_name": "Ravi", "expert_corrected_types": ["Ravelling"], "reviewed_at": "2026-11-06T10:00:00+00:00"})
    r = cl.post(f"/expert-api/citizen/review/{aid}", json=body)
    assert r.status_code == 200 and r.json()["replayed"] is True
    # the same content from a DIFFERENT reviewer with a new op_id is a real conflict
    r = cl.post(f"/expert-api/citizen/review/{aid}", json={**body, "session": sess("cuB"), "reviewer_name": "Kiran",
                                                           "op_id": op()})
    assert r.status_code == 409


def test_citizen_log_failure_then_retry(cenv):
    from fastapi.testclient import TestClient
    from fastapi import HTTPException
    _clean_extra(cenv)
    row = _citizen_row()
    aid = row["id"]
    sb = FlakySupabase(row)
    cl = TestClient(_app(sb))
    body = {"session": sess("lfA"), "reviewer_name": "Asha", "types": ["Potholes"], "severity": "High",
            "queue": "citizen_pending", "notes": "", "seen_reviewed_at": None, "op_id": op()}
    real = ER.finish_citizen_review
    calls = {"n": 0}

    def broken(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            from app.expert_routes import _Unavailable
            raise _Unavailable("simulated log failure")
        return real(*a, **k)

    ER.finish_citizen_review = broken
    try:
        r = cl.post(f"/expert-api/citizen/review/{aid}", json=body)
        assert r.status_code == 200 and r.json()["saved"] and r.json()["log_error"], r.text
        assert sb.row["expert_reviewed"] and not _log_rows(cenv, assessment_id=aid)
        r = cl.post(f"/expert-api/citizen/review/{aid}", json=body)
        assert r.status_code == 200 and not r.json().get("log_error") and r.json()["log_id"], r.text
    finally:
        ER.finish_citizen_review = real
    assert len(_log_rows(cenv, assessment_id=aid)) == 1
    assert _wait_events(cenv, "server_save_error") >= 1
    _ = HTTPException


# ============================================================
# 7. Deadlocks
# ============================================================

def test_db_retries_deadlocks(cenv):
    import psycopg
    from app import expert_routes as R
    calls = {"n": 0}

    def flaky(conn):
        calls["n"] += 1
        if calls["n"] < 3:
            raise psycopg.errors.DeadlockDetected("simulated deadlock")
        return conn.execute("SELECT 41 + 1").fetchone()[0]

    assert R._db(flaky) == 42 and calls["n"] == 3

    def always(conn):
        raise psycopg.errors.LockNotAvailable("simulated lock timeout")

    try:
        R._db(always)
        raise AssertionError("expected _Unavailable")
    except R._Unavailable as e:
        assert "busy" in str(e)


def test_real_deadlock_resolves(cenv):
    """Two units of work lock two rows in opposite orders: PostgreSQL aborts one; _db reruns it."""
    from app import expert_routes as R
    c = cenv.conn
    c.execute("CREATE TABLE IF NOT EXISTS public.zz_deadlock_test (id int PRIMARY KEY, n int)")
    with c.transaction():
        c.execute("DELETE FROM public.zz_deadlock_test")
        c.execute("INSERT INTO public.zz_deadlock_test VALUES (1, 0), (2, 0)")
    barrier = threading.Barrier(2)
    errors = []

    def work(first, second):
        state = {"tries": 0}

        def fn(conn):
            state["tries"] += 1
            with conn.transaction():
                conn.execute("UPDATE public.zz_deadlock_test SET n = n + 1 WHERE id = %s", (first,))
                if state["tries"] == 1:
                    barrier.wait(timeout=10)
                conn.execute("UPDATE public.zz_deadlock_test SET n = n + 1 WHERE id = %s", (second,))
            return state["tries"]
        try:
            return R._db(fn)
        except Exception as e:                       # noqa: BLE001
            errors.append(repr(e))

    out = {}
    t1 = threading.Thread(target=lambda: out.__setitem__("a", work(1, 2)))
    t2 = threading.Thread(target=lambda: out.__setitem__("b", work(2, 1)))
    t1.start(); t2.start(); t1.join(30); t2.join(30)
    try:
        assert not errors, errors
        n = dict(c.execute("SELECT id, n FROM public.zz_deadlock_test").fetchall())
        assert n == {1: 2, 2: 2}, n                 # both units of work done exactly once
        assert max(out.values()) >= 2                # one of them was aborted and rerun
    finally:
        c.execute("DROP TABLE IF EXISTS public.zz_deadlock_test")


# ============================================================
# 8. Stress: several experts, retries and resends at once
# ============================================================

def test_stress_many_experts(cenv):
    _clean_extra(cenv)
    from app import expert_routes as R
    names = ["Asha", "Ravi", "Kiran", "Meera", "Arjun", "Divya"]
    errors, ops_done = [], []
    lock = threading.Lock()

    def expert(i):
        rnd = random.Random(1000 + i)
        s, name = sess(f"st{i}"), names[i]
        held, last_ops = [], []
        for _ in range(25):
            act = rnd.random()
            try:
                if act < 0.3 or not held:
                    d = R._db(lambda c: ER.claim_next_video(c, session=s, reviewer_name=name,
                                                            queue=rnd.choice(["needs_review", "auto", "random"]),
                                                            n=3, drive=BIG))
                    held = [x["segment_id"] for x in d["items"]]
                elif act < 0.45:
                    R._db(lambda c: ER.claims_heartbeat(c, session=s))
                elif act < 0.85:
                    item = held.pop(0)
                    o = op()
                    q = R._db(lambda c: c.execute("SELECT queue FROM research.review_claims WHERE item_id = %s "
                                                  "AND reviewer_session = %s", (item, s)).fetchone())
                    body = _video_body(s, name, o, types=(rnd.choice(["Normal", "Potholes"]),),
                                       queue=q[0] if q else "needs_review")
                    for _k in range(rnd.choice([1, 1, 2, 3])):      # resends of the same save
                        try:
                            R._db(lambda c: ER.save_video_review_claimed(c, item, body))
                            with lock:
                                ops_done.append(o)
                        except ER.Conflict:
                            pass
                    last_ops.append(o)
                else:
                    item = held.pop(0)
                    R._db(lambda c: ER.claims_release(c, session=s, items=[{"source": "video", "item_id": item}],
                                                      skip=True, reviewer_name=name))
            except ER.Conflict:
                pass
            except Exception as e:                   # noqa: BLE001
                with lock:
                    errors.append(f"{name}: {e!r}\n{traceback.format_exc()}")

    ts = [threading.Thread(target=expert, args=(i,)) for i in range(len(names))]
    [t.start() for t in ts]
    [t.join(120) for t in ts]
    assert not errors, errors[:3]
    dup = cenv.conn.execute("SELECT op_id, COUNT(*) FROM research.expert_review_log WHERE op_id IS NOT NULL "
                            "GROUP BY op_id HAVING COUNT(*) > 1").fetchall()
    assert not dup, dup
    logged = {r[0] for r in cenv.conn.execute("SELECT op_id FROM research.expert_review_log").fetchall()}
    assert set(ops_done) <= logged
    double = cenv.conn.execute("SELECT item_id, COUNT(*) FROM research.review_claims GROUP BY item_id "
                               "HAVING COUNT(*) > 1").fetchall()
    assert not double
    print(f"    stress: {len(set(ops_done))} distinct saves logged once each, {len(ops_done)} save calls")


# ============================================================
# 9. Events and leaderboard
# ============================================================

def test_events_and_leaderboard(cenv):
    from fastapi.testclient import TestClient
    _clean_extra(cenv)
    cl = TestClient(_app())
    o_retry, o_ok = op(), op()
    ev = [{"kind": "save_retry", "op_id": o_retry, "source": "video", "item_id": bseg(1), "reviewer_name": "Asha",
           "http_status": 503, "attempts": 1, "message": "server unavailable"},
          {"kind": "save_retry", "op_id": o_ok, "source": "video", "item_id": bseg(2), "attempts": 1},
          {"kind": "save_recovered", "level": "info", "op_id": o_ok, "attempts": 3}]
    r = cl.post("/expert-api/events", json={"events": ev})
    assert r.status_code == 200 and r.json()["stored"] == 3, r.text
    assert cl.post("/expert-api/events", json={"events": [{"kind": "nope"}]}).status_code == 400
    assert cl.post("/expert-api/events", json={"events": [{"kind": "client_error"}] * 21}).status_code == 400
    assert cl.post("/expert-api/events", json={"events": "x"}).status_code == 400
    # o_ok is in the review log (it was saved after retries), o_retry is not
    a = sess("lbA")
    item = cl.post("/expert-api/claims/next", json={"source": "video", "queue": "needs_review", "session": a,
                                                     "reviewer_name": "Asha", "n": 3,
                                                     "filters": {"drive": BIG}}).json()["items"]
    for k, it in enumerate(item):
        o = o_ok if k == 0 else op()
        nm = "Asha" if k < 2 else "  asha "                          # same expert, sloppy spelling
        assert cl.post(f"/expert-api/video/review/{it['segment_id']}",
                       json=_video_body(a, nm, o)).status_code == 200
    s = cl.get("/expert-api/events/summary").json()
    assert s["counts"]["save_retry"] == 2 and s["counts"]["save_recovered"] == 1
    assert s["retrying_not_yet_saved"] == 1 and s["recent"][0]["kind"] == "save_recovered"
    b = sess("lbB")
    it2 = cl.post("/expert-api/claims/next", json={"source": "video", "queue": "needs_review", "session": b,
                                                    "reviewer_name": "Ravi", "n": 1,
                                                    "filters": {"drive": BIG}}).json()["items"][0]
    assert cl.post(f"/expert-api/video/review/{it2['segment_id']}", json=_video_body(b, "Ravi", op())).status_code == 200
    lb = cl.get("/expert-api/leaderboard").json()
    by = {e["key"]: e for e in lb["experts"]}
    assert by["asha"]["total"] == 3 and by["asha"]["today"] == 3 and by["asha"]["video"] == 3
    assert by["ravi"]["total"] == 1 and lb["team_total"] == 4 and lb["experts"][0]["key"] == "asha"
    # an update of one's own review does not count again
    seen = cenv.conn.execute("SELECT a.reviewed_at FROM video_segments v JOIN assessments a ON a.id = v.assessment_id "
                             "WHERE v.segment_id = %s", (it2["segment_id"],)).fetchone()[0].isoformat()
    assert cl.post(f"/expert-api/video/review/{it2['segment_id']}",
                   json=_video_body(b, "Ravi", op(), queue="reviewed", seen=seen)).status_code == 200
    assert {e["key"]: e for e in cl.get("/expert-api/leaderboard").json()["experts"]}["ravi"]["total"] == 1


# ============================================================
# Standalone runner
# ============================================================

def main() -> int:
    if not TC.DSN:
        print("SKIP: DATABASE_URL is not set")
        return 0
    tests = [test_video_resend_is_replay, test_video_simultaneous_copies, test_citizen_unreachable_lost_answer_and_replay,
             test_citizen_log_failure_then_retry, test_db_retries_deadlocks, test_real_deadlock_resolves,
             test_stress_many_experts, test_events_and_leaderboard]
    env = TC.CEnv()
    failed = 0
    try:
        for t in tests:
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
