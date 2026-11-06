"""
Several experts at once: review claims (leases) and conditional saves
(app/expert_review.py claims section, app/expert_routes.py /claims/*, /citizen/review, migration 007).

What must hold:
  1. Racing hand-outs never give one item to two live sessions: 8 sessions x 50 calls on three
     lists at once, mixed with grants and releases, checked against a shadow model that is
     conservative (an item leaves the model before its lease is released in the database and
     enters it only after the database granted it, so a model clash is a real double grant).
  2. An expired lease goes to the next session; the old holder's heartbeat reports it lost.
  3. A save while another session holds the item -> Conflict (409); the holder's save works,
     releases the lease and is logged; an old page (no session) may save only unclaimed items.
  4. A save whose seen_reviewed_at is stale -> Conflict; with the current value it updates.
  5. Release and re-claim; a skipped item is not handed back to the same session, but is to others.
  6. The random-audit list is handed out in its fixed order, skipping only items others hold.
  7. Citizen saves through the server: the Supabase PATCH carries the conditional filter
     (expert_reviewed=eq.false, or reviewed_at=eq.<seen>), zero rows changed -> 409, a lease held
     by someone else -> 409 before any PATCH; the DB_BACKEND=postgres branch does the same in SQL.
  8. Over HTTP (TestClient): every /claims endpoint, sendBeacon-style text/plain release, 409 bodies.

Scratch database geoai_test only (dropped and re-created; never `geoai`). No GPU.

Usage:
    venv/Scripts/python.exe scripts/tests/test_expert_concurrency.py
    venv/Scripts/python.exe -m pytest scripts/tests/test_expert_concurrency.py -q
"""

from __future__ import annotations

import os
import random
import sys
import threading
import traceback
import uuid
from pathlib import Path
from urllib.parse import unquote

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
        if DSN:
            os.environ["DATABASE_URL"] = DSN
    except ImportError:
        pass

if not DSN and pytest is not None and __name__ != "__main__":
    pytest.skip("DATABASE_URL is not set: local PostgreSQL tests skipped", allow_module_level=True)

from app import expert_review as ER  # noqa: E402

BIG = "20261027_101500_s21"           # a second fake drive with many answered segments
N_BIG = 40


def sess(tag: str) -> str:
    return f"test-session-{tag}".ljust(20, "x")


def bseg(k: int) -> str:
    return f"{BIG}_seg{k:05d}"


if DSN:
    import hashlib
    import json

    from app import video_store  # noqa: E402
    from scripts.tests.test_local_db import Env, answer, fake_record, png_bytes  # noqa: E402

    def build_big_run(root: Path) -> tuple[Path, dict]:
        """N_BIG sent segments; every 3rd is low-confidence (expert_review), the rest classified."""
        run = root / BIG
        (run / "sidecars").mkdir(parents=True)
        (run / "frames").mkdir()
        answers = {}
        for k in range(N_BIG):
            data = png_bytes(k + 7)
            sha = hashlib.sha256(data + bytes([k])).hexdigest()
            # distinct bytes per segment: re-encode with the k in the PNG text so the SHA is distinct
            from PIL import Image, PngImagePlugin
            import io
            im = Image.open(io.BytesIO(data))
            meta = PngImagePlugin.PngInfo()
            meta.add_text("k", str(k))
            buf = io.BytesIO()
            im.save(buf, format="PNG", pnginfo=meta)
            data = buf.getvalue()
            sha = hashlib.sha256(data).hexdigest()
            (run / "frames" / f"{BIG}_seg{k:05d}.png").write_bytes(data)
            rec = fake_record(BIG, k, "send", sha)
            (run / "sidecars" / f"{rec['id']}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
            review = k % 3 == 0
            answers[bseg(k)] = answer(types=("Potholes",) if k % 2 else ("Ravelling",), severity="Medium",
                                      s2c=0.55 if review else 0.92, review=review)
        return run, answers

    class CEnv(Env):
        def __init__(self):
            os.environ.setdefault("DATABASE_URL", DSN)
            super().__init__()
            run, answers = build_big_run(self.runs)
            video_store.store_session(self.conn, run, answers, image_root=self.image_root)
            self.prev_url = os.environ.get("DATABASE_URL")
            os.environ["DATABASE_URL"] = self.dsn
            ER._schema_ready = False
            ER._claims_schema_ready = False
            ER.ensure_claims_schema(self.conn)

        def close(self):
            os.environ["DATABASE_URL"] = self.prev_url
            super().close()

        def fresh(self):
            """Each test starts with no leases, skips or reviews on the big drive."""
            with self.conn.transaction():
                self.conn.execute("DELETE FROM research.review_claims")
                self.conn.execute("DELETE FROM research.review_skips")
                self.conn.execute("UPDATE assessments SET expert_reviewed = false, expert_corrected_types = NULL, "
                                  "expert_corrected_severity = NULL, reviewer_name = NULL, reviewed_at = NULL")

        def connect(self):
            return video_store.connect(self.dsn)

    if pytest is not None:
        @pytest.fixture(scope="module")
        def cenv():
            e = CEnv()
            yield e
            e.close()


def _raises(fn, exc):
    try:
        fn()
    except exc as e:
        return e
    raise AssertionError(f"{fn} did not raise {exc.__name__}")


def _holder(conn, item):
    r = conn.execute("SELECT reviewer_session FROM research.review_claims WHERE source = 'video' AND item_id = %s "
                     "AND expires_at > now()", (item,)).fetchone()
    return r[0] if r else None


# ============================================================
# 1. The race
# ============================================================

def test_race_never_double_grants(cenv):
    cenv.fresh()
    model: dict = {}                       # item -> session believed to hold it
    mlock = threading.Lock()
    clashes: list = []
    errors: list = []
    stats = {"calls": 0, "granted": 0}
    queues = ["random", "needs_review", "auto"]

    def worker(t: int):
        s = sess(f"race{t}")
        rng = random.Random(1000 + t)
        conn = cenv.connect()
        held: set = set()
        try:
            for _ in range(50):
                op = rng.random()
                if op < 0.6:
                    out = ER.claim_next_video(conn, session=s, reviewer_name=f"Expert {t}",
                                              queue=rng.choice(queues), n=rng.randint(1, 4))
                    got = [i["segment_id"] for i in out["items"]]
                elif op < 0.75:
                    cand = [bseg(rng.randrange(N_BIG)) for _ in range(5)]
                    out = ER.claims_grant(conn, source="video", session=s, reviewer_name=f"Expert {t}",
                                          queue="random", items=cand, n=2)
                    got = out["granted"]
                else:
                    if held:
                        drop = rng.sample(sorted(held), k=min(len(held), rng.randint(1, 3)))
                        with mlock:                       # leave the model BEFORE the database
                            for i in drop:
                                if model.get(i) == s:
                                    del model[i]
                        ER.claims_release(conn, session=s, items=[{"source": "video", "item_id": i} for i in drop])
                        held.difference_update(drop)
                    continue
                with mlock:                               # enter the model AFTER the database granted
                    stats["calls"] += 1
                    for i in got:
                        other = model.get(i)
                        if other is not None and other != s:
                            clashes.append((i, other, s))
                        model[i] = s
                        stats["granted"] += 1
                held.update(got)
                # everything handed to me is still mine in the database (TTL is 10 min; only I release)
                for i in got:
                    h = _holder(conn, i)
                    if h != s:
                        clashes.append((i, h, s, "db"))
        except Exception as e:                            # noqa: BLE001
            errors.append(f"thread {t}: {type(e).__name__}: {e}")
        finally:
            conn.close()

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=120)
    assert not errors, errors
    assert not clashes, clashes[:5]
    assert stats["granted"] > 100, stats            # the race actually happened
    # the table itself: at most one live lease per item (primary key) and none for unknown items
    n_live = cenv.conn.execute("SELECT COUNT(*) FROM research.review_claims WHERE expires_at > now()").fetchone()[0]
    assert n_live == len(model), (n_live, len(model))
    print(f"    race: {stats['calls']} hand-outs, {stats['granted']} items granted, 0 double grants")


# ============================================================
# 2-6. Leases, saves, order
# ============================================================

def test_expiry_and_heartbeat(cenv):
    cenv.fresh()
    a, b = sess("expA"), sess("expB")
    c = cenv.conn
    got = ER.claim_next_video(c, session=a, reviewer_name="Asha", queue="random", n=2)["items"]
    ids = [i["segment_id"] for i in got]
    assert len(ids) == 2
    hb = ER.claims_heartbeat(c, session=a, items=[{"source": "video", "item_id": i} for i in ids])
    assert len(hb["held"]) == 2 and not hb["lost"]
    with c.transaction():
        c.execute("UPDATE research.review_claims SET expires_at = now() - interval '1 second' WHERE item_id = %s",
                  (ids[0],))
    out = ER.claims_grant(c, source="video", session=b, reviewer_name="Ravi", queue="random", items=ids)
    assert out["granted"] == [ids[0]], out                  # the expired one only
    assert out["held_by_others"] == [{"item_id": ids[1], "reviewer_name": "Asha"}]
    hb = ER.claims_heartbeat(c, session=a, items=[{"source": "video", "item_id": i} for i in ids])
    assert hb["lost"] == [{"source": "video", "item_id": ids[0]}] and len(hb["held"]) == 1
    # an expired lease nobody took is renewed by its holder's heartbeat
    with c.transaction():
        c.execute("UPDATE research.review_claims SET expires_at = now() - interval '1 second' WHERE item_id = %s",
                  (ids[1],))
    hb = ER.claims_heartbeat(c, session=a)
    assert hb["held"] == [{"source": "video", "item_id": ids[1]}]
    assert _holder(c, ids[1]) == a


def test_save_requires_claim_and_fresh_view(cenv):
    cenv.fresh()
    a, b = sess("saveA"), sess("saveB")
    c = cenv.conn
    item = ER.claim_next_video(c, session=a, reviewer_name="Asha", queue="needs_review", n=1)["items"][0]
    sid = item["segment_id"]
    assert item["reviewed_at"] is None and item["image_v"] and len(item["image_v"]) == 16
    body = {"reviewer_name": "Ravi", "types": ["Potholes"], "severity": "High", "queue": "needs_review"}
    e = _raises(lambda: ER.save_video_review_claimed(c, sid, {**body, "session": b}), ER.Conflict)
    assert e.info["held_by"] == "Asha"
    e = _raises(lambda: ER.save_video_review_claimed(c, sid, body), ER.Conflict)        # old page, no session
    assert c.execute("SELECT COUNT(*) FROM research.expert_review_log WHERE segment_id = %s",
                     (sid,)).fetchone()[0] == 0
    out = ER.save_video_review_claimed(c, sid, {**body, "reviewer_name": "Asha", "session": a})
    assert out["action"] == "confirm" and out["reviewed_at"]
    assert _holder(c, sid) is None                                          # released by the save
    # 4. optimistic concurrency on edits
    e = _raises(lambda: ER.save_video_review_claimed(c, sid, {**body, "session": b, "queue": "reviewed"}),
                ER.Conflict)                                                 # b saw it unreviewed
    assert e.info["reviewed_by"] == "Asha" and e.info["reviewed_at"] == out["reviewed_at"]
    e = _raises(lambda: ER.save_video_review_claimed(c, sid, {**body, "session": b, "queue": "reviewed",
                                                              "seen_reviewed_at": "2020-01-01T00:00:00+00:00"}),
                ER.Conflict)
    out2 = ER.save_video_review_claimed(c, sid, {**body, "session": b, "queue": "reviewed",
                                                 "seen_reviewed_at": out["reviewed_at"]})
    assert out2["action"] == "update"
    _raises(lambda: ER.save_video_review_claimed(c, sid, {**body, "session": b, "queue": "reviewed",
                                                          "seen_reviewed_at": "yesterday"}), ER.ReviewError)
    # an old page may still save an item nobody holds
    free = ER.claim_next_video(c, session=a, reviewer_name="Asha", queue="auto", n=1)["items"][0]["segment_id"]
    ER.claims_release(c, session=a, items=[{"source": "video", "item_id": free}])
    assert ER.save_video_review_claimed(c, free, {**body, "queue": "auto"})["action"] == "confirm"
    # reviewed items leave the review lists; nobody is handed them again
    for s in (a, b):
        got = ER.claim_next_video(c, session=s, reviewer_name="X Y", queue="random", n=30)["items"]
        assert sid not in [i["segment_id"] for i in got] and free not in [i["segment_id"] for i in got]
    log = c.execute("SELECT action FROM research.expert_review_log WHERE segment_id = %s ORDER BY id",
                    (sid,)).fetchall()
    assert [r[0] for r in log] == ["confirm", "update"]


def test_release_skip_and_reclaim(cenv):
    cenv.fresh()
    a, b = sess("skipA"), sess("skipB")
    c = cenv.conn
    first = [i["segment_id"] for i in ER.claim_next_video(c, session=a, reviewer_name="Asha", queue="random",
                                                          n=3)["items"]]
    again = [i["segment_id"] for i in ER.claim_next_video(c, session=a, reviewer_name="Asha", queue="random",
                                                          n=3)["items"]]
    assert again == first                                   # a reload gets the same items back
    ER.claims_release(c, session=a, items=[{"source": "video", "item_id": first[0]}], skip=True,
                      reviewer_name="Asha")
    ER.claims_release(c, session=a, items=[{"source": "video", "item_id": first[1]}])
    mine = [i["segment_id"] for i in ER.claim_next_video(c, session=a, reviewer_name="Asha", queue="random",
                                                         n=3)["items"]]
    assert first[0] not in mine and first[1] in mine and first[2] in mine
    theirs = [i["segment_id"] for i in ER.claim_next_video(c, session=b, reviewer_name="Ravi", queue="random",
                                                           n=1)["items"]]
    assert theirs == [first[0]]                             # someone else does get the skipped one
    assert c.execute("SELECT COUNT(*) FROM research.review_skips").fetchone()[0] == 1
    ER.claims_release(c, session=a)
    assert c.execute("SELECT COUNT(*) FROM research.review_claims WHERE reviewer_session = %s",
                     (a,)).fetchone()[0] == 0


def test_random_order_preserved(cenv):
    cenv.fresh()
    c = cenv.conn
    order = [r[0] for r in c.execute(
        "SELECT v.segment_id FROM video_segments v JOIN assessments a ON a.id = v.assessment_id "
        "WHERE v.session_id = %s AND NOT COALESCE(a.expert_reviewed, false) "
        "ORDER BY md5('expert-review-random-v1:' || v.segment_id)", (BIG,)).fetchall()]
    a, b, d = sess("ordA"), sess("ordB"), sess("ordD")
    ga = [i["segment_id"] for i in ER.claim_next_video(c, session=a, reviewer_name="Asha", queue="random", n=3,
                                                       drive=BIG)["items"]]
    gb = [i["segment_id"] for i in ER.claim_next_video(c, session=b, reviewer_name="Ravi", queue="random", n=3,
                                                       drive=BIG)["items"]]
    assert ga == order[:3] and gb == order[3:6], (ga, gb, order[:6])
    ER.claims_release(c, session=a)
    gd = [i["segment_id"] for i in ER.claim_next_video(c, session=d, reviewer_name="Dev", queue="random", n=4,
                                                       drive=BIG)["items"]]
    assert gd == order[:3] + order[6:7]                      # skips only what b holds
    act = ER.claims_active(c, me=d, with_items=True)
    assert act["experts_online"] == 2 and act["experts"][0]["mine"] is True
    assert {i["item_id"] for i in act["items"]} == set(gb)   # others' items only, never session ids
    assert all("reviewer_session" not in i for i in act["items"])


# ============================================================
# 7. Citizen saves through the server
# ============================================================

class FakeResp:
    def __init__(self, status, data):
        self.status_code, self._data = status, data

    @property
    def is_success(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._data


class FakeSupabase:
    """Records PATCH calls; answers GET with the current row; PATCH changes the row only when the
    filter in the URL matches, like PostgREST."""

    def __init__(self, row):
        self.row, self.patches = dict(row), []

    async def get(self, url):
        return FakeResp(200, [dict(self.row)])

    async def patch(self, url, json=None, headers=None):
        self.patches.append((url, json, headers))
        q = dict(p.split("=", 1) for p in url.split("?", 1)[1].split("&"))
        ok = q.get("id") == f"eq.{self.row['id']}"
        if "expert_reviewed" in q:
            ok = ok and q["expert_reviewed"] == "eq.false" and not self.row.get("expert_reviewed")
        if "reviewed_at" in q:
            ok = ok and unquote(q["reviewed_at"]) == "eq." + str(self.row.get("reviewed_at"))
        if not ok:
            return FakeResp(200, [])
        self.row.update(json)
        return FakeResp(200, [dict(self.row)])


def _citizen_app(client):
    from fastapi import FastAPI
    from app import expert_routes
    app = FastAPI()
    expert_routes.install(app)
    app.state.supabase_client = client
    return app


def test_citizen_conditional_save(cenv):
    from fastapi.testclient import TestClient
    cenv.fresh()
    aid = str(uuid.uuid4())
    sb = FakeSupabase({"id": aid, "image_url": "https://res.cloudinary.com/x/image/upload/a.jpg",
                       "status": "expert_review", "stage1_label": "Distress", "stage1_confidence": 0.7,
                       "distress_types": ["Potholes"], "severity": "High", "stage2_confidence": 0.6,
                       "needs_expert_review": True, "model_version": "m", "processed_at": None,
                       "expert_reviewed": False, "expert_corrected_types": None, "expert_corrected_severity": None,
                       "reviewer_name": None, "reviewed_at": None})
    cl = TestClient(_citizen_app(sb))
    a, b = sess("citA"), sess("citB")
    body = {"reviewer_name": "Asha", "types": ["Potholes"], "severity": "High", "queue": "citizen_pending",
            "notes": "ok", "seen_reviewed_at": None}
    g = cl.post("/expert-api/claims/next", json={"source": "citizen", "queue": "citizen_pending", "session": a,
                                                  "reviewer_name": "Asha", "n": 5, "candidates": [aid]}).json()
    assert g["granted"] == [aid]
    r = cl.post(f"/expert-api/citizen/review/{aid}", json={**body, "session": b, "reviewer_name": "Ravi"})
    assert r.status_code == 409 and r.json()["detail"]["held_by"] == "Asha" and sb.patches == []
    r = cl.post(f"/expert-api/citizen/review/{aid}", json={**body, "session": a})
    assert r.status_code == 200, r.text
    url, patch, headers = sb.patches[-1]
    assert "expert_reviewed=eq.false" in url and headers["Prefer"] == "return=representation"
    assert patch["expert_reviewed"] is True and patch["expert_corrected_types"] == ["Potholes"]
    assert r.json()["comparison"]["verdict"] == "right" and r.json()["log_id"]
    assert cenv.conn.execute("SELECT COUNT(*) FROM research.review_claims WHERE item_id = %s",
                             (aid,)).fetchone()[0] == 0
    # a second pending save (someone else saw it unreviewed): zero rows -> 409, nothing logged
    r = cl.post(f"/expert-api/citizen/review/{aid}", json={**body, "session": b, "reviewer_name": "Ravi"})
    assert r.status_code == 409 and r.json()["detail"]["reviewed_by"] == "Asha"
    assert "expert_reviewed=eq.false" in sb.patches[-1][0]
    # an edit with the reviewed_at it saw: conditional on that value; a stale one -> 409
    seen = sb.row["reviewed_at"]
    r = cl.post(f"/expert-api/citizen/review/{aid}", json={**body, "session": b, "reviewer_name": "Ravi",
                                                           "queue": "citizen_reviewed",
                                                           "seen_reviewed_at": "2020-01-01T00:00:00+00:00"})
    assert r.status_code == 409
    r = cl.post(f"/expert-api/citizen/review/{aid}", json={**body, "session": b, "reviewer_name": "Ravi",
                                                           "queue": "citizen_reviewed", "seen_reviewed_at": seen,
                                                           "types": ["Ravelling"], "severity": "Low"})
    assert r.status_code == 200, r.text
    assert "reviewed_at=eq." in sb.patches[-1][0] and "%2B" in sb.patches[-1][0]   # '+' encoded
    rows = cenv.conn.execute("SELECT action, reviewer_name FROM research.expert_review_log "
                             "WHERE assessment_id = %s ORDER BY id", (aid,)).fetchall()
    assert [tuple(x) for x in rows] == [("confirm", "Asha"), ("update", "Ravi")]
    assert cl.post(f"/expert-api/citizen/review/{aid}", json={**body, "session": "short"}).status_code == 400
    assert cl.post("/expert-api/citizen/review/not-a-uuid", json=body).status_code == 400


def test_citizen_save_local_backend(cenv):
    """DB_BACKEND=postgres: the same conditional write, in SQL on the local assessments table."""
    from fastapi.testclient import TestClient
    cenv.fresh()
    c = cenv.conn
    with c.transaction():
        aid = str(c.execute(
            "INSERT INTO assessments (image_url, status, stage1_label, stage1_confidence, is_distressed, "
            "distress_types, severity, stage2_confidence, needs_expert_review) VALUES "
            "('https://res.cloudinary.com/x/image/upload/b.jpg', 'expert_review', 'Distress', 0.7, true, "
            "'[\"Potholes\"]'::jsonb, 'High', 0.6, true) RETURNING id").fetchone()[0])

    class LocalClient:
        IS_LOCAL_DB = True

        async def get_assessment_by_id(self, i):
            from psycopg.rows import dict_row
            with cenv.connect() as conn:
                r = conn.cursor(row_factory=dict_row).execute(
                    "SELECT id::text AS id, status, reviewer_name, reviewed_at, expert_reviewed FROM assessments "
                    "WHERE id = %s::uuid", (i,)).fetchone()
            return r

    cl = TestClient(_citizen_app(LocalClient()))
    a, b = sess("locA"), sess("locB")
    body = {"reviewer_name": "Asha", "types": ["Normal"], "queue": "citizen_pending", "seen_reviewed_at": None}
    r = cl.post(f"/expert-api/citizen/review/{aid}", json={**body, "session": a})
    assert r.status_code == 200, r.text
    row = c.execute("SELECT expert_reviewed, needs_expert_review, expert_corrected_types, reviewer_name FROM "
                    "assessments WHERE id = %s::uuid", (aid,)).fetchone()
    assert row[0] is True and row[1] is False and row[2] == ["Normal"] and row[3] == "Asha"
    r = cl.post(f"/expert-api/citizen/review/{aid}", json={**body, "session": b, "reviewer_name": "Ravi"})
    assert r.status_code == 409


# ============================================================
# 8. HTTP
# ============================================================

def test_http_claims(cenv):
    from fastapi.testclient import TestClient
    cenv.fresh()
    from fastapi import FastAPI
    from app import expert_routes
    app = FastAPI()
    expert_routes.install(app)
    cl = TestClient(app)
    a, b = sess("httpA"), sess("httpB")
    r = cl.post("/expert-api/claims/next", json={"source": "video", "queue": "needs_review", "session": a,
                                                  "reviewer_name": "Asha", "n": 3, "filters": {"drive": BIG}})
    assert r.status_code == 200, r.text
    d = r.json()
    ids = [i["segment_id"] for i in d["items"]]
    assert len(ids) == 3 and d["held"] == 3 and d["available"] == d["total"] and d["ttl_s"] == ER.CLAIM_TTL_S
    assert all(i["claim_expires_at"] for i in d["items"])
    rb = cl.post("/expert-api/claims/next", json={"source": "video", "queue": "needs_review", "session": b,
                                                   "reviewer_name": "Ravi", "n": 30, "filters": {"drive": BIG}}).json()
    assert not set(ids) & {i["segment_id"] for i in rb["items"]}
    assert rb["in_progress_elsewhere"] == 3
    # marked list: who holds what
    lst = cl.get("/expert-api/video/list", params={"queue": "needs_review", "session": BIG, "me": a,
                                                    "limit": 100}).json()
    marks = {i["segment_id"]: (i["claim_name"], i["claim_mine"]) for i in lst["items"]}
    assert all(marks[i] == ("Asha", True) for i in ids)
    assert all(marks[i["segment_id"]] == ("Ravi", False) for i in rb["items"])
    # saves through HTTP: 409 body
    r = cl.post(f"/expert-api/video/review/{ids[0]}", json={"session": b, "reviewer_name": "Ravi",
                                                              "types": ["Normal"], "queue": "needs_review"})
    assert r.status_code == 409 and r.json()["detail"]["held_by"] == "Asha"
    r = cl.post(f"/expert-api/video/review/{ids[0]}", json={"session": a, "reviewer_name": "Asha",
                                                              "types": ["Normal"], "queue": "needs_review",
                                                              "seen_reviewed_at": None})
    assert r.status_code == 200 and r.json()["reviewed_at"]
    # heartbeat + sendBeacon-style release (text/plain body)
    hb = cl.post("/expert-api/claims/heartbeat", json={"session": a}).json()
    assert len(hb["held"]) == 2
    import json as _json
    r = cl.post("/expert-api/claims/release", content=_json.dumps({"session": a, "source": "video"}),
                headers={"Content-Type": "text/plain;charset=UTF-8"})
    assert r.status_code == 200 and len(r.json()["released"]) == 2
    act = cl.get("/expert-api/claims/active", params={"me": a, "items": "true"}).json()
    assert act["experts_online"] == 1 and act["experts"][0]["reviewer_name"] == "Ravi"
    assert len(act["items"]) == len(rb["items"])
    g = cl.post("/expert-api/claims/grant", json={"source": "video", "queue": "reviewed", "session": a,
                                                   "reviewer_name": "Asha", "items": [rb["items"][0]["segment_id"]]})
    assert g.json()["granted"] == [] and g.json()["held_by_others"][0]["reviewer_name"] == "Ravi"
    # bad input
    assert cl.post("/expert-api/claims/next", json={"source": "video", "queue": "reviewed", "session": a,
                                                     "reviewer_name": "Asha"}).status_code == 400
    assert cl.post("/expert-api/claims/next", json={"queue": "random", "session": "x",
                                                     "reviewer_name": "Asha"}).status_code == 400
    assert cl.post("/expert-api/claims/heartbeat", json={"session": a, "items": [{"source": "nope"}]}).status_code == 400
    assert cl.get("/expert-api/claims/active", params={"source": "x"}).status_code == 400


# ============================================================
# Standalone runner
# ============================================================

def main() -> int:
    if not DSN:
        print("SKIP: DATABASE_URL is not set")
        return 0
    tests = [test_race_never_double_grants, test_expiry_and_heartbeat, test_save_requires_claim_and_fresh_view,
             test_release_skip_and_reclaim, test_random_order_preserved, test_citizen_conditional_save,
             test_citizen_save_local_backend, test_http_claims]
    env = CEnv()
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
