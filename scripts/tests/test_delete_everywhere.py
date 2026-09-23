"""
Tests for the operator delete (app/deletion.py), network-free.

The property that matters: a delete either removes the upload from Cloudinary
AND Supabase (photos + assessments), or deletes nothing. The failure to rule
out is the one that already happened in production: rows gone or kept while
the image is left behind (or the reverse), with nothing reporting it.

Supabase and Cloudinary are replaced by httpx.MockTransport handlers that
record every request, so each test can assert not only the outcome but that
no DELETE reached the database when the Cloudinary step failed.

What is NOT covered: the real Cloudinary signature. That is checked against
the live API without deleting anything: destroying a public_id that does not
exist returns "not found" only when the signature is valid (see
--live-signature below).

Usage:
    venv/Scripts/python.exe scripts/tests/test_delete_everywhere.py
    venv/Scripts/python.exe scripts/tests/test_delete_everywhere.py --live-signature
    venv/Scripts/python.exe scripts/tests/test_delete_everywhere.py --live-e2e  (needs the server)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import parse_qs

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from app import deletion  # noqa: E402
from app.deletion import (DeleteError, delete_everywhere,  # noqa: E402
                          parse_cloudinary_url, sign)

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))
        FAILURES.append(name)


CLOUD = "testcloud"
IMG = f"https://res.cloudinary.com/{CLOUD}/image/upload/v1745000000/abc123.jpg"
AID = "11111111-2222-3333-4444-555555555555"


class FakeWorld:
    """Supabase + Cloudinary state behind two mock transports."""

    def __init__(self, *, image_url=IMG, status="classified", photo=True,
                 destroy=("ok", 200), served_after=False, fail_photo_delete=False,
                 cascade=True):
        self.assessment = {"id": AID, "status": status, "image_url": image_url,
                           "photo_id": 42 if photo else None}
        self.photo = {"id": 42, "image_url": image_url} if photo else None
        self.destroy = destroy
        self.served_after = served_after
        self.fail_photo_delete = fail_photo_delete
        self.cascade = cascade
        self.requests: list[tuple[str, str]] = []
        self.destroy_bodies: list[dict] = []

    # ---- Supabase PostgREST ----
    def supabase(self, req: httpx.Request) -> httpx.Response:
        self.requests.append((req.method, str(req.url)))
        path, q = req.url.path, str(req.url.query, "utf-8") if isinstance(req.url.query, bytes) else str(req.url.query)
        if path.endswith("/assessments") and req.method == "GET":
            return httpx.Response(200, json=[self.assessment] if self.assessment else [])
        if path.endswith("/photos") and req.method == "GET":
            return httpx.Response(200, json=[self.photo] if self.photo else [])
        if path.endswith("/photos") and req.method == "DELETE":
            if self.fail_photo_delete:
                return httpx.Response(409, json={"message": "violates foreign key constraint"})
            gone = [self.photo] if self.photo else []
            self.photo = None
            if self.cascade:
                self.assessment = None
            return httpx.Response(200, json=gone)
        if path.endswith("/assessments") and req.method == "DELETE":
            gone = [self.assessment] if self.assessment else []
            self.assessment = None
            return httpx.Response(200, json=gone)
        return httpx.Response(404, json={"message": f"unexpected {req.method} {path}?{q}"})

    # ---- Cloudinary API + CDN ----
    def http(self, req: httpx.Request) -> httpx.Response:
        self.requests.append((req.method, str(req.url)))
        if req.url.host == "api.cloudinary.com":
            self.destroy_bodies.append({k: v[0] for k, v in parse_qs(req.content.decode()).items()})
            result, code = self.destroy
            if code != 200:
                return httpx.Response(code, json={"error": {"message": result}})
            return httpx.Response(200, json={"result": result})
        if req.method == "HEAD":
            return httpx.Response(200 if self.served_after else 404)
        return httpx.Response(404)

    def db_deletes(self) -> int:
        return sum(1 for m, u in self.requests if m == "DELETE")


def run(world: FakeWorld, assessment_id: str = AID):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(world.supabase),
                                     base_url="https://db.example") as sb, \
                httpx.AsyncClient(transport=httpx.MockTransport(world.http)) as http:
            return await delete_everywhere(sb, http, assessment_id)
    try:
        return asyncio.run(go()), None
    except DeleteError as e:
        return None, e


def live_e2e() -> None:
    """Delete a throwaway upload through the RUNNING server, end to end.

    Uploads a fixture image to Cloudinary under a unique test public_id,
    inserts an assessment row pointing at it (status rejected_non_pavement,
    so the worker never claims it and no map shows it), calls
    DELETE /dashboard/delete/{id}, then checks the row is gone and the image
    is no longer served. Only data this test created is touched; anything
    left behind by a failure is removed in the finally block.

    The photos-row path is not exercised live: inserting into `photos` would
    publish the test upload in the RoadSide app. It is covered by [3] and [7].
    """
    import time
    import uuid

    from dotenv import dotenv_values
    print("\n[11] live end-to-end delete through the running server")
    env = dotenv_values(PROJECT_ROOT / ".env")
    for k in ("CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET"):
        os.environ[k] = env.get(k) or ""
    cloud, api_key, secret = deletion.cloudinary_credentials()
    sb_url = env["SUPABASE_URL"].rstrip("/")
    sb_h = {"apikey": env["SUPABASE_SERVICE_KEY"],
            "Authorization": f"Bearer {env['SUPABASE_SERVICE_KEY']}"}
    api = os.environ.get("API_BASE", "http://127.0.0.1:8000")
    pid = f"geoai-delete-e2e-{uuid.uuid4().hex[:12]}"
    aid = image_url = None
    with httpx.Client(timeout=60) as c:
        try:
            params = {"public_id": pid, "timestamp": str(int(time.time()))}
            with open(PROJECT_ROOT / "test_fixtures" / "pothole_D40.jpg", "rb") as f:
                r = c.post(f"https://api.cloudinary.com/v1_1/{cloud}/image/upload",
                           data={**params, "api_key": api_key, "signature": sign(params, secret)},
                           files={"file": f})
            check("fixture uploaded to Cloudinary", r.status_code == 200, r.text[:200])
            image_url = r.json()["secure_url"]
            check("fixture is served", c.get(image_url).status_code == 200)

            r = c.post(f"{sb_url}/rest/v1/assessments",
                       headers={**sb_h, "Prefer": "return=representation"},
                       json={"image_url": image_url, "status": "rejected_non_pavement",
                             "address": "TEST-delete-e2e", "model_version": "delete-e2e-test"})
            check("test assessment inserted", r.status_code in (200, 201), r.text[:200])
            aid = r.json()[0]["id"]

            r = c.delete(f"{api}/dashboard/delete/{aid}?confirm=true")
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            check("endpoint returns 200 deleted", r.status_code == 200 and body.get("deleted"),
                  f"{r.status_code} {r.text[:200]}")
            check("Cloudinary reported ok",
                  (body.get("cloudinary") or [{}])[0].get("result") == "ok", str(body)[:200])
            rows = c.get(f"{sb_url}/rest/v1/assessments?id=eq.{aid}&select=id", headers=sb_h).json()
            check("assessment row gone from Supabase", rows == [], str(rows))
            if rows == []:
                aid = None
            gone = c.head(deletion.origin_url(image_url)).status_code
            check("image gone from Cloudinary (origin, CDN bypassed)", gone == 404, f"HTTP {gone}")
            if gone == 404:
                cleanup_url, image_url = image_url, None
                # The URL was fetched above, so a CDN edge holds a copy.
                # invalidate=true purges it; Cloudinary says that takes
                # minutes at most. Measured 5-20 s.
                t0 = time.time()
                while time.time() - t0 < 120:
                    if c.head(cleanup_url).status_code == 404:
                        break
                    time.sleep(5)
                check(f"CDN copy purged ({time.time() - t0:.0f} s)",
                      c.head(cleanup_url).status_code == 404)
        finally:
            if aid:
                c.delete(f"{sb_url}/rest/v1/assessments?id=eq.{aid}", headers=sb_h)
                print(f"  [cleanup] removed leftover test row {aid}")
            if image_url:
                p2 = {"public_id": pid, "timestamp": str(int(time.time())), "invalidate": "true"}
                c.post(f"https://api.cloudinary.com/v1_1/{cloud}/image/destroy",
                       data={**p2, "api_key": api_key, "signature": sign(p2, secret)})
                print(f"  [cleanup] destroyed leftover test image {pid}")


def main() -> int:
    os.environ["CLOUDINARY_CLOUD_NAME"] = CLOUD
    os.environ["CLOUDINARY_API_KEY"] = "123456789012345"
    os.environ["CLOUDINARY_API_SECRET"] = "test-secret"

    print("\n[1] URL parsing")
    a = parse_cloudinary_url(IMG)
    check("versioned URL", a and (a.cloud_name, a.resource_type, a.delivery_type, a.public_id)
          == (CLOUD, "image", "upload", "abc123"), str(a))
    a = parse_cloudinary_url(f"https://res.cloudinary.com/{CLOUD}/image/upload/c_fill,w_300/v17/road/x%20y.png")
    check("transformation + folder + encoded space", a and a.public_id == "road/x y", str(a))
    a = parse_cloudinary_url(f"https://res.cloudinary.com/{CLOUD}/image/upload/w_200/road/pic.jpg")
    check("transformation without version", a and a.public_id == "road/pic", str(a))
    a = parse_cloudinary_url(f"https://res.cloudinary.com/{CLOUD}/raw/upload/v1/data.csv")
    check("raw keeps its extension", a and a.public_id == "data.csv", str(a))
    check("non-Cloudinary URL -> None",
          parse_cloudinary_url("https://erepublic.brightspotcdn.com/x/y.jpg") is None)
    check("empty -> None", parse_cloudinary_url(None) is None)

    print("\n[2] signature covers every signed param, sorted")
    import hashlib
    expect = hashlib.sha1(b"invalidate=true&public_id=abc123&timestamp=100test-secret").hexdigest()
    check("sorted params + secret", sign({"timestamp": "100", "public_id": "abc123",
                                          "invalidate": "true"}, "test-secret") == expect)

    print("\n[3] happy path: Cloudinary, then photo (cascade), then verify")
    w = FakeWorld()
    res, err = run(w)
    check("deleted", res is not None and res["deleted"] and err is None, str(err))
    check("photo row deleted", res and res["photo_deleted"])
    check("assessment gone", w.assessment is None)
    check("image destroyed with invalidate", w.destroy_bodies and
          w.destroy_bodies[0].get("invalidate") == "true" and
          w.destroy_bodies[0].get("public_id") == "abc123")
    body = w.destroy_bodies[0]
    signed = {k: body[k] for k in ("invalidate", "public_id", "timestamp")}
    check("request signature verifies", body.get("signature") == sign(signed, "test-secret"))
    first_delete = next(i for i, (m, _u) in enumerate(w.requests) if m == "DELETE")
    destroy_at = next(i for i, (_m, u) in enumerate(w.requests) if "api.cloudinary.com" in u)
    check("Cloudinary before any database delete", destroy_at < first_delete)

    print("\n[4] Cloudinary failures delete nothing")
    for label, world in [
        ("API error (401 bad signature)", FakeWorld(destroy=("Invalid Signature", 401))),
        ("'not found' but image still served", FakeWorld(destroy=("not found", 200), served_after=True)),
        ("image in another Cloudinary account",
         FakeWorld(image_url="https://res.cloudinary.com/other/image/upload/v1/z.jpg")),
    ]:
        res, err = run(world)
        check(f"{label}: refused", err is not None and res is None, str(res))
        check(f"{label}: no database DELETE sent", world.db_deletes() == 0, str(world.requests))
        check(f"{label}: rows intact", world.assessment is not None and world.photo is not None)

    saved = os.environ.pop("CLOUDINARY_API_SECRET")
    w = FakeWorld()
    res, err = run(w)
    check("missing credentials: 503, nothing deleted",
          err is not None and err.status_code == 503 and w.db_deletes() == 0, str(err))
    os.environ["CLOUDINARY_API_SECRET"] = saved

    print("\n[5] already gone from Cloudinary -> rows are removed")
    w = FakeWorld(destroy=("not found", 200), served_after=False)
    res, err = run(w)
    check("deleted", res is not None and w.assessment is None and w.photo is None, str(err))
    check("reported as already_gone", res and res["cloudinary"][0]["result"] == "already_gone")

    print("\n[6] image not on Cloudinary -> rows removed, nothing sent to Cloudinary")
    w = FakeWorld(image_url="https://erepublic.brightspotcdn.com/a/b.jpg")
    res, err = run(w)
    check("deleted", res is not None and w.assessment is None, str(err))
    check("no Cloudinary call", not w.destroy_bodies)
    check("reported not_cloudinary", res and res["cloudinary"][0]["result"] == "not_cloudinary")

    print("\n[7] rows without a photo, and a database without the cascade")
    w = FakeWorld(photo=False)
    res, err = run(w)
    check("assessment-only row deleted", res is not None and w.assessment is None, str(err))
    w = FakeWorld(cascade=False)
    res, err = run(w)
    check("no cascade: assessment still deleted explicitly",
          res is not None and w.assessment is None and w.photo is None, str(err))

    print("\n[8] refusals")
    res, err = run(FakeWorld(status="processing"))
    check("row being processed -> 409", err is not None and err.status_code == 409, str(err))
    w = FakeWorld()
    w.assessment = None
    res, err = run(w)
    check("unknown id -> 404", err is not None and err.status_code == 404, str(err))
    res, err = run(FakeWorld(), assessment_id="x&id=neq.0")
    check("id with query syntax -> 400", err is not None and err.status_code == 400, str(err))

    print("\n[9] database failure after the image is gone is reported, not hidden")
    w = FakeWorld(fail_photo_delete=True)
    res, err = run(w)
    check("502 that says to retry", err is not None and err.status_code == 502
          and "Retry" in err.detail, str(err))

    if "--live-signature" in sys.argv:
        print("\n[10] live: Cloudinary accepts our signature (deletes nothing)")
        from dotenv import dotenv_values
        env = dotenv_values(PROJECT_ROOT / ".env")
        for k in ("CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET"):
            os.environ[k] = env.get(k) or ""
        creds = deletion.cloudinary_credentials()
        asset = deletion.CloudinaryAsset(creds[0], "image", "upload",
                                         "signature-check-does-not-exist-7f3e9c")

        async def live():
            async with httpx.AsyncClient() as http:
                return await deletion.cloudinary_destroy(http, asset, creds)
        r = asyncio.run(live())
        check("destroy of a nonexistent id -> 'not found' (signature valid)",
              r.get("result") == "not found", json.dumps(r)[:200])

    if "--live-e2e" in sys.argv:
        live_e2e()

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("FAILED:", ", ".join(FAILURES))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
