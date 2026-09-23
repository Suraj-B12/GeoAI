"""
Operator delete: remove an upload from Supabase AND Cloudinary.

One upload lives in three places:

    photos         the row the RoadSide app wrote; the app and the WebGIS
                   read it (public read policy)
    assessments    the pipeline's row, linked by assessments.photo_id
    Cloudinary     the image file both rows point at

The FK is assessments.photo_id REFERENCES photos(id) ON DELETE CASCADE, so
the cascade runs photos -> assessments only. Deleting the assessment, which
is what the dashboard used to do, left the photos row behind. The app kept
showing the upload with a dead image, and 16 such rows had built up by
2026-09-23. Deleting the photos row removes both.

Order: Cloudinary first, then the database. If the database step fails
afterwards, the row is still there pointing at a dead image. The operator
sees the error and can retry, and on retry Cloudinary reports "not found",
which is treated as done. The other order fails worse: an image whose rows
are gone can only be found again by listing the whole Cloudinary account.

Nothing is deleted from the database unless the image is confirmed gone
from Cloudinary, or is not hosted there at all. Missing credentials, a
foreign Cloudinary account, an API error or an unverifiable "not found"
abort the whole delete, so a failure never leaves an orphaned image.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import unquote, urlparse

import httpx

from app.supabase_client import SupabaseError, _check


class DeleteError(Exception):
    """A delete that was refused or failed. status_code maps to HTTP."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# ============================================================
# Cloudinary
# ============================================================

@dataclass(frozen=True)
class CloudinaryAsset:
    cloud_name: str
    resource_type: str   # image | video | raw
    delivery_type: str   # upload | private | authenticated
    public_id: str


_VERSION = re.compile(r"v\d+")


def parse_cloudinary_url(url: Optional[str]) -> Optional[CloudinaryAsset]:
    """The asset a Cloudinary delivery URL points at, or None if it is not one.

    https://res.cloudinary.com/<cloud>/<resource_type>/<type>/
        [<transformations>/...][v<version>/]<public_id>.<ext>

    The public_id is everything after the version segment. Without a version
    segment, leading transformation segments (they contain "_" parameters
    such as "c_fill,w_300") are skipped. The file extension is not part of
    an image's public_id.
    """
    if not url:
        return None
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.netloc.endswith("cloudinary.com"):
        return None
    parts = [unquote(p) for p in u.path.split("/") if p]
    if len(parts) < 4:
        return None
    cloud, resource_type, delivery_type, rest = parts[0], parts[1], parts[2], parts[3:]
    versions = [i for i, p in enumerate(rest) if _VERSION.fullmatch(p)]
    if versions:
        rest = rest[versions[0] + 1:]
    else:
        while len(rest) > 1 and re.match(r"^[a-z]{1,3}_", rest[0]):
            rest = rest[1:]
    if not rest:
        return None
    path = "/".join(rest)
    if resource_type != "raw" and "." in rest[-1]:
        path = path.rsplit(".", 1)[0]
    return CloudinaryAsset(cloud, resource_type, delivery_type, path) if path else None


def cloudinary_credentials() -> Optional[tuple]:
    cloud = os.environ.get("CLOUDINARY_CLOUD_NAME")
    key = os.environ.get("CLOUDINARY_API_KEY")
    secret = os.environ.get("CLOUDINARY_API_SECRET")
    return (cloud, key, secret) if (cloud and key and secret) else None


def sign(params: dict, api_secret: str) -> str:
    """Cloudinary request signature: SHA-1 of the sorted params + secret."""
    to_sign = "&".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha1((to_sign + api_secret).encode("utf-8")).hexdigest()


async def cloudinary_destroy(http: httpx.AsyncClient, asset: CloudinaryAsset,
                             creds: tuple) -> dict:
    """Delete one asset. Returns {"result": "ok" | "not found" | <error>}.

    invalidate=true also purges the CDN copies. Without it the deleted image
    stays downloadable from the CDN cache for a while.
    """
    cloud, api_key, api_secret = creds
    params = {"public_id": asset.public_id, "timestamp": str(int(time.time())),
              "invalidate": "true"}
    if asset.delivery_type != "upload":
        params["type"] = asset.delivery_type
    body = {**params, "api_key": api_key, "signature": sign(params, api_secret)}
    url = f"https://api.cloudinary.com/v1_1/{cloud}/{asset.resource_type}/destroy"
    try:
        r = await http.post(url, data=body, timeout=20.0)
    except httpx.HTTPError as e:
        return {"result": "error", "detail": f"{type(e).__name__}: {e}"}
    try:
        j = r.json()
    except ValueError:
        j = {}
    if r.is_success and "result" in j:
        return {"result": j["result"], "status_code": r.status_code}
    msg = (j.get("error") or {}).get("message") if isinstance(j, dict) else None
    return {"result": "error", "status_code": r.status_code,
            "detail": msg or r.text[:200]}


def origin_url(url: str) -> str:
    """The same delivery URL with a fresh random version segment.

    Cloudinary looks assets up by public_id and ignores the version number,
    but the CDN caches by full path, so a never-seen version always reaches
    the origin. A query string does NOT do this: measured 2026-09-23, after a
    successful destroy the original URL and URL?x=<random> both kept serving
    the edge copy for 5-20 s, while the random-version URL returned 404
    immediately.
    """
    fresh = f"v{random.randint(10 ** 9, 10 ** 10 - 1)}"
    u = urlparse(url)
    parts = u.path.split("/")
    for i, p in enumerate(parts):
        if _VERSION.fullmatch(p):
            parts[i] = fresh
            break
    else:
        # No version segment: insert one after /<resource_type>/<type>/ and
        # any transformation segments, i.e. right before the public_id.
        i = 4
        while i < len(parts) - 1 and re.match(r"^[a-z]{1,3}_", parts[i]):
            i += 1
        parts.insert(i, fresh)
    return u._replace(path="/".join(parts), query="").geturl()


async def still_served(http: httpx.AsyncClient, url: str) -> bool:
    """True if the origin still has the image (CDN copies are bypassed)."""
    try:
        r = await http.head(origin_url(url), timeout=15.0, follow_redirects=True)
    except httpx.HTTPError:
        return False
    return r.status_code == 200


# ============================================================
# Orchestration
# ============================================================

async def _fetch_one(supabase: httpx.AsyncClient, path: str) -> Optional[dict]:
    r = await supabase.get(path)
    _check(r)
    rows = r.json() or []
    return rows[0] if rows else None


async def delete_everywhere(supabase: httpx.AsyncClient, http: httpx.AsyncClient,
                            assessment_id: str) -> dict:
    """Delete an upload from Cloudinary, photos and assessments.

    Raises DeleteError when nothing was deleted, or when the image was
    deleted but the database step failed; the detail says which.
    """
    if not assessment_id or len(assessment_id) > 64 or \
            not re.fullmatch(r"[0-9A-Za-z-]+", assessment_id):
        raise DeleteError(400, f"invalid assessment id: {assessment_id!r}")

    a = await _fetch_one(supabase, f"/rest/v1/assessments?id=eq.{assessment_id}"
                                   "&select=id,status,image_url,photo_id&limit=1")
    if a is None:
        raise DeleteError(404, "assessment not found")
    if a.get("status") == "processing":
        # The worker would write its result to a row that no longer exists.
        raise DeleteError(409, "this image is being classified right now - "
                               "try again in a few seconds")

    photo = None
    if a.get("photo_id") is not None:
        photo = await _fetch_one(supabase, f"/rest/v1/photos?id=eq.{a['photo_id']}"
                                           "&select=id,image_url&limit=1")

    # ---- 1. Cloudinary: every distinct image the rows point at ----
    urls = []
    for u in (a.get("image_url"), (photo or {}).get("image_url")):
        if u and u not in urls:
            urls.append(u)

    creds = cloudinary_credentials()
    cloudinary = []
    for u in urls:
        asset = parse_cloudinary_url(u)
        if asset is None:
            # Not hosted on Cloudinary (e.g. a test row with an external URL):
            # there is nothing of ours to delete.
            cloudinary.append({"url": u, "result": "not_cloudinary"})
            continue
        if creds is None:
            raise DeleteError(503, "CLOUDINARY_CLOUD_NAME / _API_KEY / _API_SECRET are "
                                   "not set on the server. Nothing was deleted: removing "
                                   "the rows would leave the image on Cloudinary.")
        if asset.cloud_name != creds[0]:
            raise DeleteError(409, f"the image is in Cloudinary account "
                                   f"'{asset.cloud_name}', not the configured one. "
                                   f"Nothing was deleted.")
        res = await cloudinary_destroy(http, asset, creds)
        if res["result"] == "not found":
            # Either deleted earlier, or the public_id is wrong. Only the
            # first is safe to treat as done.
            if await still_served(http, u):
                raise DeleteError(502, f"Cloudinary reports '{asset.public_id}' as not "
                                       f"found but the image is still being served. "
                                       f"Nothing was deleted.")
            res["result"] = "already_gone"
        elif res["result"] != "ok":
            raise DeleteError(502, f"Cloudinary delete failed "
                                   f"({res.get('detail') or res['result']}). "
                                   f"Nothing was deleted.")
        cloudinary.append({"url": u, "public_id": asset.public_id, **res})

    # ---- 2. Database: the photos row cascades to the assessment ----
    try:
        photo_deleted = False
        if photo is not None:
            r = await supabase.delete(f"/rest/v1/photos?id=eq.{photo['id']}",
                                      headers={"Prefer": "return=representation"})
            _check(r)
            photo_deleted = bool(r.json())
        # The cascade should already have removed the assessment. Delete it
        # explicitly anyway: covers rows without a photo, and a database whose
        # FK lacks ON DELETE CASCADE.
        r = await supabase.delete(f"/rest/v1/assessments?id=eq.{assessment_id}",
                                  headers={"Prefer": "return=representation"})
        _check(r)
        remaining = await _fetch_one(supabase, f"/rest/v1/assessments?id=eq.{assessment_id}"
                                               "&select=id&limit=1")
        if remaining is not None:
            raise DeleteError(502, "the image was removed from Cloudinary but the "
                                   "assessment row is still in Supabase. Retry the delete.")
    except SupabaseError as e:
        raise DeleteError(502, f"the image was removed from Cloudinary but the Supabase "
                               f"delete failed ({e}). Retry the delete.") from e

    return {
        "deleted": True,
        "assessment_id": assessment_id,
        "photo_id": a.get("photo_id"),
        "photo_deleted": photo_deleted,
        "image_url": a.get("image_url"),
        "cloudinary": cloudinary,
    }
