"""
Find RoadSide `photos` rows whose assessment was deleted, and optionally
remove them (and their Cloudinary image, if it is still there).

Before 2026-09-23 the operator dashboard's Delete removed the assessment and
the Cloudinary image but not the `photos` row. The FK cascades photos ->
assessments, not the other way, so the photos row survived and the RoadSide
app kept showing an upload whose image was gone. The dashboard now deletes
all three (app/deletion.py); this cleans up what the old version left.

An orphan is a photos row with no assessment. The trigger that creates the
assessment runs in the same transaction as the photo insert, so a photo
without one means the assessment was deleted. Photos are read BEFORE
assessments, so an upload arriving mid-run is never mistaken for an orphan,
and rows younger than --min-age-minutes are skipped as well.

Dry run by default: it lists what it would delete and deletes nothing.

Usage:
    venv/Scripts/python.exe scripts/cleanup_orphan_photos.py            # report
    venv/Scripts/python.exe scripts/cleanup_orphan_photos.py --apply    # delete
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from app.deletion import (cloudinary_credentials, cloudinary_destroy,  # noqa: E402
                          parse_cloudinary_url, still_served)
from app.supabase_client import _check, build_supabase_client  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete")
    ap.add_argument("--min-age-minutes", type=int, default=10)
    args = ap.parse_args()

    async with build_supabase_client() as sb, httpx.AsyncClient() as http:
        r = await sb.get("/rest/v1/photos?select=id,image_url,created_at,address&order=id.asc")
        _check(r)
        photos = r.json() or []
        r = await sb.get("/rest/v1/assessments?select=photo_id&photo_id=not.is.null")
        _check(r)
        linked = {a["photo_id"] for a in r.json() or []}

        cutoff = datetime.now(timezone.utc) - timedelta(minutes=args.min_age_minutes)
        orphans = [p for p in photos if p["id"] not in linked
                   and datetime.fromisoformat(p["created_at"].replace("Z", "+00:00")) < cutoff]
        print(f"{len(photos)} photos, {len(linked)} with an assessment, "
              f"{len(orphans)} orphaned (older than {args.min_age_minutes} min)")
        if not orphans:
            return 0

        creds = cloudinary_credentials()
        plan = []
        for p in orphans:
            url = p.get("image_url")
            asset = parse_cloudinary_url(url)
            served = await still_served(http, url) if url else False
            plan.append((p, asset, served))
            print(f"  photo {p['id']:>5}  {p['created_at'][:10]}  "
                  f"image {'STILL ON CLOUDINARY' if served else 'already gone'}  "
                  f"{(p.get('address') or '')[:50]}")

        if not args.apply:
            print("\nDry run - nothing deleted. Re-run with --apply to delete these rows "
                  "(and any image still on Cloudinary).")
            return 0

        failed = 0
        for p, asset, served in plan:
            if served:
                if asset is None or creds is None or asset.cloud_name != creds[0]:
                    print(f"  photo {p['id']}: image still served but cannot be deleted "
                          f"with the configured Cloudinary account - row kept")
                    failed += 1
                    continue
                res = await cloudinary_destroy(http, asset, creds)
                if res["result"] not in ("ok", "not found"):
                    print(f"  photo {p['id']}: Cloudinary delete failed ({res}) - row kept")
                    failed += 1
                    continue
            r = await sb.delete(f"/rest/v1/photos?id=eq.{p['id']}",
                                headers={"Prefer": "return=representation"})
            if r.is_success and r.json():
                print(f"  photo {p['id']}: deleted")
            else:
                print(f"  photo {p['id']}: Supabase delete failed ({r.status_code} {r.text[:120]})")
                failed += 1
        print(f"\n{len(plan) - failed}/{len(plan)} orphaned photos removed")
        return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
