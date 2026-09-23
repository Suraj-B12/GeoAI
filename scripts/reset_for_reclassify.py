"""
One-shot: reset ALL Supabase assessments rows to pending so the new
Improved Baseline pipeline re-processes everything.

Clears all prediction fields (distress_types, severity, stage*_confidence,
pavement_filter_*, processed_at, etc.) but preserves identifying fields
(image_url, address, lat/lng, created_at, photo_id).

Usage:
    python scripts/reset_for_reclassify.py            # dry-run, shows what would change
    python scripts/reset_for_reclassify.py --apply    # actually performs the update

Codex-scrutinized invariants:
  - photo_id, image_url, address, latitude, longitude, created_at NEVER touched
  - status reset only for rows currently in {classified, expert_review,
    done, failed, rejected_non_pavement} — does NOT touch 'pending' or
    'processing' rows (those are either already queued or in-flight)
  - prints summary counts before + after
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_key() -> tuple[str, str]:
    env = PROJECT_ROOT / ".env"
    if not env.exists():
        sys.exit("ERROR: .env not found")
    url = key = None
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k == "SUPABASE_URL":
            url = v
        elif k == "SUPABASE_SERVICE_KEY":
            key = v
    if not url or not key:
        sys.exit("ERROR: SUPABASE_URL or SUPABASE_SERVICE_KEY missing")
    return url, key


def request_json(url: str, key: str, method: str = "GET", body: dict | None = None) -> list | dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        body_text = r.read()
    return json.loads(body_text) if body_text else []


def image_reachable(url, timeout: float = 20.0) -> bool:
    """True if the image can be fetched. HEAD first; some hosts reject HEAD,
    so fall back to a ranged GET for the first byte."""
    if not url:
        return False
    for method, headers in (("HEAD", {}), ("GET", {"Range": "bytes=0-0"})):
        try:
            req = urllib.request.Request(url, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if 200 <= resp.status < 300:
                    return True
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                return False  # definitively gone - no point retrying with GET
        except Exception:
            pass
    return False


def main():
    parser = argparse.ArgumentParser(description="Reset all rows for re-classification")
    parser.add_argument("--apply", action="store_true",
                        help="Actually perform the reset (default: dry-run)")
    parser.add_argument("--include-pending", action="store_true",
                        help="Also reset rows currently in 'pending' status "
                             "(useful if a previous run touched them mid-stream)")
    parser.add_argument("--include-unreachable", action="store_true",
                        help="Also reset rows whose image can no longer be "
                             "downloaded. OFF by default: resetting clears the "
                             "row's results, and a row whose image is gone "
                             "(deleted from Cloudinary) can never be "
                             "re-classified, so its only copy of the results "
                             "would be destroyed and it would end as 'failed'.")
    parser.add_argument("--include-expert-reviewed", action="store_true",
                        help="Also re-queue rows an expert has already reviewed. "
                             "OFF by default: an expert correction is ground "
                             "truth for Phase 3 retraining, and re-queueing it "
                             "lets the model overwrite the status a human set.")
    args = parser.parse_args()

    base, key = load_key()

    # 1. Inspect current state
    print("Fetching current row counts...")
    rows = request_json(f"{base}/rest/v1/assessments?select=id,status,expert_reviewed,image_url", key)
    counts = Counter(r.get("status", "unknown") for r in rows)
    print(f"Total rows: {len(rows)}")
    for s, n in counts.most_common():
        print(f"  {s:30s} {n}")

    # 2. Decide target rows
    target_statuses = {"classified", "expert_review", "done", "failed", "rejected_non_pavement"}
    if args.include_pending:
        target_statuses.add("pending")
    targets = [r for r in rows if r.get("status") in target_statuses]
    n_reviewed = sum(1 for r in targets if r.get("expert_reviewed"))
    if not args.include_expert_reviewed:
        targets = [r for r in targets if not r.get("expert_reviewed")]
    # Image reachability. Checked BEFORE anything is cleared: a row reset for
    # re-classification loses its current results, and if its image has since
    # been deleted from the host the worker can never produce new ones.
    unreachable = []
    if not args.include_unreachable:
        print(f"Checking that {len(targets)} images can still be downloaded...")
        reachable = []
        for r in targets:
            if image_reachable(r.get("image_url")):
                reachable.append(r)
            else:
                unreachable.append(r)
        targets = reachable
        if unreachable:
            print(f"  protecting {len(unreachable)} rows whose image is gone - "
                  f"their current results are kept (pass --include-unreachable "
                  f"to reset them anyway)")
    n_target = len(targets)
    print()
    print(f"Will reset {n_target} rows in statuses: {sorted(target_statuses)}")
    print(f"  (skipping {sum(1 for r in rows if r.get('status') == 'processing')} in-flight)")
    if n_reviewed:
        if args.include_expert_reviewed:
            print(f"  WARNING: {n_reviewed} expert-reviewed rows ARE included "
                  f"(--include-expert-reviewed) - the worker will overwrite the "
                  f"status a human set")
        else:
            print(f"  (protecting {n_reviewed} expert-reviewed rows - pass "
                  f"--include-expert-reviewed to re-queue them too)")

    if not args.apply:
        print()
        print("DRY-RUN — no changes made. Re-run with --apply to perform reset.")
        return

    # 3. Build payload — clear all prediction fields, set status=pending.
    # Column names match the live schema (see migration 001 + 004):
    #   retry_count  (NOT attempt_count)
    #   claimed_by   (NOT worker_id)
    payload = {
        "status": "pending",
        "processed_at": None,
        "stage1_label": None,
        "stage1_confidence": None,
        "is_distressed": None,
        "distress_types": None,
        "severity": None,
        "description": None,
        "stage2_confidence": None,
        "needs_expert_review": False,
        "processing_time_ms": None,
        "raw_response": None,
        "pavement_filter_decision": None,
        "pavement_filter_raw": None,
        "pavement_filter_at": None,
        "error_message": None,
        "retry_count": 0,
        "claimed_by": None,
        "claimed_at": None,
    }

    # 4. Apply in batches (PostgREST has URL-length limits on large IN clauses)
    print()
    print("Applying reset...")
    BATCH = 200
    ids = [r["id"] for r in targets]
    t0 = time.time()
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        id_list = ",".join(f'"{x}"' for x in chunk)
        url = f"{base}/rest/v1/assessments?id=in.({id_list})"
        try:
            request_json(url, key, method="PATCH", body=payload)
            print(f"  batch {i // BATCH + 1}: reset {len(chunk)} rows "
                  f"({i + len(chunk)}/{len(ids)})")
        except Exception as e:
            print(f"  batch {i // BATCH + 1}: FAILED — {e}")
            sys.exit(1)

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s. Verifying...")

    # 5. Verify
    rows2 = request_json(f"{base}/rest/v1/assessments?select=status", key)
    counts2 = Counter(r.get("status", "unknown") for r in rows2)
    print(f"Post-reset row counts:")
    for s, n in counts2.most_common():
        print(f"  {s:30s} {n}")


if __name__ == "__main__":
    main()
