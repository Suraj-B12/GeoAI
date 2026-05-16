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


def main():
    parser = argparse.ArgumentParser(description="Reset all rows for re-classification")
    parser.add_argument("--apply", action="store_true",
                        help="Actually perform the reset (default: dry-run)")
    parser.add_argument("--include-pending", action="store_true",
                        help="Also reset rows currently in 'pending' status "
                             "(useful if a previous run touched them mid-stream)")
    args = parser.parse_args()

    base, key = load_key()

    # 1. Inspect current state
    print("Fetching current row counts...")
    rows = request_json(f"{base}/rest/v1/assessments?select=id,status", key)
    counts = Counter(r.get("status", "unknown") for r in rows)
    print(f"Total rows: {len(rows)}")
    for s, n in counts.most_common():
        print(f"  {s:30s} {n}")

    # 2. Decide target rows
    target_statuses = {"classified", "expert_review", "done", "failed", "rejected_non_pavement"}
    if args.include_pending:
        target_statuses.add("pending")
    targets = [r for r in rows if r.get("status") in target_statuses]
    n_target = len(targets)
    print()
    print(f"Will reset {n_target} rows in statuses: {sorted(target_statuses)}")
    print(f"  (skipping {sum(1 for r in rows if r.get('status') == 'processing')} in-flight)")

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
