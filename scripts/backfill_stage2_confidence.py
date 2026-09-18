"""
Repoint the Stage 2 confidence column at the field-restricted metric for rows
that already ran, WITHOUT re-running inference.

Context
-------
Until 2026-09-17 the `assessments.stage2_confidence` column held the
whole-sequence geometric mean, and the 0.80 expert-review gate ran on it.
That metric was measured anti-correlated with correctness (AUC 0.231 over 37
labelled images — see eval_results/confidence_calibration_qwen25vl7b.json), so
production switched to the field-restricted metric.

Since commit a18f4fc the worker has written BOTH numbers into
`raw_response.stage2_confidence_field` and `..._sequence` on every row. For
those rows the switch is a pure database rewrite: copy the field value into
the column and replay the routing gate. No GPU, no model load, no risk of a
different model version changing the prediction.

Rows processed BEFORE a18f4fc have no field value stored. They cannot be
backfilled and are reported separately — use scripts/reset_for_reclassify.py
to queue them for the worker.

Safety
------
  - Dry-run by default. --apply is required to write.
  - NEVER touches a row an expert has already reviewed (expert_reviewed=true)
    or a row in status 'done'. A human decision outranks a metric change.
  - NEVER touches 'pending' / 'processing' rows (not classified yet / in
    flight) or 'rejected_non_pavement' (Stage 2 never ran).
  - Only writes stage2_confidence, needs_expert_review and status. Predictions
    (distress_types, severity, description) are not modified — the model said
    what it said; only our confidence in it is being re-measured.
  - Records the previous value in raw_response.stage2_confidence_backfill so
    the change is reversible and auditable.

Usage:
    venv/Scripts/python.exe scripts/backfill_stage2_confidence.py
    venv/Scripts/python.exe scripts/backfill_stage2_confidence.py --apply
    venv/Scripts/python.exe scripts/backfill_stage2_confidence.py --apply --mode sequence
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import CONFIDENCE_THRESHOLD  # noqa: E402

# Stage 2 only ran for rows in these statuses; everything else is out of scope.
ELIGIBLE_STATUSES = ("classified", "expert_review")


def load_credentials() -> tuple:
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
            url = v.rstrip("/")
        elif k == "SUPABASE_SERVICE_KEY":
            key = v
    if not url or not key:
        sys.exit("ERROR: SUPABASE_URL / SUPABASE_SERVICE_KEY missing from .env")
    return url, key


def request_json(url: str, key: str, method: str = "GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal" if method == "PATCH" else "return=representation",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        sys.exit(f"ERROR: HTTP {e.code} on {method} {url}\n{e.read().decode()[:500]}")
    except Exception as e:
        sys.exit(f"ERROR: {type(e).__name__} on {method} {url}: {e}\n"
                 f"(If this is a connection failure, the Supabase project may be "
                 f"paused — resume it in the dashboard and re-run.)")


def route(stage1_conf, stage2_conf, is_distressed: bool, threshold: float) -> str:
    """Replay app/worker.py _do_inference routing."""
    s1 = stage1_conf or 0.0
    s2 = stage2_conf or 0.0
    if not is_distressed:
        return "classified" if s1 >= threshold else "expert_review"
    return "classified" if (s1 >= threshold and s2 >= threshold) else "expert_review"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually write (default is a dry run)")
    ap.add_argument("--mode", default="field", choices=["field", "sequence"],
                    help="which stored metric to promote into the column")
    ap.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    args = ap.parse_args()

    url, key = load_credentials()
    stored_key = f"stage2_confidence_{args.mode}"

    rows = request_json(
        f"{url}/rest/v1/assessments"
        "?select=id,status,stage1_confidence,is_distressed,stage2_confidence,"
        "needs_expert_review,expert_reviewed,raw_response"
        f"&status=in.({','.join(ELIGIBLE_STATUSES)})"
        "&order=created_at.asc",
        key,
    )
    print(f"{len(rows)} rows in {ELIGIBLE_STATUSES}")

    updates = []
    skipped = Counter()
    transitions = Counter()

    for r in rows:
        if r.get("expert_reviewed"):
            skipped["expert_already_reviewed"] += 1
            continue
        raw = r.get("raw_response") or {}
        if not isinstance(raw, dict):
            skipped["raw_response_not_an_object"] += 1
            continue
        new_conf = raw.get(stored_key)
        if new_conf is None:
            # Classified before the dual-metric worker (commit a18f4fc).
            skipped["no_stored_metric_needs_reinference"] += 1
            continue
        if not r.get("is_distressed"):
            # Stage 2 never ran; the column is 0 and the gate uses Stage 1 only.
            skipped["not_distressed_stage2_never_ran"] += 1
            continue

        old_conf = r.get("stage2_confidence")
        new_status = route(r.get("stage1_confidence"), new_conf, True, args.threshold)
        needs_review = new_status == "expert_review"

        if (old_conf == new_conf
                and r["status"] == new_status
                and bool(r.get("needs_expert_review")) == needs_review):
            skipped["already_correct"] += 1
            continue

        transitions[f"{r['status']} -> {new_status}"] += 1
        patch = {
            "stage2_confidence": new_conf,
            "status": new_status,
            "needs_expert_review": needs_review,
            "raw_response": {
                **raw,
                "stage2_confidence_mode": args.mode,
                # Reversible + auditable: what the column held before, and why.
                "stage2_confidence_backfill": {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "from_mode": raw.get("stage2_confidence_mode", "sequence"),
                    "to_mode": args.mode,
                    "previous_stage2_confidence": old_conf,
                    "previous_status": r["status"],
                    "threshold": args.threshold,
                },
            },
        }
        updates.append((r["id"], patch))

    print("\n--- skipped ---")
    for k, v in skipped.most_common():
        print(f"  {k:<38} {v}")
    print("\n--- status transitions if applied ---")
    for k, v in transitions.most_common():
        print(f"  {k:<38} {v}")
    print(f"\n{len(updates)} rows would be updated "
          f"(promoting raw_response.{stored_key} into stage2_confidence)")

    need_reinf = skipped["no_stored_metric_needs_reinference"]
    if need_reinf:
        print(f"\nNOTE: {need_reinf} rows were classified before the worker "
              f"recorded both metrics. They cannot be backfilled — re-queue "
              f"them with scripts/reset_for_reclassify.py so the worker "
              f"re-runs them under the new gate.")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to commit.")
        return 0

    print(f"\nApplying {len(updates)} updates...")
    for i, (row_id, patch) in enumerate(updates, 1):
        request_json(f"{url}/rest/v1/assessments?id=eq.{row_id}", key,
                     method="PATCH", body=patch)
        if i % 10 == 0 or i == len(updates):
            print(f"  {i}/{len(updates)}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
