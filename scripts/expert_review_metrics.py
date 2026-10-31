"""
Model vs expert, from the expert review log (research.expert_review_log, migration 006).

Writes eval_results/expert_review_metrics.json and .md: the same numbers the expert UI's
Metrics tab shows (both call app.expert_review.metrics), with reviewer names replaced by
R1, R2, ... so the report can be committed. Read-only; no inference.

Definitions are in app/expert_review.py and repeated in the report. The headline accuracy
for the paper is the "random audit" sample (video segments reviewed in a fixed random order);
the needs-review list holds the model's uncertain answers and understates its accuracy.

Usage:
    venv/Scripts/python.exe scripts/expert_review_metrics.py
    venv/Scripts/python.exe scripts/expert_review_metrics.py --out-dir <folder>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUEUE_LABEL = {"needs_review": "Video: needs review (gate-flagged)", "auto": "Video: auto-classified",
               "random": "Video: random audit (unbiased)", "reviewed": "Video: reviewed list",
               "citizen_pending": "Citizen: pending", "citizen_reviewed": "Citizen: reviewed list"}


def pct(sh) -> str:
    if not sh or not sh.get("n"):
        return "–"
    ci = sh.get("ci95")
    c = f" [{ci[0] * 100:.0f}–{ci[1] * 100:.0f}]" if ci else ""
    return f"{sh['rate'] * 100:.1f}% ({sh['k']}/{sh['n']}){c}"


def summary_md(title: str, s: dict) -> list[str]:
    v, s1, b, g = s["verdict"], s["stage1"], s["both_distress"], s["gate"]
    out = [f"### {title} (n = {s['n']})", "",
           "| Measure | Value [95% Wilson CI] |", "|---|---:|",
           f"| Right | {pct(v['right'])} |", f"| Close | {pct(v['close'])} |", f"| Wrong | {pct(v['wrong'])} |",
           f"| Stage 1 agreement | {pct(s1['agree'])} |",
           f"| Stage 1 recall (distress found) | {pct(s1['recall'])} |",
           f"| Stage 1 precision | {pct(s1['precision'])} |",
           f"| Stage 1 specificity | {pct(s1['specificity'])} |",
           f"| Main label correct (both distress, n = {b['n']}) | {pct(b['first_label_right'])} |",
           f"| Exact same types (both distress) | {pct(b['types_exact'])} |",
           f"| Severity exact / one step / two steps | {pct(b['severity']['exact'])} / "
           f"{pct(b['severity']['close'])} / {pct(b['severity']['wrong'])} |",
           f"| Model wrong, gate-flagged items (n = {g['flagged']['n']}) | {pct(g['flagged']['wrong'])} |",
           f"| Model wrong, auto-classified items (n = {g['auto_classified']['n']}) | "
           f"{pct(g['auto_classified']['wrong'])} |", ""]
    out += [f"Stage 1 confusion (expert as truth): TP {s1['tp']}, FP {s1['fp']}, FN {s1['fn']}, TN {s1['tn']}.", ""]
    if s["per_type"]:
        out += ["| Type | Both | Model only | Expert only | Precision | Recall |", "|---|---:|---:|---:|---:|---:|"]
        for t, x in s["per_type"].items():
            out.append(f"| {t} | {x['tp']} | {x['fp']} | {x['fn']} | {pct(x['precision'])} | {pct(x['recall'])} |")
        out.append("")
    h = s["heatmap"]
    out += [f"Overlay (\"where the model looked\") opened on {pct(h['viewed'])} of reviews; "
            f"{h['rated_overlays']} overlays rated: on the damage {pct(h['rating']['on'])}, "
            f"partly {pct(h['rating']['partly'])}, off {pct(h['rating']['off'])}.", ""]
    t = s["time_ms"]
    if t["n"]:
        def sec(x):
            return "–" if x is None else f"{x / 1000:.1f} s"
        out += [f"Median time per review (first click on the card to save, n = {t['n']}): {sec(t['median_all'])}; "
                f"with the overlay {sec(t['median_with_heatmap'])}, without {sec(t['median_without_heatmap'])}.", ""]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "eval_results"))
    a = ap.parse_args(argv)
    if not os.environ.get("DATABASE_URL"):
        try:
            from dotenv import dotenv_values
            url = dotenv_values(PROJECT_ROOT / ".env").get("DATABASE_URL")
            if url:
                os.environ["DATABASE_URL"] = url
        except ImportError:
            pass
    from app import expert_review as ER
    from app import video_store
    conn = video_store.connect(connect_timeout=5)
    try:
        m = ER.metrics(conn)
    finally:
        conn.close()
    names = sorted(m["reviewers"], key=lambda n: (-m["reviewers"][n], n))
    alias = {n: f"R{i + 1}" for i, n in enumerate(names)}
    m["reviewers"] = {alias[n]: c for n, c in m["reviewers"].items()}
    m["generated_by"] = "scripts/expert_review_metrics.py"
    m["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "expert_review_metrics.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    md = ["# Model vs expert (expert review log)", "",
          f"Generated by `scripts/expert_review_metrics.py` at {m['generated_at']} from "
          f"`research.expert_review_log` ({m['log_rows']} log entries, {m['reviewed_items']} reviewed items, "
          f"reviewers {', '.join(f'{k} ({v})' for k, v in m['reviewers'].items()) or 'none'}).", "",
          "**Definitions.** " + " ".join(f"*{k}*: {v}." for k, v in m["definitions"].items()), ""]
    if not m["reviewed_items"]:
        md += ["No reviews logged yet.", ""]
    else:
        if "random" in m["by_queue"]:
            md += ["## Headline: random audit sample", ""] + summary_md(QUEUE_LABEL["random"], m["by_queue"]["random"])
        md += ["## By how items were sampled", ""]
        for q, s in m["by_queue"].items():
            if q != "random":
                md += summary_md(QUEUE_LABEL.get(q, q), s)
        md += ["## By source", ""]
        for src, s in m["by_source"].items():
            md += summary_md("Video segments" if src == "video" else "Citizen uploads", s)
        md += ["## All reviews (mixed sample, not an accuracy estimate)", ""] + summary_md("All", m["all"])
    (out / "expert_review_metrics.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{m['reviewed_items']} reviewed items, {m['log_rows']} log entries -> {out / 'expert_review_metrics.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
