"""Drive-video survey metrics over every session the video_pipeline toggle processed (paper §8.7).

No inference: reads each run's segments_results.csv and vlm_results_batch_*.csv under
VIDEO_RUNS_ROOT, the toggle's per-session metrics (GET /operator/video/sessions, or a saved
snapshot), and the original send-all batch (batch_001: the first three drives) for comparison.

Writes eval_results/video_survey.json, eval_results/video_survey.md and figures
eval_results/plots/paper/fig_v10..v13.

Definitions (same as GeoAI_Video geoai/analyze.py):
- a segment is "distress" when the VLM's answer has at least one valid non-Normal IRC:82 label;
- shares are length-weighted over ANSWERED segments (gated runs do not send skip segments);
- skip segments carry no VLM answer, so the survey-wide distress share is a lower bound unless the
  audit sample shows skips are clean.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from generate_video_figures import (BLUE, GRID, INK, MUTED, ORANGE, RED, STATUS_COLORS,  # noqa: E402
                                    enu, title)

OUT_PLOTS = PROJECT_ROOT / "eval_results" / "plots" / "paper"
OUT_JSON = PROJECT_ROOT / "eval_results" / "video_survey.json"
OUT_MD = PROJECT_ROOT / "eval_results" / "video_survey.md"
SNAPSHOT = PROJECT_ROOT / "eval_results" / "video_survey_sessions_snapshot.json"
SPLIT = re.compile(r"\s*[+;|]\s*")


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def labels_of(row: dict) -> list[str]:
    return [x for x in SPLIT.split(row.get("vlm_distress_labels") or "") if x]


def load_sessions(url: str, key_file: Path) -> dict:
    """Toggle per-session metrics by session_id; refreshes the snapshot when the server answers."""
    try:
        key = key_file.read_text(encoding="utf-8").strip() if key_file.exists() else ""
        req = urllib.request.Request(url.rstrip("/") + "/operator/video/sessions",
                                     headers={"X-API-Key": key} if key else {})
        with urllib.request.urlopen(req, timeout=15) as r:
            items = json.load(r)["items"]
        keep = [{k: s.get(k) for k in ("session_id", "state", "metrics", "first_seen_utc", "updated_utc")}
                for s in items]
        for s in keep:  # deleted_paths is long and machine-specific
            ((s.get("metrics") or {}).get("storage") or {}).pop("deleted_paths", None)
        SNAPSHOT.write_text(json.dumps(keep, indent=1), encoding="utf-8")
        print("  sessions from server; snapshot", SNAPSHOT.relative_to(PROJECT_ROOT))
    except OSError as e:
        print(f"  WARNING: server not usable ({e}) - using the saved snapshot, which may be older than the runs")
        keep = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return {s["session_id"]: s for s in keep}


def summarise(rows: list[dict]) -> dict:
    ans = [r for r in rows if truthy(r.get("answered"))]
    dis = [r for r in ans if r.get("vlm_class") == "distress"]
    m_ans = sum(float(r["length_m"]) for r in ans)
    m_dis = sum(float(r["length_m"]) for r in dis)
    lab_m = defaultdict(float)
    lab_n = Counter()
    for r in dis:
        for l in set(labels_of(r)):
            lab_m[l] += float(r["length_m"])
            lab_n[l] += 1
    status = Counter(r["status"] for r in rows)
    aud = [r for r in ans if r["status"] == "audit"]
    aud_d = sum(r.get("vlm_class") == "distress" for r in aud)
    snd = [r for r in ans if r["status"] == "send"]
    snd_d = sum(r.get("vlm_class") == "distress" for r in snd)
    return {
        "segments": len(rows), "route_m": round(sum(float(r["length_m"]) for r in rows), 1),
        "status": dict(status), "answered": len(ans), "answered_m": round(m_ans, 1),
        "distress_segments": len(dis), "distress_m": round(m_dis, 1),
        "distress_share_of_answered_m": round(m_dis / m_ans, 4) if m_ans else None,
        "distress_share_of_answered_n": round(len(dis) / len(ans), 4) if ans else None,
        "distress_share_n_ci95": wilson(len(dis), len(ans)),
        "labels_m_share": {l: round(v / m_ans, 4) for l, v in sorted(lab_m.items(), key=lambda kv: -kv[1])},
        "labels_segments": dict(lab_n.most_common()),
        "send_distress_rate": round(snd_d / len(snd), 4) if snd else None,
        "send_distress_ci95": wilson(snd_d, len(snd)),
        "audit_n": len(aud), "audit_distress": aud_d,
        "audit_distress_rate": round(aud_d / len(aud), 4) if aud else None,
        "audit_distress_ci95": wilson(aud_d, len(aud)),
        "expert_review_share": round(sum(truthy(r.get("vlm_needs_expert_review")) for r in ans) / len(ans), 4)
        if ans else None,
    }


def seconds(run: Path) -> list[tuple[float, str]]:
    out = []
    for p in run.glob("vlm_results_batch_*.csv"):
        for r in read_csv(p):
            if truthy(r.get("answered")) and r.get("seconds"):
                out.append((float(r["seconds"]), r.get("stage1_label") or ""))
    return out


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / name, dpi=200)
    plt.close(fig)
    print("  wrote", (OUT_PLOTS / name).relative_to(PROJECT_ROOT))


def short(sid: str) -> str:
    return f"{sid[6:8]} Oct {sid[9:11]}:{sid[11:13]}Z · {sid[-4:]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=Path(r"C:\GeoAI\runs"))
    ap.add_argument("--baseline", type=Path,
                    default=PROJECT_ROOT.parent / "GeoAI_Video" / "field" / "batches" / "batch_001")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key-file", type=Path, default=Path(r"C:\GeoAI\operator_api_key.txt"))
    a = ap.parse_args()
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    meta = load_sessions(a.url, a.key_file)
    runs = sorted(p.parent.parent.parent for p in a.runs_root.glob("*/vlm/analysis/segments_results.csv"))
    runs = [r for r in runs if meta.get(r.name, {}).get("state") in ("stored", "cleaned")]
    if not runs:
        print("no stored runs under", a.runs_root)
        return 1

    per, all_rows, secs = [], [], []
    for run in runs:
        rows = read_csv(run / "vlm" / "analysis" / "segments_results.csv")
        all_rows += rows
        m = meta[run.name].get("metrics") or {}
        t = m.get("timings_s") or {}
        s = seconds(run)
        secs += s
        summ = summarise(rows)
        t = dict(t)
        extract_note = None
        if t.get("extract") is None:  # extraction finished outside the toggle's timer (server restart)
            rep = json.loads((run / "report.json").read_text(encoding="utf-8"))
            t["extract"] = round(float((rep.get("timings_s") or rep.get("timings") or {}).get("total")), 1)
            extract_note = "extract time from report.json (finished across a server restart)"
        wall = sum(v for v in t.values() if isinstance(v, (int, float)))
        per.append({
            "session_id": run.name, "label": short(run.name), **summ, "extract_note": extract_note,
            "extract_s": t.get("extract"), "classify_s": t.get("classify"), "wall_s": round(wall, 1),
            "wall_min_per_km": round(wall / 60 / (summ["route_m"] / 1000), 1) if summ["route_m"] else None,
            "vlm_s_median": m.get("vlm_s_median"), "vlm_s_p95": m.get("vlm_s_p95"),
            "low_light_share": m.get("low_light_share"), "coverage_pct": m.get("coverage_pct"),
            "storage_after_gb": (m.get("storage") or {}).get("after_gb"),
            "image_store_gb": round(((m.get("storage") or {}).get("after_bytes") or {}).get("image_store", 0) / 1e9, 3),
            "verify_ok": (m.get("verify") or {}).get("ok"),
            "db_rows": (m.get("verify") or {}).get("rows"),
        })

    total = summarise(all_rows)
    normal_s = [x for x, l in secs if l == "Normal"]
    dist_s = [x for x, l in secs if l != "Normal"]
    walls = [p["wall_s"] for p in per]
    total.update({
        "sessions": len(per), "dates": sorted({p["session_id"][:8] for p in per}),
        "vlm_seconds": {
            "n": len(secs), "median": round(st.median(x for x, _ in secs), 2) if secs else None,
            "normal_median": round(st.median(normal_s), 2) if normal_s else None, "normal_n": len(normal_s),
            "distress_median": round(st.median(dist_s), 2) if dist_s else None, "distress_n": len(dist_s),
            "profile": "video"},
        "wall_s_total": round(sum(walls), 1),
        "wall_h_per_km": round(sum(walls) / 3600 / (total["route_m"] / 1000), 3),
        "all_verified": all(p["verify_ok"] for p in per),
        "db_rows": sum(p["db_rows"] or 0 for p in per),
        "image_store_gb": round(sum(p["image_store_gb"] for p in per), 2),
        "low_light_sessions": sum(1 for p in per if (p["low_light_share"] or 0) > 0.5),
    })
    base = None
    bpath = a.baseline / "analysis" / "segments_results.csv"
    if bpath.exists():
        base = summarise(read_csv(bpath))
        base["note"] = "batch_001: first three drives (26-27 Oct), send-all, default /classify profile"

    by_day = {}
    for day in sorted({p["session_id"][:8] for p in per}):
        rows_d = [r for r in all_rows if r["session_id"][:8] == day]
        pd_ = [p for p in per if p["session_id"][:8] == day]
        sd = summarise(rows_d)
        sd.update({"drives": len(pd_), "wall_s": round(sum(p["wall_s"] for p in pd_), 1),
                   "vlm_s_median_of_drives": round(float(np.median([p["vlm_s_median"] for p in pd_
                                                                    if p["vlm_s_median"]])), 2)})
        by_day[day] = sd
    total["by_day"] = by_day
    out = {"generated_by": "scripts/video_survey_metrics.py", "toggle_sessions": total, "per_session": per,
           "baseline_batch_001": base}
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("  wrote", OUT_JSON.relative_to(PROJECT_ROOT))

    figures(per, all_rows, runs, total, base, secs)
    write_md(per, total, base)
    return 0


def figures(per, all_rows, runs, total, base, secs) -> None:
    # v10: route per session, stacked by gate status
    fig, ax = plt.subplots(figsize=(7.2, 0.26 * len(per) + 1.2))
    y = np.arange(len(per))
    by_sid = defaultdict(lambda: Counter())
    for r in all_rows:
        by_sid[r["session_id"]][r["status"]] += float(r["length_m"])
    left = np.zeros(len(per))
    for s in ("send", "audit", "skip", "not_covered"):
        v = np.array([by_sid[p["session_id"]][s] for p in per])
        if v.sum():
            ax.barh(y, v, left=left, color=STATUS_COLORS[s], label=s.replace("_", " "), height=0.7)
            left += v
    for i, p in enumerate(per):
        ax.text(left[i] + 8, i, f"{p['distress_share_of_answered_m'] * 100:.0f} % distress" if p[
            "distress_share_of_answered_m"] is not None else "", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y, [p["label"] for p in per], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("route length (m)")
    ax.set_xlim(0, left.max() * 1.22)
    ax.legend(loc="lower right", fontsize=7, ncol=4)
    title(ax, f"Figure 8.10  {len(per)} drives, {total['route_m'] / 1000:.2f} km: gate decision by length, "
              f"VLM distress share")
    save(fig, "fig_v10_survey_sessions.png")

    # v11: distress label length share, toggle sessions vs first three drives
    labs = list(total["labels_m_share"])[:10]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    yy = np.arange(len(labs))
    ax.barh(yy - 0.2, [100 * total["labels_m_share"].get(l, 0) for l in labs], 0.4, color=BLUE,
            label=f"{total['sessions']} new drives ({total['answered']} answered segments)")
    if base:
        ax.barh(yy + 0.2, [100 * base["labels_m_share"].get(l, 0) for l in labs], 0.4, color=ORANGE,
                label=f"first 3 drives, send-all ({base['answered']} segments)")
    ax.set_yticks(yy, labs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("share of answered length carrying the label (%)")
    ax.legend(fontsize=7, loc="lower right")
    title(ax, "Figure 8.11  IRC:82 labels named by the VLM (a segment can carry several)")
    save(fig, "fig_v11_survey_labels.png")

    # v12: map of all drives, local metres from the first segment
    pts = [(float(r["centre_lat"]), float(r["centre_lon"]), r) for r in all_rows
           if r.get("centre_lat") and r.get("centre_lon")]
    lat0 = np.mean([p[0] for p in pts]); lon0 = np.mean([p[1] for p in pts])
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    groups = {"skip / not covered (no answer)": [], "normal": [], "distress": []}
    for la, lo, r in pts:
        e, n = enu(la, lo, lat0, lon0)
        key = "distress" if r.get("vlm_class") == "distress" else (
            "normal" if truthy(r.get("answered")) else "skip / not covered (no answer)")
        groups[key].append((e, n))
    for key, col, sz in (("skip / not covered (no answer)", "#b8b2ac", 4), ("normal", BLUE, 4), ("distress", RED, 9)):
        if groups[key]:
            e, n = zip(*groups[key])
            ax.scatter(e, n, s=sz, color=col, label=f"{key} ({len(groups[key])})", linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    ax.legend(fontsize=7, loc="best", markerscale=2)
    title(ax, "Figure 8.12  Segment centres of all new drives (5 m segments), coloured by VLM answer")
    save(fig, "fig_v12_survey_map.png")

    # v13: processing time per km per session, and VLM seconds per image
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
    km = np.array([p["route_m"] / 1000 for p in per])
    ex = np.array([(p["extract_s"] or 0) / 60 for p in per]) / km
    cl = np.array([(p["classify_s"] or 0) / 60 for p in per]) / km
    ot = np.array([p["wall_s"] / 60 for p in per]) / km - ex - cl
    xx = np.arange(len(per))
    a1.bar(xx, ex, color=ORANGE, label="extract")
    a1.bar(xx, cl, bottom=ex, color=BLUE, label="classify (VLM)")
    a1.bar(xx, ot, bottom=ex + cl, color="#b8b2ac", label="import, store, verify, clean")
    a1.set_xticks(xx, [p["session_id"][-4:] for p in per], rotation=90, fontsize=6)
    a1.set_ylabel("minutes per km")
    a1.legend(fontsize=6.5)
    title(a1, "Figure 8.13a  Wall time per km, per drive")
    n_s = [x for x, l in secs if l == "Normal"]
    d_s = [x for x, l in secs if l != "Normal"]
    bins = np.linspace(0, max(x for x, _ in secs) * 1.02, 40)
    a2.hist(n_s, bins=bins, color=BLUE, alpha=0.85, label=f"Stage 1 Normal (n={len(n_s)})")
    a2.hist(d_s, bins=bins, color=RED, alpha=0.75, label=f"Distress (n={len(d_s)})")
    a2.set_xlabel("VLM seconds per image (profile=video)")
    a2.set_ylabel("images")
    a2.legend(fontsize=6.5)
    title(a2, "Figure 8.13b  VLM time per image")
    save(fig, "fig_v13_survey_throughput.png")


def fmt0(x) -> str:
    return "—" if x is None else f"{x:.0f}"


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f} %"


def ci(c) -> str:
    return "" if not c else f" [{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def write_md(per, t, base) -> None:
    L = [f"# Drive-video survey: {t['sessions']} drives processed by the video_pipeline toggle",
         "", "Generated by `scripts/video_survey_metrics.py` (no inference). JSON: `video_survey.json`.",
         "The VLM is the reference, not ground truth (Stage 1 recall on Attain distress frames: 51.8 %).", "",
         "## Totals", "",
         "| Metric | New drives (toggle, gated) | First 3 drives (batch_001, send-all) |", "|---|---|---|"]
    b = base or {}

    def row(name, v1, v2=""):
        L.append(f"| {name} | {v1} | {v2} |")
    row("Route", f"{t['route_m'] / 1000:.2f} km, {t['segments']} segments",
        f"{b.get('route_m', 0) / 1000:.2f} km, {b.get('segments', '')} segments" if b else "")
    s = t["status"]
    row("Gate: send / audit / skip / not covered",
        f"{s.get('send', 0)} / {s.get('audit', 0)} / {s.get('skip', 0)} / {s.get('not_covered', 0)}",
        "all sent" if b else "")
    row("Answered by the VLM", f"{t['answered']} ({t['answered_m'] / 1000:.2f} km)",
        f"{b.get('answered', '')}" if b else "")
    row("Distress, share of answered length", pct(t["distress_share_of_answered_m"]),
        pct(b.get("distress_share_of_answered_m")) if b else "")
    row("Distress, share of answered segments (95 % CI)",
        pct(t["distress_share_of_answered_n"]) + ci(t["distress_share_n_ci95"]),
        (pct(b.get("distress_share_of_answered_n")) + ci(b.get("distress_share_n_ci95"))) if b else "")
    row("Distress among audit samples of gate-skipped segments",
        f"{t['audit_distress']}/{t['audit_n']} = {pct(t['audit_distress_rate'])}{ci(t['audit_distress_ci95'])}")
    row("Flagged for expert review (of answered)", pct(t["expert_review_share"]),
        pct(b.get("expert_review_share")) if b else "")
    vs = t["vlm_seconds"]
    row("VLM s/image, median (Normal / Distress)",
        f"{vs['median']} ({vs['normal_median']} / {vs['distress_median']}), profile=video",
        "13.06 (13.0 / 21.7), default profile" if b else "")
    row("Wall time (validate → clean up)", f"{t['wall_s_total'] / 3600:.2f} h = {t['wall_h_per_km'] * 60:.1f} min/km")
    row("Stored and verified", f"{'all' if t['all_verified'] else 'NOT all'} sessions; {t['db_rows']} DB rows; "
        f"{t['image_store_gb']} GB of frames")
    if t.get("by_day"):
        days = sorted(t["by_day"])
        L += ["", "## By recording day (toggle drives)", "",
              "| | " + " | ".join(f"{d[6:8]} {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][int(d[4:6]) - 1]}" for d in days) + " |",
              "|---|" + "---|" * len(days)]
        def drow(name, fn):
            L.append(f"| {name} | " + " | ".join(fn(t["by_day"][d]) for d in days) + " |")
        drow("Drives", lambda x: str(x["drives"]))
        drow("Route", lambda x: f"{x['route_m'] / 1000:.2f} km, {x['segments']} segments")
        drow("Answered", lambda x: f"{x['answered']}")
        drow("Distress, share of answered segments [95 % CI]",
             lambda x: pct(x["distress_share_of_answered_n"]) + ci(x["distress_share_n_ci95"]))
        drow("Distress, share of answered length", lambda x: pct(x["distress_share_of_answered_m"]))
        drow("Potholes, share of answered length", lambda x: pct(x["labels_m_share"].get("Potholes")))
        drow("Flagged for expert review", lambda x: pct(x["expert_review_share"]))
        drow("Wall time", lambda x: f"{x['wall_s'] / 3600:.2f} h = {x['wall_s'] / 60 / (x['route_m'] / 1000):.1f} min/km")
    L += ["", "## Labels (share of answered length; a segment can carry several)", "",
          "| IRC:82 label | New drives | First 3 drives |", "|---|---|---|"]
    for l, v in list(t["labels_m_share"].items())[:12]:
        L.append(f"| {l} | {pct(v)} | {pct((b.get('labels_m_share') or {}).get(l)) if b else ''} |")
    L += ["", "## Per drive", "",
          "| Drive | km | segments | send/audit/skip | distress (length) | VLM s med | extract s | classify s | min/km |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in per:
        s = p["status"]
        L.append(f"| {p['label']} | {p['route_m'] / 1000:.2f} | {p['segments']} | "
                 f"{s.get('send', 0)}/{s.get('audit', 0)}/{s.get('skip', 0)} | "
                 f"{pct(p['distress_share_of_answered_m'])} | {p['vlm_s_median']} | {fmt0(p['extract_s'])} | "
                 f"{fmt0(p['classify_s'])} | {p['wall_min_per_km']} |")
    L += ["", "Figures: `plots/paper/fig_v10_survey_sessions.png`, `fig_v11_survey_labels.png`, "
          "`fig_v12_survey_map.png`, `fig_v13_survey_throughput.png`.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("  wrote", OUT_MD.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    sys.exit(main())
