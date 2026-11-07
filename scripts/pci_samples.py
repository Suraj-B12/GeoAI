"""
Pre-registered sample reports for the WebGIS portal (DESIGN §9): chosen by a rule fixed BEFORE any rating is
looked at, written to eval_results/pci/samples_rule.json, then generated and saved with sample = true.

Rule (seed 20261107, Python random.Random):
  (a) "longest_drive": the drive with the most segments, A = its first segment with a position, B = its last.
  (b) "random_1km_<date>_<i>": for each survey day 2026-10-27 and 2026-10-29, the day's trip = its drives in
      time order, segments in k order, cumulative distance along the trip (gaps as straight lines). Three
      start segments drawn uniformly among those with >= 1000 m of trip after them; B = the first segment
      >= 1000 m further along the trip. No via points (what a user clicking A and B would get). A draw the
      router refuses is recorded with its reason and replaced by the next draw (at most 20 draws per day).
  (c) "two_gaps": every run of three drives d1 -> d2 -> d3 joined by two time-adjacent gap edges (same device,
      dt < 10 min) is a candidate; one is drawn at random; A = the segment 100 m before the end of d1 (or its
      first), B = the segment 100 m after the start of d3 (or its last).
  Category urban (Table 5.3), lane width 3.5 m (default), no road chainage calibration, no free text.

Idempotent: when samples_rule.json exists its chosen inputs are reused (never re-drawn); each report is saved
through app.pci_routes.save_report (dedupe by input hash), and older sample rows of the same rule label are
deleted when a regenerated one is stored ("regenerated, never edited").

Usage (the database must be named explicitly; the coordinator runs it on production later; --regenerate after a
code change re-renders the stored sample PDFs):
    venv/Scripts/python.exe scripts/pci_samples.py --database geoai_test
    venv/Scripts/python.exe scripts/pci_samples.py --database geoai --select-only     # write the rule file only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import setup_local_db  # noqa: E402

RULE_FILE = PROJECT_ROOT / "eval_results" / "pci" / "samples_rule.json"
SEED = 20261107
DAYS = ("2026-10-27", "2026-10-29")
PER_DAY = 3
STRETCH_M = 1000.0
MAX_DRAWS = 20
IST = timezone(timedelta(hours=5, minutes=30))
RULE_TEXT = __doc__.split("Rule (", 1)[1].split("Idempotent:", 1)[0].strip()


def _atomic_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _pt(n):
    return {"lat": round(n.lat, 7), "lon": round(n.lon, 7)}


def _req(a, b, rule):
    return {"a": _pt(a), "b": _pt(b), "category": "urban", "_rule": rule,
            "_a_segment": a.segment_id, "_b_segment": b.segment_id}


def select(g) -> dict:
    from app import pci_chainage as PC
    rnd = random.Random(SEED)
    by: dict[str, list] = {}
    for n in g.nodes:
        by.setdefault(n.session_id, []).append(n)
    for ns in by.values():
        ns.sort(key=lambda n: n.k)
    chosen, refused = [], []
    # (a)
    sid = max(sorted(by), key=lambda s: len(by[s]))
    chosen.append(_req(by[sid][0], by[sid][-1], "longest_drive"))

    def start(s):
        t = [n.utc for n in by[s] if n.utc]
        return min(t) if t else None
    # (b)
    for day in DAYS:
        drives = sorted([s for s in by if start(s) and start(s).astimezone(IST).date().isoformat() == day], key=start)
        trip = [n for s in drives for n in by[s]]
        cum = [0.0]
        for p, q in zip(trip, trip[1:]):
            cum.append(cum[-1] + math.hypot(p.x - q.x, p.y - q.y))
        eligible = [i for i in range(len(trip)) if cum[-1] - cum[i] >= STRETCH_M]
        got = 0
        for draw in range(MAX_DRAWS):
            if got >= PER_DAY or not eligible:
                break
            i = rnd.choice(eligible)
            j = next(j for j in range(i, len(trip)) if cum[j] - cum[i] >= STRETCH_M)
            req = _req(trip[i], trip[j], f"random_1km_{day}_{got + 1}")
            req["_draw"] = draw + 1
            try:
                PC.find_path(g, req["a"], req["b"])
            except PC.PciError as e:
                refused.append({"day": day, "draw": draw + 1, "a_segment": trip[i].segment_id,
                                "b_segment": trip[j].segment_id, "reason": str(e)})
                continue
            chosen.append(req)
            got += 1
    # (c)
    gap_pairs = {(gp["from_session"], gp["to_session"]) for gp in g.gaps}
    nxt = {a: b for a, b in gap_pairs}
    cands = sorted((a, nxt[a], nxt[nxt[a]]) for a in nxt if nxt[a] in nxt)
    if cands:
        d1, d2, d3 = rnd.choice(cands)

        def back(ns, metres):
            acc, i = 0.0, len(ns) - 1
            while i > 0 and acc < metres:
                acc += math.hypot(ns[i].x - ns[i - 1].x, ns[i].y - ns[i - 1].y)
                i -= 1
            return ns[i]

        def fwd(ns, metres):
            acc, i = 0.0, 0
            while i < len(ns) - 1 and acc < metres:
                acc += math.hypot(ns[i + 1].x - ns[i].x, ns[i + 1].y - ns[i].y)
                i += 1
            return ns[i]
        req = _req(back(by[d1], 100.0), fwd(by[d3], 100.0), "two_gaps")
        req["_drives"] = [d1, d2, d3]
        req["_candidates"] = len(cands)
        chosen.append(req)
    return {"chosen": chosen, "refused_draws": refused, "two_gap_candidates": len(cands)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database", required=True, help="database to read and to save the samples to")
    ap.add_argument("--select-only", action="store_true", help="write samples_rule.json, generate nothing")
    ap.add_argument("--regenerate", action="store_true",
                    help="delete the stored sample rows first, so the PDFs are re-rendered with the current code "
                         "(otherwise an unchanged input on unchanged data returns the stored snapshot)")
    a = ap.parse_args(argv)
    os.environ["DATABASE_URL"] = setup_local_db.with_database(setup_local_db.database_url(), a.database)
    from app import pci_chainage as PC
    from app import pci_routes as R
    g = PC.load_graph(force=True)
    if RULE_FILE.exists():
        rule = json.loads(RULE_FILE.read_text(encoding="utf-8"))
        print(f"reusing the pre-registered selection in {RULE_FILE.name} ({len(rule['chosen'])} samples)")
    else:
        sel = select(g)
        rule = {"rule": RULE_TEXT, "seed": SEED, "rule_sha256": hashlib.sha256(RULE_TEXT.encode()).hexdigest(),
                "selected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "selected_on_database": a.database, "data_as_of": g.data_as_of,
                "note": "written BEFORE any rating was computed; reused unchanged on every later run", **sel}
        _atomic_write(RULE_FILE, rule)
        print(f"wrote {RULE_FILE} ({len(rule['chosen'])} samples, {len(rule['refused_draws'])} refused draws)")
    if a.select_only:
        return 0
    import psycopg
    results = []
    for req in rule["chosen"]:
        body = {k: v for k, v in req.items() if not k.startswith("_")}
        label = req["_rule"]
        if a.regenerate:
            with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as c:
                R.ensure_schema(c)
                c.execute("DELETE FROM research.pci_reports WHERE sample AND sample_rule = %s", (label,))
        try:
            out = R.save_report(body, sample=True, sample_rule=label, cap=None, graph=g)
        except Exception as e:      # an HTTPException from the builder: record and go on
            detail = getattr(e, "detail", None) or str(e)
            print(f"  {label}: NOT generated: {detail}")
            results.append({"rule": label, "error": str(detail)})
            continue
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as c:
            c.execute("DELETE FROM research.pci_reports WHERE sample AND sample_rule = %s AND id <> %s",
                      (label, out["id"]))
            routed, gaps = c.execute("SELECT (report->'summary'->>'length_m')::float, "
                                     "(report->'summary'->>'gaps')::int FROM research.pci_reports WHERE id = %s",
                                     (out["id"],)).fetchone()
        requested = STRETCH_M if label.startswith("random_1km") else None
        print(f"  {label}: report {out['id']} ({'new' if out['created'] else 'existing'}), routed {routed:.0f} m"
              + (f" for a requested {requested:.0f} m of trip" if requested else "") + f", {gaps} gap(s)")
        results.append({"rule": label, "id": out["id"], "created": out["created"],
                        "requested_trip_m": requested, "routed_length_m": routed, "gaps": gaps})
    gen = {"database": a.database, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "data_as_of": g.data_as_of, "note": "requested = metres along the drives' own trip (rule b); routed = "
           "the A->B path the router chose (it may use other drives, gaps and links, so it can differ)",
           "reports": results}
    _atomic_write(RULE_FILE.with_name(f"samples_generated_{a.database}.json"), gen)
    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
