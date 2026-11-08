"""Applies pci_iri.iri_interval to the stored survey drives (100 m own-track blocks and ~1 km stretches).
Writes interval_summary.json. Descriptive only."""
import json, os, sys
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent)); sys.path.insert(0, str(HERE))
import numpy as np
import validate_iri as V
from app import pci_iri as P, pci_irc2023 as C
from scripts import setup_local_db

rows, _ = V.load(setup_local_db.with_database(setup_local_db.database_url(), "geoai"), P.method_fp())
by = {}
for r in rows:
    by.setdefault(r["session_id"], []).append({**r, "flags": r["flags"] or []})


def pci(x):
    return float(C.EQUATIONS["urban"]["iri"](max(x, C.IRI_LEFT)))


out = {}
for unit in (100.0, 1000.0):
    ivs = []
    for v in by.values():
        for b in P.sections_along_session(v, unit):
            if b["length_m"] >= 0.8 * unit:
                iv = P.iri_interval(b)
                if iv["point"] is not None:
                    ivs.append(iv)
    pt = np.array([i["point"] for i in ivs]); lo = np.array([i["lo90"] for i in ivs]); hi = np.array([i["hi90"] for i in ivs])
    q = lambda a: [round(float(np.percentile(a, k)), 2) for k in (5, 50, 95)]
    out[f"{int(unit)}m"] = {"n": len(ivs), "point_p5_50_95": q(pt), "lo90_p5_50_95": q(lo), "hi90_p5_50_95": q(hi),
                            "hi_over_lo_median": round(float(np.median(hi / lo)), 2),
                            "pci_iri_urban_at_median_section": [round(pci(float(np.median(hi))), 1), round(pci(float(np.median(pt))), 1), round(pci(float(np.median(lo))), 1)],
                            "share_lo90_below_IRC_good_2.55": round(float(np.mean(lo < 2.55)), 3),
                            "share_hi90_above_IRC_fair_3.30": round(float(np.mean(hi > 3.30)), 3)}
(HERE / "interval_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
