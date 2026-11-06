"""
Data preparation for the post-training study (paper §6.11). CPU only.

1. Contamination decision (eval_results/finetune/contamination.json): a training
   frame is dropped if EITHER independent keypoint check (ORB or SIFT,
   scripts/ft_contamination_check.py) flags it against any WS_V2.0 frame, and
   so is every frame within +-NEIGHBOURS indices of a flagged frame in the same
   subset: the subsets are numbered video frames, so a run of consecutive
   frames shows the same place (seen: WS_V1 269-277 all match WS_V2 445).
2. Manifests (targets per image, sha256 per file):
     manifest_armA.json  WS_V1.0 + OS_V1.0 after removal (fully external)
     manifest_armB.json  arm A + the 2/3 "train" part of the WS_V2.0 DEV blocks
   plus splits.json (dev / dev_train / dev_cal / test frame lists).
3. Bengaluru sample (bengaluru_items.json): 200 drive-video segments (seeded
   random from C:\\GeoAI\\images) + up to 60 citizen uploads (read-only
   Supabase select of id,image_url only; images cached under
   C:\\GeoAI\\finetune\\bengaluru). Used only for behaviour-shift and format
   checks - there are no labels.

Usage:
    venv/Scripts/python.exe scripts/ft_prepare.py
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ft_common import (ART_DIR, FT_DIR, arm_b_dev_split, atomic_write_json,  # noqa: E402
                               build_manifest, load_train_subset, load_v2_rows, read_json,
                               sha256_file, utc_now)

NEIGHBOURS = 5
SEED = 20261106
VIDEO_DIR = Path(r"C:\GeoAI\images")


def frame_no(name: str) -> int:
    return int(re.search(r"_(\d+)\.\w+$", name).group(1))


def contamination_decision() -> dict:
    checks = {d: read_json(FT_DIR / f"contamination_{d}.json") for d in ("orb", "sift")}
    missing = [d for d, v in checks.items() if not v]
    if missing:
        sys.exit(f"contamination check output missing for {missing}; run ft_contamination_check.py")
    flagged = {}
    for d, c in checks.items():
        for name, f in c["flagged"].items():
            flagged.setdefault(name, {})[d] = {"inliers": f["inliers"], "v2_frame": f["v2_frame"]}
    dropped = {}
    subsets = {"WS_V1.0": "Attain_SMP_WS_v1_", "OS_V1.0": "Attain_SMP_OS_v1_"}
    for name, ev in flagged.items():
        sub = "WS_V1.0" if "_WS_v1_" in name else "OS_V1.0"
        n = frame_no(name)
        for k in range(n - NEIGHBOURS, n + NEIGHBOURS + 1):
            nb = f"{subsets[sub]}{k:06d}.jpg"
            dropped.setdefault(nb, {"reason": [], "subset": sub})
            dropped[nb]["reason"].append(name if k != n else f"flagged: {ev}")
    return {
        "rule": (f"drop a training frame if the ORB or the SIFT keypoint check flags it against any "
                 f"WS_V2.0 frame (each check's own threshold, see contamination_<detector>.json), plus "
                 f"every frame within +-{NEIGHBOURS} frame numbers of a flagged frame in the same subset"),
        "thresholds": {d: c["threshold_inliers"] for d, c in checks.items()},
        "controls": {d: c["controls"] for d, c in checks.items()},
        "n_flagged": {d: len(c["flagged"]) for d, c in checks.items()},
        "n_flagged_union": len(flagged),
        "flagged": flagged,
        "dropped": dropped,
        "decided_at": utc_now(),
    }


def bengaluru_items(n_video: int, n_citizen: int) -> dict:
    rng = random.Random(SEED)
    vids = sorted(VIDEO_DIR.rglob("*.png"))
    pick = sorted(rng.sample(vids, min(n_video, len(vids))))
    items = [{"key": p.stem, "path": str(p), "kind": "video_segment"} for p in pick]
    note = None
    try:
        import httpx
        from dotenv import dotenv_values
        env = dotenv_values(PROJECT_ROOT / ".env")
        url, key = env.get("SUPABASE_URL"), env.get("SUPABASE_SERVICE_KEY")
        q = (f"{url}/rest/v1/assessments?select=id,image_url,status"
             f"&status=in.(classified,expert_review)&order=id.asc&limit=1000")
        cache = ART_DIR / "bengaluru" / "citizen"
        cache.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=60, follow_redirects=True) as c:
            rows = c.get(q, headers={"apikey": key, "Authorization": f"Bearer {key}"}).json()
            rows = [r for r in rows if (r.get("image_url") or "").startswith("https://res.cloudinary.com/")]
            for r in sorted(rng.sample(rows, min(n_citizen, len(rows))), key=lambda r: r["id"]):
                dst = cache / f"{r['id']}.jpg"
                if not dst.exists():
                    resp = c.get(r["image_url"])
                    resp.raise_for_status()
                    tmp = dst.with_suffix(".part")
                    tmp.write_bytes(resp.content)
                    tmp.replace(dst)
                items.append({"key": f"citizen_{r['id']}", "path": str(dst), "kind": "citizen",
                              "status": r["status"]})
    except Exception as e:      # the video sample alone still answers the question
        note = f"citizen photos unavailable: {type(e).__name__}: {e}"
        print(f"[warn] {note}")
    return {"seed": SEED, "n_video": sum(i["kind"] == "video_segment" for i in items),
            "n_citizen": sum(i["kind"] == "citizen" for i in items), "note": note,
            "items": items, "made_at": utc_now()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bengaluru", action="store_true")
    a = ap.parse_args()

    dec = contamination_decision()
    atomic_write_json(FT_DIR / "contamination.json", dec)
    print(f"[contamination] flagged {dec['n_flagged']} union {dec['n_flagged_union']}; "
          f"dropped incl. neighbours {len(dec['dropped'])}")

    train_rows = load_train_subset("WS_V1.0") + load_train_subset("OS_V1.0")
    kept = [r for r in train_rows if r["image"] not in dec["dropped"]]
    v2 = load_v2_rows()
    dev = [r for r in v2 if r["split"] == "dev"]
    test = [r for r in v2 if r["split"] == "test"]
    dev_train, dev_cal = arm_b_dev_split(dev)
    for r in dev_train:
        r["subset"] = "WS_V2.0"
        r["annotated"] = sorted({"Linear crack", "Alligator crack", "Pothole", "Raveling", "Weathering",
                                 "Patch and utility cut", "Block crack"})

    mA = build_manifest(kept, "armA: WS_V1.0 + OS_V1.0 after contamination removal")
    mB = build_manifest(kept + dev_train, "armB: arm A + WS_V2.0 dev_train blocks")
    # hard guarantee: no test frame in either manifest
    test_names = {r["image"] for r in test}
    test_sha = {sha256_file(Path(r["image_path"])) for r in test}
    for m in (mA, mB):
        bad = [it["image"] for it in m["items"] if it["image"] in test_names or it["sha256"] in test_sha]
        if bad:
            sys.exit(f"ERROR: test frames in {m['name']}: {bad[:5]}")
    atomic_write_json(FT_DIR / "manifest_armA.json", mA)
    atomic_write_json(FT_DIR / "manifest_armB.json", mB)
    atomic_write_json(FT_DIR / "splits.json", {
        "source": "stage2_probe_report.split (20-frame blocks, seed 20260923, 35% dev) + "
                  "ft_common.arm_b_dev_split (dev blocks 2/3 train : 1/3 cal, seed 20261106)",
        "dev": sorted(r["image"] for r in dev), "dev_train": sorted(r["image"] for r in dev_train),
        "dev_cal": sorted(r["image"] for r in dev_cal), "test": sorted(test_names),
        "test_sha256": sorted(test_sha)})
    print(f"[manifest] A {mA['n_images']} images {mA['target_counts']}")
    print(f"[manifest] B {mB['n_images']} images")
    print(f"[splits] dev {len(dev)} (train {len(dev_train)} / cal {len(dev_cal)}), test {len(test)}")

    if not a.skip_bengaluru:
        b = bengaluru_items(200, 60)
        atomic_write_json(FT_DIR / "bengaluru_items.json", b)
        print(f"[bengaluru] {b['n_video']} video segments + {b['n_citizen']} citizen photos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
