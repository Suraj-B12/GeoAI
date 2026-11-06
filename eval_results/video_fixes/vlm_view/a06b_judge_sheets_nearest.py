"""Sheets for the visual judgement of the nearest-vs-sharpest flips (JPEG, for people only; CPU).

For every flip in nearest_vs_sharpest.json: the two whole frames of the segment side by side (left: the
sharpest arm's frame, the one batch_001 answered; right: the nearest_sharp arm's frame), each 900 px wide with
the segment's far/near edges (green), captioned with both answers. Two flips per sheet -> judge_nearest/NN.jpg.
Which arm is left is fixed (sharpest left), so the judge knows which answer belongs to which frame; the question
per flip is only whether the SEGMENT's road (between the green lines) shows IRC:82 distress.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:/GeoAI/analysis/fixes")
import atomic_io as aio  # noqa: E402,F401  (kept for symmetry; JPEGs are written via temp + rename below)

OUT = Path(r"C:/GeoAI/analysis/fixes/vlm_view")
REPO = Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video")
SID = "20261027T050257Z_SM-G991B-F0D1_d47c"
RUNS = {"sharpest": REPO / "field/runs_segments_sendall" / SID, "nearest": REPO / "field/runs_eval/nearest" / SID}
W = 900
flips = json.loads((OUT / "nearest_vs_sharpest.json").read_text(encoding="utf-8"))["flips"]
dst = OUT / "judge_nearest"
dst.mkdir(exist_ok=True)


def panel(arm, rid, labels):
    side = json.loads((RUNS[arm] / "sidecars" / f"{rid}.json").read_text(encoding="utf-8"))
    img = cv2.imread(str(RUNS[arm] / side["frame"]["file"]), cv2.IMREAD_COLOR)
    s = W / img.shape[1]
    th = cv2.resize(img, (W, round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    vt, vb = side["frame"]["rows_of_segment"]
    for v in (vt, min(vb, img.shape[0] - 1)):
        cv2.line(th, (0, round(v * s)), (W - 1, round(v * s)), (0, 200, 0), 2)
    cap = np.full((34, W, 3), 255, np.uint8)
    txt = f"{arm.upper()} ({side['frame']['near_edge_ahead_m']:.2f} m ahead): {labels or 'Normal'}"
    cv2.putText(cap, txt[:95], (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([cap, th])


rows = []
for i, fl in enumerate(flips):
    rid = fl["id"]
    a = panel("sharpest", rid, fl["sharpest_labels"])
    b = panel("nearest", rid, fl["nearest_labels"])
    head = np.full((30, 2 * W + 10, 3), 235, np.uint8)
    cv2.putText(head, f"#{i + 1}  {rid[-15:]}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    rows.append(np.vstack([head, np.hstack([a, np.full((a.shape[0], 10, 3), 255, np.uint8), b])]))
for j in range(0, len(rows), 2):
    sheet = np.vstack(rows[j:j + 2])
    p = dst / f"judge_nearest_{j // 2 + 1:02d}.jpg"
    tmp = p.with_name("." + p.stem + ".tmp.jpg")
    cv2.imwrite(str(tmp), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    tmp.replace(p)
print(f"{len(flips)} flips -> {len(range(0, len(rows), 2))} sheets in {dst}")
