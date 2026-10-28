"""Compact sheets for the visual judgement of flips (JPEG, for people only): two flips per image, each the
whole frame at 1280 px with the segment's far/near edge (green) and the band (orange), captioned with both answers."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

OUT = Path(r"C:/GeoAI/analysis/fixes/vlm_view")
B = Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/field/batches")
(OUT / "judge").mkdir(exist_ok=True)
flips = list(csv.DictReader(open(OUT / "flips.csv", encoding="utf-8")))
W = 1280
cells = []
for i, fl in enumerate(flips):
    rid = fl["id"]
    rec = json.loads((B / "ab_band_001" / "records" / f"{rid}.json").read_text(encoding="utf-8"))
    img = cv2.imread(str(B / "ab_frame_001" / "images" / f"{rid}.png"), cv2.IMREAD_COLOR)
    s = W / img.shape[1]
    th = cv2.resize(img, (W, round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    vt, vb = rec["frame"]["rows_of_segment"]
    for v in (vt, min(vb, img.shape[0] - 1)):
        cv2.line(th, (0, round(v * s)), (W - 1, round(v * s)), (0, 200, 0), 2)
    y0, y1 = rec["vlm_input"]["box"][1], rec["vlm_input"]["box"][3]
    cv2.rectangle(th, (1, round(y0 * s)), (W - 2, round(y1 * s) - 1), (0, 140, 255), 1)
    cap = np.full((40, W, 3), 255, np.uint8)
    cv2.putText(cap, f"#{i + 1} {rid[-15:]} FRAME: {fl['frame_labels'] or 'Normal'} | BAND: {fl['band_labels'] or 'Normal'}",
                (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cells.append(np.vstack([cap, th]))
for j in range(0, len(cells), 2):
    pair = cells[j:j + 2]
    sheet = np.vstack([pair[0], np.full((10, W, 3), 255, np.uint8)] + pair[1:])
    cv2.imwrite(str(OUT / "judge" / f"judge_{j // 2 + 1:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(len(cells), "flips on", (len(cells) + 1) // 2, "sheets")
