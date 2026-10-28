"""Annotated thumbnails: segment rows (green) on the 1344x756 VLM view (CPU)."""
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(r"C:/Users/PRO-LAB-3/Documents/GeoAI_Video/field/runs_segments_sendall")
OUT = Path(r"C:/GeoAI/analysis/discriminator/out/frames")
for sid in sys.argv[1:]:
    sess = sid.rsplit("_seg", 1)[0]
    rec = json.loads((ROOT / sess / "sidecars" / f"{sid}.json").read_text())
    img = cv2.imread(str(ROOT / sess / rec["frame"]["file"]), cv2.IMREAD_COLOR)
    small = cv2.resize(img, (1344, 756), interpolation=cv2.INTER_AREA)
    s = 1344 / 3840
    vt, vb = rec["frame"]["rows_of_segment"]
    for v, c in ((vt, (0, 255, 0)), (min(vb, 2159), (0, 255, 0))):
        cv2.line(small, (0, int(v * s)), (1343, int(v * s)), c, 2)
    cv2.putText(small, f"{sid[-12:]} {rec['frame']['near_edge_ahead_m']:.2f}-{rec['frame']['far_edge_ahead_m']:.2f} m", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(str(OUT / f"{sid[-12:]}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
