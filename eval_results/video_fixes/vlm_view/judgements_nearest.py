"""Visual judgement of the 25 nearest-vs-sharpest flips (judge_nearest/judge_nearest_01..13.jpg), by the analysing
agent on 6 Nov 2026, after the decision rule (nearest_decision_rule.json, 10:17:28Z) and the answers existed.

Question per flip: does the SEGMENT's own road (between the green lines, carriageway) show IRC:82 distress?
"nearest" = the nearest_sharp arm's answer looks right, "sharpest" = the sharpest arm's answer looks right,
"unclear" = cannot tell at this resolution. NOT ground truth: one non-expert look per flip, and NOT blind (the
sheet names each arm and its answer; sharpest is always on the left). Notes give what decided it.
"""
JN = {
    "d47c_seg00080": ("unclear", "dark oil-like stains and a darker area by the parked car's wheel; no clear pothole or bleeding"),
    "d47c_seg00083": ("nearest", "clean shaded asphalt; nothing like bleeding or potholes in the segment"),
    "d47c_seg00089": ("nearest", "clean road with sun/shade pattern beside a parked car"),
    "d47c_seg00098": ("unclear", "large darker irregular area (wet, overlay or patch?); ravelled edge at bottom right"),
    "d47c_seg00114": ("unclear", "dark stain plus thin lines near the left edge; sharpest frame too dark to tell"),
    "d47c_seg00138": ("nearest", "cracked surface on the left part of the carriageway in both frames (twig pile is not a pothole)"),
    "d47c_seg00168": ("nearest", "polygonal crack pattern visible near the near edge in the nearest frame; blurred in the sharpest"),
    "d47c_seg00186": ("sharpest", "clean asphalt; no pothole or bleeding; only kerb-edge wear outside the lane"),
    "d47c_seg00195": ("unclear", "light dried trail across the road (spill/sand) on a coarse surface; streaking? ravelling?"),
    "d47c_seg00203": ("unclear", "raised slab/ramp edge at the left and a worn surface; no clear pothole"),
    "d47c_seg00235": ("nearest", "round water/oil stains only; no pothole"),
    "d47c_seg00255": ("unclear", "dark stain by the drain plus shadows; spill or bleeding cannot be told"),
    "d47c_seg00257": ("unclear", "very dark frame; wet patch/shadow; road edge by construction debris may be broken"),
    "d47c_seg00305": ("sharpest", "manhole cover and small stains; no pothole"),
    "d47c_seg00312": ("nearest", "clean carriageway; broken footpath edge lies outside the segment's road"),
    "d47c_seg00325": ("unclear", "dark wet broken patch at the left road margin; edge breaking or a wet stain"),
    "d47c_seg00338": ("nearest", "several crack lines across the shaded road, visible in both frames"),
    "d47c_seg00353": ("nearest", "clean road; cracks only on the footpath/gutter outside the carriageway"),
    "d47c_seg00359": ("nearest", "clean road under tree shade"),
    "d47c_seg00368": ("nearest", "utility cover and a patch line (condition indicator, not a distress type); road clean"),
    "d47c_seg00378": ("nearest", "a crack line runs diagonally through the segment, clear in the nearest frame"),
    "d47c_seg00391": ("unclear", "patched surface with patch outlines and a ravelled area at the left; pothole doubtful"),
    "d47c_seg00402": ("sharpest", "a longitudinal crack/joint with a step runs along the lane in both frames"),
    "d47c_seg00406": ("sharpest", "heavily damaged, patched surface with depressions; the nearest frame is badly defocused"),
    "d47c_seg00442": ("nearest", "clean asphalt with leaf shadows and a manhole cover"),
}
