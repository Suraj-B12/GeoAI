"""
Integration test: the per-type probe inside the real PavementClassifier (GPU).

What must hold in production:
  1. With a valid config the probe is enabled at load, and its fast
     shared-prefix path agrees with the slow one-pass-per-question path.
  2. SHADOW mode (production): distress_types and the gate are exactly the
     free-form answer; the probe's full answer is recorded with applied=false.
  3. PROBE mode: the probe decides the calibrated types, every type's P(yes)
     is recorded, no condition indicator ever enters distress_types, and the
     API response model accepts the result.
  4. Stage 1 safety net: an Attain frame that Stage 1 wrongly calls Normal is
     flagged (-> expert review); a blank image is not.
  5. FAIL-SAFE: an exception in the probe keeps the free-form answer for that
     image and says why; the gate falls back to 'field'. A failing safety
     net leaves the Normal decision exactly as it was.
  6. FAIL-SAFE: a missing config disables the probe at load with a reason.
  7. No allocator build-up across repeated probe calls.

Usage (loads a second model copy; ~6 GB of VRAM):
    venv/Scripts/python.exe scripts/tests/test_stage2_probe_integration.py
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FAILS: list = []


def check(name, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    os.environ["STAGE2_MODE"] = "shadow"
    os.environ["STAGE1_SAFETY_NET"] = "true"
    os.environ["STAGE2_CONFIDENCE_MODE"] = "field"
    os.environ["STAGE2_PROBE_CONFIG"] = str(PROJECT_ROOT / "configs" / "stage2_probe.json")
    os.environ["PROMPTS_VERSION"] = "v2"
    os.environ.setdefault("MAX_IMAGE_PIXELS", "1048576")

    import torch
    from PIL import Image
    import app.model as appmodel
    from app.model import PavementClassifier
    from app.schemas import ClassificationResponse
    from scripts.irc82_taxonomy import IRC82_CONDITION_INDICATORS, IRC82_DISTRESS_TAXONOMY

    clf = PavementClassifier("Qwen/Qwen2.5-VL-7B-Instruct", None, 4)
    st = clf.probe_status
    check("1. probe enabled at load", st.get("enabled") is True, str(st.get("reason")))
    check("1. parity self-test passed", (st.get("parity") or {}).get("ok") is True, str(st.get("parity")))
    check("1. safety net active", clf.safety_net_active, str(st))

    imgs = PROJECT_ROOT / "Attain" / "Attain" / "Attain" / "Attain_SMP_WS_V2.0" / "Images"
    img = Image.open(imgs / "Attain_SMP_WS_v2_000001.jpg").convert("RGB")
    torch.cuda.empty_cache()
    base_reserved = torch.cuda.memory_reserved()

    # 2. shadow mode
    s2 = clf.predict_stage2(img)
    pr = s2.get("stage2_probe") or {}
    check("2. shadow: labels are the free-form list",
          s2["distress_types"] == s2["stage2_generated_types"] and s2["stage2_mode"] == "generate")
    check("2. shadow: probe recorded, not applied",
          pr.get("applied") is False and len(pr.get("p", {})) >= 19 and s2["stage2_probe_mode"] == "shadow")
    check("2. shadow: gate is field", s2["stage2_confidence"] == s2["stage2_confidence_field"])
    check("2. shadow: no indicators emitted", s2["condition_indicators"] == [])
    rec = (pr.get("views") or {}).get("recorded") or {}
    check("2. shadow: per-tile scores recorded (decision still full photo)",
          rec.get("mode") == "tile" and len(rec.get("views", [])) >= 2
          and (pr.get("views") or {}).get("mode") == "full", str(pr.get("views"))[:300])
    print(f"    shadow: kept {s2['distress_types']}; probe would say {pr.get('types')}")

    # 3. probe mode
    appmodel.STAGE2_MODE = "probe"
    irc = set(IRC82_DISTRESS_TAXONOMY)
    s2 = clf.predict_stage2(img)
    pr = s2.get("stage2_probe") or {}
    check("3. probe: probe path used", s2.get("stage2_mode") == "probe" and pr.get("applied") is True,
          str(s2.get("stage2_probe_error")))
    check("3. probe: types are IRC names", set(s2["distress_types"]) <= irc, str(s2["distress_types"]))
    check("3. probe: no indicator inside distress_types",
          not (set(s2["distress_types"]) & set(IRC82_CONDITION_INDICATORS)))
    full = clf.predict(img)
    try:
        ClassificationResponse(**full)
        ok = True
    except Exception as e:
        ok = False
        print("   ", e)
    check("3. probe: API response model accepts predict()", ok)
    print(f"    probe: {s2['stage2_generated_types']} -> {s2['distress_types']} "
          f"added={pr.get('added_by_probe')} removed={pr.get('removed_by_probe')} {pr.get('time_ms')} ms")
    appmodel.STAGE2_MODE = "shadow"

    # 4. safety net
    missed = Image.open(imgs / "Attain_SMP_WS_v2_000037.jpg").convert("RGB")
    sn = clf.stage1_safety_check(missed)
    check("4. safety net flags a Stage 1 miss", bool(sn and sn.get("flagged")), str(sn and {k: v for k, v in sn.items() if k != 'p'}))
    blank = Image.new("RGB", (640, 640), (128, 128, 128))
    sn = clf.stage1_safety_check(blank)
    check("4. safety net leaves a blank image alone", bool(sn) and sn.get("flagged") is False, str(sn and {k: v for k, v in sn.items() if k != 'p'}))
    full = clf.predict(missed)
    if full["stage1_label"] == "Normal":
        check("4. predict(): flagged Normal needs expert review",
              full["needs_expert_review"] is True and full["stage1_safety_net"]["flagged"])
    else:
        print("    (Stage 1 said Distress on this frame in this run; predict() path not exercised)")
    missed.close()

    # 5. runtime failures
    real_probe = clf._prober.probe

    def boom(_img):
        raise RuntimeError("injected failure")
    clf._prober.probe = boom
    appmodel.STAGE2_MODE = "probe"
    s2 = clf.predict_stage2(img)
    check("5. probe failure falls back to the free-form list",
          s2["stage2_mode"] == "generate" and "injected failure" in (s2["stage2_probe_error"] or "")
          and s2["distress_types"] == s2["stage2_generated_types"])
    sn = clf.stage1_safety_check(blank)
    check("5. safety-net failure leaves the Normal decision alone",
          sn.get("flagged") is False and "injected failure" in sn.get("error", ""))
    clf._prober.probe = real_probe
    appmodel.STAGE2_MODE = "shadow"

    # 6. missing config
    saved = appmodel.STAGE2_PROBE_CONFIG
    appmodel.STAGE2_PROBE_CONFIG = str(PROJECT_ROOT / "configs" / "does_not_exist.json")
    clf._init_probe()
    st = clf.probe_status
    check("6. missing config disables the probe with a reason",
          st.get("enabled") is False and "not found" in (st.get("reason") or ""), str(st))
    s2 = clf.predict_stage2(img)
    check("6. disabled probe -> free-form path, no error, no safety net",
          s2["stage2_mode"] == "generate" and s2["stage2_probe_error"] is None
          and s2["stage2_probe"] is None and clf.stage1_safety_check(blank) is None)
    appmodel.STAGE2_PROBE_CONFIG = saved
    clf._init_probe()
    check("6. re-enabled after restoring the config", clf.probe_status.get("enabled") is True)

    # 8. views: tiles and zoom crops, and a failure inside a view
    from scripts.stage2_views import DamageLocator
    saved_views = clf._probe_cfg.get("views")
    clf._probe_cfg["views"] = {"mode": "tile", "aggregate": "+full", "upscale_limit": 2.0,
                               "tile": {"target": 4, "overlap": 0.1}}
    s2 = clf.predict_stage2(img)
    vi = (s2.get("stage2_probe") or {}).get("views") or {}
    check("8. tile views probed", vi.get("mode") == "tile" and vi.get("n_views", 0) >= 2, str(vi))
    clf._probe_cfg["views"] = {"mode": "zoom", "aggregate": "+full", "upscale_limit": 2.0,
                               "zoom": {"max_boxes": 4, "max_area_frac": 0.5, "margin": 0.2,
                                        "min_side_frac": 0.25, "nms_iou": 0.3}}
    clf._locator = DamageLocator(clf._model, clf._processor)
    s2 = clf.predict_stage2(img)
    vi = (s2.get("stage2_probe") or {}).get("views") or {}
    check("8. zoom views ran (boxes may be zero)", vi.get("mode") == "zoom" and "located_boxes" in vi, str(vi))
    real_locate = clf._locator.locate
    clf._locator.locate = lambda _i: (_ for _ in ()).throw(RuntimeError("injected locator failure"))
    s2 = clf.predict_stage2(img)
    vi = (s2.get("stage2_probe") or {}).get("views") or {}
    check("8. a failing view falls back to the full photo, probe still recorded",
          vi.get("fell_back_to") == "full" and "injected" in vi.get("views_error", "")
          and s2["stage2_probe"] is not None)
    clf._locator.locate = real_locate
    clf._locator = None
    if saved_views is None:
        clf._probe_cfg.pop("views", None)
    else:
        clf._probe_cfg["views"] = saved_views

    # 7. allocator
    for _ in range(8):
        clf._prober.probe(img)
    gc.collect()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_reserved()
    check("7. reserved memory returns to baseline after release",
          after <= base_reserved + 256 * 1024 ** 2,
          f"baseline {base_reserved / 1e9:.2f} GB, after {after / 1e9:.2f} GB")
    img.close()

    print(f"\n{'ALL PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
