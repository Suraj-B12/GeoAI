"""
Tests for the /classify request profile and the loopback rate-limit exemption (GPU-free).

1. PavementClassifier.predict(profile=...) on the REAL predict / predict_stage2 /
   stage1_safety_check / stage2_views.probe_scores / stage2_probe_rules code and the REAL
   configs/stage2_probe.json, with only the model replaced (fake Stage 1 answer, fake Stage 2
   text, and a fake prober whose P(yes) depends on the pixels it is shown):
     - profile 'video' probes the full photo only (no record-only tiles);
     - everything that decides labels or review is identical to the default path
       (Normal, Normal-but-flagged-by-the-safety-net, and Distress images);
     - no profile = no 'profile' / 'views_recorded' keys (the citizen result is unchanged);
     - an unknown profile is refused.
2. The HTTP layer (TestClient, fake classifier):
     - no profile: predict() is called exactly as before and the reply bytes equal the plain
       ClassificationResponse serialisation (FastAPI's JSON encoding);
     - profile=video (form field, and the base64 JSON key) adds only 'profile' and
       'views_recorded'; bad profile -> 400; empty profile -> default path.
3. Rate limit: 10/minute still applies to remote clients and to loopback requests that carry a
   proxy header (the Cloudflare tunnel connects from 127.0.0.1); unproxied loopback is exempt.

Expected values come from the setup (counts of requests, tiles from the config, thresholds),
not from the code under test.

Usage:
    venv/Scripts/python.exe scripts/tests/test_classify_profile.py
"""

from __future__ import annotations

import base64
import copy
import io
import json
import os
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("SKIP_MODEL_SELFTEST", "1")
# The production settings this module reads at import time (CLAUDE.md START HERE).
os.environ["STAGE2_MODE"] = "shadow"
os.environ["STAGE1_SAFETY_NET"] = "true"
os.environ["STAGE2_PROBE_CONFIG"] = str(PROJECT_ROOT / "configs" / "stage2_probe.json")

from PIL import Image  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))
        FAILURES.append(name)


# ============================================================
# 1. Model level
# ============================================================

class FakeProber:
    """P(yes) per type from the mean brightness of the view, so tiles differ from the full photo."""

    def __init__(self, keys: list[str], hot: dict[str, float] | None = None):
        self.keys = keys
        self.hot = hot or {}
        self.calls = 0

    def probe(self, image: Image.Image) -> list[dict]:
        self.calls += 1
        g = image.convert("L").resize((8, 8))
        mean = sum(g.getdata()) / 64 / 255
        out = []
        for i, k in enumerate(self.keys):
            p = self.hot.get(k, 0.01 + 0.02 * ((i * 7) % 5) * mean)
            out.append({"key": k, "p_yes": p, "yes_no_mass": 0.97})
        return out


def make_classifier(stage1_distressed: bool, hot: dict | None = None):
    from app import model as m
    from scripts import stage2_probe_rules as rules
    from scripts.stage2_probe import all_probe_types

    cfg = rules.load_config(m.STAGE2_PROBE_CONFIG)
    types = all_probe_types()
    rules.validate_config(cfg, known_keys={t.key for t in types})
    clf = m.PavementClassifier.__new__(m.PavementClassifier)
    clf.model_path, clf.adapter_path = "fake", None
    clf._lock = threading.Lock()
    clf._load_info = {"max_pixels": 1024 * 1024}
    clf._prober = FakeProber([t.key for t in types], hot)
    clf._locator = None
    clf._probe_cfg = cfg
    clf._probe_labels = {t.key: t.output_label for t in types}
    clf._probe_status = {"enabled": True}
    clf._model = None

    def stage1(image):
        return {"is_distressed": stage1_distressed, "stage1_label": "Distress" if stage1_distressed else "Normal",
                "stage1_confidence": 0.91, "needs_expert_review": False, "stage1_time_ms": 1.0,
                "stage1_raw": "Distress" if stage1_distressed else "Normal"}

    clf.predict_stage1 = stage1
    clf._run_inference_with_scores = lambda image, s, u: (
        "DISTRESS_TYPES: Potholes, Ravelling\nSEVERITY: Medium\nDESCRIPTION: two potholes", None, None)
    clf._stage2_confidences = lambda scores, ids, parsed: {
        "stage2_confidence": 0.86, "stage2_confidence_field": 0.86, "stage2_confidence_sequence": 0.7,
        "stage2_confidence_primary": 0.9, "stage2_type_confidences": {"Potholes": 0.9, "Ravelling": 0.8}}
    clf._has_active_adapter = lambda: False
    return clf


def image() -> Image.Image:
    # 1600x900 gradient: tiles have different brightness from the whole photo.
    im = Image.new("RGB", (1600, 900))
    im.putdata([((x * 255) // 1600, (y * 255) // 900, 90) for y in range(900) for x in range(1600)])
    return im


def strip_views(d):
    """The decision part of a result: drop timings, the views block and the profile keys."""
    if isinstance(d, dict):
        return {k: strip_views(v) for k, v in d.items()
                if k not in ("views", "time_ms", "processing_time_ms", "stage2_time_ms", "profile", "views_recorded")}
    return d


def test_model_level() -> None:
    print("\n[1] PavementClassifier.predict(profile=...) on the real pipeline code")
    from app import model as m
    cfg_views = json.loads((PROJECT_ROOT / "configs" / "stage2_probe.json").read_text(encoding="utf-8"))["views"]
    check("production config decides on the full photo and records tiles",
          cfg_views.get("mode") == "full" and cfg_views["record"]["mode"] == "tile", str(cfg_views)[:120])
    img = image()
    sn_keys = json.loads((PROJECT_ROOT / "configs" / "stage2_probe.json").read_text(encoding="utf-8"))[
        "stage1_safety_net"]
    cases = [("Normal, not flagged", False, None, False),
             # A headline type above the safety-net threshold: Normal must go to review under BOTH profiles.
             ("Normal, flagged by the safety net", False, {sn_keys["keys"][0]: sn_keys["threshold"] + 0.2}, True),
             ("Distress", True, None, None)]
    for name, distressed, hot, flagged in cases:
        c_def = make_classifier(distressed, hot)
        c_vid = make_classifier(distressed, hot)
        c_cit = make_classifier(distressed, hot)
        r_def = c_def.predict(img.copy())
        r_vid = c_vid.predict(img.copy(), profile="video")
        r_cit = c_cit.predict(img.copy(), profile="citizen")
        check(f"{name}: default result has no profile keys", "profile" not in r_def and "views_recorded" not in r_def)
        check(f"{name}: video probes the full photo once", c_vid._prober.calls == 1, str(c_vid._prober.calls))
        block = (r_def.get("stage2_probe") or r_def.get("stage1_safety_net") or {}).get("views", {})
        n_tiles = len(block.get("recorded", {}).get("views", []))
        check(f"{name}: default probes full photo + recorded tiles", n_tiles >= 2 and c_def._prober.calls == 1 + n_tiles,
              f"calls {c_def._prober.calls}, tiles {n_tiles}")
        check(f"{name}: explicit citizen == default probing", c_cit._prober.calls == c_def._prober.calls)
        check(f"{name}: video says profile video, views_recorded false",
              r_vid.get("profile") == "video" and r_vid.get("views_recorded") is False)
        check(f"{name}: citizen says views_recorded true",
              r_cit.get("profile") == "citizen" and r_cit.get("views_recorded") is True)
        check(f"{name}: every decision field identical (video vs default)", strip_views(r_vid) == strip_views(r_def),
              json.dumps(strip_views(r_vid))[:300] + " vs " + json.dumps(strip_views(r_def))[:300])
        check(f"{name}: citizen == default", strip_views(r_cit) == strip_views(r_def))
        if flagged is not None:
            check(f"{name}: needs_expert_review = {flagged} under both", r_def["needs_expert_review"] is flagged
                  and r_vid["needs_expert_review"] is flagged, str((r_def["needs_expert_review"], r_vid["needs_expert_review"])))
            check(f"{name}: safety net flagged = {flagged} under both",
                  r_def["stage1_safety_net"]["flagged"] is flagged and r_vid["stage1_safety_net"]["flagged"] is flagged)
            check(f"{name}: video safety net has no recorded tiles",
                  "recorded" not in r_vid["stage1_safety_net"]["views"] and "recorded" in r_def["stage1_safety_net"]["views"])
        else:
            check(f"{name}: same labels, severity, Stage 2 confidence",
                  (r_vid["distress_types"], r_vid["severity"], r_vid["stage2_confidence"]) ==
                  (r_def["distress_types"], r_def["severity"], r_def["stage2_confidence"]) == (["Potholes", "Ravelling"], "Medium", 0.86),
                  str((r_vid["distress_types"], r_vid["severity"], r_vid["stage2_confidence"])))
            check(f"{name}: shadow probe P(yes) identical", r_vid["stage2_probe"]["p"] == r_def["stage2_probe"]["p"])
            check(f"{name}: shadow probe not applied under either", r_vid["stage2_probe"]["applied"] is False
                  and r_def["stage2_probe"]["applied"] is False)
    # The config object is not mutated by a video request (a later citizen request still records).
    c = make_classifier(False)
    before = copy.deepcopy(c._probe_cfg)
    c.predict(img.copy(), profile="video")
    check("video request leaves the probe config untouched", c._probe_cfg == before)
    r_after = c.predict(img.copy())
    check("a citizen request after a video one still records tiles", "recorded" in r_after["stage1_safety_net"]["views"])
    try:
        make_classifier(False).predict(img.copy(), profile="drone")
        check("unknown profile refused", False, "no error")
    except ValueError as e:
        check("unknown profile refused", "drone" in str(e))
    check("REQUEST_PROFILES: citizen records, video does not",
          m.REQUEST_PROFILES == {"citizen": {"record_views": True}, "video": {"record_views": False}})


# ============================================================
# 2 + 3. HTTP layer
# ============================================================

RESULT = {"is_distressed": False, "stage1_label": "Normal", "stage1_confidence": 0.93, "distress_types": [],
          "severity": "None", "description": "", "stage2_confidence": 0.0, "needs_expert_review": True,
          "stage1_time_ms": 1500.0, "stage2_time_ms": 0.0, "stage1_raw": "Normal", "stage2_raw": "",
          "stage1_safety_net": {"flagged": True, "type": "Potholes", "p_yes": 0.31},
          "processing_time_ms": 3100.0}


class FakeClassifier:
    def __init__(self):
        self.calls: list[tuple] = []

    def predict(self, image, **kw):
        self.calls.append(tuple(sorted(kw.items())))
        r = dict(RESULT)
        if kw.get("profile") is not None:
            r["profile"] = kw["profile"]
            r["views_recorded"] = kw["profile"] != "video"
        return r


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


def build(main, clf):
    from contextlib import asynccontextmanager

    main.get_classifier = lambda *a, **k: clf

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    main.app.router.lifespan_context = noop_lifespan


def test_http() -> None:
    print("\n[2] /classify and /classify/base64 profile plumbing")
    from fastapi.testclient import TestClient
    import app.main as main
    from app.schemas import ClassificationResponse

    clf = FakeClassifier()
    build(main, clf)
    main.limiter.enabled = False
    client = TestClient(main.app)
    files = {"file": ("a.png", png_bytes(), "image/png")}
    r = client.post("/classify", files=files)
    expected = json.dumps(ClassificationResponse(**RESULT).model_dump(mode="json"), ensure_ascii=False,
                          allow_nan=False, indent=None, separators=(",", ":")).encode("utf-8")
    check("default: 200", r.status_code == 200, r.text[:200])
    check("default: predict() called with no keyword (the path every citizen request takes)", clf.calls[-1] == ())
    check("default: reply bytes = plain ClassificationResponse", r.content == expected,
          r.content[:200].decode() + " vs " + expected[:200].decode())
    r = client.post("/classify", files=files, data={"profile": "video"})
    body = r.json()
    check("video: 200", r.status_code == 200, r.text[:200])
    check("video: predict(profile='video')", clf.calls[-1] == (("profile", "video"),))
    check("video: reply says profile video, views_recorded false",
          body.get("profile") == "video" and body.get("views_recorded") is False, str(body)[:200])
    check("video: every other key equal to the default reply",
          {k: v for k, v in body.items() if k not in ("profile", "views_recorded")} == json.loads(expected))
    r = client.post("/classify", files=files, data={"profile": " Video "})
    check("profile is case/space-insensitive", r.status_code == 200 and clf.calls[-1] == (("profile", "video"),))
    r = client.post("/classify", files=files, data={"profile": ""})
    check("empty profile = default path", r.status_code == 200 and clf.calls[-1] == () and r.content == expected)
    n = len(clf.calls)
    r = client.post("/classify", files=files, data={"profile": "turbo"})
    check("unknown profile -> 400, nothing classified", r.status_code == 400 and len(clf.calls) == n
          and "turbo" in r.text, r.text[:200])
    r = client.post("/classify", files=files, data={"profile": "citizen"})
    check("explicit citizen: views_recorded true", r.status_code == 200 and r.json().get("views_recorded") is True)
    b64 = base64.b64encode(png_bytes()).decode()
    r = client.post("/classify/base64", json={"image": b64})
    check("base64 default: same bytes as /classify default", r.status_code == 200 and r.content == expected
          and clf.calls[-1] == ())
    r = client.post("/classify/base64", json={"image": b64, "profile": "video"})
    check("base64 video", r.status_code == 200 and r.json().get("profile") == "video"
          and clf.calls[-1] == (("profile", "video"),), r.text[:200])
    r = client.post("/classify/base64", json={"image": b64, "profile": 5})
    check("base64 non-string profile -> 400", r.status_code == 400)

    print("\n[3] rate limit: loopback exempt, remote and proxied keep 10/minute")
    main.limiter.enabled = True

    def burst(client_addr, headers=None, n=13, path="/classify"):
        main.limiter.reset()
        c = TestClient(main.app, client=client_addr)
        codes = []
        for _ in range(n):
            if path == "/classify":
                codes.append(c.post(path, files=files, headers=headers or {}).status_code)
            else:
                codes.append(c.post(path, json={"image": b64}, headers=headers or {}).status_code)
        return codes

    try:
        codes = burst(("203.0.113.7", 40000))
        check("remote client: 10 answered, then 429", codes[:10] == [200] * 10 and codes[10:] == [429] * 3, str(codes))
        codes = burst(("127.0.0.1", 40000))
        check("unproxied 127.0.0.1: 13 of 13 answered", codes == [200] * 13, str(codes))
        codes = burst(("::1", 40000))
        check("unproxied ::1: 13 of 13 answered", codes == [200] * 13, str(codes))
        codes = burst(("127.0.0.1", 40000), path="/classify/base64")
        check("unproxied loopback exempt on /classify/base64 too", codes == [200] * 13, str(codes))
        for hdr in ({"Cf-Connecting-Ip": "198.51.100.4"}, {"X-Forwarded-For": "198.51.100.4"}):
            codes = burst(("127.0.0.1", 40000), headers=hdr)
            check(f"loopback via a proxy ({list(hdr)[0]}): still 10/minute",
                  codes[:10] == [200] * 10 and codes[10:] == [429] * 3, str(codes))
        codes = burst(("203.0.113.7", 40000), path="/classify/base64")
        check("remote client on /classify/base64: 10 then 429", codes[10:] == [429] * 3, str(codes))
    finally:
        main.limiter.reset()
        main.limiter.enabled = False

    from starlette.requests import Request

    def req(host, headers=()):
        return Request({"type": "http", "client": (host, 1), "headers": [(k.encode(), v.encode()) for k, v in headers]})

    check("is_local_unproxied: 127.0.0.5 (loopback /8)", main.is_local_unproxied(req("127.0.0.5")))
    check("is_local_unproxied: ::ffff:127.0.0.1", main.is_local_unproxied(req("::ffff:127.0.0.1")))
    check("is_local_unproxied: LAN address is not local", not main.is_local_unproxied(req("192.168.1.20")))
    check("is_local_unproxied: 'testclient' (not an IP) is not local", not main.is_local_unproxied(req("testclient")))
    check("is_local_unproxied: Forwarded header counts as proxied",
          not main.is_local_unproxied(req("127.0.0.1", [("forwarded", "for=1.2.3.4")])))
    check("remote key unchanged (get_remote_address)", main.classify_rate_key(req("203.0.113.7")) == "203.0.113.7")
    check("limit strings", main.classify_rate_limit(main.classify_rate_key(req("203.0.113.7"))) == "10/minute"
          and main.classify_rate_limit(main.classify_rate_key(req("127.0.0.1"))) != "10/minute")


def main() -> int:
    test_model_level()
    test_http()
    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("FAILED: " + "; ".join(FAILURES))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
