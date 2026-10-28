# VLM time per drive-video frame: `profile=video`

**Result.** Skipping the recording of per-tile probe scores for video frames cuts a Stage-1-Normal answer from 13.72 s to 3.76 s. End to end through `run_vlm_batch.py`, time falls from 14.4 to 5.2 s per image (2.76×). Every answer stays identical: 60 of 60 frames matched in every decision field.

## Change

- **What `profile=video` does:** an opt-in per-request option on `/classify` (form field) and `/classify/base64` (JSON key). It skips only the *recorded* probe tiles, in Stage 2 and in the Stage 1 safety net.
- **What it leaves alone:** labels, confidences, the full-photo shadow probe and the safety-net decision come from the same code. A request without a profile runs and answers byte-for-byte as before. The tiles are kept for citizen photos, for future calibration; the video project never reads them.
- **Rate-limit exemption:** loopback requests that carry no proxy header skip the 10/min limit. The Cloudflare tunnel also connects from 127.0.0.1 but adds `Cf-Connecting-Ip`/`X-Forwarded-For`, so tunnel users keep the limit.
- **Runner:** `run_vlm_batch.py` v3 gains `--profile`. `--min-interval` defaults to 0 for a loopback URL. `--skip-batch-check` lets the toggle verify the batch's checksums once per classify pass instead of every 10 images; each image is still hashed as it is uploaded.
- **Bug fix:** the toggle read `report["timings"]`, but the key is `timings_s`, so extraction timings were missing from session metrics. Now fixed.

## Method

- **Sample:** 60 frames from `batch_001` (seed 20261028): 40 with Stage 1 Normal and 20 with Stage 1 Distress, spread over the 3 drives.
- **Procedure:** each frame was sent with and without the profile, alternating which went first, on the A5000 at 4-bit and the production configuration (`/health` P4, parity ok). 95 % CIs by bootstrap over frames.
- **End-to-end run:** a separate set of 30 frames (24 Normal, 6 Distress), not among the 60.

## Results

| Answer | Before: median [95 % CI], p90 | After: median [95 % CI], p90 | Saving per frame |
|---|---|---|---|
| Stage 1 Normal (n = 40) | 13.72 s [13.38, 14.29], 15.33 | 3.76 s [3.62, 3.94], 4.18 | 10.02 s [9.67, 10.34] |
| Stage 1 Distress (n = 20) | 24.41 s [22.80, 25.74], 27.59 | 14.43 s [13.88, 15.69], 16.74 | 9.80 s [9.41, 10.22] |
| All (n = 60) | 14.48 s, p90 25.41 | 4.00 s, p90 15.45 | 9.84 s [9.66, 10.24] |

- **Where the saving comes from:** the saving equals the recorded-tile time (9.92 s [9.65, 10.25]). Stage 1 itself takes 1.58 s.
- **Survey mix:** at `batch_001`'s mix (649 Normal, 164 Distress), the mean per image falls from 16.1 s to 6.0 s, so VLM time per km in send-all mode falls from about 0.85 h to about 0.33 h.
- **End to end:** the runner took 432 s before (14.4 s/image; default profile, 6.5 s pacing) and 156 s after (5.2 s/image; `--profile video`, no pacing on loopback). The pacing alone accounts for about 2.4 s of that difference.
- **Batch re-hash:** re-hashing `batch_001` (1,633 files, 4.90 GB) takes 16–19 s. Done once per 10-image chunk, that cost 1.6–1.9 s per image; it now happens once per pass.
- **Identity:**
  - 60/60 paired answers are identical in Stage 1 label, confidence and raw text, distress types, severity, description, Stage 2 confidences and raw text, the review flag, and the safety net's flag, score and 20 per-type P(yes) values.
  - The default answers also equal `batch_001`'s recorded answers for the same 60 frames (repeatability across days).
  - 30/30 frames agree between the two runner runs.

## Tests

- **Capstone:** `test_classify_profile` 66/66 (new), `test_video_pipeline` 161/161, `validate_all` 18/18, `test_gis_routes` 160/160; the other GPU-free suites unchanged.
- **GeoAI_Video:** `test_vlm` 18 passed (5 new).

## Open points

1. **Sign-off:** the VLM team should approve the opt-in profile. It changes no answer and only skips work nobody reads for video frames.
2. **Other proxies:** the loopback exemption assumes any local proxy adds a forwarding header (cloudflared does).
3. **Time-left estimate:** `geoai/analyze.py` still assumes at least 6.5 s per image, which now overstates the time left on loopback.
4. **Keep the repos in step:** the toggle passes `--profile`, which an older GeoAI_Video runner rejects.

Files: `results.json`, `ab_raw.jsonl`, `ab_sample.json`, `hash_time.json`, `runner_before/`, `runner_after/`, `fig_vlm_speed.png`, scripts `ab_measure.py`, `make_runner_batches.py`, `time_runner.py`, `hash_time.py`, `analyze.py`.
