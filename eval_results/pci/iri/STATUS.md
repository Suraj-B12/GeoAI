# STATUS — IRI (roughness) from phone IMU (workstream 2026-11-08) — ALL STEPS DONE

Owned files: app/pci_iri.py, scripts/iri_from_imu.py, migrations/011_segment_roughness.sql,
scripts/tests/test_pci_iri.py, eval_results/pci/iri/* (RESEARCH.md, VALIDATION.md, validate_iri.py,
explore_iri.py, validation.json, explore.json, run.log). Nothing committed (coordinator commits).
Migration 011 is NOT registered in scripts/setup_local_db.py (coordinator does); iri_from_imu.py applies it itself.

## Facts a new session needs
- IMU: <session>/imu.csv `t_ns,sensor,x,y,z,w,accuracy`; acc/gyro 200 Hz, gravity/game_rv/mag 50 Hz; t_ns =
  CLOCK_BOOTTIME, the clock of frames.csv, gnss_fixes.csv and segment records' frame.t_ns.
  Found in $VIDEO_INBOX (C:\GeoAI\inbox)\<session>\ (40 drives) or GeoAI_Video\field\sessions\<session>\ (3).
- Segment -> IMU samples: anchors (frame.t_ns, s_cam = s0_m − frame.near_edge_ahead_m) from every covered
  segment = the video pipeline's own track distance s(t); between anchors the GNSS Doppler distance is warped
  onto them (handles stops); a segment's window = grid times with s(t) + axle_offset_m (0) in [s0_m, s1_m).
- Spectrum: engine line at ~25 Hz dominates; main band 0.5–15 Hz (RESEARCH §5).
- method_fp e9e1bc016ba5fa70 (current Params). Table research.segment_roughness: 8,402 rows for 43 drives;
  research.session_roughness: 43 rows.

## Steps
1. [x] data look
2. [x] RESEARCH.md
3. [x] app/pci_iri.py + scripts/tests/test_pci_iri.py (9/9 pass, CPU, ~1 min)
4. [x] migration 011 + scripts/iri_from_imu.py, run on geoai (all 43 drives, ~1 min total)
5. [x] VALIDATION.md: criteria written 09:30Z before results; results: S1/S2/S4 PASS, V2 noise floor FAIL
   (0.39 > 0.25), V3 repeatability not testable (3 blocks), V4 uninformative, V5 potholes PASS (AUC 0.68).

## Resume / re-run
    venv/Scripts/python.exe scripts/iri_from_imu.py --database geoai            # skips stored drives; --redo recomputes
    venv/Scripts/python.exe eval_results/pci/iri/validate_iri.py --database geoai
    CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe scripts/tests/test_pci_iri.py

## Open (not this workstream's to decide)
- Repeated same-lane passes (3–5 over a few km, two speeds) to test 100 m repeatability and the speed exponent.
- Survey calibration half -> pci_iri.fit_calibration(rows of aggregate() + iri_ref, source=...) ->
  eval_results/pci/iri/iri_calibration.json. Until then report_field gives the uncalibrated scooter interval.
- Wiring into app/pci_irc2023.py / the report: use pci_iri.aggregate() over a section's segment rows, then
  pci_iri.report_field().

# Task 2 (2026-11-09)
VEHICLE CORRECTION (owner, 2026-11-09): drives were on a SCOOTER, phone on handlebar/front — not a car/windshield.
RESEARCH.md/VALIDATION.md v1 text assumed a car; corrected (RESEARCH §6, VALIDATION header).
Goal: IRI estimate with honest uncertainty + noise floor re-check + rut prior
Owned additionally: eval_results/pci/rut/*, app/pci_rut.py (new).
1. [x] UNCERTAINTY.md (literature incl. scooter-vs-car 1.9x, apps 3.7x apart; Bengaluru IRI 1.5-5.6; IRC SP:16)
2. [x] iri_interval(agg, calibration=None) + report_field(agg, calibration=None) in app/pci_iri.py; sensitivity.py ->
   sensitivity.json (scooter R p05/p50/p95 0.77/1.59/3.08 pooled; SENS_R embedded, test checks it);
   interval_summary.py -> interval_summary.json (100 m: median lo/point/hi 1.69/5.4/17.6 m/km)
3. [x] noise-floor re-check (NOISE_FLOOR.md, noise_floor.py/json): neither subtraction adopted (representativeness 69% < 80%);
   C2 floor 0.257 m/s2 = lower end of interval (median 100 m index 7.94 -> 7.33). No new drives; 3 repeat blocks.
4. [x] rut: eval_results/pci/rut/RUT_PRIOR.md + app/pci_rut.py rut_prior()/score_stats() (prior 0-25 mm, point None)
5. [x] tests 13/13 (scripts/tests/test_pci_iri.py); iri_from_imu.py --redo run so summaries carry vehicle.
   RESEARCH.md §6 + VALIDATION.md header corrected for the scooter. DONE; nothing committed.
