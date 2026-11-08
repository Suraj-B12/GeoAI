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
  eval_results/pci/iri/iri_calibration.json. Until then every report field says "uncalibrated".
- Wiring into app/pci_irc2023.py / the report: use pci_iri.aggregate() over a section's segment rows, then
  pci_iri.report_field().
