-- ============================================================
-- Migration 011: road roughness features per drive-video segment (local PostgreSQL only)
-- ============================================================
-- scripts/iri_from_imu.py reads each drive's imu.csv (phone accelerometer/gyro, 200 Hz) and writes one
-- row per (5 m video segment, method). app/pci_iri.py is the method; eval_results/pci/iri/RESEARCH.md
-- the reasoning and VALIDATION.md the checks.
--
-- These are RAW features plus an UNCALIBRATED Golden-Car-equivalent index, NOT an IRI. A calibrated IRI
-- (m/km) is computed at read time by app/pci_iri.calibrated_iri() from eval_results/pci/iri/
-- iri_calibration.json (fitted on the survey's calibration half), on 100 m aggregates; without that file
-- the report says "uncalibrated".
--
-- method_fp: hash of app/pci_iri.Params (bands, filter, resampling, speed limits, Golden Car constants,
-- version); a method change writes new rows next to the old ones.
-- error: why a segment has no features (not covered, outside the IMU record, no imu.csv ...).
-- In schema `research`, like migrations 006-010. Additive and idempotent.
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE IF NOT EXISTS research.segment_roughness (
    segment_id TEXT NOT NULL,
    method_fp TEXT NOT NULL CHECK (method_fp ~ '^[0-9a-f]{16}$'),
    session_id TEXT NOT NULL,
    k INTEGER,
    imu_sha256 TEXT CHECK (imu_sha256 IS NULL OR imu_sha256 ~ '^[0-9a-f]{64}$'),
    t0_ns BIGINT,                          -- window on the session clock (CLOCK_BOOTTIME), first/last sample
    t1_ns BIGINT,
    duration_s REAL,
    n_samples INTEGER,
    coverage REAL,                         -- share of the window with IMU samples
    speed_mps REAL,                        -- segment length / time on it
    speed_min_mps REAL,                    -- lowest GNSS Doppler speed inside the window
    rms_av REAL,                           -- RMS vertical acceleration, 0.5-15 Hz (m/s^2)  [primary feature]
    rms_body REAL,                         -- 0.5-5 Hz
    rms_wheel REAL,                        -- 5-15 Hz
    rms_rot REAL,                          -- roll+pitch rate, 0.5-15 Hz (rad/s)
    peak_av REAL,                          -- max |a_v| in the window (m/s^2)
    rms_av_norm REAL,                      -- rms_av / sqrt(speed_mps)
    iri_gc_uncal REAL,                     -- Golden-Car-equivalent index (m/km), UNCALIBRATED, NULL below min speed
    flags TEXT[] NOT NULL DEFAULT '{}',    -- low_speed, stop_in_window, filter_edge, imu_gap, sparse_anchors, too_few_samples
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (segment_id, method_fp),
    CHECK ((error IS NULL) OR (rms_av IS NULL AND iri_gc_uncal IS NULL))
);

CREATE INDEX IF NOT EXISTS idx_segment_roughness_session
    ON research.segment_roughness(session_id, method_fp);

-- One row per (drive, method): the idle noise floor and run facts, for validation.
CREATE TABLE IF NOT EXISTS research.session_roughness (
    session_id TEXT NOT NULL,
    method_fp TEXT NOT NULL CHECK (method_fp ~ '^[0-9a-f]{16}$'),
    imu_path TEXT,
    imu_sha256 TEXT,
    summary JSONB,                         -- {idle: {idle_s, rms_av, n_stops}, acc_one_g, gravity_source, n_anchors, ...}
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (session_id, method_fp)
);

COMMIT;
