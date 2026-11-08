-- ============================================================
-- Migration 010: per-type probe scores for drive-video segments (local PostgreSQL only)
-- ============================================================
-- The video segments were classified with the free-form Stage 2 list only, which never names
-- Ravelling, Patching or Settlement (0 of 7,050 segments). scripts/video_probe_scores.py runs the
-- per-type Yes/No probe (scripts/stage2_probe.py, the production variant, full photo) on every stored
-- segment image and writes P(yes) per type here, one row per (segment, probe configuration).
--
-- These are RAW scores, not decisions: app/pci_core turns them into labels only through a calibration
-- file (eval_results/pci/param_calibration.json) whose thresholds come from a survey/expert split.
--
-- probe_fp: hash of the model, probe variant, question set and max_pixels; a change writes new rows.
-- One INSERT per image = the scoring run's checkpoint (re-runs skip rows already present).
-- In schema `research`, like migrations 006-009. Additive and idempotent.
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE IF NOT EXISTS research.video_probe_scores (
    segment_id TEXT NOT NULL,
    probe_fp TEXT NOT NULL CHECK (probe_fp ~ '^[0-9a-f]{16}$'),
    image_sha256 TEXT,
    scores JSONB,                          -- {probe key: P(yes)}; NULL when error is set
    error TEXT,
    time_s REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (segment_id, probe_fp),
    CHECK ((scores IS NULL) <> (error IS NULL))
);

COMMIT;
