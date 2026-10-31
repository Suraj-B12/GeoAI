-- ============================================================
-- Migration 006: expert review log (local PostgreSQL only)
-- ============================================================
-- One row per expert save, append-only: a later "Update review" adds a row
-- and never edits an earlier one. Each row holds what the model answered
-- (read server-side from the assessment at save time, before the expert's
-- values are written), what the expert answered, and the comparison
-- (app/expert_review.compare). This is the record the paper's
-- "model vs expert" numbers are computed from (scripts/expert_review_metrics.py).
--
-- Both sources are logged here:
--   video    drive-video segments (assessments in this database; the expert
--            UI's Video tab saves through POST /expert-api/video/review)
--   citizen  RoadSide uploads (assessments in Supabase; the expert UI saves
--            to Supabase as before, then POST /expert-api/citizen/log)
--
-- In its own schema `research`, outside `public`, so the local schema stays
-- identical to Supabase's (scripts/tests/test_local_db.py compares `public`).
-- Additive and idempotent. Applied by scripts/setup_local_db.py, and by
-- app/expert_review.ensure_schema() on first use.
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE IF NOT EXISTS research.expert_review_log (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    source TEXT NOT NULL CHECK (source IN ('video', 'citizen')),
    assessment_id TEXT NOT NULL,          -- assessments.id (local for video, Supabase for citizen)
    segment_id TEXT,                      -- video_segments.segment_id (video only)
    session_id TEXT,                      -- the drive (video only)
    queue TEXT NOT NULL,                  -- which list the item was reviewed from (sampling!)
    action TEXT NOT NULL CHECK (action IN ('confirm', 'update')),
    reviewer_name TEXT NOT NULL,

    -- The model's answer when the expert saved (status keeps the gate's decision)
    vlm JSONB NOT NULL,
    -- The expert's answer; types = ["Normal"] means no distress
    expert_types JSONB NOT NULL,
    expert_severity TEXT NOT NULL CHECK (expert_severity IN ('None', 'Low', 'Medium', 'High')),
    expert_notes TEXT,
    -- app/expert_review.compare(vlm, expert): stage1 / types / severity verdicts
    comparison JSONB NOT NULL,

    -- "Where the model looked" overlay: shown?, which types, the expert's rating
    heatmap JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- UI facts: ms from the first interaction with the card to the save
    ui JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_expert_review_log_assessment
    ON research.expert_review_log (source, assessment_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_expert_review_log_created
    ON research.expert_review_log (created_at DESC);

COMMIT;
