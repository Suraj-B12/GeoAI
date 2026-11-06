-- ============================================================
-- Migration 008: idempotent expert saves + an event log of save problems (local PostgreSQL only)
-- ============================================================
-- op_id: a random id the expert page gives each save. The page keeps the save in a
-- browser-side queue until the server confirms it, so a save may arrive twice (the answer
-- to the first try was lost, the network came back, two tabs flushed the same queue).
-- The unique index makes the second arrival a replay of the first: the server answers
-- with the stored result instead of writing the review again or refusing it as a
-- conflict with itself.
--
-- research.expert_events: anything that did not go to plan with a save, from the page
-- (a retry, a save that succeeded only after retries, a conflict, a rejected save) and
-- from the server (Supabase or the database unreachable, a deadlock retried). Read by
-- GET /expert-api/events/summary and status.ps1. Never holds review content beyond ids.
--
-- In schema `research`, like migrations 006 and 007. Additive and idempotent. Applied by
-- scripts/setup_local_db.py, and by app/expert_review.ensure_schema() on first use.
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS research;

ALTER TABLE research.expert_review_log ADD COLUMN IF NOT EXISTS op_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS uq_expert_review_log_op_id
    ON research.expert_review_log (op_id) WHERE op_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_expert_review_log_reviewer
    ON research.expert_review_log (lower(btrim(reviewer_name)), created_at);

CREATE TABLE IF NOT EXISTS research.expert_events (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    origin TEXT NOT NULL CHECK (origin IN ('client', 'server')),
    kind TEXT NOT NULL,                   -- save_retry, save_recovered, save_conflict, save_rejected, ...
    level TEXT NOT NULL DEFAULT 'warn' CHECK (level IN ('info', 'warn', 'error')),
    op_id TEXT,
    source TEXT,                          -- video | citizen
    item_id TEXT,
    reviewer_name TEXT,
    reviewer_session TEXT,
    http_status INTEGER,
    attempts INTEGER,
    message TEXT,
    detail JSONB
);

CREATE INDEX IF NOT EXISTS idx_expert_events_created ON research.expert_events (created_at);
CREATE INDEX IF NOT EXISTS idx_expert_events_op ON research.expert_events (op_id) WHERE op_id IS NOT NULL;

COMMIT;
