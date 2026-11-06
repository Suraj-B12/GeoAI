-- ============================================================
-- Migration 007: review claims, so several experts can review at once (local PostgreSQL only)
-- ============================================================
-- A claim is a short lease on one item (a video segment or a citizen
-- assessment): while it is live, only the browser session that holds it is
-- shown the item in a review list and can save it. Leases last about ten
-- minutes and the page renews them every minute; a save, a skip or closing
-- the page releases them, and an expired lease is free for anyone.
--
-- The server is the only writer (app/expert_review.py). Handing out items is
-- one INSERT ... ON CONFLICT (source, item_id) DO UPDATE ... WHERE the old
-- lease expired or belongs to the same session, so two sessions can never
-- hold the same item at the same time.
--
-- reviewer_session is a random id each browser keeps in localStorage; it is
-- never shown to other experts (only reviewer_name is).
--
-- review_skips remembers what a session skipped, so the same expert is not
-- handed it straight back (others still are). The random-audit sample should
-- not be skipped by eye; skips are counted so the paper can report them.
--
-- In schema `research`, outside `public`, like migration 006. Additive and
-- idempotent. Applied by scripts/setup_local_db.py, and by
-- app/expert_review.ensure_schema() on first use.
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE IF NOT EXISTS research.review_claims (
    source TEXT NOT NULL CHECK (source IN ('video', 'citizen')),
    item_id TEXT NOT NULL,                -- video_segments.segment_id, or the Supabase assessments.id
    reviewer_session TEXT NOT NULL,       -- random per-browser id (never shown to others)
    reviewer_name TEXT NOT NULL,
    queue TEXT,                           -- the list the item was handed out from
    claimed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (source, item_id)
);

CREATE INDEX IF NOT EXISTS idx_review_claims_session ON research.review_claims (reviewer_session);
CREATE INDEX IF NOT EXISTS idx_review_claims_expires ON research.review_claims (expires_at);

CREATE TABLE IF NOT EXISTS research.review_skips (
    source TEXT NOT NULL CHECK (source IN ('video', 'citizen')),
    item_id TEXT NOT NULL,
    reviewer_session TEXT NOT NULL,
    reviewer_name TEXT NOT NULL,
    queue TEXT,
    skipped_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source, item_id, reviewer_session)
);

COMMIT;
