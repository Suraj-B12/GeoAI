-- ============================================================
-- Migration 001: Operator Pipeline Support
-- ============================================================
-- Adds the worker queue infrastructure on top of the existing
-- assessments table. Safe to run on a populated database:
--   - All ALTERs are additive
--   - All CREATEs use IF NOT EXISTS
--   - Existing rows are backfilled with sensible defaults
--   - Re-runnable (idempotent)
--
-- What this enables:
--   - RoadSide app inserts a row with image+GPS only (status='pending')
--   - Operator worker atomically claims a batch (status -> 'processing')
--   - Worker fills in stage1/stage2 fields, sets status -> 'classified' / 'expert_review' / 'failed'
--   - Heartbeat table lets the dashboard detect worker stalls
--   - Crash recovery: rows stuck in 'processing' > 5 min get reset to 'pending'
-- ============================================================

BEGIN;

-- ============================================================
-- 1. Make stage1/stage2 columns nullable (RoadSide app inserts without them)
-- ============================================================
-- These were NOT NULL in the original schema, but we now insert pending
-- rows BEFORE classification runs.

ALTER TABLE assessments ALTER COLUMN stage1_label DROP NOT NULL;
ALTER TABLE assessments ALTER COLUMN stage1_confidence DROP NOT NULL;
ALTER TABLE assessments ALTER COLUMN is_distressed DROP NOT NULL;


-- ============================================================
-- 2. Add status enum + worker tracking columns
-- ============================================================

ALTER TABLE assessments
    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'classified', 'expert_review', 'done', 'failed'));

-- When the worker claimed this row (used for stale-claim detection)
ALTER TABLE assessments
    ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMPTZ;

-- Which worker instance claimed it (heartbeat correlation)
ALTER TABLE assessments
    ADD COLUMN IF NOT EXISTS claimed_by TEXT;

-- Number of failed attempts before giving up
ALTER TABLE assessments
    ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0;

-- Last error if processing failed
ALTER TABLE assessments
    ADD COLUMN IF NOT EXISTS error_message TEXT;

-- When pipeline finished (success or failure)
ALTER TABLE assessments
    ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ;


-- ============================================================
-- 3. Backfill status for existing rows
-- ============================================================
-- Rows already classified (stage1_label populated) -> 'done'
-- Rows flagged for expert review and not yet reviewed -> 'expert_review'
-- Everything else -> 'pending' (default already applied)

UPDATE assessments
SET status = CASE
    WHEN expert_reviewed = TRUE THEN 'done'
    WHEN needs_expert_review = TRUE AND expert_reviewed = FALSE THEN 'expert_review'
    WHEN stage1_label IS NOT NULL THEN 'classified'
    ELSE 'pending'
END
WHERE status = 'pending';  -- only touch rows we haven't already migrated


-- ============================================================
-- 4. Indexes for the worker queue
-- ============================================================

-- Hot index: worker's main query "give me pending rows ordered by oldest first"
CREATE INDEX IF NOT EXISTS idx_assessments_status_pending
    ON assessments(created_at)
    WHERE status = 'pending';

-- Stale claim recovery query
CREATE INDEX IF NOT EXISTS idx_assessments_status_processing
    ON assessments(claimed_at)
    WHERE status = 'processing';

-- Operator dashboard: "show me recent failures"
CREATE INDEX IF NOT EXISTS idx_assessments_status_failed
    ON assessments(processed_at DESC)
    WHERE status = 'failed';

-- General-purpose status filter (for metrics counts)
CREATE INDEX IF NOT EXISTS idx_assessments_status
    ON assessments(status);


-- ============================================================
-- 5. Atomic batch claim function
-- ============================================================
-- PostgREST cannot do UPDATE ... LIMIT, so we expose this as an RPC.
-- Uses FOR UPDATE SKIP LOCKED to safely claim N rows in one round-trip,
-- even if multiple workers are running concurrently.

CREATE OR REPLACE FUNCTION claim_pending_assessments(
    batch_size INTEGER,
    worker_id TEXT
)
RETURNS SETOF assessments
LANGUAGE plpgsql
SECURITY DEFINER  -- runs with table owner privileges, bypasses RLS for service_role calls
SET search_path = public
AS $$
BEGIN
    RETURN QUERY
    UPDATE assessments
    SET status = 'processing',
        claimed_at = NOW(),
        claimed_by = worker_id
    WHERE id IN (
        SELECT id FROM assessments
        WHERE status = 'pending'
        ORDER BY created_at ASC
        FOR UPDATE SKIP LOCKED
        LIMIT batch_size
    )
    RETURNING *;
END;
$$;

-- Allow service_role to call it (RoadSide app can't claim work, only the worker)
GRANT EXECUTE ON FUNCTION claim_pending_assessments(INTEGER, TEXT) TO service_role;


-- ============================================================
-- 6. Stale claim recovery function
-- ============================================================
-- If a worker crashes mid-claim, rows stuck in 'processing' should
-- eventually be reset to 'pending' so another worker can pick them up.
-- Call this on worker startup AND periodically from the worker loop.

CREATE OR REPLACE FUNCTION reset_stale_processing_assessments(
    stale_threshold_seconds INTEGER DEFAULT 300  -- 5 minutes
)
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    reset_count INTEGER;
BEGIN
    UPDATE assessments
    SET status = 'pending',
        claimed_at = NULL,
        claimed_by = NULL,
        retry_count = retry_count + 1
    WHERE status = 'processing'
      AND claimed_at < NOW() - (stale_threshold_seconds || ' seconds')::INTERVAL;

    GET DIAGNOSTICS reset_count = ROW_COUNT;
    RETURN reset_count;
END;
$$;

GRANT EXECUTE ON FUNCTION reset_stale_processing_assessments(INTEGER) TO service_role;


-- ============================================================
-- 7. Worker heartbeat table
-- ============================================================
-- The operator dashboard reads this to know if the worker is alive,
-- when it last processed an image, and what its current state is.

CREATE TABLE IF NOT EXISTS worker_state (
    worker_id TEXT PRIMARY KEY,
    is_running BOOLEAN NOT NULL DEFAULT FALSE,
    started_at TIMESTAMPTZ,
    stopped_at TIMESTAMPTZ,
    last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    current_image_id UUID REFERENCES assessments(id) ON DELETE SET NULL,

    -- Cumulative metrics (since worker started)
    images_processed INTEGER NOT NULL DEFAULT 0,
    images_failed INTEGER NOT NULL DEFAULT 0,
    images_flagged_review INTEGER NOT NULL DEFAULT 0,

    -- Rolling stats (in milliseconds)
    avg_stage1_time_ms REAL DEFAULT 0,
    avg_stage2_time_ms REAL DEFAULT 0,

    -- Last status info
    last_error TEXT,
    last_processed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_worker_state_heartbeat
    ON worker_state(last_heartbeat DESC);

-- RLS: only service_role can write to worker_state
ALTER TABLE worker_state ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Service role full access worker_state" ON worker_state;
CREATE POLICY "Service role full access worker_state"
    ON worker_state FOR ALL
    USING (auth.role() = 'service_role');


-- ============================================================
-- 8. Aggregate metrics view (single query for dashboard)
-- ============================================================
-- The operator dashboard polls this view to get queue depth,
-- per-class distribution, recent throughput, etc.

CREATE OR REPLACE VIEW pipeline_metrics AS
SELECT
    -- Queue depth
    COUNT(*) FILTER (WHERE status = 'pending')         AS pending_count,
    COUNT(*) FILTER (WHERE status = 'processing')      AS processing_count,
    COUNT(*) FILTER (WHERE status = 'classified')      AS classified_count,
    COUNT(*) FILTER (WHERE status = 'expert_review')   AS expert_review_count,
    COUNT(*) FILTER (WHERE status = 'done')            AS done_count,
    COUNT(*) FILTER (WHERE status = 'failed')          AS failed_count,
    COUNT(*)                                           AS total_count,

    -- Throughput in last hour
    COUNT(*) FILTER (
        WHERE processed_at >= NOW() - INTERVAL '1 hour'
        AND status IN ('classified', 'expert_review', 'done')
    ) AS throughput_last_hour,

    -- Throughput in last minute
    COUNT(*) FILTER (
        WHERE processed_at >= NOW() - INTERVAL '1 minute'
        AND status IN ('classified', 'expert_review', 'done')
    ) AS throughput_last_minute,

    -- Average confidences (only over recently classified)
    AVG(stage1_confidence) FILTER (
        WHERE stage1_confidence IS NOT NULL
        AND processed_at >= NOW() - INTERVAL '1 hour'
    ) AS avg_stage1_confidence_last_hour,
    AVG(stage2_confidence) FILTER (
        WHERE stage2_confidence > 0
        AND processed_at >= NOW() - INTERVAL '1 hour'
    ) AS avg_stage2_confidence_last_hour,

    -- Distress vs Normal split (last hour)
    COUNT(*) FILTER (
        WHERE is_distressed = TRUE
        AND processed_at >= NOW() - INTERVAL '1 hour'
    ) AS distressed_last_hour,
    COUNT(*) FILTER (
        WHERE is_distressed = FALSE
        AND processed_at >= NOW() - INTERVAL '1 hour'
    ) AS normal_last_hour
FROM assessments;

GRANT SELECT ON pipeline_metrics TO service_role, authenticated, anon;


-- ============================================================
-- 9. Per-class distribution view (for dashboard chart)
-- ============================================================

CREATE OR REPLACE VIEW pipeline_class_distribution AS
SELECT
    distress_type,
    COUNT(*) AS count
FROM assessments,
     LATERAL jsonb_array_elements_text(distress_types) AS distress_type
WHERE status IN ('classified', 'expert_review', 'done')
  AND processed_at >= NOW() - INTERVAL '24 hours'
GROUP BY distress_type
ORDER BY count DESC;

GRANT SELECT ON pipeline_class_distribution TO service_role, authenticated, anon;


COMMIT;

-- ============================================================
-- VERIFICATION QUERIES (run separately to confirm migration)
-- ============================================================
-- SELECT column_name, data_type, is_nullable, column_default
--   FROM information_schema.columns
--   WHERE table_name = 'assessments'
--   ORDER BY ordinal_position;
--
-- SELECT status, COUNT(*) FROM assessments GROUP BY status;
--
-- SELECT * FROM pg_proc WHERE proname IN
--     ('claim_pending_assessments', 'reset_stale_processing_assessments');
--
-- SELECT * FROM pipeline_metrics;
