-- ============================================================
-- Migration 005: video pipeline (GeoAI_Video segment mode)
-- ============================================================
-- Stores the drive-video survey next to the RoadSide uploads, in the same
-- tables the operator dashboard already reads (GeoAI_Video
-- docs/HANDOFF_PLAN.md section 7, docs/DB_MAPPING.md):
--
--   video_sessions   one row per recording: pipeline state for the operator UI
--                    (queued -> extracting -> classifying -> storing -> stored
--                    -> cleaned, or failed with the error), per-stage
--                    timestamps, metrics and the run's report.json.
--   video_segments   one row per 5 m road segment, every status (send, audit,
--                    skip, not_covered), keyed by the record id
--                    "<session_id>_seg<k:05d>". Holds the full geoai.segment.v1
--                    record, the raw /classify answer, the stored image path +
--                    SHA-256, and links to the photos / assessments rows of
--                    segments the VLM answered.
--
-- A classified segment is ALSO a normal photos + assessments pair (photos
-- insert -> trigger trg_sync_photo_to_assessment -> assessment), so the
-- dashboard, metrics views and expert UI show it with no change. Its
-- assessments.road_segment_id is the segment id; the partial unique index
-- below makes that the idempotency key (a re-run updates, never duplicates).
--
-- Additive only: no existing column, constraint or policy changes. Idempotent.
-- Applied to the local PostgreSQL by scripts/setup_local_db.py. To keep the
-- Supabase schema identical, the owner runs this same file in the Supabase
-- SQL editor (the duplicate check below aborts it, changing nothing, if
-- assessments.road_segment_id already holds duplicates).
-- ============================================================

BEGIN;

-- ------------------------------------------------------------
-- 0. Refuse to build the unique index over duplicate segment ids
-- ------------------------------------------------------------
DO $$
DECLARE
    dup TEXT;
BEGIN
    SELECT road_segment_id INTO dup
    FROM assessments
    WHERE road_segment_id IS NOT NULL
    GROUP BY road_segment_id
    HAVING COUNT(*) > 1
    LIMIT 1;
    IF dup IS NOT NULL THEN
        RAISE EXCEPTION 'assessments.road_segment_id has duplicates (e.g. %); resolve them before migration 005', dup;
    END IF;
END
$$;

CREATE UNIQUE INDEX IF NOT EXISTS uq_assessments_road_segment_id
    ON assessments(road_segment_id)
    WHERE road_segment_id IS NOT NULL;


-- ------------------------------------------------------------
-- 1. video_sessions
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS video_sessions (
    session_id TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'queued'
        CHECK (state IN ('queued', 'extracting', 'classifying', 'storing',
                         'stored', 'cleaned', 'failed')),
    error TEXT,                       -- last failure (with the stage), NULL when not failed
    metrics JSONB,                    -- per-session metrics (HANDOFF_PLAN section 9)
    report JSONB,                     -- the run's report.json

    -- When the session last ENTERED each state (set by app/video_store.set_session_state)
    queued_at TIMESTAMPTZ,
    extracting_at TIMESTAMPTZ,
    classifying_at TIMESTAMPTZ,
    storing_at TIMESTAMPTZ,
    stored_at TIMESTAMPTZ,
    cleaned_at TIMESTAMPTZ,
    failed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_video_sessions_updated_at
    ON video_sessions(updated_at DESC);


-- ------------------------------------------------------------
-- 2. video_segments
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS video_segments (
    segment_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES video_sessions(session_id) ON DELETE CASCADE,
    k INTEGER NOT NULL CHECK (k >= 0),
    status TEXT NOT NULL CHECK (status IN ('send', 'audit', 'skip', 'not_covered')),

    -- Copied out of the record for queries; the record stays the source of truth
    length_m DOUBLE PRECISION,
    s0_m DOUBLE PRECISION,
    s1_m DOUBLE PRECISION,
    centre_lat DOUBLE PRECISION,
    centre_lon DOUBLE PRECISION,
    chainage_start_km DOUBLE PRECISION,
    chainage_end_km DOUBLE PRECISION,
    utc TIMESTAMPTZ,                  -- frame.utc_ms; NULL when not covered or no UTC fit

    -- Local image store (replaces Cloudinary): absolute path + SHA-256 of the PNG
    image_path TEXT,
    image_sha256 TEXT CHECK (image_sha256 IS NULL OR image_sha256 ~ '^[0-9a-f]{64}$'),

    -- The photos / assessments pair of a segment the VLM answered
    photo_id BIGINT REFERENCES photos(id) ON DELETE SET NULL,
    assessment_id UUID REFERENCES assessments(id) ON DELETE SET NULL,

    record JSONB NOT NULL,            -- full geoai.segment.v1 record (sidecar JSON)
    vlm_response JSONB,               -- raw /classify JSON, NULL until answered

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- "<session_id>_seg<k:05d>" (Python's :05d widens past 99999, so does this)
    CONSTRAINT video_segments_id_format CHECK (
        segment_id = session_id || '_seg' ||
            CASE WHEN k < 100000 THEN lpad(k::text, 5, '0') ELSE k::text END
    ),
    CONSTRAINT video_segments_session_k_key UNIQUE (session_id, k)
);

CREATE INDEX IF NOT EXISTS idx_video_segments_status
    ON video_segments(session_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_video_segments_photo_id
    ON video_segments(photo_id) WHERE photo_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_video_segments_assessment_id
    ON video_segments(assessment_id) WHERE assessment_id IS NOT NULL;


-- ------------------------------------------------------------
-- 3. updated_at maintenance
-- ------------------------------------------------------------
CREATE OR REPLACE FUNCTION video_touch_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = public
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_video_sessions_updated_at ON video_sessions;
CREATE TRIGGER trg_video_sessions_updated_at
    BEFORE UPDATE ON video_sessions
    FOR EACH ROW EXECUTE FUNCTION video_touch_updated_at();

DROP TRIGGER IF EXISTS trg_video_segments_updated_at ON video_segments;
CREATE TRIGGER trg_video_segments_updated_at
    BEFORE UPDATE ON video_segments
    FOR EACH ROW EXECUTE FUNCTION video_touch_updated_at();


-- ------------------------------------------------------------
-- 4. RLS: same pattern as worker_state (only the server writes)
-- ------------------------------------------------------------
ALTER TABLE video_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE video_segments ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Service role full access video_sessions" ON video_sessions;
CREATE POLICY "Service role full access video_sessions"
    ON video_sessions FOR ALL
    USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "Service role full access video_segments" ON video_segments;
CREATE POLICY "Service role full access video_segments"
    ON video_segments FOR ALL
    USING (auth.role() = 'service_role');

COMMIT;
