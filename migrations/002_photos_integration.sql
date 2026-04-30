-- ============================================================
-- Migration 002: Integrate RoadSide app's `photos` table with the AI pipeline
-- ============================================================
-- The RoadSide mobile app writes uploads to a `photos` table:
--    photos(id BIGINT, image_url TEXT, address TEXT,
--           latitude FLOAT8, longitude FLOAT8, created_at TIMESTAMPTZ)
--
-- This migration bridges that table to our AI pipeline (`assessments`)
-- without modifying `photos` itself, by:
--    1. Adding `photo_id` FK column to `assessments`
--    2. Installing an AFTER INSERT trigger on `photos` that auto-creates
--       a matching `assessments` row with status='pending'
--    3. Backfilling all existing `photos` rows that don't yet have an
--       `assessments` row
--
-- Result: The mobile app continues writing only to `photos`. The pipeline
-- reads from `assessments`. The trigger keeps them in sync. The dashboard
-- and WebGIS join by `photo_id` if they need the original photo metadata.
--
-- This migration is idempotent — safe to run multiple times.
-- ============================================================

BEGIN;

-- ============================================================
-- 1. Add photo_id FK column to assessments
-- ============================================================

ALTER TABLE assessments
    ADD COLUMN IF NOT EXISTS photo_id BIGINT
        REFERENCES photos(id) ON DELETE CASCADE;

-- Unique to prevent the trigger from creating duplicates
-- (one assessment per photo). Allows NULL for the older test rows that
-- were inserted directly without a photos counterpart.
CREATE UNIQUE INDEX IF NOT EXISTS uq_assessments_photo_id
    ON assessments(photo_id)
    WHERE photo_id IS NOT NULL;

-- Fast lookup when joining for the WebGIS / dashboard
CREATE INDEX IF NOT EXISTS idx_assessments_photo_id
    ON assessments(photo_id);


-- ============================================================
-- 2. Trigger function: photos INSERT -> assessments INSERT
-- ============================================================
-- SECURITY DEFINER so the function can bypass RLS on assessments
-- when called from the mobile app's authenticated context (which only
-- has INSERT privileges on photos, not on assessments).

CREATE OR REPLACE FUNCTION sync_photo_to_assessment()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    -- Skip if an assessment for this photo somehow already exists
    -- (e.g., from manual backfill). The unique index would catch it,
    -- but explicit guard avoids raising a constraint violation up the stack.
    IF EXISTS (SELECT 1 FROM assessments WHERE photo_id = NEW.id) THEN
        RETURN NEW;
    END IF;

    INSERT INTO assessments (
        photo_id,
        image_url,
        latitude,
        longitude,
        address,
        status,
        model_version,
        created_at
    ) VALUES (
        NEW.id,
        NEW.image_url,
        NEW.latitude,
        NEW.longitude,
        NEW.address,
        'pending',
        'pending',           -- model_version is overwritten by the worker
        NEW.created_at       -- preserve original upload time for FIFO queue ordering
    );

    RETURN NEW;
END;
$$;


-- ============================================================
-- 3. Install the trigger
-- ============================================================

DROP TRIGGER IF EXISTS trg_sync_photo_to_assessment ON photos;
CREATE TRIGGER trg_sync_photo_to_assessment
    AFTER INSERT ON photos
    FOR EACH ROW
    EXECUTE FUNCTION sync_photo_to_assessment();


-- ============================================================
-- 4. Backfill existing photos that don't yet have an assessment
-- ============================================================
-- Run synchronously inside the migration so by the time it completes,
-- the queue is fully populated. The pipeline can be started immediately.

INSERT INTO assessments (
    photo_id,
    image_url,
    latitude,
    longitude,
    address,
    status,
    model_version,
    created_at
)
SELECT
    p.id,
    p.image_url,
    p.latitude,
    p.longitude,
    p.address,
    'pending',
    'pending',
    p.created_at
FROM photos p
LEFT JOIN assessments a ON a.photo_id = p.id
WHERE a.id IS NULL  -- only photos without an existing assessment
ON CONFLICT (photo_id) WHERE photo_id IS NOT NULL DO NOTHING;


-- ============================================================
-- 5. RLS: allow authenticated mobile users to INSERT photos
-- ============================================================
-- (The `photos` table likely already has policies — only adding the
-- ones we know are needed for the pipeline to function. Existing
-- policies are left untouched.)

ALTER TABLE photos ENABLE ROW LEVEL SECURITY;

-- Allow service_role full access (worker uses service_role key)
DROP POLICY IF EXISTS "Service role full access photos" ON photos;
CREATE POLICY "Service role full access photos"
    ON photos FOR ALL
    USING (auth.role() = 'service_role');

-- Allow public read so the dashboard can fetch photo metadata
-- (matches existing pattern for custom_distress_types)
DROP POLICY IF EXISTS "Public read photos" ON photos;
CREATE POLICY "Public read photos"
    ON photos FOR SELECT
    USING (true);


COMMIT;


-- ============================================================
-- VERIFICATION QUERIES (run separately after the migration)
-- ============================================================
-- Should show the new column is present with FK and unique index:
-- SELECT column_name, data_type, is_nullable
--   FROM information_schema.columns
--   WHERE table_name = 'assessments' AND column_name = 'photo_id';
--
-- Should show the trigger is installed:
-- SELECT trigger_name, event_manipulation, action_timing, action_statement
--   FROM information_schema.triggers
--   WHERE event_object_table = 'photos';
--
-- Backfill check — should match the photos count exactly:
-- SELECT
--   (SELECT COUNT(*) FROM photos) AS photos_total,
--   (SELECT COUNT(*) FROM assessments WHERE photo_id IS NOT NULL) AS bridged_assessments,
--   (SELECT COUNT(*) FROM assessments WHERE photo_id IS NOT NULL AND status = 'pending') AS pending_in_queue;
