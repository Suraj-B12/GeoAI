-- Run once against the same Supabase project used by the app.
-- Repeatable: existing photos, upload times, queue order and policies are unchanged.
BEGIN;

ALTER TABLE public.photos
    ADD COLUMN IF NOT EXISTS captured_at TIMESTAMPTZ;

COMMENT ON COLUMN public.photos.captured_at IS
    'Original photo capture time when known. NULL means unavailable; created_at remains upload time.';

-- Make the additional column visible to PostgREST after commit.
NOTIFY pgrst, 'reload schema';

COMMIT;
