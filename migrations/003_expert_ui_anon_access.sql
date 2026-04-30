-- ============================================================
-- Migration 003: Expert Review UI — anon access + reviewer name
-- ============================================================
-- The expert UI runs in a browser with the Supabase anon key (no real
-- auth). We need to:
--   1. Add a reviewer_name TEXT column (since reviewed_by is a UUID FK
--      to auth.users which we're not populating)
--   2. Allow the anon role to:
--        - SELECT rows where needs_expert_review = TRUE  (Pending tab)
--        - SELECT rows where expert_reviewed     = TRUE  (Reviewed tab)
--        - UPDATE rows where needs_expert_review = TRUE  (submit corrections)
--        - INSERT into custom_distress_types  (add new type from UI)
--
-- Security note: this is a PRAGMATIC tradeoff for a capstone demo. Anyone
-- with the public anon key can read/update review-flagged rows. Anonymous
-- users CANNOT see classified/done/pending/processing rows — those stay
-- locked down by the existing "Users view own assessments" policy.
--
-- When you ship this for real, switch back to Supabase Auth (magic link
-- or password) and gate these policies on `auth.role() = 'authenticated'`
-- instead of `TO anon`.
-- ============================================================

BEGIN;

-- ============================================================
-- 1. Reviewer name column (no auth.users FK needed)
-- ============================================================
ALTER TABLE assessments
    ADD COLUMN IF NOT EXISTS reviewer_name TEXT;


-- ============================================================
-- 2. Anon SELECT — Pending tab (rows awaiting review)
-- ============================================================
DROP POLICY IF EXISTS "Anon read pending review assessments" ON assessments;
CREATE POLICY "Anon read pending review assessments"
    ON assessments
    FOR SELECT
    TO anon
    USING (needs_expert_review = TRUE);


-- ============================================================
-- 3. Anon SELECT — Reviewed tab (rows already reviewed)
-- ============================================================
DROP POLICY IF EXISTS "Anon read reviewed assessments" ON assessments;
CREATE POLICY "Anon read reviewed assessments"
    ON assessments
    FOR SELECT
    TO anon
    USING (expert_reviewed = TRUE);


-- ============================================================
-- 4. Anon UPDATE — submit a correction
-- ============================================================
-- USING controls which rows the anon user can attempt to update
-- (only rows currently flagged for review). WITH CHECK = TRUE because
-- the workflow is to flip needs_expert_review->false and
-- expert_reviewed->true, which is the exact intent.
DROP POLICY IF EXISTS "Anon update pending review assessments" ON assessments;
CREATE POLICY "Anon update pending review assessments"
    ON assessments
    FOR UPDATE
    TO anon
    USING (needs_expert_review = TRUE)
    WITH CHECK (TRUE);


-- ============================================================
-- 5. Anon INSERT — add a new custom distress type from the UI
-- ============================================================
DROP POLICY IF EXISTS "Anon insert custom distress types" ON custom_distress_types;
CREATE POLICY "Anon insert custom distress types"
    ON custom_distress_types
    FOR INSERT
    TO anon
    WITH CHECK (TRUE);


COMMIT;


-- ============================================================
-- VERIFICATION (run separately)
-- ============================================================
-- SELECT polname, cmd, roles, qual, with_check
--   FROM pg_policies
--   WHERE tablename IN ('assessments','custom_distress_types')
--     AND polname LIKE 'Anon%'
--   ORDER BY tablename, polname;
--
-- SELECT column_name, data_type
--   FROM information_schema.columns
--   WHERE table_name = 'assessments' AND column_name = 'reviewer_name';
