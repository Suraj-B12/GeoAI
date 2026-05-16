-- ============================================================
-- Migration 004 — Pavement pre-filter + dashboard tooling
-- ============================================================
-- Adds two things to the assessments table:
--   1. New status value: 'rejected_non_pavement' — used by the worker's
--      pre-filter when the VLM confidently says the photo is NOT pavement
--      (selfies, indoor scenes, signs, sky, random objects, etc).
--      Operator can either delete (true non-pavement) or re-classify
--      (false-positive rejection — model mistakenly rejected a real road).
--   2. Pavement-check audit columns:
--        pavement_filter_decision  TEXT     'yes' | 'no' | 'unsure'
--        pavement_filter_raw       TEXT     full VLM response (for audit)
--        pavement_filter_at        TIMESTAMPTZ
--      Reversible: operator can re-classify rejected rows back to 'pending'.
--
-- Backward compatible: existing rows untouched, no NOT NULL added.

BEGIN;

-- 1. Add the new status to the CHECK constraint
ALTER TABLE assessments DROP CONSTRAINT IF EXISTS assessments_status_check;
ALTER TABLE assessments ADD CONSTRAINT assessments_status_check
    CHECK (status IN (
        'pending',
        'processing',
        'classified',
        'expert_review',
        'done',
        'failed',
        'rejected_non_pavement'
    ));

-- 2. Add audit columns for the pavement pre-filter
ALTER TABLE assessments
    ADD COLUMN IF NOT EXISTS pavement_filter_decision TEXT
        CHECK (pavement_filter_decision IS NULL
               OR pavement_filter_decision IN ('yes', 'no', 'unsure'));

ALTER TABLE assessments
    ADD COLUMN IF NOT EXISTS pavement_filter_raw TEXT;

ALTER TABLE assessments
    ADD COLUMN IF NOT EXISTS pavement_filter_at TIMESTAMPTZ;

-- 3. Index for dashboard rejected-tab filter
CREATE INDEX IF NOT EXISTS idx_assessments_rejected
    ON assessments (created_at DESC)
    WHERE status = 'rejected_non_pavement';

COMMIT;
