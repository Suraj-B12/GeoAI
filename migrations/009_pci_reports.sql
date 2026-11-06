-- ============================================================
-- Migration 009: stored A->B chainage + IRC:82-2015 condition-rating reports (local PostgreSQL only)
-- ============================================================
-- A stored report is a SNAPSHOT: the report JSON and the PDF bytes as generated, never re-rendered,
-- so a later expert review or image change cannot alter an old report. GET /gis/pci/reports/{id}
-- compares data_as_of with the segments' current updated_at / reviewed_at and returns a stale flag.
--
-- input_hash: SHA-256 of the canonical request (+ data_as_of); a second save of the same request on
-- the same data returns the stored row (dedupe). The API refuses saves beyond 200 rows (hard cap).
-- sample: set only by scripts/pci_samples.py (pre-registered sample reports), never by the API.
--
-- In schema `research`, like migrations 006-008. Additive and idempotent. Applied by
-- scripts/setup_local_db.py, and by app/pci_routes.ensure_schema() on first use.
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE IF NOT EXISTS research.pci_reports (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    title TEXT NOT NULL,
    input JSONB NOT NULL,
    input_hash TEXT NOT NULL CHECK (input_hash ~ '^[0-9a-f]{64}$'),
    report JSONB NOT NULL,
    pdf BYTEA NOT NULL,
    sample BOOLEAN NOT NULL DEFAULT FALSE,
    sample_rule TEXT,                     -- which pre-registered rule chose it (sample reports only)
    data_as_of TIMESTAMPTZ NOT NULL,      -- newest segment updated_at / reviewed_at the report saw
    schema_version TEXT NOT NULL DEFAULT 'geoai.pci.v1'
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_pci_reports_input_hash ON research.pci_reports (input_hash);
CREATE INDEX IF NOT EXISTS idx_pci_reports_created ON research.pci_reports (created_at DESC);

COMMIT;
