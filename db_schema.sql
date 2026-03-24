-- ============================================================
-- Pavement Distress Classification — Supabase Schema
-- ============================================================
-- Run this in the Supabase SQL Editor to create all tables.
-- ============================================================

-- ============================================================
-- ASSESSMENTS: Every image classification result
-- ============================================================
CREATE TABLE IF NOT EXISTS assessments (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,

    -- Image
    image_url TEXT NOT NULL,
    image_width INTEGER,
    image_height INTEGER,

    -- Location (from phone GPS)
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    address TEXT,
    road_segment_id TEXT,

    -- Stage 1: Binary detection
    stage1_label TEXT NOT NULL CHECK (stage1_label IN ('Normal', 'Distress')),
    stage1_confidence REAL NOT NULL CHECK (stage1_confidence BETWEEN 0 AND 1),
    is_distressed BOOLEAN NOT NULL,

    -- Stage 2: Type classification (NULL/empty if Normal)
    distress_types JSONB DEFAULT '[]'::jsonb,
    severity TEXT CHECK (severity IN ('None', 'Low', 'Medium', 'High', 'Unknown')),
    description TEXT DEFAULT '',
    stage2_confidence REAL DEFAULT 0 CHECK (stage2_confidence BETWEEN 0 AND 1),

    -- Expert review
    needs_expert_review BOOLEAN DEFAULT FALSE,
    expert_reviewed BOOLEAN DEFAULT FALSE,
    expert_corrected_types JSONB,
    expert_corrected_severity TEXT,
    expert_notes TEXT,
    reviewed_by UUID REFERENCES auth.users(id),
    reviewed_at TIMESTAMPTZ,

    -- Retraining
    included_in_retrain BOOLEAN DEFAULT FALSE,

    -- Metadata
    processing_time_ms REAL,
    model_version TEXT DEFAULT '1.0.0',
    raw_response JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_assessments_user_id ON assessments(user_id);
CREATE INDEX idx_assessments_created_at ON assessments(created_at DESC);
CREATE INDEX idx_assessments_needs_review ON assessments(needs_expert_review)
    WHERE needs_expert_review = TRUE AND expert_reviewed = FALSE;
CREATE INDEX idx_assessments_distressed ON assessments(is_distressed)
    WHERE is_distressed = TRUE;
CREATE INDEX idx_assessments_road_segment ON assessments(road_segment_id)
    WHERE road_segment_id IS NOT NULL;

-- ============================================================
-- CUSTOM DISTRESS TYPES: Expert-added categories
-- ============================================================
CREATE TABLE IF NOT EXISTS custom_distress_types (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT DEFAULT '',
    added_by UUID REFERENCES auth.users(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Seed with known types so the dropdown is pre-populated
INSERT INTO custom_distress_types (name, description) VALUES
    ('Longitudinal Crack (D00)', 'Cracks running parallel to the road direction'),
    ('Transverse Crack (D10)', 'Cracks running perpendicular to the road direction'),
    ('Alligator Crack (D20)', 'Interconnected cracks forming alligator skin pattern'),
    ('Pothole (D40)', 'Bowl-shaped holes in the pavement surface'),
    ('Block Crack (D43)', 'Rectangular cracks dividing pavement into blocks'),
    ('Inlaid Patch (D44)', 'Repair patches flush with surrounding pavement'),
    ('Open Joint (D50)', 'Gaps or separations at pavement joints'),
    ('Rutting', 'Longitudinal surface depressions in the wheel path'),
    ('Raveling', 'Loss of aggregate particles from the pavement surface'),
    ('Bleeding', 'Excess asphalt binder creating shiny reflective areas'),
    ('Edge Crack', 'Cracks along the pavement edge near the shoulder'),
    ('Depression', 'Localized low areas in the pavement surface')
ON CONFLICT (name) DO NOTHING;

-- ============================================================
-- RETRAIN JOBS: Track retraining history
-- ============================================================
CREATE TABLE IF NOT EXISTS retrain_jobs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    corrections_count INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    progress_pct REAL DEFAULT 0,
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 0,
    error_message TEXT,
    adapter_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- Row Level Security
-- ============================================================
ALTER TABLE assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE custom_distress_types ENABLE ROW LEVEL SECURITY;
ALTER TABLE retrain_jobs ENABLE ROW LEVEL SECURITY;

-- Users can view and insert their own assessments
CREATE POLICY "Users view own assessments"
    ON assessments FOR SELECT
    USING (user_id = auth.uid());

CREATE POLICY "Users insert own assessments"
    ON assessments FOR INSERT
    WITH CHECK (user_id = auth.uid());

-- Experts can view ALL assessments needing review + update them
-- (Implement via a custom role or service_role key on the backend)
CREATE POLICY "Service role full access assessments"
    ON assessments FOR ALL
    USING (auth.role() = 'service_role');

-- Everyone can read distress types
CREATE POLICY "Public read distress types"
    ON custom_distress_types FOR SELECT
    USING (true);

-- Only service role can insert new types
CREATE POLICY "Service role manage distress types"
    ON custom_distress_types FOR ALL
    USING (auth.role() = 'service_role');

-- Only service role manages retrain jobs
CREATE POLICY "Service role manage retrain jobs"
    ON retrain_jobs FOR ALL
    USING (auth.role() = 'service_role');
