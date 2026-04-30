-- ============================================================
-- SMOKE TEST DATA — 3 mock pending rows
-- ============================================================
-- Inserts three test assessments pointing at local FastAPI test_fixtures.
-- After running this, start the FastAPI server with:
--   ENABLE_TEST_FIXTURES=1 SUPABASE_URL=... SUPABASE_SERVICE_KEY=... uvicorn app.main:app ...
-- Then visit http://localhost:8000/operator and click Start.
--
-- IMPORTANT: The image_url uses host.docker.internal so it works whether
-- the worker runs locally or in a container. For pure local testing, replace
-- with http://localhost:8000 if needed.
--
-- To CLEAN UP after testing:
--   DELETE FROM assessments WHERE image_url LIKE '%test_fixtures%';
-- ============================================================

INSERT INTO assessments (
    image_url,
    latitude,
    longitude,
    address,
    status,
    model_version
) VALUES
    -- Bengaluru landmarks for plausible GPS coordinates
    (
        'http://localhost:8000/test_fixtures/longitudinal_D00.jpg',
        12.9716, 77.5946,
        'MG Road, Bengaluru (test fixture)',
        'pending',
        'smoke-test'
    ),
    (
        'http://localhost:8000/test_fixtures/alligator_D20.jpg',
        12.9352, 77.6245,
        'Koramangala, Bengaluru (test fixture)',
        'pending',
        'smoke-test'
    ),
    (
        'http://localhost:8000/test_fixtures/pothole_D40.jpg',
        12.9784, 77.6408,
        'Indiranagar, Bengaluru (test fixture)',
        'pending',
        'smoke-test'
    );

-- Verify they were inserted
SELECT id, image_url, status, latitude, longitude, address, created_at
FROM assessments
WHERE image_url LIKE '%test_fixtures%'
ORDER BY created_at DESC;
