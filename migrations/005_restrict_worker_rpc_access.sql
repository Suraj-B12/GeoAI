-- The queue RPCs run as the table owner and are intended for the backend's
-- service_role only. GRANT to service_role does not remove PostgreSQL's
-- default PUBLIC EXECUTE privilege on newly created functions.
-- Apply through the same administrator connection used for migrations 001-004.
BEGIN;

REVOKE ALL ON FUNCTION public.claim_pending_assessments(INTEGER, TEXT)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.reset_stale_processing_assessments(INTEGER)
    FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION public.claim_pending_assessments(INTEGER, TEXT)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.reset_stale_processing_assessments(INTEGER)
    TO service_role;

-- Resolve table names predictably inside these SECURITY DEFINER functions.
ALTER FUNCTION public.claim_pending_assessments(INTEGER, TEXT)
    SET search_path = public, pg_temp;
ALTER FUNCTION public.reset_stale_processing_assessments(INTEGER)
    SET search_path = public, pg_temp;

COMMIT;
