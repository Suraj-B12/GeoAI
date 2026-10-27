"""
Local PostgreSQL backend for the operator pipeline (DB_BACKEND=postgres).

DB_BACKEND selects where the pipeline's rows live:

    supabase  (default, also when unset)  Supabase PostgREST via httpx - production, unchanged
    postgres                              this module, a local PostgreSQL at DATABASE_URL

The local database has the same schema as the Supabase project
(scripts/setup_local_db.py builds it from db_schema.sql + migrations, unchanged).

LocalDb offers every operation app/supabase_client.py performs, with the SAME
return shapes. Each public function in supabase_client.py checks
`is_local_db(client)` on its first line and hands over to the method of the
same name here, so app/worker.py, the dashboard endpoints and the read cache
run unchanged on either backend: the FastAPI lifespan simply passes a LocalDb
where it would pass the Supabase httpx client.

Same shapes, by construction: every row leaves PostgreSQL through
json_agg / row_to_json, which is how PostgREST renders rows. Timestamps,
UUIDs, real/float4 values and jsonb therefore arrive exactly as from Supabase
(ISO strings with +00:00 in a UTC session, string UUIDs, "0.9" not
0.899999976). Errors are raised as SupabaseError with an HTTP-like status
(404 no row, 400 bad input, 503 database unreachable), so callers' error
handling is unchanged too.

Why a thread pool and not psycopg's async pool: uvicorn 0.46 runs a single
worker on Windows on the ProactorEventLoop, and psycopg's async mode refuses
that loop. Switching the server's loop policy would affect everything else in
the process, so each call runs a short synchronous query on a pooled
connection via asyncio.to_thread - the same pattern the server already uses
for model inference. The queries are small; the pool is bounded.
"""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from typing import Any, Callable, Optional

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool, PoolTimeout

from app.supabase_client import SupabaseError, is_local_db  # noqa: F401  (is_local_db re-exported)

BACKENDS = ("supabase", "postgres")

# Tables the pipeline writes through update/upsert. Columns are checked against
# the catalogue, so a typo is a 400 (as from PostgREST), never SQL injection.
_WRITABLE = ("assessments", "worker_state", "photos")

_DASHBOARD_COLS = (
    "id, image_url, address, latitude, longitude, status, "
    "stage1_label, stage1_confidence, distress_types, severity, "
    "stage2_confidence, needs_expert_review, created_at, processed_at, "
    "raw_response->'condition_indicators' AS condition_indicators, "
    "pavement_filter_decision"
)
_INFLIGHT_COLS = (
    "id, image_url, latitude, longitude, address, status, "
    "stage1_label, stage1_confidence, distress_types, severity, "
    "stage2_confidence, error_message, claimed_at, processed_at"
)
_RECENT_COLS = (
    "id, status, stage1_label, stage1_confidence, distress_types, "
    "severity, stage2_confidence, processed_at, error_message"
)
# Same payload as supabase_client.reset_for_reclassify.
_RECLASSIFY_RESET = {
    "status": "pending", "processed_at": None, "stage1_label": None, "stage1_confidence": None,
    "is_distressed": None, "distress_types": None, "severity": None, "description": None,
    "stage2_confidence": None, "needs_expert_review": False, "processing_time_ms": None,
    "raw_response": None, "pavement_filter_decision": None, "pavement_filter_raw": None,
    "pavement_filter_at": None, "error_message": None, "retry_count": 0,
    "claimed_by": None, "claimed_at": None,
}


def db_backend() -> str:
    """DB_BACKEND, lower-cased; 'supabase' when unset. ValueError for anything else,
    so a typo cannot silently send local work to the production database."""
    v = (os.environ.get("DB_BACKEND") or "supabase").strip().lower()
    if v not in BACKENDS:
        raise ValueError(f"DB_BACKEND={v!r} is not one of {BACKENDS}")
    return v


def _wrap(value: Any) -> Any:
    """dict/list -> Jsonb (every dict/list column in this schema is jsonb)."""
    return Jsonb(value) if isinstance(value, (dict, list)) else value


class LocalDb:
    """Pooled local PostgreSQL with the supabase_client operations as async methods."""

    IS_LOCAL_DB = True

    def __init__(self, dsn: Optional[str] = None, *, min_size: int = 1, max_size: int = 5,
                 timeout: float = 10.0):
        dsn = dsn or os.environ.get("DATABASE_URL")
        if not dsn:
            raise RuntimeError("DB_BACKEND=postgres needs DATABASE_URL (see SETUP.md, 'Local PostgreSQL').")
        # UTC session: timestamps render with +00:00, exactly like Supabase.
        self._pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size, timeout=timeout,
                                    kwargs={"autocommit": True, "options": "-c TimeZone=UTC"},
                                    open=False, name="geoai-localdb")
        self._columns: dict[str, dict[str, str]] = {}

    # ---------------------------------------------------------- lifecycle
    async def open(self, check: bool = True) -> None:
        """Open the pool; with check, fail now (not on the first claim) if the DB is unreachable."""
        await asyncio.to_thread(self._pool.open, False)
        if check:
            await self.ping()

    async def aclose(self) -> None:
        """Same name as httpx.AsyncClient.aclose, so the lifespan's shutdown closes either."""
        await asyncio.to_thread(self._pool.close)

    async def ping(self) -> None:
        await self._run(lambda c: c.execute("SELECT 1").fetchone())

    # ---------------------------------------------------------- plumbing
    async def _run(self, fn: Callable[[psycopg.Connection], Any]) -> Any:
        return await asyncio.to_thread(self._call, fn)

    def _call(self, fn: Callable[[psycopg.Connection], Any]) -> Any:
        try:
            with self._pool.connection() as conn:
                return fn(conn)
        except SupabaseError:
            raise
        except PoolTimeout as e:
            raise SupabaseError(503, f"local database unavailable: {e}") from e
        except psycopg.OperationalError as e:
            raise SupabaseError(503, f"local database unavailable: {e}") from e
        except (psycopg.DataError, psycopg.IntegrityError, psycopg.ProgrammingError) as e:
            raise SupabaseError(400, f"{type(e).__name__}: {e}".strip()) from e
        except psycopg.Error as e:
            raise SupabaseError(500, f"{type(e).__name__}: {e}".strip()) from e

    @staticmethod
    def _json(conn: psycopg.Connection, query, params=()) -> Any:
        """First column of the first row (a json value built by PostgreSQL)."""
        row = conn.execute(query, params).fetchone()
        return row[0] if row else None

    def _table_columns(self, conn: psycopg.Connection, table: str) -> dict[str, str]:
        """{column: SQL type} of a public table, cached."""
        if table not in self._columns:
            rows = conn.execute(
                "SELECT a.attname, format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
                "JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'public' AND c.relname = %s AND a.attnum > 0 AND NOT a.attisdropped",
                (table,)).fetchall()
            if not rows:
                raise SupabaseError(404, f"relation public.{table} does not exist")
            self._columns[table] = dict(rows)
        return self._columns[table]

    def _assignments(self, conn, table: str, fields: dict) -> tuple[list, list]:
        """(SET items, params) with each value cast to its column type; 400 on unknown columns."""
        if table not in _WRITABLE:
            raise SupabaseError(400, f"table {table!r} is not writable through LocalDb")
        cols = self._table_columns(conn, table)
        unknown = sorted(set(fields) - set(cols))
        if unknown:
            raise SupabaseError(400, f"Could not find the {unknown[0]!r} column of {table!r}")
        items, params = [], []
        for k, v in fields.items():
            items.append((sql.Identifier(k), sql.SQL(cols[k])))
            params.append(_wrap(v))
        return items, params

    # ---------------------------------------------------------- worker
    async def claim_pending_batch(self, batch_size: int, worker_id: str) -> list[dict]:
        def q(conn):
            return self._json(conn, "SELECT COALESCE(json_agg(c ORDER BY c.created_at), '[]'::json) "
                                    "FROM claim_pending_assessments(%s, %s) c",
                              (int(batch_size), worker_id)) or []
        return await self._run(q)

    async def update_assessment(self, assessment_id: str, fields: dict) -> dict:
        def q(conn):
            if not fields:
                row = self._json(conn, "SELECT row_to_json(a) FROM assessments a WHERE id = %s::uuid",
                                 (assessment_id,))
            else:
                items, params = self._assignments(conn, "assessments", fields)
                query = sql.SQL("UPDATE assessments SET {} WHERE id = %s::uuid "
                                "RETURNING row_to_json(assessments.*)").format(
                    sql.SQL(", ").join(sql.SQL("{} = %s::{}").format(c, t) for c, t in items))
                row = self._json(conn, query, (*params, assessment_id))
            if row is None:
                raise SupabaseError(404, f"No assessment with id={assessment_id}")
            return row
        return await self._run(q)

    async def reset_stale_processing(self, stale_threshold_seconds: int = 300) -> int:
        return int(await self._run(lambda c: self._json(
            c, "SELECT reset_stale_processing_assessments(%s)", (int(stale_threshold_seconds),))) or 0)

    async def upsert_worker_state(self, worker_id: str, fields: dict) -> dict:
        """INSERT ... ON CONFLICT (worker_id) DO UPDATE of the given columns
        (PostgREST's Prefer: resolution=merge-duplicates)."""
        def q(conn):
            payload = {"worker_id": worker_id, **fields}
            items, params = self._assignments(conn, "worker_state", payload)
            cols = [c for c, _ in items]
            values = sql.SQL(", ").join(sql.SQL("%s::{}").format(t) for _, t in items)
            # Only the columns sent are overwritten (merge-duplicates); worker_id
            # is the conflict key, and SET worker_id keeps RETURNING working when
            # nothing else was sent.
            updates = [sql.SQL("{c} = EXCLUDED.{c}").format(c=sql.Identifier(n)) for n in payload
                       if n != "worker_id"] or [sql.SQL("worker_id = EXCLUDED.worker_id")]
            conflict = sql.SQL("DO UPDATE SET ") + sql.SQL(", ").join(updates)
            query = sql.SQL("INSERT INTO worker_state ({}) VALUES ({}) ON CONFLICT (worker_id) {} "
                            "RETURNING row_to_json(worker_state.*)").format(
                sql.SQL(", ").join(cols), values, conflict)
            return self._json(conn, query, params) or {}
        return await self._run(q)

    async def get_worker_state(self, worker_id: str) -> Optional[dict]:
        return await self._run(lambda c: self._json(
            c, "SELECT row_to_json(w) FROM worker_state w WHERE worker_id = %s", (worker_id,)))

    # ---------------------------------------------------------- dashboard reads
    async def get_pipeline_metrics(self) -> dict:
        """pipeline_metrics + the gaps supabase_client._fetch_get_pipeline_metrics fills."""
        def q(conn):
            out = self._json(conn, "SELECT row_to_json(m) FROM pipeline_metrics m") or {}
            extra = self._json(conn, """
                SELECT json_build_object(
                  'rejected', (SELECT COUNT(*) FROM assessments WHERE status = 'rejected_non_pavement'),
                  's1', (SELECT AVG(stage1_confidence) FROM assessments
                          WHERE processed_at IS NOT NULL AND stage1_confidence IS NOT NULL),
                  's2', (SELECT AVG(stage2_confidence) FROM assessments
                          WHERE processed_at IS NOT NULL AND stage2_confidence > 0),
                  'n',  (SELECT COUNT(*) FROM assessments WHERE processed_at IS NOT NULL),
                  't0', (SELECT AVG((raw_response->>'stage0_time_ms')::float8) FROM assessments
                          WHERE processed_at IS NOT NULL AND jsonb_typeof(raw_response->'stage0_time_ms') = 'number'
                            AND (raw_response->>'stage0_time_ms')::float8 > 0),
                  't1', (SELECT AVG((raw_response->>'stage1_time_ms')::float8) FROM assessments
                          WHERE processed_at IS NOT NULL AND jsonb_typeof(raw_response->'stage1_time_ms') = 'number'
                            AND (raw_response->>'stage1_time_ms')::float8 > 0),
                  't2', (SELECT AVG((raw_response->>'stage2_time_ms')::float8) FROM assessments
                          WHERE processed_at IS NOT NULL AND jsonb_typeof(raw_response->'stage2_time_ms') = 'number'
                            AND (raw_response->>'stage2_time_ms')::float8 > 0))""")
            out["rejected_non_pavement_count"] = extra["rejected"]
            out["confidence_window"] = "last_hour"
            if out.get("avg_stage1_confidence_last_hour") is None and extra["s1"] is not None:
                out["avg_stage1_confidence_last_hour"] = extra["s1"]
                out["avg_stage2_confidence_last_hour"] = extra["s2"]
                out["confidence_window"] = "all_time"
            out["stage_times_ms"] = {"stage0": extra["t0"], "stage1": extra["t1"], "stage2": extra["t2"],
                                     "n": extra["n"]}
            return out
        return await self._run(q)

    async def get_class_distribution(self) -> list[dict]:
        """The 24 h view when it has rows, else all-time (as supabase_client does)."""
        def q(conn):
            rows = self._json(conn, "SELECT COALESCE(json_agg(d), '[]'::json) FROM pipeline_class_distribution d")
            if rows:
                for r in rows:
                    r["window"] = "24h"
                return rows
            return self._json(conn, """
                SELECT COALESCE(json_agg(json_build_object('distress_type', t.distress_type,
                                                           'count', t.count, 'window', 'all_time')
                                         ORDER BY t.count DESC), '[]'::json)
                FROM (SELECT distress_type, COUNT(*) AS count
                      FROM assessments, LATERAL jsonb_array_elements_text(distress_types) AS distress_type
                      WHERE status IN ('classified', 'expert_review', 'done')
                        AND jsonb_typeof(distress_types) = 'array' AND distress_type <> ''
                      GROUP BY distress_type) t""") or []
        return await self._run(q)

    async def get_assessment_by_id(self, assessment_id: str) -> Optional[dict]:
        if not assessment_id or len(assessment_id) > 64:
            return None
        return await self._run(lambda c: self._json(
            c, f"SELECT row_to_json(t) FROM (SELECT {_INFLIGHT_COLS} FROM assessments "
               f"WHERE id = %s::uuid LIMIT 1) t", (assessment_id,)))

    async def get_recent_processed(self, limit: int = 20) -> list[dict]:
        return await self._run(lambda c: self._json(
            c, f"SELECT COALESCE(json_agg(t), '[]'::json) FROM (SELECT {_RECENT_COLS} FROM assessments "
               f"WHERE processed_at IS NOT NULL ORDER BY processed_at DESC LIMIT %s) t", (int(limit),)) or [])

    async def list_assessments_for_dashboard(self, status: Optional[str] = None, date_from: Optional[str] = None,
                                             date_to: Optional[str] = None, limit: int = 50,
                                             offset: int = 0) -> dict:
        limit = max(1, min(200, int(limit)))
        offset = max(0, int(offset))
        where, params = [], []
        if status and status != "all":
            statuses = [s.strip() for s in status.split(",") if s.strip()]
            where.append("status = ANY(%s)")
            params.append(statuses)
        if date_from:
            where.append("created_at >= %s::timestamptz")
            params.append(date_from)
        if date_to:
            where.append("created_at <= %s::timestamptz")
            params.append(date_to)
        cond = (" WHERE " + " AND ".join(where)) if where else ""

        def q(conn):
            total = conn.execute(f"SELECT COUNT(*) FROM assessments{cond}", params).fetchone()[0]
            items = self._json(conn, f"SELECT COALESCE(json_agg(t), '[]'::json) FROM (SELECT {_DASHBOARD_COLS} "
                                     f"FROM assessments{cond} ORDER BY created_at DESC LIMIT %s OFFSET %s) t",
                               (*params, limit, offset)) or []
            return {"items": items, "total": int(total), "limit": limit, "offset": offset}
        return await self._run(q)

    async def reset_for_reclassify(self, assessment_id: str) -> dict:
        if not assessment_id or len(assessment_id) > 64:
            raise ValueError(f"invalid assessment_id: {assessment_id!r}")
        try:
            await self.update_assessment(assessment_id, dict(_RECLASSIFY_RESET))
            return {"reset": True, "id": assessment_id}
        except SupabaseError as e:
            if e.status_code == 404:
                return {"reset": False, "id": assessment_id}
            raise

    async def dashboard_summary(self) -> dict:
        def q(conn):
            rows = conn.execute(
                "SELECT status, to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD') FROM assessments").fetchall()
            statuses = Counter(r[0] or "unknown" for r in rows)
            dates = Counter(r[1] for r in rows if r[1])
            return {"total": len(rows), "by_status": dict(statuses.most_common()),
                    "by_date": sorted(dates.items(), reverse=True)[:60]}
        return await self._run(q)

    # ---------------------------------------------------------- operator delete (app/deletion.py)
    async def deletion_assessment(self, assessment_id: str) -> Optional[dict]:
        return await self._run(lambda c: self._json(
            c, "SELECT row_to_json(t) FROM (SELECT id, status, image_url, photo_id FROM assessments "
               "WHERE id = %s::uuid LIMIT 1) t", (assessment_id,)))

    async def deletion_photo(self, photo_id) -> Optional[dict]:
        return await self._run(lambda c: self._json(
            c, "SELECT row_to_json(t) FROM (SELECT id, image_url FROM photos WHERE id = %s LIMIT 1) t",
            (int(photo_id),)))

    async def deletion_delete_photo(self, photo_id) -> bool:
        return await self._run(lambda c: c.execute(
            "DELETE FROM photos WHERE id = %s", (int(photo_id),)).rowcount > 0)

    async def deletion_delete_assessment(self, assessment_id: str) -> bool:
        return await self._run(lambda c: c.execute(
            "DELETE FROM assessments WHERE id = %s::uuid", (assessment_id,)).rowcount > 0)

    async def deletion_assessment_exists(self, assessment_id: str) -> bool:
        return await self._run(lambda c: c.execute(
            "SELECT 1 FROM assessments WHERE id = %s::uuid", (assessment_id,)).fetchone() is not None)


async def build_local_db() -> LocalDb:
    """LocalDb from DATABASE_URL, opened and checked (the lifespan calls this)."""
    db = LocalDb()
    try:
        await db.open(check=True)
    except Exception:
        await db.aclose()
        raise
    return db
