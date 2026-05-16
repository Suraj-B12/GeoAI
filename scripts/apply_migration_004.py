"""
Helper for applying migration 004 to the live Supabase project.

Supabase's Management API requires a Personal Access Token (different from
the service-role JWT we use for data ops). Service-role JWT cannot run DDL.

This script either:
  - tries any pre-configured exec_sql RPC (if you've set one up)
  - or prints the one-click URL + SQL the operator pastes into Supabase
    SQL editor

Usage:
    python scripts/apply_migration_004.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIGRATION = PROJECT_ROOT / "migrations" / "004_pavement_filter_and_dashboard.sql"


def load_env() -> dict:
    env = {}
    for line in (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def project_ref_from_url(url: str) -> str:
    """Extract 'abcdefg' from 'https://abcdefg.supabase.co'."""
    return urllib.parse.urlparse(url).hostname.split(".")[0]


def try_rpc(url: str, key: str, sql: str) -> bool:
    """Try common RPC names — return True if any worked."""
    for fn in ["exec_sql", "execute_sql", "run_sql"]:
        try:
            req = urllib.request.Request(
                f"{url}/rest/v1/rpc/{fn}",
                data=json.dumps({"query": sql}).encode(),
                method="POST",
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
            )
            resp = urllib.request.urlopen(req, timeout=10)
            print(f"OK: applied via rpc/{fn}: {resp.read().decode()[:200]}")
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue  # function doesn't exist, try next
            print(f"rpc/{fn} returned HTTP {e.code}: {e.read().decode()[:200]}")
            continue
        except Exception:
            continue
    return False


def main():
    if not MIGRATION.exists():
        sys.exit(f"ERROR: {MIGRATION} not found")
    sql = MIGRATION.read_text(encoding="utf-8")

    env = load_env()
    url = env.get("SUPABASE_URL")
    key = env.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        sys.exit("ERROR: SUPABASE_URL / SUPABASE_SERVICE_KEY missing from .env")

    ref = project_ref_from_url(url)
    print(f"Project: {ref}")
    print(f"Migration file: {MIGRATION}")
    print()

    # 1. Try RPC
    print("Attempting automated apply via RPC...")
    if try_rpc(url, key, sql):
        print()
        print("SUCCESS — migration applied automatically.")
        return

    # 2. Fall back to manual instructions
    print("No exec_sql RPC found. Apply manually (30 seconds):")
    print()
    print("=" * 70)
    print(f"1. Open: https://supabase.com/dashboard/project/{ref}/sql/new")
    print("2. Paste this SQL (or copy-paste from migrations/004_*.sql):")
    print("=" * 70)
    print()
    print(sql)
    print()
    print("=" * 70)
    print("3. Click Run.")
    print("4. Verify: query 'SELECT column_name FROM information_schema.columns")
    print("           WHERE table_name=\\'assessments\\' AND column_name LIKE \\'pavement%\\';'")
    print("           — should return 3 rows.")
    print("=" * 70)


if __name__ == "__main__":
    main()
