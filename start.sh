#!/usr/bin/env bash
# ============================================================
# GeoAI / Pavement Distress Pipeline — failsafe startup (Linux)
# ============================================================
# Single command to launch the full production stack on the A5000.
#
# Usage:
#   ./start.sh                       # server only
#   ./start.sh --auto-start          # also start the worker
#   ./start.sh --tunnel              # also expose via Cloudflare
#   ./start.sh --watchdog            # auto-restart on crash
#   ./start.sh --auto-start --tunnel --watchdog   # full prod
#
# First-time:  chmod +x start.sh
# Stop:        Ctrl+C
# ============================================================

set -uo pipefail

AUTO_START=0
TUNNEL=0
WATCHDOG=0
PORT=8000
HOST=0.0.0.0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --auto-start) AUTO_START=1; shift ;;
        --tunnel)     TUNNEL=1; shift ;;
        --watchdog)   WATCHDOG=1; shift ;;
        --port)       PORT="$2"; shift 2 ;;
        --host)       HOST="$2"; shift 2 ;;
        -h|--help)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

ok()   { printf '  \033[32m[OK]\033[0m   %s\n' "$1"; }
warn() { printf '  \033[33m[WARN]\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m[FAIL]\033[0m %s\n' "$1"; }
step() { printf '\n\033[36m==> %s\033[0m\n' "$1"; }

# ----- Pre-flight -----
step "Pre-flight checks"

# 1. venv
VENV_PY="venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    fail "venv not found at $VENV_PY"
    echo "    Recreate it:  python -m venv venv && source venv/bin/activate && uv pip install -r requirements.txt"
    exit 1
fi
ok "venv found"

# 2. .env
if [[ ! -f ".env" ]]; then
    fail ".env file missing"
    echo "    Copy .env.example -> .env and fill in SUPABASE_SERVICE_KEY"
    exit 1
fi
for key in SUPABASE_URL SUPABASE_SERVICE_KEY; do
    if ! grep -qE "^${key}=\S" .env; then
        fail ".env missing required key: $key"
        exit 1
    fi
done
ok ".env present with required keys"

# 3. Production switches
if grep -qE '^DISABLE_ADAPTER=true' .env; then
    ok "DISABLE_ADAPTER=true (Improved Baseline production)"
else
    warn "DISABLE_ADAPTER is not 'true' — adapter pathway will run"
fi
if grep -qE '^PROMPTS_VERSION=v\d' .env; then
    ok "$(grep -E '^PROMPTS_VERSION=' .env)"
fi

# 4. Cloudinary
if grep -qE '^CLOUDINARY_CLOUD_NAME=\S' .env \
   && grep -qE '^CLOUDINARY_API_KEY=\S' .env \
   && grep -qE '^CLOUDINARY_API_SECRET=\S' .env; then
    ok "Cloudinary creds present"
else
    warn "Cloudinary creds missing — dashboard delete will only remove Supabase row"
fi

# 5. Port free
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${PORT}\$"; then
    warn "Port $PORT in use — killing stale uvicorn"
    pkill -f "uvicorn.*app.main" || true
    sleep 2
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${PORT}\$"; then
        fail "Port $PORT still in use. Find offender:  ss -ltnp 'sport = :${PORT}'"
        exit 1
    fi
fi
ok "Port $PORT is free"

# 6. GPU
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu=$(nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU: $gpu"
else
    warn "nvidia-smi not found — running on CPU only?"
fi

# ----- Tunnel (background) -----
if [[ $TUNNEL -eq 1 ]]; then
    step "Cloudflare tunnel"
    if command -v cloudflared >/dev/null 2>&1; then
        ok "Launching cloudflared (logs -> /tmp/cloudflared.log)"
        cloudflared tunnel --url "http://localhost:$PORT" >/tmp/cloudflared.log 2>&1 &
        echo "    Tunnel PID: $!  (tail -f /tmp/cloudflared.log for the public URL)"
    else
        warn "cloudflared not installed — skipping tunnel"
        echo "    Install via tunnel/setup_tunnel.sh"
    fi
fi

# ----- Worker auto-start (background) -----
if [[ $AUTO_START -eq 1 ]]; then
    step "Worker auto-start scheduled"
    (
        for i in $(seq 1 60); do
            if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
                break
            fi
            sleep 3
        done
        if curl -fsS -X POST "http://127.0.0.1:$PORT/operator/start" >/dev/null 2>&1; then
            echo "[auto-start] worker started"
        else
            echo "[auto-start] FAILED — start manually from /operator UI"
        fi
    ) &
    ok "worker will auto-start once /health is reachable"
fi

# ----- Banner -----
step "Endpoints (open in browser)"
echo "  Operator UI:      http://localhost:$PORT/operator     (Start/Stop pipeline, metrics)"
echo "  Image Dashboard:  http://localhost:$PORT/dashboard    (browse, delete, re-classify)"
echo "  Health:           http://localhost:$PORT/health       (JSON status)"
echo "  API docs:         http://localhost:$PORT/docs         (Swagger)"
echo ""
echo "  Stop:             Ctrl+C"
if [[ $WATCHDOG -eq 1 ]]; then
    echo "  Watchdog:         ON — server will auto-restart on crash"
fi
echo ""

# ----- Run uvicorn -----
run_uvicorn() {
    "$VENV_PY" -m uvicorn app.main:app \
        --env-file .env --host "$HOST" --port "$PORT" --workers 1
}

if [[ $WATCHDOG -eq 1 ]]; then
    crash_count=0
    while true; do
        step "Starting uvicorn (attempt #$((crash_count + 1)))"
        run_uvicorn
        exit_code=$?
        # 0 = clean exit, 130 = Ctrl+C (SIGINT)
        if [[ $exit_code -eq 0 || $exit_code -eq 130 ]]; then
            echo "uvicorn exited cleanly. Stopping watchdog."
            break
        fi
        crash_count=$((crash_count + 1))
        backoff=$(( 5 * crash_count ))
        [[ $backoff -gt 60 ]] && backoff=60
        fail "uvicorn crashed (exit $exit_code). Restarting in ${backoff}s (crash #${crash_count})"
        pkill -f "uvicorn.*app.main" 2>/dev/null || true
        sleep "$backoff"
    done
else
    run_uvicorn
fi

echo ""
echo "Shutdown complete."
