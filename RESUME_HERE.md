# Resume here (work in flight since 2026-11-06)

If a session was cut off (usage limit, reboot), continue from this file. Delete it when every item
below is done and committed.

## First, every time
1. `.\status.ps1`. If the server is down: `.\start.ps1 -AutoStart -Tunnel -Watchdog -Detach`, wait about a
   minute, run `.\status.ps1` again, and give the user the new tunnel URL from `C:\GeoAI\tmp\tunnel_url.txt`.
2. GPU lock: `status.ps1` reports a lock whose owner PID is gone as STALE. Remove it only after
   `nvidia-smi` shows nothing but the idle server.
3. `git status` in both repos. Uncommitted files belong to the workstreams below. Do not commit the
   user's own `.gitignore` change (the `exports/` line) unless they ask.

## What the user asked (2026-11-06)
1. Finish the unfinished work (`eval_results/video_fixes/STATUS.md`).
2. Expert review backend for several experts at once:
   - no clashes or race conditions;
   - the same item never shown to two experts;
   - photo enlarge must be instant.
3. Research the paper on tuning open-weight model weights. This is PostTrainBench: Rank et al. 2026,
   arXiv:2603.08640, where LLM agents post-train open models under a GPU budget with anti-cheating
   rules. Do the same on our VLM in sandboxed tests, show a real held-out improvement, and update
   results, figures and the paper.
4. Update CLAUDE.md and the agent notes so restarts need no fiddling (done; see below).
5. Make everything safe to continue after a usage limit (this file, plus each workstream's STATUS file).

## Workstreams (each keeps its own STATUS file, overwritten after each milestone)

| Workstream | Files it owns | Its status file |
|---|---|---|
| A. Multi-expert claims + instant enlarge | `app/expert_review.py`, `app/expert_routes.py`, `expert_ui/index.html`, `migrations/007_review_claims.sql`, `scripts/setup_local_db.py`, `scripts/tests/test_expert_*.py`, `test_local_db.py`, paper §6.4 (one paragraph) | `eval_results/expert_claims_STATUS.md` |
| B. PostTrainBench-style tuning study | `scripts/ft_*.py`, `scripts/tests/test_ft_data.py`, `eval_results/finetune/`, big artifacts in `C:\GeoAI\finetune\`, figures `fig_6_11*`, paper §6.11 / §7.12 | `eval_results/finetune/STATUS.md` |
| C. Leftover drive-video studies | `C:\GeoAI\analysis\fixes\...`, `eval_results/video_fixes/`, figures `fig_8_*`, paper §8.6, GeoAI_Video repo (`docs/research/METRICS.md` §9, code) | `eval_results/video_fixes/STATUS.md` |

Rules for all of them:
- `app/` must stay importable after every saved edit, because the watchdog reloads from disk.
- Long GPU jobs run as detached processes that hold `C:\GeoAI\gpu.lock` (owner.txt has the PID) and
  release it in a finally block.
- Result files are written atomically.
- B's pre-registration is never edited (addenda only), and every test-split evaluation is counted.

## Coordinator steps still to do (after A, B and C report)
1. Review each workstream's diff and results, and re-run its tests plus `scripts/validate_all.py`.
2. Deploy A to production with the restart recipe in CLAUDE.md, using the new `start.ps1`:
   - `.\stop.ps1`, then `.\start.ps1 -AutoStart -Tunnel -Watchdog -Detach`;
   - check `status.ps1` and `eval_results\autostart.log`;
   - check that the worker started without a manual POST;
   - test the expert page with two browser sessions.
3. Paper: check that §6.4, §6.11, §7.12 and §8.6 read consistently, and that the revision notes and the
   reading guide (P4 §6.5–6.11) are right.
4. CLAUDE.md:
   - key-metrics and decision rows for the tuning study and the video studies;
   - next steps;
   - expert UI claims.
   Also update memory (project_expert_ui, a new project_posttrain_study note, project_geoai_video_repo)
   and GeoAI_Video's CLAUDE.md if its contract changed.
5. Commit Capstone (no Co-Authored-By lines; user's rule) and push with `git push origin HEAD`. Check
   that GeoAI_Video is committed and push it with `GCM_INTERACTIVE=Never`.

## Done so far (2026-11-06)
- Demo restored after the reboot: server, worker, video pipeline and a new tunnel.
- `start.ps1`:
  - auto-start sends the API key (it used to get a silent 401) and fires after every watchdog relaunch,
    logging to `eval_results/autostart.log`;
  - `-Tunnel` runs cloudflared hidden and saves the URL;
  - new `-Detach` switch.
- New `status.ps1`.
- CLAUDE.md: START HERE, the multi-agent rules, the restart recipe, and the Attain WS_V1/OS_V1 facts.
- Memory: live demo, restart, Grad-CAM description, and a new feedback note on ops friction.
- Contamination pre-check of Attain WS_V1/OS_V1 against WS_V2: no duplicates or crops found. B runs
  the stricter check.
