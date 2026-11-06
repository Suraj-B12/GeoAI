# Expert review: multi-expert claims + instant enlarge — status

Updated: 2026-11-06 (milestone 6 of 6: DONE except the production restart, which the main session does)

**Production local DB:** migration 007 applied 2026-11-06 10:13 UTC via `scripts/setup_local_db.py`; ledger
`geoai_local.schema_migrations` sha `2685607c…18fdb3` = the file; re-run says already_applied; tables empty.
Production still runs the OLD code until it is restarted (UI and backend then go live together).
validate_all 18/18 PASS (2026-11-06).

## Goal
Several experts review at once without clashes (server + local PostgreSQL are the arbiter: leases on
items, conditional saves), and the photo lightbox shows something on the first frame after a click.

## Done
- `migrations/007_review_claims.sql` (research.review_claims, research.review_skips), registered in
  `scripts/setup_local_db.py`; `scripts/tests/test_local_db.py` expects 9 files and re-runs 007.
  The code also creates the tables on first use (`ensure_claims_schema`), like 006.
- `app/expert_review.py` "Claims" section: claim_next_video, claims_grant, claims_heartbeat,
  claims_release (skip), claims_active, take_claim, save_video_review_claimed (lease + seen_reviewed_at;
  Conflict -> 409), finish_citizen_review, list_video_marked. Old list_video/save_video_review kept.
- `app/expert_routes.py`: /claims/next, /claims/grant, /claims/heartbeat, /claims/release,
  /claims/active, /citizen/review/{id} (conditional Supabase PATCH, exactly one row or 409, then log);
  /video/list = list_video_marked (+ `me`, claim_name, claim_mine, image_v);
  /video/review = save_video_review_claimed (409 when held by another or stale);
  claims/next pre-renders the 1920-px previews of handed-out segments (disk cache) in the background.
- `app/gis_routes.py` /gis/image: `?v=<16+ hex of the image SHA-256>` -> `Cache-Control: immutable, 1 year`
  (only when v matches; otherwise unchanged 1 day).
- `expert_ui/index.html`: per-tab session (sessionStorage); review lists = batches of 12 leases with
  top-up after save/skip; heartbeat 60 s; sendBeacon release on pagehide; Skip on video cards; presence
  line; "X has this item open" + Save disabled on Reviewed items held elsewhere; 409 messages; citizen
  saves through /expert-api/citizen/review when served by the server (file:// = single-reviewer mode as
  before); citizen cards keyed by id (fixes edits landing on the wrong card after a save/skip); video save
  no longer re-renders other cards (their unsaved notes stay); instant lightbox (thumbnail bitmap drawn on
  a canvas in the click handler, decoded full-size swapped into the same box, zoom kept); prefetch of the
  visible cards' full images; image URLs carry v. Fixed: citizen card image passed its id as the overlay.
- Paper §6.4: paragraph "Several experts at once (2026-11-06)".

## Tests (all passing, 2026-11-06)
- `scripts/tests/test_expert_concurrency.py` 8/8 (race: 8 sessions x 50 ops, ~830 grants, 0 double
  grants; mutation check: removing BOTH the advisory lock and the upsert guard is caught).
- `scripts/tests/test_expert_review.py` 6/6. `scripts/tests/test_local_db.py` 12/12 (with DATABASE_URL).
- validate_all: see final report.
- Browser (harness on :8010, scratch DB geoai_harness, real 4K frames): two tabs get disjoint batches;
  edit lease shows "Harness B has this item open"; forced save -> 409; stale edit -> 409 + reload;
  skip replaces the card; leaving the page releases all leases; 375 px no horizontal scroll.

## Enlarge timings (harness, 8 opens each, cold server cache, median)
| Page / condition | picture on screen | full-size image |
|---|---|---|
| Before (old page), 6 s after the list loads | 506 ms (407-612), blank until then | 506 ms |
| After, 6 s after the list loads (prefetch on) | 6.8 ms | 7 ms (6/8 prefetched; 2 below the fold 140-149 ms) |
| After, click at once (prefetch on) | 6 ms | 7 ms (worst 521 ms, thumbnail at 5.6 ms) |
| After, prefetch off, click at once (worst case) | 5.9 ms (0.6-7.2) | 184 ms (165-756) |

## Next
1. Done: 007 applied to production local PG; production /expert-api/video/list still answers.
2. The main session restarts production to deploy (UI + backend go live together), then a two-browser check.

## How to resume
Read this file, `git diff` the files above, run
`CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -c "import app.expert_routes, app.expert_review"`,
`venv/Scripts/python.exe scripts/tests/test_expert_review.py`, `.../test_expert_concurrency.py`.
