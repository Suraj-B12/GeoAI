# WebGIS Layer — Handoff Brief

**Paste the entire contents of this file as the first message in a new Claude Code chat.**
Everything below is self-contained context for building the WebGIS layer of the GeoAI pavement-distress system.

---

## Project Context (1-paragraph)

**GeoAI** is a pavement-distress detection system being built as a college capstone in Bengaluru. Citizens snap photos of damaged roads with a mobile app called **RoadSide**. Each photo is uploaded to **Cloudinary** (CDN-backed image storage) and the photo metadata (Cloudinary URL, GPS coordinates, address) is written to a **Supabase Postgres** database. A two-stage AI pipeline (Qwen2.5-VL-7B vision-language model running on an A5000 GPU) classifies each photo into a distress type (longitudinal crack, transverse crack, alligator crack, pothole, or normal) with severity (Low/Medium/High). The **WebGIS layer is what civic authorities (BBMP — Bruhat Bengaluru Mahanagara Palike) will use to see where damaged roads are concentrated**. Build that.

The phone app, image storage, database, AI pipeline, and expert-review UI are all already done. Only the WebGIS is missing. **You do not need to touch any of the existing components.** Read from the database, render a map.

---

## Your goal

Build a **standalone web app** that:
1. Reads classified pavement assessments from Supabase
2. Renders them as points on a map of Bengaluru, color-coded by distress type and sized/intensified by severity
3. Lets users filter by date range, distress type, severity, and review status
4. Shows a detail panel when a marker is clicked (image + classification + confidence + address)
5. Optionally shows a heatmap layer for spotting hotspots
6. Optionally exports filtered results as CSV for the road department

The audience is **civic authorities and road maintenance departments**, not citizens or the AI team. Design accordingly: information-dense, professional, no marketing.

---

## Database (Supabase / Postgres)

### Connection details

```
PROJECT_URL  = https://vtlkitpoffudiefuoijb.supabase.co
ANON_KEY     = <ask the user — public anon key, safe for browser use>
```

**Important:** for the WebGIS, use the **Supabase public anon key**, NOT the service-role key. The anon key is safe for the browser. Service-role bypasses RLS and must stay server-side only.

You can interact via:
- The official `@supabase/supabase-js` JavaScript client library (best DX)
- Direct PostgREST REST calls (no SDK needed)
- GraphQL via Supabase's pg_graphql extension (if you prefer)

### Tables you will read FROM

#### `assessments` — the main table you visualize

| Column | Type | What it is |
|---|---|---|
| `id` | UUID PK | Internal AI-pipeline assessment ID |
| `photo_id` | BIGINT FK → photos.id | Links to the original photo upload |
| `image_url` | TEXT | Cloudinary URL — embeddable directly into `<img src="...">` |
| `latitude`, `longitude` | DOUBLE PRECISION | GPS coordinates from the phone (these drive the map markers) |
| `address` | TEXT | Reverse-geocoded human-readable address |
| `road_segment_id` | TEXT | (currently NULL) future field for grouping by road |
| `status` | TEXT enum | `pending` / `processing` / `classified` / `expert_review` / `done` / `failed` |
| `stage1_label` | TEXT | `'Normal'` or `'Distress'` |
| `stage1_confidence` | REAL (0-1) | How sure the AI is about Stage 1 (≥0.80 = high) |
| `is_distressed` | BOOLEAN | Convenience flag from Stage 1 |
| `distress_types` | JSONB array of strings | Stage 2 output, e.g. `["Longitudinal Crack (D00)", "Pothole (D40)"]`. Empty array if not distressed. |
| `severity` | TEXT enum | `None` / `Low` / `Medium` / `High` / `Unknown` |
| `description` | TEXT | AI-generated one-sentence description |
| `stage2_confidence` | REAL (0-1) | How sure the AI is about the type. 0 if Stage 2 didn't run. |
| `needs_expert_review` | BOOLEAN | Flagged for human verification (when confidence < 80%) |
| `expert_reviewed` | BOOLEAN | A human has confirmed this |
| `expert_corrected_types` | JSONB | Expert's correction (use this if present, else use `distress_types`) |
| `expert_corrected_severity` | TEXT | Expert's severity correction (use if present) |
| `expert_notes` | TEXT | Free-form notes from the expert |
| `processed_at` | TIMESTAMPTZ | When the AI finished (null if still pending/processing) |
| `created_at` | TIMESTAMPTZ | When the assessment was queued (matches photo upload time) |

**Filter rule of thumb:** for the public WebGIS, show only rows where `status IN ('classified', 'done')`. Skip `pending`, `processing`, `expert_review` (incomplete), and `failed`.

#### `photos` — original uploads (rarely needed by WebGIS)

The `photos` table is what the mobile app writes to. The pipeline trigger then auto-creates an `assessments` row with the same image_url/lat/lng. **You usually don't need to query `photos` directly** — every row has a corresponding `assessments` row with the same image_url. Use it only if you need to show "raw" uploads not yet processed.

| Column | Type |
|---|---|
| `id` | BIGINT PK (auto-increment) |
| `image_url` | TEXT (Cloudinary URL) |
| `address` | TEXT |
| `latitude`, `longitude` | DOUBLE PRECISION |
| `created_at` | TIMESTAMPTZ |

#### `custom_distress_types` — reference list (for filter dropdown)

Pre-seeded list of all known distress types. Use it to populate a "filter by type" dropdown so it stays in sync if more types are added.

| Column | Purpose |
|---|---|
| `id` | UUID |
| `name` | e.g. `"Pothole (D40)"` — matches the strings in `assessments.distress_types` |
| `description` | Human-readable description (good for tooltip on the filter chip) |

### Helper SQL views (already exist in Supabase)

#### `pipeline_metrics` (single row, aggregate counts)

Useful for showing a compact stats bar: "5,432 photos analyzed · 89% confidence average · 12 high-severity hotspots".

| Column | What it is |
|---|---|
| `pending_count` / `processing_count` / `classified_count` / `expert_review_count` / `done_count` / `failed_count` / `total_count` | Status breakdowns |
| `throughput_last_hour` / `throughput_last_minute` | Recent pipeline throughput (probably not interesting for civic users) |
| `avg_stage1_confidence_last_hour` / `avg_stage2_confidence_last_hour` | Average AI confidence |
| `distressed_last_hour` / `normal_last_hour` | Recent distress vs normal split |

#### `pipeline_class_distribution` (multi-row)

Counts per distress type over the last 24h. Good for a small bar chart showing "what kind of damage is most common right now".

| Column | What it is |
|---|---|
| `distress_type` | e.g. `"Pothole (D40)"` |
| `count` | Number of recent occurrences |

### Tables you will NOT touch

- `worker_state` — internal AI worker heartbeat
- `retrain_jobs` — internal LoRA retraining queue
- `auth.users` — Supabase auth (irrelevant unless you build a login)

### RLS (Row Level Security) — important

`assessments` has RLS enabled. The anon key has the policies:
- ✅ Users can SELECT their own rows (rows where `user_id = auth.uid()`)
- ❌ Users cannot SELECT rows they don't own

**For a public WebGIS this is a problem.** Two options:
1. **Add a public read policy** (recommended): ask the user to run this SQL in Supabase, which lets the anon key see classified rows for the WebGIS:
   ```sql
   CREATE POLICY "Public read classified assessments"
       ON assessments FOR SELECT
       USING (status IN ('classified', 'done'));
   ```
2. **Use auth** (heavier): require civic users to log in.

**Default to option 1** unless the user asks for auth.

`photos` already has a "Public read photos" policy (added in migration 002), so anon-key reads work there.

---

## Sample queries you will run

### Fetch all completed assessments for the map
```sql
-- via PostgREST:
GET /rest/v1/assessments
    ?status=in.(classified,done)
    &select=id,latitude,longitude,address,image_url,
            distress_types,severity,stage2_confidence,
            stage1_confidence,description,processed_at,
            expert_reviewed,expert_corrected_types,
            expert_corrected_severity
    &order=processed_at.desc
    &limit=5000
```

In the JS client:
```javascript
const { data } = await supabase
  .from('assessments')
  .select(`
    id, latitude, longitude, address, image_url,
    distress_types, severity, stage2_confidence, stage1_confidence,
    description, processed_at,
    expert_reviewed, expert_corrected_types, expert_corrected_severity
  `)
  .in('status', ['classified', 'done'])
  .order('processed_at', { ascending: false })
  .limit(5000);
```

### Filter by distress type
```javascript
.contains('distress_types', ['Pothole (D40)'])
```

### Filter by severity
```javascript
.eq('severity', 'High')
```

### Filter by date range
```javascript
.gte('processed_at', '2026-04-01')
.lt('processed_at', '2026-05-01')
```

### Filter by bounding box (when user pans the map)
```javascript
.gte('latitude', swLat).lte('latitude', neLat)
.gte('longitude', swLng).lte('longitude', neLng)
```

### Count of distress in a polygon (for choropleth or summary)
PostgREST doesn't natively do PostGIS spatial queries via the REST API. Options:
- Compute in JavaScript after fetching points
- Or write a Postgres RPC function `count_distress_in_polygon(geom)` and call via `/rpc/`

For a typical WebGIS, in-browser computation on ≤10k points is fine.

---

## What to display

### Per-marker information
When the user clicks a marker, show:
- **The photo itself** — `<img src={image_url}>` (Cloudinary serves it directly with proper CORS)
- **Address** (`address` field)
- **Distress type(s)** — if `expert_corrected_types` is non-empty use it, else use `distress_types`
- **Severity** — same priority: expert correction first
- **AI description** (`description` field — one sentence)
- **Confidence** — `stage2_confidence` if Stage 2 ran, else `stage1_confidence`
- **Reviewed by expert?** — small badge showing `expert_reviewed=true`
- **Date** — `processed_at` formatted nicely

### Map color coding (suggested)

| Distress type | Suggested color |
|---|---|
| Longitudinal Crack (D00) | Yellow / amber |
| Transverse Crack (D10) | Orange |
| Alligator Crack (D20) | Red |
| Pothole (D40) | Dark red / crimson |
| Block Crack (D43) | Purple |
| Other / Unknown | Gray |

Override if multiple types — pick the most severe (potholes > alligator > transverse > longitudinal).

### Marker size by severity

| Severity | Marker radius |
|---|---|
| Low | 6px |
| Medium | 9px |
| High | 13px |
| None / Unknown | 4px (small, deemphasized) |

### Filters needed
- Date range picker (default: last 30 days)
- Distress type multi-select (populated from `custom_distress_types`)
- Severity buttons (Low, Medium, High — multi-toggle)
- "Only show expert-reviewed" toggle
- "Only show high-confidence (≥80%)" toggle

### Optional features (nice-to-haves)
- **Heatmap toggle** — for spotting density without resolving individual markers
- **Cluster markers** at low zoom levels (use `leaflet.markercluster`)
- **CSV export** — current filtered set, with columns: id, latitude, longitude, address, distress_types, severity, processed_at, image_url
- **Bengaluru ward boundaries overlay** — if you can find a public GeoJSON of BBMP wards, overlay them and let users see counts per ward
- **Time slider** — animate distress accumulation over time
- **Real-time updates** — Supabase has a Realtime subscription feature; you can subscribe to INSERT events on assessments

---

## Tech stack suggestion (open to alternatives)

- **Map library:** [Leaflet.js](https://leafletjs.com/) — free, no API key, mature, plenty of plugins. Mapbox/Google Maps cost money.
- **Tiles:** OpenStreetMap (free) is fine to start. Switch to a styled tile provider later if needed.
- **Framework:** Plain HTML/CSS/JS works. Or use React/Vue/Svelte — your call. The existing project's other UIs (`expert_ui`, `operator_ui`, `test_website`) are all vanilla HTML+JS single files, so following that convention keeps the project consistent. **But the user said the WebGIS can have its own design language**, so feel free to use a framework if it speeds you up.
- **Supabase client:** `@supabase/supabase-js` — official, well-maintained.
- **Heatmap plugin:** `leaflet.heat` if you want a heatmap layer.
- **Cluster plugin:** `leaflet.markercluster` for dense areas.
- **Charts (for stats panel):** Chart.js or any small lib.

---

## Design freedom

**You are not bound to the existing UI design language.** The other UIs in this project (`expert_ui/`, `operator_ui/`, `test_website/`) use a specific design — SK Modernist font, monochrome with green/red/amber accents. **Do not feel obligated to match that.** Civic dashboards typically have their own look (blue/white government-y, or whatever serves the data best). Make decisions that serve the WebGIS users, not consistency with internal tools.

---

## Where things are in the project (in case you need to look)

```
Capstone/
├── db_schema.sql                    — initial DB schema
├── migrations/
│   ├── 001_operator_pipeline.sql    — adds status enum, RPC, views
│   └── 002_photos_integration.sql   — photos→assessments trigger
├── app/
│   ├── main.py                      — FastAPI server (you don't touch this)
│   └── ...                          — all backend code
├── operator_ui/index.html           — example design language (don't have to match)
├── expert_ui/index.html             — same
├── test_website/index.html          — same
├── paper/                           — research paper, technical handbook
├── SETUP.md                         — full operations guide (read if curious)
└── CLAUDE.md                        — project-wide context (read for full context)
```

---

## Suggested project structure for the WebGIS

```
Capstone/
└── webgis/                  ← create this folder
    ├── index.html           ← entry point
    ├── styles.css           ← styles (whatever design you choose)
    ├── app.js               ← main logic
    ├── supabase.js          ← Supabase client wrapper
    ├── filters.js           ← filter UI logic
    ├── markers.js           ← map marker logic + color/size mapping
    ├── detail-panel.js      ← click-to-show photo + classification
    └── README.md            ← short note on how to run it locally
```

Keep it self-contained. No build step required ideally.

---

## Things to verify with the user before you start coding

1. **Public anon key** — Ask for the Supabase anon key (it's different from the service_role key, safe for the browser).
2. **Should you add the public RLS policy** for `assessments` (so the anon key can SELECT classified rows)? If yes, give them the SQL to paste into Supabase SQL Editor.
3. **Map provider** — Confirm Leaflet + OSM is fine, or do they want Mapbox/Google?
4. **Default city center** — Bengaluru's lat/lng is `12.9716, 77.5946` (MG Road). Use this as the default map center.
5. **Whether to include the optional features** — heatmap, clustering, CSV export, ward boundaries overlay, time slider, real-time subscription. Confirm scope.

---

## Things you should NOT do

- ❌ Don't modify any backend code (`app/`, migrations, scripts).
- ❌ Don't write to the `assessments` table from the WebGIS. It's read-only here.
- ❌ Don't use the service_role key in browser code.
- ❌ Don't hard-code coordinates or dummy data — read from Supabase.
- ❌ Don't try to run the AI model — it lives on the GPU server, you don't need it.

---

## How to test what you build

1. Insert some test photos into the `photos` table (the trigger creates `assessments` rows automatically). Then run the AI pipeline (`POST /operator/start` on the existing FastAPI server) to populate classifications.
2. Or skip step 1 if there are already some `classified`/`done` rows in `assessments` (most likely there are — the team has run smoke tests).
3. Open your WebGIS in a browser. Verify markers appear at the right Bengaluru locations.
4. Click markers, check the detail panel.
5. Use filters, verify they actually filter.
6. Mobile view test — civic users will check this on phones too.

---

## What deliverables the user expects

1. A working `webgis/` folder with the app
2. A `webgis/README.md` explaining how to run it (e.g. "open `index.html` in a browser, or serve with `python -m http.server`")
3. Screenshots in the README showing the working dashboard
4. A note on what's done vs what's optional/pending

---

That's everything. Build something useful for the road maintenance team.
