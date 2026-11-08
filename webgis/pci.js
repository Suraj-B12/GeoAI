/* A→B condition report: chainage + IRC:82-2023 Pavement Condition Index (PCI 0–100, screening estimate).
   Talks to geoai.pci.v1 (/gis/pci/*). Shares the map and helpers through window.GeoAIApp. */
(() => {
  'use strict';
  const G = window.GeoAIApp, D = window.SurveyData, $ = id => document.getElementById(id);
  if (!G || !D || !window.L) return;
  const esc = G.esc, api = G.api;
  const fmtM = m => D.formatDistance(Number(m));
  const num = v => (v === null || v === undefined || v === '' || !Number.isFinite(Number(v))) ? null : Number(v);
  const pct = v => num(v) === null ? '—' : `${Number(v).toFixed(1)} %`;
  const chain = m => { const v = Math.max(0, Math.round(Number(m) || 0)); return `${Math.floor(v / 1000)}+${String(v % 1000).padStart(3, '0')}`; };
  const KEY_STORE = 'geoaiPciOperatorKey';
  // IRC:82-2023 facts used by the form and the result card. GET /gis/pci/meta → irc2023 replaces them when the
  // server provides it (same values; app/pci_irc2023.py is the source).
  const P23 = {
    classes: [['Excellent', 90], ['Good', 80], ['Satisfactory', 60], ['Fair', 40], ['Poor', 20], ['Fail', -Infinity]],
    rec: { Excellent: 'Routine maintenance', Good: 'Preventive maintenance', Satisfactory: 'Renewal (as per section 9.5)',
      Fair: 'Minor rehabilitation (based on structural evaluation)', Poor: 'Major rehabilitation / structural overlay', Fail: 'Reconstruction' },
    roads: [
      { id: 'urban', name: 'Urban road', surfaces: ['BC', 'SDBC'], def: 'BC' },
      { id: 'highway', name: 'Highway (NH/SH)', surfaces: ['BC', 'SDBC'], def: 'BC' },
      { id: 'mdr', name: 'Major District Road (MDR)', surfaces: ['SD', 'OGPC', 'MSS', 'SDBC', 'BC'], def: 'BC' },
      { id: 'rural', name: 'Rural road (ODR, VR)', surfaces: ['SD', 'OGPC', 'MSS', 'SDBC'], def: 'SDBC' }],
    surfaceNames: { SD: 'Surface Dressing', OGPC: 'Open Graded Premix Carpet', MSS: 'Mix Seal Surfacing', SDBC: 'Semi Dense Bituminous Concrete', BC: 'Bituminous Concrete' },
    params: [
      { id: 'cracking', name: 'Cracking', unit: '%', hint: '% of carriageway area', w: .12, lim: [0, 100], step: '0.1' },
      { id: 'ravelling', name: 'Ravelling', unit: '%', hint: '% of carriageway area', w: .10, lim: [0, 100], step: '0.1' },
      { id: 'potholes', name: 'Potholes', unit: 'number', hint: 'number (area ÷ 0.1 m²)', w: .16, lim: [0, 1000], step: '1' },
      { id: 'patching', name: 'Patching', unit: '%', hint: '% of carriageway area', w: .08, lim: [0, 100], step: '0.1' },
      { id: 'rut_depth', name: 'Rut depth', unit: 'mm', hint: 'mm, mean of lanes', w: .14, lim: [0, 200], step: '0.5' },
      { id: 'iri', name: 'Roughness (IRI)', unit: 'm/km', hint: 'm/km, mean of lanes', w: .40, lim: [0.5, 30], step: '0.01' }] };
  const CLASS_KEYS = P23.classes.map(c => c[0].toLowerCase());
  const classOf = v => num(v) === null ? null : P23.classes.find(([, lo]) => Number(v) > lo)[0];
  const ckey = c => (c && CLASS_KEYS.includes(String(c).toLowerCase())) ? String(c).toLowerCase() : 'none';
  const road = id => P23.roads.find(r => r.id === id) || P23.roads[0];
  const paramDef = id => P23.params.find(p => p.id === id) || { id, name: id, unit: '', w: 0 };
  async function loadMeta() {
    if (S.metaLoaded) return; S.metaLoaded = true;
    try {
      const m = (await get('/gis/pci/meta'))?.irc2023; if (!m) return;
      if (Array.isArray(m.road_classes)) for (const rc of m.road_classes) { const r = P23.roads.find(x => x.id === rc.id); if (r && Array.isArray(rc.surfaces) && rc.surfaces.length) { r.surfaces = rc.surfaces; r.def = rc.default_surface || r.def; } }
      if (Array.isArray(m.params)) for (const p of m.params) { const d = P23.params.find(x => x.id === p.id); if (d && Array.isArray(p.entered_limits)) d.lim = p.entered_limits.map(Number); }
      if (S.open && S.view === 'build') render();
    } catch { /* fallback constants stay */ }
  }

  const S = { colorBy: 'pci', open: false, view: 'build', tab: 'new', pick: null, points: { a: null, b: null, via: [] },
    path: null, pathError: '', pathLoading: false, report: null, reportSource: null, savedId: null,
    error: '', loading: false, section: null, frame: null, saved: [], savedLoaded: false, controller: null,
    features: new Map(), minimized: false, savedError: '' };
  window.PCI = { get active() { return S.open; }, get state() { return S; } };

  /* ---------- map layers ---------- */
  const lmap = () => G.map?.map;
  let layers = null;
  function ensureLayers() {
    const m = lmap(); if (!m || layers) return;
    m.createPane('pci'); m.getPane('pci').style.zIndex = '440';
    m.createPane('pciMarkers'); m.getPane('pciMarkers').style.zIndex = '640';
    const renderer = L.svg({ pane: 'pci', padding: .3 });
    layers = { renderer, path: L.featureGroup([], { pane: 'pci' }), sections: L.featureGroup([], { pane: 'pci' }),
      ticks: L.featureGroup([], { pane: 'pci' }), marks: L.featureGroup([], { pane: 'pciMarkers' }),
      snaps: L.featureGroup([], { pane: 'pci' }), frame: L.featureGroup([], { pane: 'pciMarkers' }) };
    m.on('click', onMapClick);
  }
  function showLayers(on) {
    const m = lmap(); if (!m || !layers) return;
    for (const key of ['path', 'sections', 'ticks', 'marks', 'snaps', 'frame']) {
      if (on && !m.hasLayer(layers[key])) layers[key].addTo(m);
      if (!on && m.hasLayer(layers[key])) m.removeLayer(layers[key]);
    }
    // Survey markers stay visible but stop taking clicks, so a click always places a point.
    for (const pane of ['survey', 'markerPane']) { const el = m.getPane(pane); if (el) el.style.pointerEvents = on ? 'none' : ''; }
    m.getContainer().classList.toggle('pci-picking', on && Boolean(S.pick));
  }

  /* ---------- panel markup ---------- */
  function buildPanel() {
    const panel = document.createElement('section');
    panel.id = 'pciPanel'; panel.className = 'pci-panel'; panel.hidden = true;
    panel.setAttribute('role', 'region'); panel.setAttribute('aria-labelledby', 'pciTitle');
    panel.innerHTML = `
      <div class="pci-head">
        <div><p class="eyebrow">A→B REPORT · IRC:82-2023 PCI</p><h2 id="pciTitle">Report a stretch</h2><p class="pci-minsum" id="pciMinSum" hidden></p></div>
        <div class="pci-head-actions"><button type="button" class="pci-min pci-expand" id="pciExpand" aria-pressed="false" aria-label="Make the report panel taller">▴ Taller</button><button type="button" class="pci-min" id="pciMinimize" aria-label="Minimise the report panel" aria-expanded="true"></button><button type="button" class="icon-button" id="pciClose" aria-label="Close the condition report">×</button></div>
      </div>
      <div class="pci-tabs" role="tablist" aria-label="Condition report">
        <button type="button" role="tab" id="pciTabNew" aria-controls="pciNew" aria-selected="true">New report</button>
        <button type="button" role="tab" id="pciTabSaved" aria-controls="pciSaved" aria-selected="false" tabindex="-1">Saved reports</button>
      </div>
      <p class="pci-prompt" id="pciPrompt" role="status" aria-live="polite"></p>
      <div class="pci-body" id="pciBody">
        <div id="pciNew" role="tabpanel" aria-labelledby="pciTabNew"></div>
        <div id="pciSaved" role="tabpanel" aria-labelledby="pciTabSaved" hidden></div>
      </div>
      <div class="pci-foot" id="pciFoot"></div>`;
    document.body.append(panel);
    $('pciClose').addEventListener('click', () => close());
    $('pciMinimize').addEventListener('click', () => setMinimized(!S.minimized));
    $('pciExpand').addEventListener('click', () => { S.tall = !S.tall; if (S.tall && S.minimized) setMinimized(false); $('pciExpand').setAttribute('aria-pressed', String(S.tall)); $('pciExpand').innerHTML = S.tall ? '▾ Shorter' : '▴ Taller'; $('pciExpand').setAttribute('aria-label', S.tall ? 'Make the report panel shorter' : 'Make the report panel taller'); layout(); setTimeout(() => { if (S.section !== null) keepSectionVisible(); }, 40); });
    $('pciTabNew').addEventListener('click', () => setTab('new'));
    $('pciTabSaved').addEventListener('click', () => setTab('saved'));
    panel.querySelector('.pci-tabs').addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
      event.preventDefault(); setTab(S.tab === 'new' ? 'saved' : 'new'); $(S.tab === 'new' ? 'pciTabNew' : 'pciTabSaved').focus();
    });
    panel.addEventListener('click', onPanelClick);
    panel.addEventListener('input', onPanelInput);
    panel.addEventListener('change', onPanelInput);
  }
  function buildEntry() {
    const actions = document.querySelector('.dashboard-actions'); if (!actions) return;
    const button = document.createElement('button');
    button.type = 'button'; button.id = 'pciButton'; button.className = 'button pci-action';
    button.setAttribute('aria-controls', 'pciPanel'); button.setAttribute('aria-expanded', 'false');
    button.setAttribute('aria-label', 'Condition report: choose a stretch from A to B');
    button.title = 'Condition report — choose a stretch A→B';
    button.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="6" cy="18" r="2.2"></circle><circle cx="18" cy="6" r="2.2"></circle><path d="M8 17c5-1 3-9 8-10"></path></svg><span class="pci-action-label">Condition report</span>';
    actions.prepend(button);
    // Icon-only: a text label would push the dashboard title onto three lines (measured at 1024–1440 px).
    button.addEventListener('click', () => S.open ? close() : open());
  }

  /* ---------- open / close ---------- */
  function open() {
    ensureLayers(); if (!lmap()) return;
    G.closeAllFlows?.(); G.closeSelection?.(); G.closeStrip?.();
    S.open = true; $('pciPanel').hidden = false; document.body.classList.add('pci-open');
    $('pciButton').setAttribute('aria-expanded', 'true'); $('pciButton').classList.add('active');
    if (!S.points.a && !S.report) S.pick = 'a';
    loadMeta();
    showLayers(true); setMinimized(false); layout(); render(); drawPoints(); drawPath(); drawReport();
    G.syncFloatingGeometry?.();
    requestAnimationFrame(() => $('pciPanel').querySelector('.pci-tabs [aria-selected=true]')?.focus({ preventScroll: true }));
  }
  function close() {
    if (!S.open) return;
    S.open = false; S.pick = null; S.controller?.abort();
    $('pciPanel').hidden = true; document.body.classList.remove('pci-open', 'pci-minimized');
    $('pciButton').setAttribute('aria-expanded', 'false'); $('pciButton').classList.remove('active');
    showLayers(false); clearNotices(); placeClosedNotice(); G.syncFloatingGeometry?.(); lmap()?.invalidateSize();
    $('pciButton').focus({ preventScroll: true });
  }
  function setMinimized(value) {
    S.minimized = Boolean(value);
    document.body.classList.toggle('pci-minimized', S.minimized);
    $('pciMinimize').setAttribute('aria-expanded', String(!S.minimized));
    $('pciMinimize').setAttribute('aria-label', S.minimized ? 'Expand the report panel' : 'Minimise the report panel');
    $('pciMinimize').innerHTML = S.minimized ? '<span aria-hidden="true">▴</span> Show panel' : '<span aria-hidden="true">▾</span> Hide';
    $('pciPanel').classList.toggle('is-min', S.minimized);
    updateMinSummary();
    G.syncFloatingGeometry?.();
    // With the card folded the map has room again: show the whole stretch (or the selected section).
    setTimeout(() => { if (!S.open) return; if (S.section !== null) keepSectionVisible(); else if (S.path || S.report) fitTo(pathFC(pathDoc())); }, 30);
  }
  function updateMinSummary() {
    const el = $('pciMinSum'); if (!el) return;
    const r = S.report, len = lengthOf(r || S.path);
    el.textContent = !S.minimized ? '' : S.pick ? PROMPTS[S.pick] : r ? `${pciLine(r.summary)} · ${fmtM(len)}` : S.path ? `Path ${fmtM(len)} · panel hidden` : 'Panel hidden — pick A on the map';
    el.hidden = !S.minimized;
  }
  const phone = () => innerWidth <= 820;
  // Place the card in the free map area: below the GEOAI bar on desktop, a bottom sheet on phones.
  // The app's two notices (#notice: data connection; #basemapNotice: map style) are placed while the card
  // is open so they never cover it, the toolbar or the dashboard card; inline styles are removed on close.
  const NOTICE_PROPS = ['left', 'right', 'top', 'bottom', 'width', 'max-width', 'transform'];
  function clearNotices() { for (const id of ['notice', 'basemapNotice']) { const el = $(id); if (el) NOTICE_PROPS.forEach(k => el.style.removeProperty(k)); } }
  const shown = el => el && !el.hidden && getComputedStyle(el).display !== 'none';
  function placeNotices() {
    const list = ['notice', 'basemapNotice'].map($).filter(shown);
    if (!list.length) return;
    const panel = $('pciPanel').getBoundingClientRect(), gap = 12;
    const set = (el, st) => { for (const [k, v] of Object.entries(st)) el.style.setProperty(k, v, 'important'); };
    if (!phone()) {
      const tool = document.querySelector('.toolbar')?.getBoundingClientRect(), card = document.querySelector('.dashboard-heading')?.getBoundingClientRect();
      const left = Math.round(panel.right + 16);
      const rightEdge = card && card.left > left + 180 ? card.left - 16 : innerWidth - 24;
      let top = Math.round(Math.max(tool ? tool.bottom : 0, panel.top) + gap);
      for (const el of list) {
        set(el, { left: left + 'px', right: 'auto', top: top + 'px', bottom: 'auto', width: Math.max(160, Math.round(rightEdge - left)) + 'px', 'max-width': 'none', transform: 'none' });
        top = Math.round(el.getBoundingClientRect().bottom + 8);
      }
    } else {
      const headEl = document.querySelector('.dashboard-heading'), heading = headEl && getComputedStyle(headEl).visibility !== 'hidden' ? headEl.getBoundingClientRect() : document.querySelector('.topbar')?.getBoundingClientRect(), tool = document.querySelector('.toolbar')?.getBoundingClientRect();
      let top = Math.round((heading ? heading.bottom : 150) + 8);
      const right = tool ? Math.round(innerWidth - tool.left + 8) : 12;
      for (const el of list) {
        set(el, { left: '12px', right: right + 'px', top: top + 'px', bottom: 'auto', width: 'auto', 'max-width': 'none', transform: 'none' });
        top = Math.round(el.getBoundingClientRect().bottom + 8);
      }
    }
  }
  // Report closed: the app's own data notice (centred at the top) sat over the drive control and the map
  // tools; put it below them, left of the dashboard card. Only #notice, only while it is shown.
  function placeClosedNotice() {
    if (S.open) return;
    clearNotices();
    const el = $('notice');
    if (phone()) {
      // Phones: both notices sat on the same spot over the zoom buttons; stack them above the bottom controls.
      const list = ['notice', 'basemapNotice'].map($).filter(shown); if (!list.length) return;
      const lows = ['.leaflet-control-zoom', '.map-status', '.legend-control', '.map-tools', '#stripPanel', '#detailPanel'].map(q => document.querySelector(q)).filter(e => shown(e) && getComputedStyle(e).visibility !== 'hidden').map(e => e.getBoundingClientRect()).filter(r => r.width && r.top > innerHeight / 2);
      let bottom = Math.round(innerHeight - Math.min(innerHeight - 70, ...lows.map(r => r.top)) + 8);
      for (const n of list) { for (const [k, v] of Object.entries({ left: '12px', right: '12px', top: 'auto', bottom: bottom + 'px', width: 'auto', 'max-width': 'none', transform: 'none' })) n.style.setProperty(k, v, 'important'); bottom = Math.round(innerHeight - n.getBoundingClientRect().top + 8); }
      return;
    }
    if (!el || !shown(el)) return;
    const set = st => { for (const [k, v] of Object.entries(st)) el.style.setProperty(k, v, 'important'); };
    const rects = ['#driveControl', '.toolbar', '.topbar'].map(q => document.querySelector(q)).filter(shown).map(e => e.getBoundingClientRect()).filter(r => r.width);
    const card = document.querySelector('.dashboard-heading')?.getBoundingClientRect();
    if (!phone()) {
      const top = Math.round(Math.max(...rects.map(r => r.bottom)) + 12), left = 24;
      const right = card && card.left > 300 ? card.left - 16 : innerWidth - 24;
      set({ left: left + 'px', right: 'auto', top: top + 'px', bottom: 'auto', transform: 'none', width: Math.round(Math.min(580, right - left)) + 'px', 'max-width': 'none' });
    }
  }
  function layout() {
    const panel = $('pciPanel'); if (!panel || panel.hidden) return;
    const bar = document.querySelector('.topbar')?.getBoundingClientRect();
    const heading = document.querySelector('.dashboard-heading')?.getBoundingClientRect();
    if (!phone()) {
      const left = bar ? bar.left : 24, top = (bar ? bar.bottom : 90) + 16;
      const room = (heading && heading.left > left + 200 ? heading.left : innerWidth) - left - 72;
      const width = Math.max(300, Math.min(410, room));
      Object.assign(panel.style, { left: left + 'px', top: top + 'px', bottom: '24px', width: width + 'px', maxHeight: '' });
      document.body.style.setProperty('--pci-right', Math.round(left + width) + 'px');
      placeNotices();
    } else {
      Object.assign(panel.style, { left: '', top: '', bottom: '', width: '' });
      placeNotices();
      // The sheet starts below the lowest notice (they may cover the map, never the card). Normal: ~60 % of
      // the room below the GEOAI bar; "Taller": up to ~88 % of the screen.
      const bar2 = document.querySelector('.topbar')?.getBoundingClientRect();
      const toolEl = document.querySelector('.toolbar'), tool2 = toolEl && getComputedStyle(toolEl).visibility !== 'hidden' ? toolEl.getBoundingClientRect() : null;
      const above = Math.max(bar2 ? bar2.bottom : 72, tool2 ? tool2.bottom : 0, ...['notice', 'basemapNotice'].map($).filter(shown).map(el => el.getBoundingClientRect().bottom)) + 8;
      const room = innerHeight - above - 16;
      const base = innerHeight - (bar2 ? bar2.bottom : 72) - 8;
      panel.style.maxHeight = Math.max(200, Math.round(S.tall ? Math.min(room, innerHeight * .88) : Math.min(room, base * .6))) + 'px';
    }
  }
  function setPick(which) {
    S.pick = which; S.pathError = '';
    // On a phone the sheet folds away while a point is placed so the road stays visible.
    if (phone()) setMinimized(Boolean(which));
    lmap()?.getContainer().classList.toggle('pci-picking', Boolean(which));
    render();
  }
  function setTab(tab) {
    S.tab = tab;
    $('pciTabNew').setAttribute('aria-selected', String(tab === 'new')); $('pciTabNew').tabIndex = tab === 'new' ? 0 : -1;
    $('pciTabSaved').setAttribute('aria-selected', String(tab === 'saved')); $('pciTabSaved').tabIndex = tab === 'saved' ? 0 : -1;
    $('pciNew').hidden = tab !== 'new'; $('pciSaved').hidden = tab !== 'saved';
    if (tab === 'saved') { S.pick = null; lmap()?.getContainer().classList.remove('pci-picking'); if (!S.savedLoaded) loadSaved(); }
    render();
  }

  /* ---------- picking ---------- */
  const PROMPTS = {
    a: 'Click the start point A on a surveyed road.',
    b: 'Now click the end point B on a surveyed road.',
    via: 'Click a via point on the surveyed road the stretch should follow.' };
  function onMapClick(event) {
    if (!S.open || !S.pick || S.tab !== 'new' || S.view !== 'build') return;
    const p = { lat: event.latlng.lat, lon: event.latlng.lng };
    if (S.pick === 'a') { S.points.a = p; setPick(S.points.b ? null : 'b'); }
    else if (S.pick === 'b') { S.points.b = p; setPick(null); }
    else if (S.pick === 'via') { if (S.points.via.length < 8) S.points.via.push(p); setPick(null); }
    if (!S.pick && phone()) setMinimized(false);
    invalidateReport(); drawPoints(); requestPath();
  }
  function markerIcon(label, kind) {
    return L.divIcon({ className: 'pci-pin-wrap', html: `<span class="pci-pin pci-pin-${kind}">${esc(label)}</span>`, iconSize: [30, 30], iconAnchor: [15, 15] });
  }
  function drawPoints() {
    if (!layers) return;
    layers.marks.clearLayers(); layers.snaps.clearLayers();
    const list = [['a', S.points.a, 'A'], ...S.points.via.map((p, i) => [`via${i}`, p, String(i + 1)]), ['b', S.points.b, 'B']];
    for (const [key, p, label] of list) {
      if (!p) continue;
      const kind = key.startsWith('via') ? 'via' : key;
      const marker = L.marker([p.lat, p.lon], { pane: 'pciMarkers', draggable: S.view === 'build' && !S.reportSource, keyboard: true, autoPan: true,
        icon: markerIcon(label, kind), title: kind === 'via' ? `Via point ${label} (drag to move)` : `Point ${label} (drag to move)`, alt: `Point ${label}` });
      marker.on('dragend', () => {
        const ll = marker.getLatLng(), q = { lat: ll.lat, lon: ll.lng };
        if (key === 'a') S.points.a = q; else if (key === 'b') S.points.b = q; else S.points.via[Number(key.slice(3))] = q;
        invalidateReport(); drawPoints(); requestPath();
      });
      layers.marks.addLayer(marker);
    }
    for (const s of snappedList(S.path)) {
      const lat = num(s.snapped_lat ?? s.lat), lon = num(s.snapped_lon ?? s.lon);
      if (lat === null || lon === null) continue;
      layers.snaps.addLayer(L.circleMarker([lat, lon], { pane: 'pci', renderer: layers.renderer, radius: 5, weight: 2, color: '#fff', fillColor: '#2563eb', fillOpacity: 1, interactive: false }));
    }
  }

  /* ---------- API ---------- */
  function apiError(status, data) {
    const e = data?.error || data?.detail || data;
    let message = typeof e === 'string' ? e : e?.message || (Array.isArray(e) ? e.map(x => x.msg || x.message).filter(Boolean).join('; ') : '');
    if (status === 429) message = 'Too many requests. Wait a minute and try again.';
    if (status === 401 || status === 403) message = message && !/^not authenticated$/i.test(message) ? message : 'The operator key was not accepted.';
    if (status === 503) message = 'The survey database is unavailable. Try again shortly.';
    if (!message) message = `The server could not produce this (HTTP ${status}).`;
    message = message.charAt(0).toUpperCase() + message.slice(1);
    message = message.replace(/\bKm (\d+)\.(\d{3})\b/g, (_, k, m) => `Km ${Number(k)}+${m}`);
    const err = new Error(message); err.status = status; err.code = e?.code; return err;
  }
  async function post(path, body, { signal, headers = {}, raw = false } = {}) {
    let response;
    try {
      response = await fetch(api(path), { method: 'POST', signal, headers: { 'Content-Type': 'application/json', Accept: raw ? '*/*' : 'application/json', ...headers }, body: JSON.stringify(body) });
    } catch (cause) {
      if (cause.name === 'AbortError') throw cause;
      throw Object.assign(new Error('The server could not be reached. Check the connection and retry.'), { status: 0 });
    }
    if (!response.ok) { let data = null; try { data = await response.json(); } catch {} throw apiError(response.status, data); }
    return raw ? response : response.json();
  }
  async function get(path, { raw = false } = {}) {
    let response;
    try { response = await fetch(api(path), { headers: { Accept: raw ? '*/*' : 'application/json' } }); }
    catch { throw Object.assign(new Error('The server could not be reached. Check the connection and retry.'), { status: 0 }); }
    if (!response.ok) { let data = null; try { data = await response.json(); } catch {} throw apiError(response.status, data); }
    return raw ? response : response.json();
  }
  const ptBody = p => ({ lat: Number(p.lat.toFixed(7)), lon: Number(p.lon.toFixed(7)) });
  function pathBody() { return { a: ptBody(S.points.a), b: ptBody(S.points.b), via: S.points.via.map(ptBody) }; }
  let pathTimer, pathVersion = 0;
  function requestPath() {
    clearTimeout(pathTimer); S.path = null; drawPath();
    if (!S.points.a || !S.points.b) { render(); return; }
    S.pathLoading = true; render();
    pathTimer = setTimeout(async () => {
      const version = ++pathVersion; S.controller?.abort(); const controller = new AbortController(); S.controller = controller;
      try {
        const data = await post('/gis/pci/path', pathBody(), { signal: controller.signal });
        if (version !== pathVersion) return;
        S.path = data; S.pathError = '';
        fitTo(pathFC(data));
      } catch (error) {
        if (error.name === 'AbortError' || version !== pathVersion) return;
        S.path = null; S.pathError = error.message;
      } finally { if (version === pathVersion) { S.pathLoading = false; drawPoints(); drawPath(); render(); } }
    }, 220);
  }
  function invalidateReport() { S.report = null; S.reportSource = null; S.savedId = null; S.section = null; S.frame = null; S.error = ''; drawReport(); }

  /* ---------- response accessors (tolerant of the backend's exact field names) ---------- */
  function pathFC(r) {
    const fc = x => x?.type === 'FeatureCollection' && Array.isArray(x.features) ? x : null;
    const c = fc(r) || fc(r?.path) || fc(r?.geojson) || fc(r?.path?.path) || fc(r?.path?.geojson) || fc(r?.path_geojson) || fc(r?.path?.path_geojson);
    return c || { type: 'FeatureCollection', features: [] };
  }
  function snappedList(r) {
    const s = r?.snapped || r?.path?.snapped; if (!s) return [];
    if (Array.isArray(s)) return s;
    return [s.a && { name: 'A', ...s.a }, ...(s.via || []).map((v, i) => ({ name: `Via ${i + 1}`, ...v })), s.b && { name: 'B', ...s.b }].filter(Boolean);
  }
  const isGap = f => { const k = String(f.properties?.kind || f.properties?.type || '').toLowerCase(); return k.includes('gap') || f.properties?.gap === true; };
  const lengthOf = r => num(r?.length_m ?? r?.summary?.length_m ?? r?.path?.length_m);
  const gapsOf = r => r?.gaps || r?.path?.gaps || [];
  const passesOf = r => r?.passes || r?.path?.passes || [];
  const summaryOf = r => r?.summary || {};
  const sectionsOf = r => r?.sections || [];
  // ---- IRC:82-2023 PCI of a unit (100 m section, km row, stretch): unit.pci = {rated, params[], pci{lo, hi, single…}}
  const has2023 = r => Boolean(summaryOf(r).pci || r?.stretch?.pci?.edition);
  const unitPci = u => (u?.pci && typeof u.pci === 'object' && 'rated' in u.pci) ? u.pci : null;
  const sectionRated = s => Boolean(unitPci(s)?.rated && unitPci(s)?.pci);
  const unitParam = (u, id) => (unitPci(u)?.params || []).find(p => p.param === id) || null;
  // A measure = {lo, hi, single} on the 0–100 PCI scale: the PCI itself, or one parameter's PCI_i.
  function pciMeasure(u, by = 'pci') {
    const P = unitPci(u); if (!P?.rated || !P.pci) return null;
    if (by === 'pci') return { lo: num(P.pci.lo), hi: num(P.pci.hi), single: num(P.pci.single) };
    const p = unitParam(u, by); if (!p) return null;
    if (!p.known) return { lo: 0, hi: 100, single: null, unknown: true };
    return { lo: num(p.pci_worst), hi: num(p.pci_best), single: num(p.pci_point) };
  }
  // Colour rule (map, strip, list): a single value → its class; a range inside one class → that class;
  // a range across classes → two colours (worst and best possible class); no data → grey.
  function colourOf(m) {
    if (!m || m.lo === null) return { a: null, b: null };
    if (m.single !== null && m.single !== undefined) return { a: classOf(m.single), b: null };
    const a = classOf(m.lo), b = classOf(m.hi);
    return { a, b: a === b ? null : b };
  }
  const r0 = v => String(Math.round(Number(v)));
  function measureText(m) {
    if (!m || m.lo === null) return '—';
    if (m.single !== null && m.single !== undefined) return r0(m.single);
    return r0(m.lo) === r0(m.hi) ? r0(m.lo) : `${r0(m.lo)}–${r0(m.hi)}`;
  }
  const colourText = c => !c.a ? 'No data' : c.b ? `${c.a}–${c.b}` : c.a;
  function fmtVal(v, unit) {
    const x = num(v); if (x === null) return '—';
    if (unit === 'number') return x >= 10 || Math.abs(x - Math.round(x)) < .05 ? String(Math.round(x)) : x.toFixed(1);
    if (unit === 'm/km') return x.toFixed(2);
    if (unit === 'mm') return Math.abs(x - Math.round(x)) < .05 ? String(Math.round(x)) : x.toFixed(1);
    return x.toFixed(1);
  }
  const unitSuffix = u => u === '%' ? ' %' : u === 'number' ? '' : ` ${u}`;
  function valueText(p) {
    if (num(p.value_lo) === null && num(p.value_hi) === null) return '—';
    const a = fmtVal(p.value_lo ?? p.value_hi, p.unit), b = fmtVal(p.value_hi ?? p.value_lo, p.unit);
    return (a === b ? a : `${a}–${b}`) + unitSuffix(p.unit);
  }
  const SOURCE = { labels: 'Image labels', labels_calibrated: 'Image labels, calibrated', probe: 'Probe, calibrated', roughness: 'Phone IRI, calibrated', entered: 'Entered', not_assessed: 'Not assessed' };
  const WHY = { ravelling: 'probe not yet calibrated on Bengaluru', patching: 'probe not yet calibrated on Bengaluru', rut_depth: 'needs a straight edge or laser profiler', iri: 'phone roughness not yet calibrated against a reference IRI' };
  // "Not assessed: Ravelling, Patching (probe not yet calibrated…) · Rut depth (…)" — one line, grouped by reason.
  function notAssessedLine(params) {
    const groups = new Map();
    for (const p of params.filter(x => !x.known)) {
      const why = p.source !== 'not_assessed' ? 'printed IRC equation not usable for this road class' : (WHY[p.param] || p.evidence || 'not assessed');
      if (!groups.has(why)) groups.set(why, []); groups.get(why).push(p.name || paramDef(p.param).name);
    }
    return [...groups].map(([why, names]) => `${names.join(', ')} (${why})`).join(' · ');
  }
  function pciLine(s) {
    const P = s?.pci; if (!P) return 'Saved before the PCI update — no PCI';
    const m = { lo: num(P.lo), hi: num(P.hi), single: num(P.single) };
    return `PCI ${measureText(m)} · ${colourText(colourOf(m))}`;
  }
  const sectionRuns = s => num(s.quantities?.runs?.potholes ?? s.pothole_runs);
  const sectionCoverage = s => { const c = num(s.quantities?.coverage ?? s.coverage_pct); return c === null ? null : (c <= 1.0001 ? c * 100 : c); };
  const coverageStatus = s => s.quantities?.coverage_status || s.rating?.coverage_status || '';
  function fitTo(fc) {
    const m = lmap(); if (!m || !fc.features.length) return;
    const b = L.geoJSON(fc).getBounds(); if (!b.isValid()) return;
    const area = G.map.availableViewport(), size = m.getSize();
    m.fitBounds(b, { paddingTopLeft: [area.left + 10, area.top + 10], paddingBottomRight: [size.x - area.right + 10, size.y - area.bottom + 10], maxZoom: 18, animate: false });
  }
  const pathDoc = () => (S.report?.path && typeof S.report.path === 'object' ? S.report.path : S.path);
  function drawPath() {
    if (!layers) return;
    layers.path.clearLayers();
    const doc = pathDoc(), fc = pathFC(doc), gaps = gapsOf(doc);
    const dark = document.body.classList.contains('dark');
    let gi = 0;
    for (const f of fc.features) {
      const gap = isGap(f), g = f.geometry; if (!g) continue;
      const lines = g.type === 'MultiLineString' ? g.coordinates : g.type === 'LineString' ? [g.coordinates] : [];
      for (const line of lines) {
        if (!line || line.length < 2) continue;
        const latlngs = line.map(c => [c[1], c[0]]);
        const layer = L.polyline(latlngs, { pane: 'pci', renderer: layers.renderer, interactive: gap, color: gap ? (dark ? '#f4f4f6' : '#1a1a1c') : '#2563eb', weight: gap ? 3 : 6, opacity: S.report ? (gap ? .9 : .28) : .85, dashArray: gap ? '2 8' : null, lineCap: 'round' });
        if (gap) {
          const info = gaps[gi++] || f.properties || {}, d = num(info.straight_m ?? info.length_m);
          layer.bindTooltip(`Gap · ${info.label || `≥ ${d === null ? '?' : Math.round(d)} m (straight line)`} · not surveyed, not rated`, { sticky: true, className: 'survey-tooltip' });
        }
        layers.path.addLayer(layer);
      }
    }
  }
  // Measured pieces of the surveyed path ([lon, lat, m] vertices split at gap steps), for ticks and cuts.
  function linePieces() {
    const doc = pathDoc(), pieces = [];
    if (Array.isArray(doc?.vertices) && doc.vertices.length >= 2) {
      const steps = doc.steps || [];
      let cur = { c: [doc.vertices[0]], m: [doc.vertices[0][2]] };
      for (let i = 1; i < doc.vertices.length; i++) {
        const v = doc.vertices[i];
        if (/gap/.test(String(steps[i - 1] || ''))) { if (cur.c.length >= 2) pieces.push(cur); cur = { c: [v], m: [v[2]] }; }
        else { cur.c.push(v); cur.m.push(v[2]); }
      }
      if (cur.c.length >= 2) pieces.push(cur);
      return pieces;
    }
    for (const f of pathFC(doc).features) {
      if (isGap(f) || f.geometry?.type !== 'LineString') continue;
      const c = f.geometry.coordinates; if (c.length < 2) continue;
      const from = num(f.properties?.from_m) ?? 0, to = num(f.properties?.to_m);
      const cum = [0]; for (let i = 1; i < c.length; i++) cum.push(cum[i - 1] + L.latLng(c[i - 1][1], c[i - 1][0]).distanceTo(L.latLng(c[i][1], c[i][0])));
      const total = cum[cum.length - 1] || 1, scale = to !== null ? (to - from) / total : 1;
      pieces.push({ c, m: cum.map(x => from + x * scale) });
    }
    return pieces;
  }
  function cut(pieces, a, b) {
    const out = [];
    for (const { c, m } of pieces) {
      if (m[m.length - 1] < a || m[0] > b) continue;
      const line = [];
      const at = x => { for (let i = 1; i < m.length; i++) if (x <= m[i]) { const t = m[i] === m[i - 1] ? 0 : (x - m[i - 1]) / (m[i] - m[i - 1]); return [c[i - 1][1] + t * (c[i][1] - c[i - 1][1]), c[i - 1][0] + t * (c[i][0] - c[i - 1][0])]; } return [c[c.length - 1][1], c[c.length - 1][0]]; };
      line.push(at(Math.max(a, m[0])));
      for (let i = 0; i < m.length; i++) if (m[i] > a && m[i] < b) line.push([c[i][1], c[i][0]]);
      line.push(at(Math.min(b, m[m.length - 1])));
      if (line.length >= 2) out.push(line);
    }
    return out;
  }
  function pointAt(pieces, x) {
    for (const { c, m } of pieces) if (x >= m[0] - .01 && x <= m[m.length - 1] + .01) return cut([{ c, m }], x, x)[0]?.[0];
    return null;
  }
  function drawReport() {
    if (!layers) return;
    layers.sections.clearLayers(); layers.ticks.clearLayers(); layers.frame.clearLayers();
    drawPath();
    if (!S.report) return;
    const pieces = linePieces();
    if (has2023(S.report)) sectionsOf(S.report).forEach((s, i) => {
      const m = pciMeasure(s, S.colorBy), col = colourOf(m), selected = S.section === i;
      const coords = s.geometry?.type === 'LineString' ? [s.geometry.coordinates.map(c => [c[1], c[0]])]
        : s.geometry?.type === 'MultiLineString' ? s.geometry.coordinates.map(l => l.map(c => [c[1], c[0]])) : cut(pieces, num(s.from_m), num(s.to_m));
      const tip = `${chain(s.from_m)} – ${chain(s.to_m)} · ${col.a ? `${colorByName()} ${measureText(m)} (${colourText(col)})` : 'Insufficient data'}`;
      for (const line of coords) {
        // casing keeps the light classes (Fair, Satisfactory) readable on any basemap; the selection is a wider casing
        layers.sections.addLayer(L.polyline(line, { pane: 'pci', renderer: layers.renderer, className: selected ? 'pci-casing selected' : 'pci-casing', weight: selected ? 14 : 10, opacity: 1, lineCap: 'butt', interactive: false }));
        const layer = L.polyline(line, { pane: 'pci', renderer: layers.renderer, className: `pci-cls-line cls-${ckey(col.a)}`, weight: selected ? 8 : 7, opacity: 1, dashArray: col.a ? null : '5 6', lineCap: 'butt' });
        layer.bindTooltip(tip, { sticky: true, className: 'survey-tooltip' });
        layer.on('click', event => { L.DomEvent.stop(event); selectSection(i, { fly: false }); });
        layers.sections.addLayer(layer);
        // a range across classes: stripes of the best possible class over the worst
        if (col.b) layers.sections.addLayer(L.polyline(line, { pane: 'pci', renderer: layers.renderer, className: `pci-cls-line cls-${ckey(col.b)}`, weight: selected ? 8 : 7, opacity: 1, dashArray: '7 7', lineCap: 'butt', interactive: false }));
      }
    });
    const total = lengthOf(S.report) || 0, step = total > 3000 ? 500 : 100;
    for (let x = 0; x <= total + .01; x += step) {
      const p = pointAt(pieces, Math.min(x, total)); if (!p) continue;
      const km = Math.abs(x % 1000) < .01;
      const marker = L.circleMarker(p, { pane: 'pci', renderer: layers.renderer, radius: km ? 5 : 3, weight: 1.5, color: '#fff', fillColor: '#1a1a1c', fillOpacity: 1, interactive: false });
      if (km || x === 0) marker.bindTooltip(chain(x), { permanent: true, direction: 'right', offset: [6, 0], className: 'pci-tick-label' });
      layers.ticks.addLayer(marker);
    }
    if (S.frame) {
      const p = S.frame.properties || {};
      if (num(p.centre_lat) !== null) layers.frame.addLayer(L.circleMarker([p.centre_lat, p.centre_lon], { pane: 'pciMarkers', radius: 9, weight: 3, color: '#2563eb', fillColor: '#fff', fillOpacity: .9, interactive: false }));
    }
  }

  /* ---------- rendering the panel ---------- */
  // Describe the focused control so it can be found again after the panel's HTML is replaced.
  function focusKey() {
    const el = document.activeElement, panel = $('pciPanel');
    if (!el || !panel?.contains(el) || el === document.body) return null;
    if (el.id) return { id: el.id };
    const d = el.dataset || {};
    if (d.act) return { act: d.act, index: d.index, id: d.id };
    if (d.actChange) return { change: d.actChange };
    if (el.tagName === 'SUMMARY') { const det = el.parentElement; return det?.dataset?.details ? { details: det.dataset.details } : null; }
    return null;
  }
  function restoreFocus(k) {
    if (!k) return;
    const panel = $('pciPanel'); let el = null;
    if (k.id) el = $(k.id);
    else if (k.act) el = [...panel.querySelectorAll(`[data-act="${k.act}"]`)].find(e => (k.index === undefined || e.dataset.index === k.index) && (k.id === undefined || e.dataset.id === k.id));
    else if (k.change) el = panel.querySelector(`[data-act-change="${k.change}"]`);
    else if (k.details) el = panel.querySelector(`[data-details="${k.details}"] > summary`);
    if (el && el !== document.activeElement && !el.disabled) el.focus({ preventScroll: true });
  }
  function render() {
    if (!$('pciPanel')) return;
    const fk = focusKey();
    layout();
    const prompt = $('pciPrompt');
    let text = '';
    if (S.tab === 'new' && S.view === 'build') {
      text = S.pick ? PROMPTS[S.pick] : S.pathLoading ? 'Finding the surveyed path…' : S.path ? 'Path found. Drag A, B or a via point to adjust, then fill in the details.' : '';
    }
    updateMinSummary();
    prompt.textContent = text; prompt.hidden = !text; prompt.classList.toggle('picking', Boolean(S.pick));
    if (S.tab === 'new') { if (S.view === 'build') renderBuild(); else renderResult(); }
    else renderSaved();
    renderFoot();
    restoreFocus(fk);
  }
  function pointRow(label, p, kind, index) {
    const list = snappedList(S.path), snap = kind === 'a' ? list[0] : kind === 'b' ? list[list.length - 1] : list[index + 1];
    const picking = S.pick === kind && kind !== 'via';
    return `<li class="pci-point ${p ? '' : 'empty'}"><span class="pci-pin pci-pin-${kind} static" aria-hidden="true">${esc(label)}</span>
      <div class="pci-point-text"><strong>${kind === 'via' ? `Via point ${index + 1}` : `Point ${label}`}</strong>
      <span>${p ? `${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}${snap && num(snap.distance_m) !== null ? ` · snapped ${Number(snap.distance_m).toFixed(1)} m` : ''}` : picking ? 'Click the map…' : 'Not set'}</span></div>
      ${kind === 'via' ? `<button type="button" class="pci-mini" data-act="remove-via" data-index="${index}" aria-label="Remove via point ${index + 1}">Remove</button>`
        : `<button type="button" class="pci-mini" data-act="pick-${kind}" aria-pressed="${picking}">${p ? 'Move' : 'Pick'}</button>`}</li>`;
  }
  function field(name, label, input, note = '') { return `<label class="pci-field" for="pci_${name}"><span>${label}${note ? ` <em>${note}</em>` : ''}</span>${input}</label>`; }
  const form = { lane_touched: false, road_name: '', road_number: '', category: 'urban', surface: '', lane_width_m: '3.5', chainage_at_a_km: '', chainage_increases: 'yes', prepared_by: '',
    entered: Object.fromEntries(P23.params.map(p => [p.id, ''])) };
  const surfaceOf = () => { const r = road(form.category); return r.surfaces.includes(form.surface) ? form.surface : r.def; };
  const enteredCount = () => P23.params.filter(p => String(form.entered[p.id]).trim() !== '').length;
  function enteredIssue(p) {
    const v = String(form.entered[p.id]).trim(); if (v === '') return '';
    const x = num(v);
    if (x === null || x < p.lim[0] || x > p.lim[1]) return `${p.name}: enter ${p.lim[0]}–${p.lim[1]}${unitSuffix(p.unit)}, or leave it empty.`;
    return '';
  }
  function measuredGroup() {
    const n = enteredCount();
    const inputs = P23.params.map(p => {
      const issue = enteredIssue(p), id = `pci_e_${p.id}`;
      return `<label class="pci-field pci-mfield" for="${id}"><span>${esc(p.name)} <em>${esc(p.hint)} · ${p.lim[0]}–${p.lim[1]}</em></span>
        <span class="pci-unit-input"><input id="${id}" name="entered.${p.id}" type="number" inputmode="decimal" min="${p.lim[0]}" max="${p.lim[1]}" step="${p.step}" value="${esc(form.entered[p.id])}" placeholder="not entered"${issue ? ` aria-invalid="true" aria-describedby="${id}_err"` : ''}><span aria-hidden="true">${esc(p.unit === 'number' ? 'no.' : p.unit)}</span></span>
        ${issue ? `<span class="pci-field-err" id="${id}_err">${esc(issue)}</span>` : ''}</label>`;
    }).join('');
    return `<details class="pci-calib" data-details="measured" ${S.measuredOpen ?? n ? 'open' : ''}><summary>Values measured on site <em>optional${n ? ` · ${n} entered` : ''}</em></summary>
      <p class="pci-note">For the whole stretch. An entered value replaces the image estimate in every section; empty = not entered. Extents are % of the area of all lanes of the carriageway (IRC:82-2023 eq. 1).</p>
      <div class="pci-grid">${inputs}</div></details>`;
  }
  function renderBuild() {
    const path = S.path, gaps = gapsOf(path);
    const pathInfo = S.pathError ? `<div class="pci-error" role="alert"><strong>No path</strong><span>${esc(S.pathError)}</span></div>`
      : S.pathLoading ? '<p class="pci-note">Finding the surveyed path…</p>'
      : path ? `<div class="pci-pathbox"><div><span>Path length</span><strong>${gaps.length ? '≥ ' : ''}${fmtM(lengthOf(path))}</strong></div><div><span>Surveyed length</span><strong>${fmtM(num(path.surveyed_m) ?? (lengthOf(path) - (num(path.gap_m) || 0)))}</strong></div><div><span>Gaps</span><strong>${gaps.length ? `${gaps.length} · ≥ ${fmtM(gaps.reduce((s, g) => s + (num(g.straight_m ?? g.length_m) || 0), 0))}` : 'None'}</strong></div><div><span>Passes</span><strong>${passesOf(path).length || '—'}</strong></div></div>${gaps.length ? `<p class="pci-note">Dashed lines are gaps between drives, drawn as straight lines (≥ the length shown); they are not surveyed and never rated.</p>` : ''}`
      : '';
    const textInput = (name, ph = '') => `<input id="pci_${name}" name="${name}" maxlength="120" autocomplete="off" value="${esc(form[name])}" placeholder="${esc(ph)}">`;
    $('pciNew').innerHTML = `
      <p class="pci-note pci-filter-note">The report uses every drive; the survey map filters are not applied to it. The survey layer is dimmed while this card is open.</p>
      <section class="pci-section" aria-labelledby="pciStretchH"><h3 id="pciStretchH">1 · Stretch</h3>
        <ol class="pci-points">${pointRow('A', S.points.a, 'a')}${S.points.via.map((p, i) => pointRow(String(i + 1), p, 'via', i)).join('')}${pointRow('B', S.points.b, 'b')}</ol>
        <div class="pci-row-actions"><button type="button" class="button" data-act="add-via" ${S.points.via.length >= 8 || !S.points.a || !S.points.b ? 'disabled' : ''} aria-pressed="${S.pick === 'via'}">${S.pick === 'via' ? 'Click the map…' : '+ Via point'}</button><button type="button" class="text-button" data-act="clear">Clear points</button></div>
        <div id="pciPathInfo" aria-live="polite">${pathInfo}</div>
      </section>
      <section class="pci-section" aria-labelledby="pciDetailsH"><h3 id="pciDetailsH">2 · Details</h3>
        <p class="pci-note">Free text is printed as entered (unverified).</p>
        <div class="pci-grid">
          ${field('road_name', 'Road name', textInput('road_name', 'e.g. 100 Feet Road'), 'as entered')}
          ${field('road_number', 'Road number', textInput('road_number', 'optional'), 'as entered')}
        </div>
        <div class="pci-grid">
          ${field('category', 'Road class', `<select id="pci_category" name="category">${P23.roads.map(c => `<option value="${c.id}" ${form.category === c.id ? 'selected' : ''}>${esc(c.name)}</option>`).join('')}</select>`)}
          ${field('surface', 'Surface', `<select id="pci_surface" name="surface" aria-describedby="pciSurfNote">${road(form.category).surfaces.map(s => `<option value="${s}" ${surfaceOf() === s ? 'selected' : ''}>${esc(s)} · ${esc(P23.surfaceNames[s] || s)}${s === road(form.category).def ? ' (default)' : ''}</option>`).join('')}</select>`)}
        </div>
        <p class="pci-note" id="pciSurfNote">${['mdr', 'rural'].includes(form.category) ? 'The surface selects the roughness (IRI) equation for this road class.' : 'For this road class the surface does not change the PCI (one roughness equation).'}</p>
        ${field('lane_width_m', 'Surveyed lane width (m)', `<input id="pci_lane_width_m" name="lane_width_m" type="number" inputmode="decimal" min="2" max="30" step="0.1" value="${esc(form.lane_width_m)}">`)}
        ${measuredGroup()}
        <details class="pci-calib" data-details="chainage" ${form.chainage_at_a_km ? 'open' : ''}><summary>Road km-stone chainage <em>optional</em></summary>
          <p class="pci-note">Without it: chainage measured by GPS from A (Km 0+000); road km-stone chainage not entered.</p>
          <div class="pci-grid">
            ${field('chainage_at_a_km', 'Km-stone chainage at A (km)', `<input id="pci_chainage_at_a_km" name="chainage_at_a_km" type="number" inputmode="decimal" min="0" max="3000" step="0.001" value="${esc(form.chainage_at_a_km)}" placeholder="e.g. 12.350">`)}
            ${field('chainage_increases', 'Road chainage increases towards B?', `<select id="pci_chainage_increases" name="chainage_increases"><option value="yes" ${form.chainage_increases === 'yes' ? 'selected' : ''}>Yes</option><option value="no" ${form.chainage_increases === 'no' ? 'selected' : ''}>No (decreases)</option></select>`)}
          </div></details>
        ${field('prepared_by', 'Prepared by', textInput('prepared_by', 'name'), 'as entered')}
      </section>
      ${S.error && !P23.params.some(p => enteredIssue(p) === S.error) ? `<div class="pci-error" id="pciGenError" tabindex="-1"><strong>Report not generated</strong><span>${esc(S.error)}</span></div>` : ''}`;
    $('pciNew').querySelector('[data-details=measured]')?.addEventListener('toggle', e => { S.measuredOpen = e.currentTarget.open; });
  }
  function formIssues() {
    const w = num(form.lane_width_m);
    if (w === null || w < 2 || w > 30) return 'Lane width must be between 2 and 30 m.';
    if (form.chainage_at_a_km !== '' && (num(form.chainage_at_a_km) === null || form.chainage_at_a_km < 0 || form.chainage_at_a_km > 3000)) return 'Km-stone chainage must be between 0 and 3000 km.';
    for (const p of P23.params) { const issue = enteredIssue(p); if (issue) return issue; }
    return '';
  }
  function reportBody() {
    const body = { ...pathBody(), road_name: form.road_name.trim() || null, road_number: form.road_number.trim() || null,
      category: form.category, ...(form.lane_touched ? { lane_width_m: num(form.lane_width_m) } : {}),
      prepared_by: form.prepared_by.trim() || null };
    // surface: sent only when it is not the class default (so the report says "entered by requester" only then)
    if (surfaceOf() !== road(form.category).def) body.surface = surfaceOf();
    const entered = {};
    for (const p of P23.params) { const v = String(form.entered[p.id]).trim(); if (v !== '' && num(v) !== null) entered[p.id] = num(v); }
    if (Object.keys(entered).length) body.entered = entered;
    if (form.chainage_at_a_km !== '' && num(form.chainage_at_a_km) !== null) {
      body.chainage_at_a_km = num(form.chainage_at_a_km); body.chainage_increases_towards_b = form.chainage_increases === 'yes';
    }
    return body;
  }
  function renderFoot() {
    const foot = $('pciFoot');
    if (S.tab === 'saved') { foot.innerHTML = '<span class="pci-note">Samples were chosen by a fixed rule before any rating was seen.</span>'; return; }
    if (S.view === 'build') {
      const ready = S.path && !S.pathLoading;
      foot.innerHTML = `${S.error ? `<p class="pci-foot-error" role="alert"><strong>Not generated:</strong> ${esc(S.error)}</p>` : `<span class="pci-note">${ready ? 'Stateless preview; nothing is saved.' : 'Pick A and B to continue.'}</span>`}<button type="button" class="button primary" data-act="generate" ${ready && !S.loading ? '' : 'disabled'}>${S.loading ? 'Generating…' : 'Generate report'}</button>`;
    } else {
      foot.innerHTML = `<button type="button" class="button" data-act="edit">${S.reportSource === 'saved' ? '← New report' : '← Edit stretch'}</button><div class="pci-downloads" role="group" aria-label="Download the report"><button type="button" class="button primary" data-act="dl-pdf">PDF</button><button type="button" class="button" data-act="dl-csv">CSV</button><button type="button" class="button" data-act="dl-json">JSON</button></div>`;
    }
  }
  // ---- what colours the map, the strip and the section list: the PCI, or one parameter's PCI_i (same 0–100 scale)
  const COLOR_BY = [['pci', 'PCI'], ['potholes', 'Potholes (PCI_i)'], ['cracking', 'Cracking (PCI_i)']];
  const colorByName = () => (COLOR_BY.find(c => c[0] === S.colorBy) || COLOR_BY[0])[1];
  const swatches = c => `<i class="pci-sw cls-${ckey(c.a)}" aria-hidden="true"></i>${c.b ? `<i class="pci-sw cls-${ckey(c.b)}" aria-hidden="true"></i>` : ''}`;
  const SKIP = s => num(s.quantities?.skip_m) || 0;
  // ---- evidence behind the pothole band: runs (backend list, else consecutive used pothole frames per pass)
  function attrRows(r) {
    if (!Array.isArray(r?.attribution)) return [];
    const cols = Array.isArray(r.attribution_columns) ? r.attribution_columns : ['segment_id', 'measure_m', 'section', 'km', 'pass', 'used'];
    return r.attribution.map(row => Array.isArray(row) ? Object.fromEntries(cols.map((c, j) => [c, row[j]])) : row);
  }
  function measureOf(id) { const row = attrRows(S.report).find(x => String(x.segment_id) === String(id) && x.used); return row ? num(row.measure_m) : null; }
  function potholeRuns(r) {
    const list = r?.summary?.pothole_runs_list;
    if (Array.isArray(list)) return list.map(x => ({ from_m: num(x.from_m ?? x.measure_m ?? x.start_m) ?? 0, to_m: num(x.to_m ?? x.end_m) ?? num(x.from_m) ?? 0,
      ids: (x.segment_ids || x.segments || (x.segment_id ? [x.segment_id] : [])).map(v => String(v.segment_id || v)), chainage: x.chainage || x.chainage_from || null }));
    const rows = attrRows(r).filter(x => x.used).sort((a, b) => a.measure_m - b.measure_m);
    if (!rows.length || rows.some(x => !S.features.has(String(x.segment_id)))) return null;     // frames not loaded yet
    const runs = []; let cur = null;
    for (const x of rows) {
      const f = S.features.get(String(x.segment_id)), pot = (f.properties.final_labels || []).includes('Potholes');
      if (pot && cur && cur.pass === x.pass && x.measure_m - cur.to_m <= 6) { cur.to_m = x.measure_m; cur.ids.push(String(x.segment_id)); }
      else if (pot) { cur = { pass: x.pass, from_m: x.measure_m, to_m: x.measure_m, ids: [String(x.segment_id)] }; runs.push(cur); }
      else cur = null;
    }
    return runs;
  }
  const kmText = m => `Km ${chain(m)}`;
  function roadKm(m) {
    const inp = S.report?.input || {}, a = num(inp.chainage_at_a_km), inc = inp.chainage_increases_towards_b;
    if (a === null || (inc !== true && inc !== false) || num(m) === null) return '';
    const v = a + (inc ? 1 : -1) * m / 1000; return v < 0 ? '' : `road Km ${chain(v * 1000)}`;
  }
  const cleanRoad = t => String(t || '').replace(/\s*\(\d+(\.\d+)?\)\s*$/, '');
  // 0–100 bar with the six Table 5.5 classes; the outlined box is the PCI range, the black tick the single value.
  function scaleBar(m) {
    const segs = [[0, 20, 'fail'], [20, 40, 'poor'], [40, 60, 'fair'], [60, 80, 'satisfactory'], [80, 90, 'good'], [90, 100, 'excellent']];
    const W = 300, x = v => 1 + Math.max(0, Math.min(100, Number(v))) * (W - 2) / 100;
    const label = m.single !== null ? `PCI ${r0(m.single)}` : `PCI between ${r0(m.lo)} and ${r0(m.hi)}`;
    let svg = `<svg class="pci-scale" viewBox="0 0 ${W} 36" role="img" aria-label="${esc(label)} on the 0–100 scale (classes: Fail up to 20, Poor to 40, Fair to 60, Satisfactory to 80, Good to 90, Excellent above 90)">`;
    for (const [a, b, k] of segs) svg += `<rect x="${x(a)}" y="7" width="${Math.max(0, x(b) - x(a) - 1)}" height="10" class="cls-${k}"/>`;
    if (m.lo !== null && m.hi !== null && (m.single === null || x(m.hi) - x(m.lo) > 2)) svg += `<rect x="${x(m.lo)}" y="3" width="${Math.max(2, x(m.hi) - x(m.lo))}" height="18" rx="2" class="pci-scale-range"/>`;
    if (m.single !== null) svg += `<rect x="${x(m.single) - 1.5}" y="0" width="3" height="24" class="pci-scale-tick"/>`;
    for (const v of [0, 20, 40, 60, 80, 90, 100]) svg += `<text x="${Math.min(W - 1, Math.max(1, x(v)))}" y="34" class="axis ${v === 0 ? 'start' : v === 100 ? 'end' : 'mid'}">${v}</text>`;
    return `${svg}</svg>`;
  }
  function headline(r) {
    const s = summaryOf(r), P = s.pci || unitPci(r.stretch)?.pci, params = unitPci(r.stretch)?.params || [];
    const cov = num(s.coverage), covPct = cov === null ? null : cov <= 1.0001 ? cov * 100 : cov;
    const ge = (s.gaps || gapsOf(r).length) ? '≥ ' : '';
    const kpis = `<div class="pci-kpis">
        <div><span>Length A→B</span><strong>${ge}${fmtM(lengthOf(r))}</strong><small>${esc(s.chainage_a || 'Km 0+000')} → ${ge}${esc(s.chainage_b || '')}</small></div>
        <div><span>Coverage</span><strong>${pct(covPct)}</strong><small>${esc(String(s.coverage_status || '').replace(/_/g, ' '))}</small></div>
        <div><span>Gaps</span><strong>${num(s.gaps) ? `${s.gaps}` : 'None'}</strong>${num(s.gaps) ? `<small>≥ ${fmtM(num(s.gap_m))} straight line, not rated</small>` : ''}</div>
        <div><span>Expert-reviewed</span><strong>${num(s.segments_expert_reviewed) ?? '—'}<small> of ${num(s.segments_used) ?? '—'}</small></strong><small>segments used</small></div>
      </div>`;
    const callout = `<p class="pci-callout"><strong>Screening estimate</strong> from images of one lane and the values entered. Confirm by field inspection before deciding on maintenance.</p>`;
    if (!P || num(P.lo) === null) return `<div class="pci-headline"><div class="pci-error" role="note"><strong>No PCI</strong><span>No 100 m section has enough imaged length to be rated.</span></div>${kpis}${callout}</div>`;
    const m = { lo: num(P.lo), hi: num(P.hi), single: num(P.single) }, col = colourOf(m);
    const cls = m.single !== null ? (P.single_class || classOf(m.single)) : null, rec = cls ? (P.recommendation || P23.rec[cls]) : null;
    // why there is no single PCI: parameters without a PCI_i, and/or the backend's note (e.g. uncalibrated label scale)
    const why = String(P.single_note || unitPci(r.stretch)?.pci?.single_note || '').replace(/^no single PCI:\s*/i, '');
    const missing = (P.missing || []).map(n => P23.params.find(p => p.name === n) || { name: n, w: 0 });
    const missW = Math.round(missing.reduce((t, p) => t + (p.w || 0), 0) * 100);
    const listNames = a => a.length <= 1 ? a.join('') : `${a.slice(0, -1).join(', ')} and ${a[a.length - 1]}`;
    const nU = num(P.units), nR = num(P.units_rated);
    const basis = nU ? `Stretch value = length-weighted mean of ${nR !== null && nR !== nU ? `${nR} rated of ${nU}` : nU} km row${nU === 1 ? '' : 's'}` : '';
    const skipNote = m.single !== null && r0(m.lo) !== r0(m.hi) ? `${r0(m.lo)}–${r0(m.hi)} with the gate-skipped frames at their audited range` : '';
    const rows = params.map(p => {
      const d = paramDef(p.param), w = num(p.weight) ?? d.w;
      if (!p.known) return `<tr class="na"><th scope="row">${esc(p.name || d.name)}</th><td>${p.value_lo !== undefined ? esc(valueText(p)) : '—'}</td><td>0–100</td><td>${w.toFixed(2)}</td><td>${p.source === 'not_assessed' ? 'Not assessed' : 'Not computable'}</td></tr>`;
      const pm = { lo: num(p.pci_worst), hi: num(p.pci_best), single: null };
      return `<tr><th scope="row">${esc(p.name || d.name)}</th><td>${esc(valueText(p))}</td><td>${esc(measureText(pm))}</td><td>${w.toFixed(2)}</td><td>${esc(SOURCE[p.source] || p.source_name || p.source || '—')}</td></tr>`;
    }).join('');
    const na = notAssessedLine(params);
    const usesLabels = params.some(p => p.known && /^labels/.test(p.source));
    const runs = potholeRuns(r), h = s.headline || {};
    const rpk = num(h.pothole_runs_per_km) ?? (runs && lengthOf(r) ? runs.length / (lengthOf(r) / 1000) : null);
    const potLabels = params.some(p => p.param === 'potholes' && /^labels/.test(p.source));
    const flags = potLabels && runs && runs.length ? `<details class="pci-flags" data-details="flags" ${S.flagsOpen ? 'open' : ''}><summary>Pothole flags · ${runs.length} run${runs.length === 1 ? '' : 's'}${rpk !== null ? ` (${rpk.toFixed(1)} per km)` : ''}</summary>
        <p class="pci-note">Image labels, not validated; shadows are often labelled as potholes. Open a frame and check it before acting.</p>
        <div class="pci-runs">${runs.slice(0, 24).map((x, i) => `<button type="button" class="pci-run" data-act="run" data-index="${i}" data-id="${esc(x.ids[0])}">${esc(x.chainage || kmText(x.from_m))}${x.ids.length > 1 ? ` · ${x.ids.length} frames` : ''}</button>`).join('')}${runs.length > 24 ? `<span class="pci-note">+ ${runs.length - 24} more (see sections)</span>` : ''}</div></details>`
      : potLabels && runs === null && sectionsOf(r).some(x => sectionRuns(x)) ? '<p class="pci-note">Loading the pothole frames…</p>' : '';
    return `<div class="pci-headline">
      <div class="pci-score">
        <div class="pci-score-top"><div><span class="pci-score-label">PCI · IRC:82-2023 (0–100)</span><strong class="pci-score-num">${esc(measureText(m))}</strong></div>
          <span class="pci-chip">${swatches(col)}${esc(cls || colourText(col))}</span></div>
        ${scaleBar(m)}
        ${rec ? `<p class="pci-rec"><span>Maintenance (Table 5.5)</span><strong>${esc(rec)}</strong></p>`
          : `<p class="pci-rec pending"><span>Not yet a single PCI</span>${missing.length ? `${esc(listNames(missing.map(p => p.name)))} (${missW} % of the weight) ${missing.length === 1 ? 'has' : 'have'} no PCI<sub>i</sub> yet, so each is taken as 0 and as 100. ` : ''}${why ? `${esc(why.charAt(0).toUpperCase() + why.slice(1))}${/[.]$/.test(why) ? '' : '.'} ` : ''}A maintenance recommendation needs a single PCI.</p>`}
        ${basis || skipNote ? `<p class="pci-note pci-basis">${esc([basis, skipNote].filter(Boolean).join(' · '))}.</p>` : ''}
      </div>
      <table class="pci-params pci-p23"><colgroup><col class="c1"><col class="c2"><col class="c3"><col class="c4"><col class="c5"></colgroup>
        <caption>Whole stretch · PCI = Σ weight × PCI<sub>i</sub></caption>
        <thead><tr><th scope="col">Parameter</th><th scope="col">Value</th><th scope="col">PCI<sub>i</sub></th><th scope="col">Weight</th><th scope="col">Source</th></tr></thead><tbody>${rows}</tbody></table>
      ${na ? `<p class="pci-na-line"><strong>Not assessed:</strong> ${esc(na)}.</p>` : ''}
      ${usesLabels ? `<p class="pci-note">Image labels: cracking = % of surveyed length (upper bound of % area); potholes = runs of flagged 5 m segments. Not yet validated on Bengaluru (${num(s.segments_expert_reviewed) ?? 0} expert-reviewed segments).</p>` : ''}
      ${flags}
      ${kpis}
      ${callout}
    </div>`;
  }
  function stripChart(r) {
    const secs = sectionsOf(r); if (!secs.length) return '';
    const total = lengthOf(r) || num(secs[secs.length - 1].to_m) || 1;
    const W = 360, left = 30, right = 8, plot = W - left - right, x = m => left + plot * (num(m) || 0) / total;
    const top = 6, H = 80, y = v => top + H - H * Math.max(0, Math.min(100, v)) / 100, cellY = top + H + 6;
    let svg = `<svg class="pci-strip" viewBox="0 0 ${W} ${cellY + 34}" role="img" aria-labelledby="pciStripT pciStripD"><title id="pciStripT">${esc(colorByName())} along the stretch, per 100 m</title><desc id="pciStripD">For each 100 m section from A to B: grey bar = possible range, coloured line = single value when known, on the 0–100 scale with lines at the class limits 20, 40, 60, 80 and 90. The row below has the map colours; select a cell to open the section.</desc>`;
    for (const v of [20, 40, 60, 80, 90]) svg += `<line x1="${left}" x2="${W - right}" y1="${y(v)}" y2="${y(v)}" class="grid"/>`;
    svg += `<line x1="${left}" x2="${W - right}" y1="${y(0)}" y2="${y(0)}" class="base"/>`;
    for (const v of [0, 20, 40, 60, 80, 100]) svg += `<text x="${left - 4}" y="${y(v) + 3.5}" class="axis end">${v}</text>`;
    secs.forEach((s, i) => {
      const x0 = x(s.from_m), w = Math.max(1, x(s.to_m) - x0 - .6), m = pciMeasure(s, S.colorBy), col = colourOf(m), sel = S.section === i;
      if (m && m.lo !== null) {
        svg += `<rect x="${x0}" y="${y(m.hi)}" width="${w}" height="${Math.max(1.5, y(m.lo) - y(m.hi))}" class="rng"/>`;
        if (m.single !== null) svg += `<rect x="${x0}" y="${y(m.single) - 1.5}" width="${w}" height="3" class="pt cls-${ckey(col.a)}"/>`;
      }
      const tip = `${esc(s.chainage_from || chain(s.from_m))} – ${esc(s.chainage_to || chain(s.to_m))}: ${col.a ? `${esc(measureText(m))} (${esc(colourText(col))})` : 'Insufficient data'}`;
      svg += `<g class="bandcell${sel ? ' sel' : ''}" data-section="${i}"><title>${tip}</title>${col.b
        ? `<rect x="${x0}" y="${cellY}" width="${w}" height="7" class="cls-${ckey(col.a)}"/><rect x="${x0}" y="${cellY + 7}" width="${w}" height="7" class="cls-${ckey(col.b)}"/>`
        : `<rect x="${x0}" y="${cellY}" width="${w}" height="14" class="cls-${ckey(col.a)}"/>`}</g>`;
      if (sel) svg += `<rect x="${x0 - 1}" y="${top - 2}" width="${w + 2}" height="${cellY + 16 - top}" class="selbox"/>`;
    });
    const step = total > 5000 ? 1000 : total > 1500 ? 500 : total > 600 ? 200 : 100;
    for (let m = 0; m <= total + .01; m += step) { if (m > 0 && total - m < step * .45) continue; svg += `<line x1="${x(m)}" x2="${x(m)}" y1="${cellY + 15}" y2="${cellY + 19}" class="base"/><text x="${x(m)}" y="${cellY + 30}" class="axis mid">${chain(m)}</text>`; }
    return `${svg}</svg>`;
  }
  // Legend: the six Table 5.5 classes, grey if a section has no data, and the two-colour rule when it is used.
  function legendHTML() {
    const secs = sectionsOf(S.report); let two = false, none = false;
    for (const s of secs) { const c = colourOf(pciMeasure(s, S.colorBy)); if (c.b) two = true; if (!c.a) none = true; }
    const cls = [['fail', 'Fail ≤ 20'], ['poor', 'Poor'], ['fair', 'Fair'], ['satisfactory', 'Satisfactory'], ['good', 'Good'], ['excellent', 'Excellent > 90']];
    return `<div class="pci-legend" role="group" aria-label="Map legend">${cls.map(([k, t]) => `<span><i class="cls-${k}"></i>${t}</span>`).join('')}${none ? '<span><i class="cls-none"></i>No data (dashed)</span>' : ''}
      ${two ? '<p class="pci-legend-rule"><i class="cls-poor"></i><i class="cls-excellent"></i>Two colours (striped on the map) = still a range: worst and best possible class.</p>' : ''}</div>`;
  }
  function sectionFacts(s) {
    if (!sectionRated(s)) return 'Insufficient data';
    const c = unitParam(s, 'cracking'), p = unitParam(s, 'potholes'), parts = [];
    if (c?.known) parts.push(`Cracking ${valueText(c)}`);
    if (p && p.value_lo !== undefined) parts.push(`Potholes ${valueText(p)}`);
    return parts.join(' · ');
  }
  function sectionsList(r) {
    const secs = sectionsOf(r);
    return `<ul class="pci-sections">${secs.map((s, i) => {
      const m = pciMeasure(s, S.colorBy), col = colourOf(m), cov = sectionCoverage(s), st = coverageStatus(s), skip = SKIP(s);
      const open = S.section === i, facts = sectionFacts(s);
      return `<li class="${open ? 'open' : ''}"><button type="button" class="pci-sec" data-act="section" data-index="${i}" aria-expanded="${open}">
        <span class="pci-sec-band" aria-hidden="true"><i class="cls-${ckey(col.a)}"></i>${col.b ? `<i class="cls-${ckey(col.b)}"></i>` : ''}</span>
        <span class="pci-sec-main"><strong>${esc(s.chainage_from || chain(s.from_m))} – ${esc(s.chainage_to || chain(s.to_m))}</strong><span>${[col.a ? `${S.colorBy === 'pci' ? '' : `${paramDef(S.colorBy).name}: `}${colourText(col)}` : '', facts, `coverage ${pct(cov)}${st === 'low_coverage' ? ' (low)' : ''}`, skip ? `${fmtM(skip)} skipped` : '', s.road_chainage_from?.text ? `road ${cleanRoad(s.road_chainage_from.text)}` : ''].filter(Boolean).map(esc).join(' · ')}</span></span>
        <span class="pci-chip sm"><span class="pci-vh">${esc(colorByName())} </span>${swatches(col)}${esc(col.a ? measureText(m) : 'n/a')}</span></button>
        ${open ? `<div class="pci-frames" id="pciFrames">${sectionDetail(s)}${framesHTML(s, i)}</div>` : ''}</li>`;
    }).join('')}</ul>`;
  }
  function sectionDetail(s) {
    const parts = [], P = unitPci(s);
    if (P?.pci) {
      const m = { lo: num(P.pci.lo), hi: num(P.pci.hi), single: num(P.pci.single) }, cls = m.single !== null ? (P.pci.single_class || classOf(m.single)) : null;
      const known = (P.params || []).filter(p => p.known).map(p => `${p.name} ${measureText({ lo: num(p.pci_worst), hi: num(p.pci_best), single: null })}`);
      parts.push(`<p class="pci-note"><strong>PCI ${esc(measureText(m))}</strong> (${esc(cls || colourText(colourOf(m)))})${cls ? ` · ${esc(P.pci.recommendation || P23.rec[cls])}` : ''}${known.length ? ` · PCI<sub>i</sub>: ${esc(known.join(', '))}` : ''}.</p>`);
    }
    const also = (s.also_surveyed || []).map(p => p.date_ist).filter(Boolean), other = (s.other_direction || []).map(p => p.date_ist).filter(Boolean);
    if (s.chosen_pass?.date_ist) parts.push(`<p class="pci-note">Rated from the pass of ${esc(s.chosen_pass.date_ist)}${also.length ? `; also surveyed on ${esc(also.join(', '))} (not averaged)` : ''}${other.length ? `; other direction on ${esc(other.join(', '))} (not rated)` : ''}.</p>`);
    return parts.join('');
  }
  // Segments of each 100 m section: the report lists the segments it used; each goes to the section
  // whose geometry is nearest to its centre (sections carry geometry, not segment lists).
  function sectionSegments(r) {
    if (r._secSegs) return r._secSegs;
    const ids = (r.segment_ids || []).map(String), secs = sectionsOf(r);
    if (secs.some(s => Array.isArray(s.segment_ids))) { r._secSegs = secs.map(s => (s.segment_ids || []).map(String)); return r._secSegs; }
    // attribution rows [segment_id, measure_m, section, km, pass, used] (array or object form): used rows per section
    if (Array.isArray(r.attribution) && r.attribution.length) {
      const out = secs.map(() => []);
      for (const row of r.attribution) {
        const cols = Array.isArray(r.attribution_columns) ? r.attribution_columns : ['segment_id', 'measure_m', 'section', 'km', 'pass', 'used'];
        const o = Array.isArray(row) ? Object.fromEntries(cols.map((c, j) => [c, row[j]])) : row;
        if (o.used && Number.isInteger(Number(o.section)) && out[Number(o.section)]) out[Number(o.section)].push({ id: String(o.segment_id), m: num(o.measure_m) ?? 0 });
      }
      const lists = out.map(list => list.sort((a, b) => a.m - b.m).map(x => x.id));
      if (lists.flat().some(id => !S.features.has(id))) return null;      // frames still loading
      r._secSegs = lists; return lists;
    }
    if (!ids.length || ids.some(id => !S.features.has(id))) return null;
    const lat0 = num(S.features.get(ids[0]).properties.centre_lat) || 12.97, kx = 111320 * Math.cos(lat0 * Math.PI / 180), ky = 110540;
    const P = c => [c[0] * kx, c[1] * ky];
    const lines = secs.map(s => { const g = s.geometry; const ls = g?.type === 'MultiLineString' ? g.coordinates : g?.type === 'LineString' ? [g.coordinates] : []; return ls.map(l => l.map(P)); });
    const segDist = (p, a, b) => { const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy; const t = L2 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2)) : 0; return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy); };
    const out = secs.map(() => []);
    for (const id of ids) {
      const f = S.features.get(id), p = P([f.properties.centre_lon, f.properties.centre_lat]);
      let best = -1, bd = Infinity;
      lines.forEach((ls, i) => { for (const l of ls) for (let j = 1; j < l.length; j++) { const d = segDist(p, l[j - 1], l[j]); if (d < bd) { bd = d; best = i; } } });
      if (best >= 0 && bd < 60) out[best].push(id);
    }
    r._secSegs = out; return out;
  }
  function framesHTML(s, i) {
    const all = sectionSegments(S.report);
    if (!all) return S.featuresError ? `<p class="pci-note">Frames unavailable: ${esc(S.featuresError)}</p>` : '<p class="pci-note">Loading frames…</p>';
    const annex = new Set((S.report.annex || []).map(a => String(a.segment_id)));
    const feats = (all[i] || []).map(id => S.features.get(id)).filter(Boolean);
    if (!feats.length) return '<p class="pci-note">No surveyed segment in this section (gap or not covered).</p>';
    const withImg = feats.filter(f => f.properties.has_image).sort((a, b) => (annex.has(D.featureId(b)) - annex.has(D.featureId(a))) || (b.properties.vlm_class === 'distress') - (a.properties.vlm_class === 'distress'));
    const limit = S.allFrames === i ? withImg.length : 12, shown = withImg.slice(0, limit);
    let html = `<p class="pci-note">${feats.length} segments · ${withImg.length} with a frame${annex.size ? ' · ★ = in the PDF photo annex' : ''}. A frame shows the road up to ~30 m <em>ahead</em> of its marked position.</p><div class="pci-thumbs">`;
    html += shown.map(f => { const p = f.properties, id = D.featureId(f), labels = p.final_labels?.length ? p.final_labels.join(', ') : className(p.vlm_class);
      const selected = S.frame && D.featureId(S.frame) === id;
      const m = measureOf(id), km = m === null ? '' : kmText(m);
      return `<button type="button" class="pci-thumb ${selected ? 'active' : ''}" data-act="frame" data-id="${esc(id)}" aria-pressed="${selected}" aria-label="Frame at ${esc(km || 'unknown chainage')}: ${esc(labels)}"><img alt="" loading="lazy" decoding="async" src="${esc(G.mediaURL(p.thumb_url))}"><span>${annex.has(id) ? '★ ' : ''}${esc(km ? chain(m) : '')}<br>${esc(p.final_primary_label || className(p.vlm_class))}</span></button>`; }).join('');
    html += '</div>';
    if (withImg.length > limit) html += `<button type="button" class="text-button pci-more-frames" data-act="all-frames" data-index="${i}">Show all ${withImg.length} frames</button>`;
    if (S.frame && (all[i] || []).includes(D.featureId(S.frame))) html += frameDetail(S.frame);
    return html;
  }
  function className(v) { return ({ distress: 'Distress', normal: 'Normal', not_sent: 'Not sent (skip)', not_covered: 'Not covered', pending: 'Pending', unknown: 'Unknown' })[v] || 'Unknown'; }
  function frameDetail(f) {
    const p = f.properties, conf = num(p.confidence);
    const review = p.expert_reviewed ? 'Expert-reviewed' : p.needs_expert_review ? 'Awaiting expert review' : 'Model result · not expert-reviewed';
    const m = measureOf(D.featureId(f)), km = m === null ? 'chainage unknown' : kmText(m), rk = m === null ? '' : roadKm(m);
    return `<figure class="pci-frame" id="pciFrameFig" tabindex="-1"><img alt="Road frame at ${esc(km)}: ${esc((p.final_labels || []).join(', ') || className(p.vlm_class))}" decoding="async" src="${esc(G.mediaURL(p.preview_url))}"><figcaption>
      <strong>${esc(p.final_primary_label || className(p.vlm_class))} · ${esc(km)}${rk ? ` (${esc(rk)})` : ''}</strong>
      <span>${esc((p.final_labels || []).join(', ') || className(p.vlm_class))}${p.vlm_class === 'distress' ? ` · ${esc(p.final_severity || '—')} severity` : ''}</span>
      <span>${esc(D.localTime(p.local_time || p.utc))} · ${Number(p.centre_lat).toFixed(5)}, ${Number(p.centre_lon).toFixed(5)}</span>
      <span>Confidence ${conf === null ? 'n/a' : `${(conf * 100).toFixed(0)} %`} · ${esc(review)}</span>
      <span>The frame shows the road ahead of this position (up to ~30 m).</span>
      ${p.image_url ? `<a href="${esc(G.mediaURL(p.image_url))}" target="_blank" rel="noopener noreferrer" class="text-button">Full resolution ↗</a>` : ''}
    </figcaption></figure>`;
  }
  function headerBlock(r) {
    const h = r.header || {}, rc = h.road_chainage_from?.text ? `${h.road_chainage_from.text} → ${h.road_chainage_to?.text || ''} · calibrated by requester at A (unverified)` : 'Chainage measured by GPS from A; road km-stone chainage not entered';
    const ent = h.entered && typeof h.entered === 'object' ? Object.entries(h.entered) : [];
    const cal = h.calibration && typeof h.calibration === 'object' ? `fitted on ${h.calibration.fitted_on || '—'} (${Object.keys(h.calibration.params || {}).map(p => paramDef(p).name).join(', ') || 'no parameter'})` : 'None (image values not calibrated against a survey)';
    const items = [
      ['Road name / number', [h.road_name, h.road_number].filter(Boolean).join(' · ') || 'Not entered', Boolean(h.road_name || h.road_number)],
      ['Road class', h.category_name || road(h.category).name || '—'],
      ['Surface', h.surface ? `${h.surface_name || P23.surfaceNames[h.surface] || ''} (${h.surface}) · ${h.surface_note || ''}` : '—'],
      ['Values entered on site', ent.length ? ent.map(([k, v]) => `${paramDef(k).name} ${fmtVal(v, paramDef(k).unit)}${unitSuffix(paramDef(k).unit)}`).join(', ') : 'None', ent.length > 0],
      ['Survey calibration', cal],
      ['Chainage', `${h.km_from || 'Km 0+000'} (A) → ${h.km_to || ''} (B), measured by GPS along the drives`],
      ['Road km-stone chainage', rc],
      ['Survey dates (IST)', (h.survey_dates_ist || []).join(', ') || '—'],
      ['Season', (h.season || []).join('; ') || '—'],
      ['Weather (survey form, as entered)', (h.weather_as_entered_on_survey_form || []).join(', ') || '—'],
      ['Lane', `${h.lane_imaged || 'one lane'} · width ${num(h.lane_width_m) ?? '—'} m${h.lane_width_note ? ` · ${h.lane_width_note}` : ''}`],
      ['Direction', (h.directions || []).map(d => d.direction).filter(Boolean).join('; ') || '—'],
      ['Method', [h.pci_name, h.method].filter(Boolean).join('. ') || '—'],
      ['Device · model', [(h.devices || []).join(', '), h.model?.pipeline, h.model?.model, h.model?.prompts].filter(Boolean).join(' · ') || '—'],
      ['Prepared by', h.prepared_by || 'Not entered', Boolean(h.prepared_by)],
      ['Report', `${r.report_id ?? r.id ?? 'preview'} · generated ${r.generated_at_ist || '—'} · data as of ${r.data_as_of ? D.localTime(r.data_as_of) : '—'}`]];
    return `<dl class="pci-header">${items.map(([k, v, free]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}${free ? ' <em>entered by requester, unverified</em>' : ''}</dd></div>`).join('')}</dl>
      ${r.accuracy ? `<p class="pci-note"><strong>Accuracy.</strong> ${esc(Array.isArray(r.accuracy) ? r.accuracy.join(' ') : typeof r.accuracy === 'string' ? r.accuracy : Object.values(r.accuracy).join(' '))}</p>` : ''}
      ${Array.isArray(r.limitations) && r.limitations.length ? `<h4 class="pci-h4">Limitations</h4><ul class="pci-list">${r.limitations.map(x => `<li>${esc(typeof x === 'string' ? x : x.text || JSON.stringify(x))}</li>`).join('')}</ul>` : ''}`;
  }
  function renderResult() {
    const r = S.report; if (!r) { S.view = 'build'; renderBuild(); return; }
    const stale = r.stale || r.is_stale;
    const h = r.header || {}, title = r.title || [h.road_name, h.road_number].filter(Boolean).join(' · ') || `${summaryOf(r).chainage_a || 'Km 0+000'} to ${summaryOf(r).chainage_b || ''}`;
    $('pciNew').innerHTML = `
      <div class="pci-result-head"><p class="eyebrow">${S.reportSource === 'saved' ? (r.sample ? 'SAMPLE REPORT' : 'SAVED REPORT') : S.reportSource === 'saved-now' ? 'SAVED REPORT' : 'PREVIEW · NOT SAVED'}${r.report_id ?? r.id ? ` · NO. ${esc(String(r.report_id ?? r.id))}` : ''}</p>
        <h3>${esc(title)}</h3>
        ${r.sample ? '<span class="pci-tag sample">Sample</span>' : ''}${stale ? `<span class="pci-tag stale">Stale${num(r.changed_segments) ? ` · ${r.changed_segments} segments changed` : ''}</span>` : ''}${r.mock ? '<span class="pci-tag">Mock API</span>' : ''}</div>
      ${stale ? `<p class="pci-note pci-stale-note">${esc(r.stale_note || 'Labels or reviews of segments in this report changed after it was generated. The stored report is a frozen snapshot.')}</p>` : ''}
      ${r.sample && r.sample_rule ? `<p class="pci-note">Sample chosen by a rule fixed before any rating was seen: ${esc(ruleText(r.sample_rule, lengthOf(r)))}. Regenerated, never edited.</p>` : ''}
      ${has2023(r) ? `${headline(r)}
      <section class="pci-section" aria-labelledby="pciStripH"><div class="pci-section-head"><h3 id="pciStripH">Along the stretch · 100 m</h3>
        <label class="pci-colorby">Colour by <select data-act-change="color-by" aria-label="Colour the map and sections by">${COLOR_BY.map(([v, t]) => `<option value="${v}" ${S.colorBy === v ? 'selected' : ''}>${esc(t)}</option>`).join('')}</select></label></div>
        ${stripChart(r)}${legendHTML()}</section>
      <section class="pci-section" aria-labelledby="pciSecH"><h3 id="pciSecH">Sections</h3><p class="pci-note">Select a section to highlight it on the map and see its frames.</p>${sectionsList(r)}</section>`
      : `<div class="pci-error" role="note"><strong>No PCI in this saved report</strong><span>It was saved before the IRC:82-2023 PCI was added. Generate it again from the same points to get the PCI; the saved copy stays as it is.</span></div>
        <button type="button" class="button primary" data-act="regenerate" ${r.input?.a && r.input?.b ? '' : 'disabled'}>Generate again with the PCI</button>`}
      <details class="pci-section pci-more" data-details="header" ${S.headerOpen ? 'open' : ''}><summary>Report header, accuracy and limitations</summary>${headerBlock(r)}</details>
      ${S.reportSource === 'saved' ? '' : saveBlock()}`;
    $('pciNew').querySelector('[data-details=header]')?.addEventListener('toggle', e => { S.headerOpen = e.currentTarget.open; });
    $('pciNew').querySelector('[data-details=flags]')?.addEventListener('toggle', e => { S.flagsOpen = e.currentTarget.open; });
  }
  function saveBlock() {
    let key = S.keyDraft || ''; try { key = key || sessionStorage.getItem(KEY_STORE) || ''; } catch {}
    return `<section class="pci-section pci-save" aria-labelledby="pciSaveH"><h3 id="pciSaveH">Save to the portal</h3>
      ${S.savedId ? `<p class="pci-ok" role="status">${S.savedExisting ? `This stretch with these details is already saved as report ${esc(String(S.savedId))} (same input and data).` : `Saved as report ${esc(String(S.savedId))}.`} It is listed under Saved reports.</p>` : `
      <label class="pci-field" for="pciKey"><span>Operator key (to save to the portal)</span><input id="pciKey" type="password" autocomplete="off" spellcheck="false" value="${esc(key)}" aria-describedby="pciKeyNote"></label>
      <p class="pci-note" id="pciKeyNote">Kept only in this browser tab (session storage). Downloads do not need a key.</p>
      ${S.saveError ? `<div class="pci-error" role="alert"><strong>Not saved</strong><span>${esc(S.saveError)}</span></div>` : ''}
      <button type="button" class="button" data-act="save" ${S.saving ? 'disabled' : ''}>${S.saving ? 'Saving…' : 'Save report'}</button>`}</section>`;
  }
  // Sample rule labels from scripts/pci_samples.py (rule fixed before any rating was seen).
  function ruleText(code, routed) {
    const c = String(code || ''); let m;
    if (c === 'longest_drive') return 'the longest drive, end to end';
    if ((m = /^random_1km_(\d{4}-\d{2}-\d{2})_(\d+)$/.exec(c))) return `random start on the ${m[1]} trip, B = first segment ≥ 1 km further along the trip (draw ${m[2]})${num(routed) !== null ? `; routed length ${fmtM(Number(routed))}` : ''}`;
    if (c === 'two_gaps') return 'a stretch across two gaps between time-adjacent drives';
    return c.replace(/_/g, ' ');
  }
  function renderSaved() {
    const list = S.saved;
    $('pciSaved').innerHTML = S.savedError ? `<div class="pci-error" role="alert"><strong>Saved reports unavailable</strong><span>${esc(S.savedError)}</span></div><button type="button" class="button" data-act="reload-saved">Retry</button>`
      : !S.savedLoaded ? '<p class="pci-note">Loading saved reports…</p>'
      : !list.length ? '<p class="pci-note">No saved reports yet.</p>'
      : `<ul class="pci-saved">${list.map(x => `<li><button type="button" class="pci-saved-item" data-act="open-saved" data-id="${esc(x.id ?? x.report_id)}">
          <span class="pci-saved-title"><strong>${esc(x.title || 'Condition report')}</strong>${x.sample ? '<span class="pci-tag sample">Sample</span>' : ''}${x.stale ? '<span class="pci-tag stale">Stale</span>' : ''}</span>
          <span class="pci-saved-bands">${x.summary?.pci ? `${swatches(colourOf({ lo: num(x.summary.pci.lo), hi: num(x.summary.pci.hi), single: num(x.summary.pci.single) }))}` : ''}${esc(pciLine(x.summary))}</span>
          ${x.sample && x.sample_rule ? `<span class="pci-saved-meta">Rule: ${esc(ruleText(x.sample_rule, x.summary?.length_m))}</span>` : ''}<span class="pci-saved-meta">${esc([num(x.summary?.length_m ?? x.length_m) !== null ? fmtM(Number(x.summary?.length_m ?? x.length_m)) : '', x.category ? road(x.category).name : '', x.created_at ? D.localTime(x.created_at) : ''].filter(Boolean).join(' · '))}</span></button></li>`).join('')}</ul>`;
  }

  /* ---------- actions ---------- */
  function onPanelInput(event) {
    const t = event.target;
    if (t.id === 'pciKey') { S.keyDraft = t.value; return; }
    if (t.dataset?.actChange === 'color-by' && event.type === 'change') { S.colorBy = t.value; drawReport(); render(); return; }
    if (t.name?.startsWith('entered.')) {
      const id = t.name.slice(8); if (!(id in form.entered)) return;
      form.entered[id] = t.value;
      if (event.type === 'change') { if (S.error && !formIssues()) S.error = ''; render(); }
      return;
    }
    if (t.name && t.name in form) {
      form[t.name] = t.value; if (t.name === 'lane_width_m') form.lane_touched = true; if (S.error) { S.error = ''; }
      if (t.name === 'category' && event.type === 'change') { if (!road(form.category).surfaces.includes(form.surface)) form.surface = ''; render(); }
      if (t.name === 'surface' && event.type === 'change') render();
    }
  }
  function onPanelClick(event) {
    const b = event.target.closest('[data-act]');
    if (!b) { const cell = event.target.closest('[data-section]'); if (cell) selectSection(Number(cell.dataset.section)); return; }
    const act = b.dataset.act;
    if (act === 'pick-a') setPick(S.pick === 'a' ? null : 'a');
    else if (act === 'pick-b') setPick(S.pick === 'b' ? null : 'b');
    else if (act === 'add-via') setPick(S.pick === 'via' ? null : 'via');
    else if (act === 'remove-via') { S.points.via.splice(Number(b.dataset.index), 1); invalidateReport(); drawPoints(); requestPath(); }
    else if (act === 'clear') { S.points = { a: null, b: null, via: [] }; S.path = null; S.pathError = ''; invalidateReport(); drawPoints(); drawPath(); setPick('a'); }
    else if (act === 'generate') generate();
    else if (act === 'edit') { if (S.reportSource === 'saved') { S.points = { a: null, b: null, via: [] }; S.path = null; invalidateReport(); S.pick = 'a'; } S.view = 'build'; S.section = null; S.frame = null; drawPoints(); drawReport(); render(); }
    else if (act === 'section') selectSection(Number(b.dataset.index));
    else if (act === 'run') openFrameById(b.dataset.id);
    else if (act === 'frame') { S.frame = S.features.get(b.dataset.id) || null; drawReport(); render(); const fig = $('pciFrameFig'); fig?.scrollIntoView({ block: 'nearest' }); fig?.focus({ preventScroll: true }); }
    else if (act === 'all-frames') { S.allFrames = Number(b.dataset.index); render(); }
    else if (act === 'dl-pdf') download('pdf', b); else if (act === 'dl-csv') download('csv', b); else if (act === 'dl-json') download('json', b);
    else if (act === 'save') save();
    else if (act === 'regenerate') regenerate();
    else if (act === 'open-saved') openSaved(b.dataset.id);
    else if (act === 'reload-saved') loadSaved();
  }
  async function generate() {
    const issue = formIssues();
    if (issue) {
      S.error = issue; if (P23.params.some(enteredIssue)) S.measuredOpen = true; render();
      const bad = $('pciNew').querySelector('[aria-invalid=true]');
      const el = bad || $('pciGenError'), B = $('pciBody');
      // scroll only the card's body (scrollIntoView could also move the fixed card itself)
      if (el && B) B.scrollTop += el.getBoundingClientRect().top - B.getBoundingClientRect().top - B.clientHeight / 3;
      bad?.focus({ preventScroll: true });
      return;
    }
    S.loading = true; S.error = ''; render();
    try {
      const body = reportBody(); const r = await post('/gis/pci/report', body); S.lastBody = body;
      S.report = r; S.reportSource = 'preview'; S.view = 'result'; S.section = null; S.frame = null; S.saveError = '';
      setPick(null); drawPoints(); drawReport(); fitTo(pathFC(r.path ? r : S.path)); render();
      $('pciBody').scrollTop = 0;
      loadFeatures(r);
    } catch (error) { if (error.name !== 'AbortError') S.error = error.message; }
    finally {
      S.loading = false; render();
      if (S.error && S.view === 'build') { $('pciGenError')?.scrollIntoView({ block: 'nearest' }); $('pciFoot').querySelector('[data-act=generate]')?.focus({ preventScroll: true }); }
    }
  }
  // A report saved before the PCI update: same points and details, generated again (a new preview; the saved copy stays).
  function regenerate() {
    const inp = S.report?.input || {}; if (!inp.a || !inp.b) return;
    S.points = { a: inp.a, b: inp.b, via: Array.isArray(inp.via) ? inp.via : [] };
    form.category = P23.roads.some(x => x.id === inp.category) ? inp.category : 'urban';
    form.surface = inp.surface || '';
    for (const k of ['road_name', 'road_number', 'prepared_by']) form[k] = inp[k] || '';
    if (num(inp.lane_width_m) !== null) { form.lane_width_m = String(inp.lane_width_m); form.lane_touched = true; }
    if (num(inp.chainage_at_a_km) !== null) { form.chainage_at_a_km = String(inp.chainage_at_a_km); form.chainage_increases = inp.chainage_increases_towards_b === false ? 'no' : 'yes'; }
    S.report = null; S.reportSource = null; S.savedId = null; S.view = 'build'; S.section = null; S.frame = null;
    drawPoints(); drawReport(); generate();
  }
  async function loadFeatures(r) {
    const sessions = new Set(passesOf(r).map(p => p.session_id).filter(Boolean));
    for (const s of sectionsOf(r)) for (const seg of s.segments || []) if (seg.session_id) sessions.add(seg.session_id);
    for (const id of [...(r.segment_ids || []), ...sectionsOf(r).flatMap(x => x.segment_ids || []), ...(Array.isArray(r.attribution) ? r.attribution.map(a => Array.isArray(a) ? a[Math.max(0, (r.attribution_columns || []).indexOf('segment_id'))] : a.segment_id) : [])]) { const m = /^(.+)_seg\d+$/.exec(String(id)); if (m) sessions.add(m[1]); }
    const missing = [...sessions].filter(s => ![...S.features.values()].some(f => f.properties.session_id === s));
    if (!missing.length) return;
    try {
      const data = await D.fetchPages(api('/gis/video/segments.geojson'), new URLSearchParams({ session: missing.join(','), geom: 'point' }), { maxFeatures: 20000 });
      for (const f of data.features) S.features.set(D.featureId(f), f);
      S.featuresError = '';
    } catch (error) { S.featuresError = error.message || 'could not load the frames'; }
    if (S.report === r) { render(); if (S.section !== null) setTimeout(() => keepSectionVisible(), 30); }
  }
  function openFrameById(id) {
    const all = sectionSegments(S.report); if (!all) return;
    const i = all.findIndex(list => list.includes(String(id))); if (i < 0) return;
    if (S.section !== i) selectSection(i);
    S.frame = S.features.get(String(id)) || null; drawReport(); render();
    const fig = $('pciFrameFig'); fig?.scrollIntoView({ block: 'nearest' }); fig?.focus({ preventScroll: true });
  }
  function sectionBounds(s) {
    const g = s?.geometry, lines = g?.type === 'MultiLineString' ? g.coordinates.map(l => l.map(c => [c[1], c[0]])) : g?.type === 'LineString' ? [g.coordinates.map(c => [c[1], c[0]])] : s ? cut(linePieces(), num(s.from_m), num(s.to_m)) : [];
    const pts = lines.flat(); return pts.length ? L.latLngBounds(pts) : null;
  }
  // Keep the selected section inside the map area that no card covers (re-checked after the card settles).
  function keepSectionVisible(force = false) {
    if (S.section === null || !S.report) return;
    const b = sectionBounds(sectionsOf(S.report)[S.section]), m = lmap(); if (!b || !m) return;
    const area = G.map.availableViewport(), size = m.getSize();
    const p1 = m.latLngToContainerPoint(b.getNorthWest()), p2 = m.latLngToContainerPoint(b.getSouthEast());
    const inside = p1.x >= area.left && p2.x <= area.right && p1.y >= area.top && p2.y <= area.bottom;
    if (inside && !force) return;
    m.fitBounds(b, { paddingTopLeft: [area.left + 30, area.top + 30], paddingBottomRight: [Math.max(0, size.x - area.right) + 30, Math.max(0, size.y - area.bottom) + 30], maxZoom: 17, animate: false });
  }
  function selectSection(i, { fly = true } = {}) {
    S.section = S.section === i ? null : i; S.frame = null; S.allFrames = null; drawReport(); render();
    if (S.section !== null && fly) { keepSectionVisible(true); setTimeout(() => keepSectionVisible(), 60); }
    if (S.section !== null) requestAnimationFrame(() => $('pciPanel').querySelector('.pci-sections li.open')?.scrollIntoView({ block: 'nearest' }));
  }
  function slug(v) { return String(v || '').normalize('NFKD').replace(/[^\w-]+/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '').slice(0, 40); }
  function filename(ext) {
    const r = S.report || {}, h = r.header || {};
    const date = String(r.generated_at_ist || r.created_at || new Date().toISOString()).slice(0, 10);
    const no = r.report_id ?? r.id; return ['IRC82-2023-PCI', slug(h.road_name || r.input?.road_name) || 'stretch', `${Math.round(lengthOf(r) || 0)}m`, date, no !== undefined && no !== null ? `no${slug(no)}` : 'preview'].filter(Boolean).join('_') + '.' + ext;
  }
  function saveBlob(blob, name) {
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = name; a.rel = 'noopener'; document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  }
  async function download(kind, button) {
    const r = S.report; if (!r) return;
    const label = button.textContent; button.disabled = true; button.textContent = kind === 'pdf' ? 'Preparing…' : '…'; button.setAttribute('aria-busy', 'true');
    S.error = '';
    try {
      const id = S.savedId && (S.reportSource === 'saved' || S.reportSource === 'saved-now') ? S.savedId : null;
      if (kind === 'json' && !id) { saveBlob(new Blob([JSON.stringify(r, null, 2)], { type: 'application/json' }), filename('json')); return; }
      const response = id ? await get(`/gis/pci/reports/${encodeURIComponent(id)}${kind === 'json' ? '' : '.' + kind}`, { raw: true })
        : await post(`/gis/pci/report.${kind}`, S.lastBody || reportBody(), { raw: true });
      const blob = await response.blob();
      if (!blob.size) throw new Error('The server returned an empty file.');
      saveBlob(blob, filename(kind));
    } catch (error) { showResultError(`Download failed: ${error.message}`); }
    finally { button.disabled = false; button.textContent = label; button.removeAttribute('aria-busy'); }
  }
  function showResultError(message) {
    let el = $('pciResultError');
    if (!el) { el = document.createElement('div'); el.id = 'pciResultError'; el.className = 'pci-error'; el.setAttribute('role', 'alert'); $('pciNew').prepend(el); }
    el.innerHTML = `<strong>Problem</strong><span>${esc(message)}</span>`; $('pciBody').scrollTop = 0;
  }
  async function save() {
    let key = $('pciKey')?.value.trim() || '';
    if (!key) { S.saveError = 'Enter the operator key to save. Downloads work without it.'; render(); $('pciKey')?.focus(); return; }
    S.saving = true; S.saveError = ''; render();
    try {
      const data = await post('/gis/pci/reports', S.lastBody || reportBody(), { headers: { 'X-API-Key': key } });
      S.savedId = data.id ?? data.report_id ?? 'saved'; S.savedExisting = data.created === false; S.savedLoaded = false;
      try { sessionStorage.setItem(KEY_STORE, key); } catch {}
      // The report now exists as no. N: downloads come from the stored report (named and stamped with its id).
      S.report = { ...S.report, id: S.savedId, report_id: S.savedId }; S.reportSource = 'saved-now';
    } catch (error) {
      if (error.status === 401 || error.status === 403) { try { sessionStorage.removeItem(KEY_STORE); } catch {} S.keyDraft = ''; }
      S.saveError = error.status === 401 || error.status === 403 ? 'The operator key was not accepted. Check it and try again.' : error.message;
    } finally { key = ''; S.saving = false; render(); }
  }
  async function loadSaved() {
    S.savedError = ''; S.savedLoaded = false; render();
    try { const data = await get('/gis/pci/reports'); S.saved = data.reports || data.items || (Array.isArray(data) ? data : []); S.savedLoaded = true; }
    catch (error) { S.savedError = error.message; }
    render();
  }
  async function openSaved(id) {
    try {
      const r = await get(`/gis/pci/reports/${encodeURIComponent(id)}`);
      const report = r.report && r.report.sections ? { ...r.report, title: r.title, stale: r.stale, stale_note: r.stale_note, changed_segments: r.changed_segments, sample: r.sample, sample_rule: r.sample_rule, id: r.id ?? id } : { ...r, id: r.id ?? id };
      S.report = report; S.reportSource = 'saved'; S.savedId = id; S.view = 'result'; S.section = null; S.frame = null; S.pick = null;
      const inp = report.input || {};
      S.points = { a: inp.a || null, b: inp.b || null, via: inp.via || [] }; S.path = report.path || null;
      setTab('new'); drawPoints(); drawReport(); fitTo(pathFC(report.path ? report : S.path)); render(); $('pciBody').scrollTop = 0;
      loadFeatures(report);
    } catch (error) { S.savedError = `Could not open the report: ${error.message}`; render(); }
  }

  /* ---------- keyboard ---------- */
  window.addEventListener('keydown', event => {
    if (event.key !== 'Escape' || !S.open || event.defaultPrevented) return;
    if (document.querySelector('.flow-card.flow-open') || document.querySelector('dialog[open]')) return;
    event.preventDefault(); event.stopPropagation();
    if (S.pick) { setPick(null); if (phone()) setMinimized(false); } else close();
  }, true);
  const relayout = () => { if (!S.open) { placeClosedNotice(); return; } layout(); clearTimeout(relayout.t); relayout.t = setTimeout(() => keepSectionVisible(), 150); };
  window.addEventListener('resize', relayout);
  window.matchMedia('(max-width: 820px)').addEventListener?.('change', () => { if (S.open) { layout(); G.syncFloatingGeometry?.(); } });
  if (window.ResizeObserver) new ResizeObserver(relayout).observe(document.documentElement);
  new MutationObserver(() => { if (S.open) layout(); else placeClosedNotice(); }).observe(document.body, { childList: true });
  const watchNotice = () => { for (const id of ['notice', 'basemapNotice']) { const el = $(id); if (el && !el.dataset.pciWatch) { el.dataset.pciWatch = '1'; new MutationObserver(() => { if (S.open) layout(); else placeClosedNotice(); }).observe(el, { attributes: true, attributeFilter: ['hidden'], childList: true, subtree: true, characterData: true }); } } };
  watchNotice(); new MutationObserver(watchNotice).observe(document.body, { childList: true });
  new MutationObserver(() => { if (S.open) drawReport(); }).observe(document.body, { attributes: true, attributeFilter: ['class'] });

  buildEntry(); buildPanel(); setTab('new');
})();
