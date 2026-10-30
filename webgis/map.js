/* Leaflet map adapter for the read-only geoai.gis.v1 survey workspace. */
(function () {
  'use strict';
  const FALLBACK = '#737780';
  const finite = (value) => typeof value === 'number' && Number.isFinite(value);
  const coordinate = (lon, lat) => finite(lon) && finite(lat) && Math.abs(lon) <= 180 && Math.abs(lat) <= 90;
  const color = (value, fallback = FALLBACK) => /^#(?:[0-9a-f]{3}|[0-9a-f]{4}|[0-9a-f]{6}|[0-9a-f]{8})$/i.test(value || '') ? value : fallback;
  const featureId = (feature) => String(feature?.properties?.segment_id ?? feature?.properties?.id ?? feature?.id ?? '');
  const safeNode = (text, className) => {
    const node = document.createElement('div');
    if (className) node.className = className;
    node.textContent = text;
    return node;
  };
  function position(feature) {
    const p = feature?.properties || {};
    const lat = p.centre_lat ?? p.latitude;
    const lon = p.centre_lon ?? p.longitude;
    if (coordinate(lon, lat)) return [lat, lon];
    const g = feature?.geometry;
    if (!g) return null;
    let coords;
    if (g.type === 'Point') coords = [g.coordinates];
    if (g.type === 'LineString') coords = Array.isArray(g.coordinates) ? g.coordinates : [];
    if (g.type === 'Polygon') coords = g.coordinates?.[0]?.slice?.(0, -1);
    if (g.type === 'MultiLineString') coords = g.coordinates?.flat?.();
    if (g.type === 'MultiPolygon') coords = g.coordinates?.[0]?.[0]?.slice?.(0, -1);
    coords = (coords || []).filter((c) => Array.isArray(c) && coordinate(c[0], c[1]));
    if (!coords.length) return null;
    return [coords.reduce((sum, c) => sum + Number(c[1]), 0) / coords.length,
      coords.reduce((sum, c) => sum + Number(c[0]), 0) / coords.length];
  }
  function usableGeometry(g) {
    if (!g || typeof g !== 'object') return false;
    const point = c => Array.isArray(c) && coordinate(c[0], c[1]);
    const line = c => Array.isArray(c) && c.length >= 2 && c.every(point);
    const polygon = c => Array.isArray(c) && c.length >= 1 && c.every(r => Array.isArray(r) && r.length >= 4 && r.every(point));
    if (g.type === 'Point') return point(g.coordinates);
    if (g.type === 'LineString') return line(g.coordinates);
    if (g.type === 'Polygon') return polygon(g.coordinates);
    if (g.type === 'MultiLineString') return Array.isArray(g.coordinates) && g.coordinates.every(line);
    if (g.type === 'MultiPolygon') return Array.isArray(g.coordinates) && g.coordinates.every(polygon);
    return false;
  }
  function inRing(lon, lat, ring) {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i], [xj, yj] = ring[j];
      if ((yi > lat) !== (yj > lat) && lon < (xj - xi) * (lat - yi) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }
  function inGeometry(lon, lat, geometry) {
    const polygons = geometry?.type === 'Polygon' ? [geometry.coordinates] : geometry?.type === 'MultiPolygon' ? geometry.coordinates : [];
    return polygons.some((rings) => rings[0] && inRing(lon, lat, rings[0]) && !rings.slice(1).some((ring) => inRing(lon, lat, ring)));
  }
  function boundsOf(geometry) {
    const bbox = [Infinity, Infinity, -Infinity, -Infinity];
    function walk(coords) {
      if (typeof coords?.[0] === 'number') {
        bbox[0] = Math.min(bbox[0], coords[0]); bbox[1] = Math.min(bbox[1], coords[1]);
        bbox[2] = Math.max(bbox[2], coords[0]); bbox[3] = Math.max(bbox[3], coords[1]);
      } else if (Array.isArray(coords)) coords.forEach(walk);
    }
    walk(geometry?.coordinates);
    return bbox;
  }
  class SurveyMap {
    constructor(elementId, callbacks = {}) {
      this.callbacks = callbacks;
      this.meta = { labels: [], severities: [], vlm_classes: [] };
      this.video = []; this.citizen = []; this.coverage = []; this.sessions = [];
      this.layers = { video: true, citizen: false, track: true, coverage: false, heat: false, wards: false };
      this.wardRecords = null; this.wardPromise = null; this.selected = null;
      this.map = L.map(elementId, { preferCanvas: true, zoomControl: false, attributionControl: true, minZoom: 1, maxZoom: 20 }).setView([12.9716, 77.5946], 12);
      this.map.attributionControl.setPrefix(false);
      L.control.zoom({ position: 'topright' }).addTo(this.map);
      L.control.scale({ position: 'bottomleft', imperial: false }).addTo(this.map);
      [['wards', 310], ['tracks', 330], ['coverage', 360], ['survey', 400], ['citizens', 420], ['selection', 450]].forEach(([name, z]) => {
        this.map.createPane(name); this.map.getPane(name).style.zIndex = String(z);
      });
      this.map.getPane('tracks').style.pointerEvents = 'none';
      this.map.getPane('selection').style.pointerEvents = 'none';
      this.DiamondMarker = L.CircleMarker.extend({
        _updatePath() {
          const p = this._point, r = this._radius;
          this._parts = [[L.point(p.x, p.y - r), L.point(p.x + r, p.y), L.point(p.x, p.y + r), L.point(p.x - r, p.y)]];
          this._renderer._updatePoly(this, true);
        }
      });
      this.canvas = L.canvas({ pane: 'survey', padding: .4 });
      // Interactive vectors share a canvas so one full-map canvas cannot steal another's pointer events.
      this.citizenCanvas = this.canvas;
      this.selectionCanvas = L.canvas({ pane: 'selection', padding: .4 });
      this.groups = {
        video: L.featureGroup(), citizen: L.featureGroup(), track: L.featureGroup(),
        coverage: L.featureGroup(), heat: L.layerGroup(), wards: L.featureGroup(), selection: L.featureGroup().addTo(this.map)
      };
      this.streetZoom = false;
      this.setTheme('light');
      this.map.on('zoomend', () => {
        const street = this.map.getZoom() >= 16;
        if (street !== this.streetZoom) { this.streetZoom = street; this.renderFeatures(); }
      });
      this.map.on('moveend', () => this.callbacks.onViewChange?.(this.getView()));
      this.map.on('move zoom resize', () => this.callbacks.onSelectionMove?.(this.anchorFor(this.selected)));
      this.map.on('unload', () => L.Util.cancelAnimFrame(this.followFrame));

      this.updateVisibility();
    }
    setMeta(meta) {
      this.meta = meta || this.meta;
      this.labelMap = new Map((this.meta.labels || []).map((x) => [x.name, x]));
      this.classMap = new Map((this.meta.vlm_classes || []).map((x) => [x.name, x]));
      this.radiusMap = new Map((this.meta.severities || []).map((x) => [x.name, x.radius_px]));
      this.renderFeatures();
    }
    setSessions(sessions) { this.sessions = sessions || []; this.renderTracks(); }
    setData(videoFeatures, citizenFeatures, coverageFeatures = []) {
      // This standalone workspace is drive-video only. Ignore any accidental
      // photo collection passed by an older integration caller.
      this.video = (videoFeatures || []).filter(feature => feature?.properties?.layer !== 'citizen');
      this.citizen = [];
      this.coverage = (coverageFeatures || []).filter(feature => feature?.properties?.layer !== 'citizen');
      this.renderFeatures(); this.updateWardCounts();
      if (this.selected) {
        const id = featureId(this.selected), layer = this.selected.properties?.layer;
        this.selected = [...this.video, ...this.citizen].find((f) => featureId(f) === id && f.properties?.layer === layer) || null;
        this.renderSelection(false);
      }
    }
    setLayers(layers) {
      const previous = this.layers;
      this.layers = { ...this.layers, ...layers, citizen: false };
      this.updateVisibility();
      if (previous.heat !== this.layers.heat) this.renderHeat();
      if (this.layers.wards) {
        this.loadWards();
        this.groups.wards.eachLayer((ward) => ward.eachLayer((shape) => shape.bringToBack?.()));
      }
      if (previous.video !== this.layers.video || previous.citizen !== this.layers.citizen) { this.updateWardCounts(); this.renderHeat(); }
      if (this.selected && !this.selectionVisible()) {
        this.selected = null;
        this.callbacks.onSelectionInvalidated?.();
      }
      this.renderSelection(false);
    }
    updateVisibility() {
      Object.entries(this.layers).forEach(([key, visible]) => {
        const group = this.groups[key];
        if (!group) return;
        if (visible && !this.map.hasLayer(group)) group.addTo(this.map);
        else if (!visible && this.map.hasLayer(group)) this.map.removeLayer(group);
      });
    }
    featureColor(feature) {
      const p = feature.properties || {};
      if (p.status === 'not_covered') return color(this.classMap?.get('not_covered')?.color);
      if (p.vlm_class && p.vlm_class !== 'distress') return color(this.classMap?.get(p.vlm_class)?.color);
      return color(this.labelMap?.get(p.marker_label)?.color);
    }
    featureRadius(feature) {
      const radius = this.radiusMap?.get(feature.properties?.final_severity) ?? this.radiusMap?.get('Unknown');
      return finite(radius) && Number(radius) > 0 ? Number(radius) : 4;
    }
    tooltip(feature) {
      const p = feature.properties || {};
      const node = document.createElement('div');
      const label = p.final_primary_label || p.vlm_class?.replaceAll('_', ' ') || 'Awaiting classification';
      const title = safeNode(label); title.style.fontWeight = '700'; node.append(title);
      node.append(safeNode(`${p.road_name || p.road_id || 'Drive survey'} · segment ${Number(p.k || 0) + 1}`));
      const info = [p.final_severity, p.expert_reviewed ? 'Expert reviewed' : p.needs_expert_review ? 'Awaiting expert review' : 'Model result'].filter(Boolean);
      node.append(safeNode(info.join(' · ')));
      if (finite(p.position_sigma_m)) node.append(safeNode(`Position ±${Number(p.position_sigma_m).toFixed(1)} m (1σ)`));
      return node;
    }
    wire(layer, feature) {
      layer.bindTooltip(() => this.tooltip(feature), { direction: 'top', sticky: true, className: 'survey-tooltip' });
      layer.on('click', () => this.callbacks.onSelect?.(feature));
      return layer;
    }
    clusterGroup(layerName) {
      if (!L.markerClusterGroup) return L.featureGroup();
      return L.markerClusterGroup({
        maxClusterRadius: 46, disableClusteringAtZoom: 16, showCoverageOnHover: false,
        chunkedLoading: true, chunkInterval: 80, chunkDelay: 25, animate: false,
        removeOutsideVisibleBounds: true, zoomToBoundsOnClick: true,
        iconCreateFunction: (cluster) => {
          const node = safeNode(String(cluster.getChildCount()));
          Object.assign(node.style, { width: '36px', height: '36px', display: 'grid', placeItems: 'center', background: '#1a1a1c', color: '#fff', border: '3px solid #fff', borderRadius: layerName === 'video' ? '10px' : '50%', boxShadow: '0 2px 8px #0003', font: '700 12px Urbanist, sans-serif' });
          node.title = `${cluster.getChildCount()} ${layerName === 'video' ? 'survey segments' : 'citizen reports'}`;
          return L.divIcon({ html: node, className: 'survey-cluster', iconSize: [36, 36], iconAnchor: [18, 18] });
        }
      });
    }
    marker(feature, type, latlng) {
      const radius = this.featureRadius(feature), fill = this.featureColor(feature);
      const node = document.createElement('span');
      const size = radius * 2;
      Object.assign(node.style, { display: 'block', width: `${size}px`, height: `${size}px`, background: fill, border: '1.5px solid #fff', boxShadow: '0 1px 3px #0005', borderRadius: type === 'video' ? '2px' : '50%', transform: type === 'video' ? 'rotate(45deg)' : '' });
      const p = feature.properties || {};
      const title = `${type === 'video' ? 'Survey segment' : 'Citizen report'}: ${p.final_primary_label || p.vlm_class || 'unclassified'}`;
      return this.wire(L.marker(latlng, { icon: L.divIcon({ html: node, className: 'survey-point', iconSize: [size, size], iconAnchor: [radius, radius] }), keyboard: true, title, alt: title }), feature);
    }
    renderFeatures() {
      if (!this.groups) return;
      this.groups.video.clearLayers(); this.groups.citizen.clearLayers(); this.groups.coverage.clearLayers();
      const street = this.map.getZoom() >= 16;
      for (const [type, features] of [['video', this.video]]) {
        const target = this.groups[type];
        const cluster = street ? null : this.clusterGroup(type);
        const markers = [];
        for (const feature of features) {
          const latlng = position(feature);
          if (!latlng) continue;
          if (cluster) { markers.push(this.marker(feature, type, latlng)); continue; }
          const paint = this.featureColor(feature), radius = this.featureRadius(feature);
          let layer;
          if (type === 'video' && usableGeometry(feature.geometry) && feature.geometry.type !== 'Point') {
            layer = L.geoJSON(feature, { renderer: this.canvas, style: { color: paint, weight: 1.5, fillColor: paint, fillOpacity: .78, opacity: .95, renderer: this.canvas } });
          } else {
            const options = { renderer: type === 'video' ? this.canvas : this.citizenCanvas, radius, color: '#fff', weight: 1.3, fillColor: paint, fillOpacity: .95 };
            layer = type === 'video' ? new this.DiamondMarker(latlng, options) : L.circleMarker(latlng, options);
          }
          this.wire(layer, feature); target.addLayer(layer);
        }
        if (cluster) {
          if (cluster.addLayers) cluster.addLayers(markers); else markers.forEach((marker) => cluster.addLayer(marker));
          target.addLayer(cluster);
        }
      }
      for (const feature of this.coverage) {
        if (!usableGeometry(feature.geometry)) continue;
        const p = feature.properties || {};
        const isMissing = p.status === 'not_covered' || p.vlm_class === 'not_covered';
        const isPending = p.vlm_class === 'pending';
        const style = { renderer: this.canvas, color: this.featureColor(feature), weight: isMissing ? 6 : 4, opacity: .9, dashArray: isMissing ? '6 5' : isPending ? '3 5' : null, fill: false };
        const layer = L.geoJSON(feature, { renderer: this.canvas, style, pointToLayer: (_, latlng) => L.circleMarker(latlng, { ...style, radius: 4 }) });
        this.wire(layer, feature); this.groups.coverage.addLayer(layer);
      }
      this.renderHeat(); this.renderSelection(false);
    }
    renderHeat() {
      this.groups.heat.clearLayers();
      if (!this.layers.heat || !L.heatLayer) return;
      const weights = { Low: 1, Medium: 2, High: 3 };
      const features = this.layers.video ? this.video : [];
      const values = features.filter((f) => f.properties?.vlm_class === 'distress').map((f) => {
        const p = position(f), weight = weights[f.properties?.final_severity] || 0;
        return p && weight ? [...p, weight] : null;
      }).filter(Boolean);
      this.groups.heat.addLayer(L.heatLayer(values, { radius: 22, blur: 18, max: 3, maxZoom: 18, minOpacity: .25 }));
    }
    renderTracks() {
      this.groups.track.clearLayers();
      this.sessions.forEach((session, i) => {
        if (!usableGeometry(session.track)) return;
        // Drive identity colours are separate from the metadata-defined distress palette.
        const key = String(session.session_id || i);
        const hue = [...key].reduce((hash, char) => (Math.imul(hash, 31) + char.charCodeAt(0)) >>> 0, 0) % 360;
        const stroke = `hsl(${hue} 40% 46%)`;
        const track = L.geoJSON(session.track, { pane: 'tracks', interactive: false, style: { color: stroke, weight: 3, opacity: .65 } });
        this.groups.track.addLayer(track);
      });
    }
    fitBBox(bbox) {
      if (!Array.isArray(bbox) || bbox.length !== 4 || !bbox.every(finite)) return;
      if (!coordinate(bbox[0], bbox[1]) || !coordinate(bbox[2], bbox[3]) || bbox[0] > bbox[2] || bbox[1] > bbox[3]) return;
      const size = this.map.getSize(), area = this.availableViewport();
      this.map.fitBounds([[Number(bbox[1]), Number(bbox[0])], [Number(bbox[3]), Number(bbox[2])]], {
        paddingTopLeft: [area.left, area.top], paddingBottomRight: [size.x - area.right, size.y - area.bottom], maxZoom: 18, animate: false
      });
    }

    selectionVisible() {
      const layer = this.selected?.properties?.layer;
      return layer === 'citizen' ? false : layer === 'video' ? (this.layers.video || this.layers.coverage) : true;
    }
    anchorFor(feature = this.selected) {
      const latlng = position(feature);
      if (!latlng) return null;
      const point = this.map.latLngToContainerPoint(latlng), size = this.map.getSize();
      const rect = this.map.getContainer().getBoundingClientRect();
      // Cards and connectors are fixed to the browser viewport, while Leaflet
      // points are relative to the inset map's own container.
      return { x: point.x + rect.left, y: point.y + rect.top, localX: point.x, localY: point.y,
        visible: point.x >= 0 && point.y >= 0 && point.x <= size.x && point.y <= size.y };
    }
    select(feature, { pan = true, focus = false } = {}) {
      L.Util.cancelAnimFrame(this.followFrame);
      this.selected = feature || null;
      const latlng = position(feature);
      // A deliberate map click may focus once. Strip/playback selection merely follows
      // when the vehicle approaches the viewport edge; it never resets the zoom.
      if (focus && latlng && this.map.getZoom() < 16) this.map.setView(latlng, 16, { animate: false });
      this.renderSelection(pan);
      this.callbacks.onSelectionMove?.(this.anchorFor());
    }
    renderSelection(pan) {
      this.groups.selection.clearLayers();
      if (!this.selected || !this.selectionVisible()) return;
      const latlng = position(this.selected);
      if (!latlng) return;
      const p = this.selected.properties || {};
      if (finite(p.position_sigma_m) && Number(p.position_sigma_m) > 0) {
        this.groups.selection.addLayer(L.circle(latlng, { renderer: this.selectionCanvas, radius: Number(p.position_sigma_m), color: '#2563eb', weight: 1, dashArray: '4 3', fillColor: '#2563eb', fillOpacity: .08, interactive: false }));
      }
      if (usableGeometry(this.selected.geometry) && this.selected.geometry.type !== 'Point') {
        this.groups.selection.addLayer(L.geoJSON(this.selected, { renderer: this.selectionCanvas, interactive: false, style: { color: this.theme === 'dark' ? '#fff' : '#111827', weight: 3, fill: false, renderer: this.selectionCanvas } }));
      }
      const node = document.createElement('span');
      if (p.layer === 'citizen') node.textContent = '●';
      else {
        const ns = 'http://www.w3.org/2000/svg';
        const svg = document.createElementNS(ns, 'svg');
        svg.setAttribute('viewBox', '0 0 24 24'); svg.setAttribute('width', '22'); svg.setAttribute('height', '22');
        svg.setAttribute('aria-hidden', 'true');
        const body = document.createElementNS(ns, 'path');
        body.setAttribute('d', 'M8 3h8l2 5v11a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V8l2-5Zm.5 3-1 4h9l-1-4h-7ZM8 14v3h8v-3H8Z');
        body.setAttribute('fill', 'currentColor'); svg.append(body); node.append(svg);
      }
      Object.assign(node.style, { display: 'grid', placeItems: 'center', width: '30px', height: '30px', background: '#fff', color: '#111827', border: '2px solid #111827', borderRadius: '50%', boxShadow: '0 2px 8px #0004', fontSize: '15px', transform: p.layer === 'citizen' ? '' : `rotate(${Number(p.camera_heading_deg) || 0}deg)` });
      this.groups.selection.addLayer(L.marker(latlng, { interactive: false, zIndexOffset: 1000, icon: L.divIcon({ html: node, className: 'survey-location', iconSize: [30, 30], iconAnchor: [15, 15] }) }));
      if (pan) {
        this.keepSelectionVisible();
        // The application reveals its card after selection. Measure once more in
        // the next frame so the very first phone selection also clears that card.
        if (this.map.getSize().x <= 820) {
          L.Util.cancelAnimFrame(this.followFrame);
          this.followFrame = L.Util.requestAnimFrame(() => this.keepSelectionVisible());
        }
      }
    }
    availableViewport({ includeDetail = false } = {}) {
      const size = this.map.getSize(), mobile = size.x <= 820;
      const container = this.map.getContainer().getBoundingClientRect();
      const bounds = { left: container.left, top: container.top, right: container.left + size.x, bottom: container.top + size.y };
      const visibleRect = element => {
        if (!element || element.hidden) return null;
        const style = window.getComputedStyle(element);
        if (style.display === 'none' || style.visibility === 'hidden') return null;
        const rect = element.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0 && rect.right > bounds.left && rect.left < bounds.right &&
          rect.bottom > bounds.top && rect.top < bounds.bottom ? rect : null;
      };
      const base = { left: mobile ? 20 : 32, top: 24, right: size.x - (mobile ? 62 : 72), bottom: size.y - (mobile ? 70 : 58) };
      base.right = Math.max(base.left + 32, base.right);
      base.bottom = Math.max(base.top + 32, base.bottom);
      // Floating cards do not occupy a full row or column. Find the largest
      // clear rectangle among their measured edges so a right-hand summary
      // does not unnecessarily remove the map's right side at every height.
      const selectors = ['.topbar', '.dashboard-heading', '.dashboard-stats', '#activeFilters', '#driveControl', '.toolbar', '#stripPanel', '.legend-control', '.map-tools', '.map-status', '.leaflet-control-zoom'];
      if (includeDetail && mobile) selectors.push('#detailPanel');
      const obstacles = selectors.map(selector => {
        const rect = visibleRect(document.querySelector(selector));
        if (!rect) return null;
        const verticalGap = selector === '#stripPanel' || selector === '#detailPanel' ? 20 : 18;
        const obstruction = {
          left: Math.max(base.left, rect.left - bounds.left - 20),
          right: Math.min(base.right, rect.right - bounds.left + 20),
          top: Math.max(base.top, rect.top - bounds.top - verticalGap),
          bottom: Math.min(base.bottom, rect.bottom - bounds.top + verticalGap)
        };
        return obstruction.right > obstruction.left && obstruction.bottom > obstruction.top ? obstruction : null;
      }).filter(Boolean);
      if (!obstacles.length) return base;
      const edges = [...new Set([base.left, base.right, ...obstacles.flatMap(rect => [rect.left, rect.right])])].sort((a, b) => a - b);
      let best = null, bestArea = -1, bestDistance = Infinity;
      const consider = (left, top, right, bottom) => {
        if (right <= left || bottom <= top) return;
        const area = (right - left) * (bottom - top);
        const distance = Math.abs(left + right - base.left - base.right) + Math.abs(top + bottom - base.top - base.bottom);
        if (area > bestArea || area === bestArea && distance < bestDistance) {
          best = { left, top, right, bottom }; bestArea = area; bestDistance = distance;
        }
      };
      for (let leftIndex = 0; leftIndex < edges.length - 1; leftIndex++) {
        for (let rightIndex = leftIndex + 1; rightIndex < edges.length; rightIndex++) {
          const left = edges[leftIndex], right = edges[rightIndex];
          const blocked = obstacles.filter(rect => rect.left < right && rect.right > left).sort((a, b) => a.top - b.top);
          let top = base.top;
          for (const rect of blocked) {
            consider(left, top, right, rect.top);
            top = Math.max(top, rect.bottom);
          }
          consider(left, top, right, base.bottom);
        }
      }
      // A full-height evidence sheet can intentionally cover the map on a phone.
      // Keep finite fit bounds until that sheet closes; no zoom is manufactured.
      return best || base;
    }
    keepSelectionVisible() {
      if (!this.selected || !this.selectionVisible() || !this.map._loaded) return;
      const latlng = position(this.selected);
      if (!latlng) return;
      const area = this.availableViewport({ includeDetail: true });
      const point = this.map.latLngToContainerPoint(latlng);
      if (point.x < area.left || point.x > area.right || point.y < area.top || point.y > area.bottom) {
        this.map.panBy([point.x - (area.left + area.right) / 2, point.y - (area.top + area.bottom) / 2], { animate: false });
      }
    }
    setTheme(theme) {
      const next = theme === 'dark' ? 'dark' : 'light';
      this.theme = next;
      this.map.getPane('tilePane').style.filter = '';
      if (this.basemapManager) {
        this.renderSelection(false);
        return this.basemapManager.select(next);
      }
      // Use the very same keyless vector cartography as the main GEOAI map.
      // Both styles stay warm for this viewport; switching only reveals a ready layer.
      if (window.GeoAIBasemaps) {
        this.basemapManager = new window.GeoAIBasemaps({
          map: this.map, L,
          styles: { light: { vector: 'positron', dark: false }, dark: { vector: 'dark', dark: true } },
          streetAttribution: '<a href="https://openfreemap.org">OpenFreeMap</a> &copy; <a href="https://openmaptiles.org">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
          fallback: { url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png', options: {
            maxNativeZoom: 19, maxZoom: 20, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          } },
          timeoutMs: 16000,
          notice: message => this.callbacks.onBasemapNotice?.(message),
          onProgress: progress => this.callbacks.onBasemapProgress?.(progress),
          onActive: active => {
            this.basemap = active.vector || active.raster;
            this.callbacks.onBasemapActive?.(active);
          }
        });
        this.basemapsReady = this.basemapManager.start(next);
        this.map.on('unload', () => this.basemapManager.destroy());
        this.renderSelection(false);
        return this.basemapsReady;
      }
      // A local script failure must still leave a conventional, attributed map.
      if (!this.basemap) {
        this.basemap = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
          maxNativeZoom: 19, maxZoom: 20,
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors', crossOrigin: true
        }).addTo(this.map);
        let failures = 0;
        this.basemap.on('tileerror', () => { if (++failures === 4) this.callbacks.onBasemapNotice?.('Basemap tiles could not load. Survey layers remain available; check the internet connection.'); });
      }
      this.renderSelection(false);
      return Promise.resolve(this.basemap);
    }
    getView() {
      const bounds = this.map.getBounds();
      const center = this.map.getCenter();
      return { zoom: this.map.getZoom(), lat: center.lat, lon: center.lng, bbox: [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()] };
    }
    invalidateSize() { this.map.invalidateSize({ animate: false }); }
    async loadWards() {
      if (this.wardRecords || this.wardPromise) return this.wardPromise;
      this.wardPromise = (async () => {
        try {
          const response = await fetch('data/bbmp-198-wards.geojson');
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const data = await response.json();
          this.wardRecords = (data.features || []).map((feature) => ({ feature, bbox: boundsOf(feature.geometry), stats: { segments: 0, distress: 0 } }));
          this.groups.wards.clearLayers();
          this.wardRecords.forEach((record) => {
            const layer = L.geoJSON(record.feature, { renderer: this.canvas, style: { renderer: this.canvas, color: '#63718a', weight: 1, fillColor: '#64748b', fillOpacity: .035, dashArray: '4 4' } });
            layer.bindTooltip(() => this.wardTooltip(record), { sticky: true, className: 'survey-tooltip' });
            layer.on('mouseover', () => layer.setStyle({ fillOpacity: .1, weight: 2 }));
            layer.on('mouseout', () => layer.setStyle({ fillOpacity: .035, weight: 1 }));
            this.groups.wards.addLayer(layer);
            layer.eachLayer((shape) => shape.bringToBack?.());
          });
          this.map.attributionControl.addAttribution('Historical wards: <a href="https://github.com/datameet/Municipal_Spatial_Data/tree/master/Bangalore">DataMeet</a> · <a href="https://creativecommons.org/licenses/by-sa/2.5/in/">CC BY-SA 2.5 IN</a>');
          this.updateWardCounts();
        } catch (error) {
          this.callbacks.onNotice?.('Ward boundaries could not load. Other survey layers remain available.');
        } finally { this.wardPromise = null; }
      })();
      return this.wardPromise;
    }
    updateWardCounts() {
      if (!this.wardRecords) return;
      this.wardRecords.forEach((record) => { record.stats = { segments: 0, distress: 0 }; });
      const features = this.layers.video ? this.video : [];
      for (const feature of features) {
        const pos = position(feature);
        if (!pos) continue;
        const [lat, lon] = pos;
        const ward = this.wardRecords.find((record) => lon >= record.bbox[0] && lat >= record.bbox[1] && lon <= record.bbox[2] && lat <= record.bbox[3] && inGeometry(lon, lat, record.feature.geometry));
        if (!ward) continue;
        ward.stats.segments++;
        if (feature.properties?.vlm_class === 'distress') ward.stats.distress += Number(feature.properties?.length_m) || 0;
      }
    }
    wardTooltip(record) {
      const p = record.feature.properties || {}, stats = record.stats;
      const node = document.createElement('div');
      const title = safeNode(`${p.WARD_NAME || 'Ward'} · ${p.WARD_NO || ''}`); title.style.fontWeight = '700';
      node.append(title, safeNode('Historical 198-ward boundaries (2012)'), safeNode(`${stats.segments.toLocaleString()} loaded survey segments · ${stats.distress.toFixed(1)} m distressed`), safeNode('Filtered, visible layers · assigned by segment centre'));
      return node;
    }
  }
  window.SurveyMap = SurveyMap;
})();
