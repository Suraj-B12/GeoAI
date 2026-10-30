/* Shared, viewport-sized basemap cache. Hidden layers remain attached so the
 * Leaflet bridge keeps them at the same position as the visible map. */
(function () {
  'use strict';

  class GeoAIBasemaps {
    constructor({ map, L, styles, streetAttribution = '', fallback, notice,
      onProgress = () => {}, onActive = () => {}, onAvailability = () => {}, timeoutMs = 20000 }) {
      Object.assign(this, { map, L, styles, streetAttribution, fallback, notice,
        onProgress, onActive, onAvailability, timeoutMs });
      this.entries = new Map();
      this.current = null;
      this.requestedKey = 'light';
      this._listeners = [];
      this._deferred = new Set();
      this._selection = null;
      this._attributions = new Set();
      this._destroyed = false;
      this._moving = false;
    }

    start(initialKey = 'light') {
      if (this._destroyed) return Promise.resolve(null);
      if (this._startup) return this._startup;
      this.requestedKey = this.styles[initialKey] ? initialKey : 'light';
      this._startup = new Promise(resolve => { this._finishStartup = resolve; });
      // Make the entire catalogue known before a synchronous SDK failure can
      // report progress or trigger a fallback.
      const vectorIds = [...new Set(Object.values(this.styles).map(style => style.vector).filter(Boolean))];
      for (const id of vectorIds) this._entry(id, 'vector');
      if (this.styles.satellite?.url) this._entry('satellite', 'raster');
      this._listen(this.map, 'movestart', () => {
        this._moving = true;
        for (const entry of this._allEntries()) {
          if (entry.state === 'failed') continue;
          entry.idle = false;
          if (entry.initialSettled) {
            clearTimeout(entry.timer);
            entry.timer = null;
          }
          this._loading(entry);
        }
      });
      this._listen(this.map, 'moveend', () => {
        this._moving = false;
        for (const entry of this._allEntries()) {
          if (entry.state === 'failed') continue;
          if (entry.state === 'loading') this._arm(entry);
          if (entry.kind === 'raster') {
            if (entry.layer?.isLoading && !entry.layer.isLoading()) entry.batchComplete = true;
            this._probeRaster(entry);
          } else {
            // The bridge will render its new viewport and emit idle. Never
            // treat the previous frame's idle event as readiness after a pan.
            entry.gl?.triggerRepaint?.();
          }
        }
        this._report();
      });
      this._listen(this.map, 'resize', () => {
        for (const entry of this._allEntries()) {
          if (entry.state === 'failed') continue;
          entry.idle = false;
          this._loading(entry);
          if (entry.kind === 'vector') entry.gl?.triggerRepaint?.();
          else if (entry.layer?.isLoading && !entry.layer.isLoading()) {
            entry.batchComplete = true;
            this._probeRaster(entry);
          }
        }
        this._report();
      });
      this._report();
      for (const id of vectorIds) this._createVector(this.entries.get(id));
      if (this.entries.has('satellite')) this._createRaster(this.entries.get('satellite'), this.styles.satellite);
      this._reconcile();
      return this._startup;
    }

    select(key) {
      if (!this.styles[key] || this._destroyed) return Promise.resolve(this.current);
      if (!this._startup) {
        this.start(key);
      }
      if (this._selection) this._selection.resolve(this.current);
      this.requestedKey = key;
      const selected = new Promise(resolve => { this._selection = { key, resolve }; });
      this._reconcile();
      return selected;
    }

    _entry(id, kind) {
      const entry = { id, kind, state: 'loading', layer: null, gl: null,
        timer: null, initialSettled: false, decoded: new Map(), listeners: [], idle: false,
        styleReady: false, loadSeen: false, batchComplete: false, originals: [], variant: null };
      this.entries.set(id, entry);
      return entry;
    }

    _allEntries() {
      return this._fallbackEntry ? [...this.entries.values(), this._fallbackEntry] : [...this.entries.values()];
    }

    _listen(target, event, callback, entry) {
      if (!target?.on) return;
      target.on(event, callback);
      const listeners = entry ? entry.listeners : this._listeners;
      listeners.push(() => target.off?.(event, callback));
    }

    _later(callback) {
      const timer = setTimeout(() => {
        this._deferred.delete(timer);
        if (!this._destroyed) callback();
      }, 0);
      this._deferred.add(timer);
    }

    _arm(entry) {
      if (entry.timer || entry.state === 'failed') return;
      entry.timer = setTimeout(() => this._fail(entry), entry.id === 'fallback' ? 8000 : this.timeoutMs);
    }

    _loading(entry) {
      if (this._destroyed || entry.state === 'failed') return;
      entry.state = 'loading';
      if (!this._moving || !entry.initialSettled) this._arm(entry);
    }

    _ready(entry) {
      if (this._destroyed || entry.state === 'failed') return;
      clearTimeout(entry.timer);
      entry.timer = null;
      entry.state = 'ready';
      entry.initialSettled = true;
      this._report();
      this._reconcile();
    }

    _fail(entry) {
      if (this._destroyed || entry.state === 'failed') return;
      clearTimeout(entry.timer);
      entry.timer = null;
      entry.state = 'failed';
      entry.initialSettled = true;
      this._report();
      this._reconcile();
      // Removing a Leaflet layer from its tile callback or a MapLibre context
      // event can interrupt the SDK's own cleanup. Defer disposal until safe.
      this._later(() => {
        if (!this._uses(entry)) this._dispose(entry);
      });
    }

    _createVector(entry) {
      if (typeof this.L.maplibreGL !== 'function' || !window.maplibregl) {
        this._fail(entry);
        return;
      }
      this._arm(entry);
      try {
        const vector = entry.layer = this.L.maplibreGL({
          style: `https://tiles.openfreemap.org/styles/${entry.id}`,
          attributionControl: false,
        });
        const L = this.L;
        const removeVector = vector.onRemove;
        vector.onRemove = function (map) {
          L.Util.cancelAnimFrame(this.transitionFrame);
          if (this.getMaplibreMap()) removeVector?.call(this, map);
          else this.getContainer()?.remove();
        };
        // Bridge 0.1.4 otherwise leaves a queued transition alive on removal.
        vector._transitionEnd = function () {
          L.Util.cancelAnimFrame(this.transitionFrame);
          this.transitionFrame = L.Util.requestAnimFrame(() => {
            const gl = this.getMaplibreMap();
            if (!this._map || !gl) return;
            const zoom = this._map.getZoom();
            const center = this._map.getCenter();
            const offset = this._map.latLngToContainerPoint(this._map.getBounds().getNorthWest());
            this._resizeContainer();
            L.DomUtil.setTransform(gl._actualCanvas, offset, 1);
            gl.once('moveend', () => {
              if (this._map && this.getMaplibreMap() === gl) this._zoomEnd();
            });
            gl.jumpTo({ center, zoom: zoom - 1 });
          });
        };
        vector.addTo(this.map);
        const container = entry.container = vector.getContainer();
        container.classList.add('map-vector-basemap');
        // Multiple bridge containers must overlap in the tile pane; its
        // default relative positioning otherwise stacks hidden maps below
        // the viewport even though their tiles report ready.
        container.style.position = 'absolute';
        container.style.top = '0';
        container.style.left = '0';
        container.style.opacity = '0';
        container.style.pointerEvents = 'none';
        container.style.zIndex = '1';
        const gl = entry.gl = vector.getMaplibreMap();
        if (!gl) throw new Error('Map renderer did not initialize');
        this._listen(gl, 'sourcedata', event => {
          if (event.coord && event.sourceId && (!event.sourceDataType || event.sourceDataType === 'content')) {
            this._rememberTile(entry, event.sourceId, event.coord.canonical || event.coord);
          }
          this._probeVector(entry);
        }, entry);
        this._listen(gl, 'style.load', () => {
          const style = gl.getStyle();
          entry.requiredSources = Object.entries(style.sources || {})
            .filter(([, source]) => source.type === 'vector').map(([id]) => id);
          entry.originals = (style.layers || []).map(layer => ({
            id: layer.id, type: layer.type,
            visibility: layer.layout?.visibility,
            text: layer.layout?.['text-field'] !== undefined,
            paint: {
              'text-color': layer.paint?.['text-color'],
              'text-halo-color': layer.paint?.['text-halo-color'],
              'text-halo-width': layer.paint?.['text-halo-width'],
            },
          }));
          entry.styleReady = true;
          entry.variant = null;
          this._probeVector(entry);
        }, entry);
        this._listen(gl, 'load', () => {
          entry.loadSeen = true;
          this._probeVector(entry);
        }, entry);
        this._listen(gl, 'idle', () => {
          entry.idle = true;
          this._probeVector(entry);
        }, entry);
        this._listen(gl, 'webglcontextlost', () => {
          // MapLibre finishes handling context loss before we switch/dispose.
          this._later(() => this._fail(entry));
        }, entry);
      } catch (error) {
        this._fail(entry);
      }
    }

    _rememberTile(entry, source, coord) {
      const valid = Number.isInteger(coord?.z) && coord.z >= 0 &&
        Number.isInteger(coord?.x) && Number.isInteger(coord?.y);
      if (!valid) return;
      if (!entry.decoded.has(source)) entry.decoded.set(source, new Map());
      const tiles = entry.decoded.get(source);
      tiles.set(`${coord.z}/${coord.x}/${coord.y}`, { z: coord.z, x: coord.x, y: coord.y });
      // Keep only a small history of visible tiles, not an offline map cache.
      if (tiles.size > 512) tiles.delete(tiles.keys().next().value);
    }

    _visibleTile(tiles) {
      if (!tiles?.size) return false;
      const bounds = this.map.getBounds?.();
      if (!bounds?.getWest || !bounds?.getEast || !bounds?.getNorth || !bounds?.getSouth) return true;
      const west = bounds.getWest(), east = bounds.getEast();
      const north = bounds.getNorth(), south = bounds.getSouth();
      for (const coord of tiles.values()) {
        const scale = 2 ** coord.z;
        const tileWest = coord.x / scale * 360 - 180;
        const tileEast = (coord.x + 1) / scale * 360 - 180;
        const latitude = y => Math.atan(Math.sinh(Math.PI * (1 - 2 * y / scale))) * 180 / Math.PI;
        if (latitude(coord.y) < south || latitude(coord.y + 1) > north) continue;
        // Account for wrapped Leaflet worlds without changing the user's view.
        const offset = Math.round(((west + east) / 2 - (tileWest + tileEast) / 2) / 360) * 360;
        if (tileEast + offset >= west && tileWest + offset <= east) return true;
      }
      return false;
    }

    _probeVector(entry) {
      if (this._moving || entry.state === 'failed' || !entry.styleReady || !entry.loadSeen || !entry.idle) return;
      const gl = entry.gl;
      if ((gl.loaded && !gl.loaded()) || (gl.areTilesLoaded && !gl.areTilesLoaded())) return;
      // MapLibre also emits load/idle after failed requests. A decoded tile
      // intersecting this viewport proves that each vector source can render.
      if (!entry.requiredSources?.length || !entry.requiredSources.every(id => this._visibleTile(entry.decoded.get(id)))) return;
      this._ready(entry);
    }

    _createRaster(entry, source) {
      if (!source?.url) { this._fail(entry); return; }
      this._arm(entry);
      try {
        const raster = entry.layer = this.L.tileLayer(source.url, {
          ...source.options, opacity: 0, zIndex: 0, attribution: '',
        });
        entry.attribution = source.options?.attribution || '';
        this._listen(raster, 'loading', () => {
          entry.batchComplete = false;
          this._loading(entry);
          this._report();
        }, entry);
        this._listen(raster, 'tileload', event => {
          this._rememberTile(entry, 'raster', event.coords || {});
          this._probeRaster(entry);
        }, entry);
        this._listen(raster, 'tileerror', event => {
          const coord = event.coords;
          if (coord) entry.decoded.get('raster')?.delete(`${coord.z}/${coord.x}/${coord.y}`);
        }, entry);
        this._listen(raster, 'load', () => {
          entry.batchComplete = true;
          if (this._visibleTile(entry.decoded.get('raster'))) this._probeRaster(entry);
          else this._later(() => this._fail(entry));
        }, entry);
        raster.addTo(this.map);
      } catch (error) {
        this._fail(entry);
      }
    }

    _probeRaster(entry) {
      if (this._moving || entry.state === 'failed' || !entry.batchComplete) return;
      if (entry.layer?.isLoading?.()) return;
      if (this._visibleTile(entry.decoded.get('raster'))) this._ready(entry);
    }

    _availability() {
      const states = {};
      for (const [key, style] of Object.entries(this.styles)) {
        const vector = this.entries.get(style.vector);
        if (style.url) {
          const raster = this.entries.get('satellite');
          states[key] = raster?.state === 'ready' && vector?.state !== 'loading' ? 'ready' : raster?.state || 'loading';
          if (raster?.state === 'ready' && vector?.state === 'loading') states[key] = 'loading';
        } else states[key] = vector?.state || 'loading';
      }
      if (this._fallbackEntry && this._fallbackEntry.state !== 'failed') states.light = this._fallbackEntry.state;
      return states;
    }

    _report() {
      if (this._destroyed) return;
      const entries = [...this.entries.values()];
      this.onProgress({ ready: entries.filter(entry => entry.state === 'ready').length,
        settled: entries.filter(entry => entry.initialSettled).length, total: entries.length });
      this.onAvailability(this._availability());
    }

    _uses(entry) {
      return !!entry.layer && !!this.current && (this.current.vector === entry.layer || this.current.raster === entry.layer);
    }

    _currentUsable() {
      if (!this.current) return false;
      const used = this._allEntries().filter(entry => this._uses(entry));
      // Satellite imagery remains useful if the label renderer fails.
      return used.some(entry => entry.state !== 'failed' && (entry.kind === 'raster' || !this.styles[this.current.key]?.labelsOnly));
    }

    _reconcile() {
      if (this._destroyed || this._reconciling) return;
      this._reconciling = true;
      try {
        const states = this._availability();
        if (states[this.requestedKey] === 'ready') {
          this._activate(this.requestedKey);
          this._resolveSelection();
        } else if (!this._currentUsable()) {
          const alternative = ['light', 'dark', 'voyager', 'satellite', ...Object.keys(this.styles)]
            .find(key => states[key] === 'ready');
          if (alternative) {
            this._activate(alternative);
            if (states[this.requestedKey] === 'failed') {
              this._showNotice('This map style is unavailable. Showing another loaded map.');
              this._resolveSelection();
            }
          } else if ([...this.entries.values()].every(entry => entry.state === 'failed')) {
            this._ensureFallback();
          }
        } else if (states[this.requestedKey] === 'failed') {
          this._showNotice('This map style is unavailable. Your current map is still visible.');
          this._resolveSelection();
        }
        if ([...this.entries.values()].every(entry => entry.initialSettled) &&
            (this._currentUsable() || this._fallbackEntry?.state === 'failed' || this._fallbackUnavailable)) {
          if (!this._currentUsable()) {
            this._showNotice(this._fallbackUnavailable || 'Map tiles could not load. Check your connection and reload.');
            this._resolveSelection();
          }
          this._finishStartup?.(this.current);
          this._finishStartup = null;
        }
      } finally {
        this._reconciling = false;
      }
    }

    _ensureFallback() {
      if (this._fallbackEntry || this._fallbackUnavailable) return;
      if (!/^https?:$/.test(window.location.protocol)) {
        this._fallbackUnavailable = 'Open this dashboard through a local web server to load the fallback map.';
        return;
      }
      // The fallback provider receives requests only if the main providers
      // fail, and only for the visible viewport.
      this._fallbackEntry = { id: 'fallback', kind: 'raster', state: 'loading',
        layer: null, timer: null, initialSettled: false, decoded: new Map(), listeners: [], batchComplete: false };
      this._showNotice('Loading the standard street map…');
      this._createRaster(this._fallbackEntry, this.fallback);
    }

    _applyVariant(entry, style) {
      const variant = style.labelsOnly ? 'labels' : style.hideLabels ? 'minimal' : 'normal';
      entry.container?.classList.toggle('map-monochrome', !!style.monochrome);
      if (entry.variant === variant) return;
      for (const layer of entry.originals) {
        const hidden = (style.labelsOnly && layer.type !== 'symbol') || (style.hideLabels && layer.type === 'symbol');
        entry.gl.setLayoutProperty(layer.id, 'visibility', hidden ? 'none' : layer.visibility || 'visible');
        if (layer.type === 'symbol' && layer.text) {
          const paint = style.labelsOnly ? { 'text-color': '#ffffff', 'text-halo-color': '#202024', 'text-halo-width': 1.5 } : layer.paint;
          for (const [name, value] of Object.entries(paint)) entry.gl.setPaintProperty(layer.id, name, value === undefined ? null : value);
        }
      }
      entry.variant = variant;
    }

    _activate(key) {
      const style = this.styles[key];
      const useFallback = key === 'light' && this.entries.get(style.vector)?.state !== 'ready' && this._fallbackEntry?.state === 'ready';
      const vectorEntry = useFallback ? null : this.entries.get(style.vector);
      const rasterEntry = useFallback ? this._fallbackEntry : style.url ? this.entries.get('satellite') : null;
      const vector = vectorEntry?.state === 'ready' ? vectorEntry.layer : null;
      const raster = rasterEntry?.state === 'ready' ? rasterEntry.layer : null;
      if (!vector && !raster) return;
      // Dark and Satellite share a renderer. When restoring Dark's full
      // paint, keep imagery underneath until that renderer has drawn a new
      // frame, so its previous labels-only canvas cannot expose a blank map.
      const previousRaster = this.current?.raster;
      const holdPreviousRaster = previousRaster && !raster && vector &&
        this.current.vector === vector && vectorEntry.variant === 'labels';
      const unchanged = this.current?.key === key && this.current.vector === vector && this.current.raster === raster;
      if (vector) this._applyVariant(vectorEntry, style);
      // Reveal the complete candidate in the same turn that hides the old
      // layer; its renderer and decoded tiles have never been discarded.
      for (const entry of this._allEntries()) {
        const visible = entry.layer === vector || entry.layer === raster;
        if (entry.kind === 'vector' && entry.container) entry.container.style.opacity = visible ? '1' : '0';
        if (entry.kind === 'raster') entry.layer?.setOpacity?.(visible || (holdPreviousRaster && entry.layer === previousRaster) ? 1 : 0);
      }
      this.current = { key, vector, raster, fallback: !!useFallback };
      if (holdPreviousRaster) {
        vectorEntry.gl.once('render', () => {
          if (!this._destroyed && this.current?.raster !== previousRaster) previousRaster.setOpacity?.(0);
        });
        vectorEntry.gl.triggerRepaint?.();
      }
      this._setAttribution([vector ? this.streetAttribution : '', rasterEntry?.attribution || '']);
      if (useFallback) this._showNotice('This map style is unavailable. Showing the standard street map.');
      else if (style.labelsOnly && !vector) this._showNotice('Street labels could not load over the satellite imagery.');
      else this._showNotice('');
      if (!unchanged) this.onActive(this.current);
      for (const entry of this._allEntries()) {
        if (entry.state === 'failed' && !this._uses(entry)) this._later(() => this._dispose(entry));
      }
    }

    _setAttribution(values) {
      const next = new Set(values.filter(Boolean));
      const control = this.map.attributionControl;
      for (const value of this._attributions) if (!next.has(value)) control?.removeAttribution?.(value);
      for (const value of next) if (!this._attributions.has(value)) control?.addAttribution?.(value);
      this._attributions = next;
    }

    _showNotice(message) {
      if (typeof this.notice === 'function') this.notice(message);
      else if (this.notice) { this.notice.textContent = message; this.notice.hidden = !message; }
    }

    _resolveSelection() {
      this._selection?.resolve(this.current);
      this._selection = null;
    }

    _dispose(entry) {
      if (entry.disposed) return;
      entry.disposed = true;
      clearTimeout(entry.timer);
      entry.timer = null;
      for (const detach of entry.listeners) detach();
      entry.listeners.length = 0;
      if (entry.layer) {
        try { this.map.removeLayer(entry.layer); }
        catch (_) { entry.container?.remove(); }
      }
    }

    destroy() {
      if (this._destroyed) return;
      this._destroyed = true;
      for (const detach of this._listeners) detach();
      for (const timer of this._deferred) clearTimeout(timer);
      for (const entry of this._allEntries()) this._dispose(entry);
      this._setAttribution([]);
      this.current = null;
      this._resolveSelection();
      this._finishStartup?.(null);
      this._finishStartup = null;
    }
  }

  window.GeoAIBasemaps = GeoAIBasemaps;
}());
