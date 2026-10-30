/* Shared, DOM-free helpers for the geoai.gis.v1 read-only API. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.SurveyData = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';
  const SCHEMA = 'geoai.gis.v1';
  const CLASS_NAMES = ['distress', 'normal', 'unknown', 'pending', 'not_sent', 'not_covered'];
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const validColor = value => /^#(?:[0-9a-f]{3}|[0-9a-f]{4}|[0-9a-f]{6}|[0-9a-f]{8})$/i.test(value || '');

  function validateMeta(meta) {
    if (!object(meta) || meta.schema !== SCHEMA || !Array.isArray(meta.labels) || !meta.labels.length ||
      !Array.isArray(meta.severities) || !meta.severities.length || !Array.isArray(meta.statuses) ||
      !Array.isArray(meta.vlm_classes) || !finite(meta.confidence_threshold) || meta.confidence_threshold < 0 ||
      meta.confidence_threshold > 1 || !Number.isInteger(meta.limits?.default_limit) || meta.limits.default_limit < 1 ||
      !Number.isInteger(meta.limits?.max_limit) || meta.limits.max_limit < meta.limits.default_limit) {
      throw new Error('The survey server returned incomplete metadata. Reload or check the API version.');
    }
    for (const [key, check] of [
      ['labels', item => validColor(item.color) && typeof item.is_distress === 'boolean'],
      ['severities', item => finite(item.radius_px) && item.radius_px > 0],
      ['statuses', () => true],
      ['vlm_classes', item => item.color === null || validColor(item.color)]
    ]) {
      const seen = new Set();
      for (const item of meta[key]) {
        if (!object(item) || typeof item.name !== 'string' || !item.name.trim() ||
          seen.has(item.name.toLowerCase()) || !check(item)) throw new Error('The survey server returned invalid ' + key + ' metadata.');
        seen.add(item.name.toLowerCase());
      }
    }
    return meta;
  }

  // Build once for an instant preview of loaded data. The server remains authoritative
  // for totals and for any rows outside a capped collection.
  function matcher(filters = {}, meta = {}, layer = 'video') {
    const query = params({ ...filters, threshold: filters.threshold ?? meta.confidence_threshold }, layer);
    const choices = (key, fold = false) => query.has(key) ? new Set(query.get(key).split(',').map(v => fold ? v.trim().toLowerCase() : v.trim())) : null;
    const labels = choices('label', true), severity = choices('severity'), classes = choices('vlm_class');
    const sessions = choices('session'), statuses = choices('status');
    const from = query.has('date_from') ? Date.parse(query.get('date_from')) : null;
    const to = query.has('date_to') ? Date.parse(query.get('date_to')) : null;
    const confidence = query.has('min_confidence') ? Number(query.get('min_confidence')) : null;
    return feature => {
      if (!object(feature)) return false;
      const p = object(feature.properties) ? feature.properties : feature;
      if (p.layer && p.layer !== layer) return false;
      if (layer === 'citizen' && query.get('include_review') !== 'true' && p.assessment_status === 'expert_review') return false;
      if (sessions && !sessions.has(p.session_id)) return false;
      if (statuses && !statuses.has(p.status)) return false;
      if (classes && !classes.has(p.vlm_class)) return false;
      if (labels && !(Array.isArray(p.final_labels) && p.final_labels.some(label => typeof label === 'string' && labels.has(label.toLowerCase())))) return false;
      if (severity && !severity.has(p.final_severity)) return false;
      if (query.has('needs_review') && Boolean(p.needs_expert_review) !== (query.get('needs_review') === 'true')) return false;
      if (query.has('reviewed') && Boolean(p.expert_reviewed) !== (query.get('reviewed') === 'true')) return false;
      if (confidence !== null && (!finite(p.confidence) || p.confidence < confidence)) return false;
      if (from !== null || to !== null) {
        const time = Date.parse(layer === 'citizen' ? p.created_at : p.utc);
        if (!Number.isFinite(time) || (from !== null && time < from) || (to !== null && time >= to)) return false;
      }
      return true;
    };
  }

  function matches(feature, filters = {}, meta = {}, layer = 'video') { return matcher(filters, meta, layer)(feature); }


  function dateBoundary(value, nextDay) {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(value));
    if (!match || Number(match[1]) < 1 || Number(match[1]) > 9998) throw new RangeError('Choose a valid calendar date.');
    const date = new Date(`${value}T00:00:00Z`);
    if (!Number.isFinite(date.getTime()) || date.toISOString().slice(0, 10) !== value) {
      throw new RangeError('Choose a valid calendar date.');
    }
    if (nextDay) date.setUTCDate(date.getUTCDate() + 1);
    return `${date.toISOString().slice(0, 10)}T00:00:00+05:30`;
  }

  function params(filters = {}, layer = 'video', extra = {}) {
    if (!['video', 'citizen'].includes(layer)) throw new TypeError('Unknown survey layer.');
    const result = new URLSearchParams();
    const put = (key, value) => {
      if (Array.isArray(value)) value = value.filter(v => v !== '' && v != null).join(',');
      if (value !== undefined && value !== null && value !== '') result.set(key, String(value));
    };
    if (filters.date_from) put('date_from', dateBoundary(filters.date_from, false));
    if (filters.date_to) put('date_to', dateBoundary(filters.date_to, true));
    if (filters.date_from && filters.date_to && filters.date_from > filters.date_to) {
      throw new RangeError('Start date must be on or before the end date.');
    }
    ['label', 'severity', 'vlm_class'].forEach(key => put(key, filters[key]));
    if (filters.review === 'needs') put('needs_review', true);
    if (filters.review === 'reviewed') put('reviewed', true);
    if (filters.highConfidence) {
      const threshold = filters.threshold;
      if (typeof threshold !== 'number' || !Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
        throw new RangeError('The confidence threshold is unavailable. Reload the survey metadata.');
      }
      put('min_confidence', threshold);
    }
    if (layer === 'video') {
      put('session', filters.session);
      put('status', filters.status);
    } else {
      put('include_review', Boolean(filters.include_review));
    }
    // Extra parameters are endpoint-specific. Do not leak UI fields to a strict API.
    const allowed = new Set(['bbox', 'limit', 'offset', 'date_from', 'date_to', 'label', 'severity',
      'vlm_class', 'reviewed', 'needs_review', 'min_confidence',
      ...(layer === 'video' ? ['session', 'status', 'answered', 'geom'] : ['include_review'])]);
    const entries = extra instanceof URLSearchParams ? extra.entries() : Object.entries(extra);
    for (const [key, value] of entries) if (allowed.has(key) || key.startsWith('_')) put(key, value);
    return result;
  }

  function color(properties = {}, meta = {}) {
    if (properties.status === 'not_covered') return (meta.vlm_classes || []).find(item => item.name === 'not_covered')?.color || meta.other_label_color || '#6b7280';
    if (properties.vlm_class === 'distress') {
      const label = (meta.labels || []).find(item => item.name === properties.marker_label);
      if (label && label.color) return label.color;
    } else {
      const item = (meta.vlm_classes || []).find(item => item.name === properties.vlm_class);
      if (item && item.color) return item.color;
    }
    return meta.other_label_color || '#6b7280';
  }

  function radius(properties = {}, meta = {}) {
    const severities = meta.severities || [];
    const severity = severities.find(item => item.name === properties.final_severity)
      || severities.find(item => item.name === 'Unknown');
    return severity && Number.isFinite(severity.radius_px) && severity.radius_px > 0 ? severity.radius_px : 4;
  }

  function featureId(feature = {}) {
    const p = feature.properties || {};
    return String(feature.id ?? p.segment_id ?? p.id ?? p.photo_id ?? '');
  }

  function formatDistance(m) {
    if (typeof m !== 'number' || !Number.isFinite(m)) return '—';
    if (Math.abs(m) >= 1000) return `${(m / 1000).toLocaleString('en-IN', { maximumFractionDigits: 2 })} km`;
    return `${m.toLocaleString('en-IN', { maximumFractionDigits: 1 })} m`;
  }

  function localTime(iso) {
    if (!iso) return 'Time unavailable';
    const date = new Date(iso);
    if (!Number.isFinite(date.getTime())) return 'Time unavailable';
    return `${new Intl.DateTimeFormat('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit',
      month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit',
      hour12: false }).format(date)} IST`;
  }

  function citizenCSV(features = []) {
    const columns = ['id', 'photo_id', 'assessment_status', 'vlm_class', 'answered', 'final_labels',
      'final_primary_label', 'marker_label', 'final_severity', 'labels', 'primary_label', 'severity',
      'confidence', 'needs_expert_review', 'expert_reviewed', 'expert_corrected_types',
      'expert_corrected_severity', 'description', 'created_at', 'processed_at', 'reviewed_at',
      'latitude', 'longitude', 'address', 'image_url'];
    const cell = value => {
      if (value == null) return '';
      if (Array.isArray(value)) value = value.join('; ');
      if (typeof value === 'boolean') value = value ? 'true' : 'false';
      let text = String(value);
      // Quoting alone does not stop spreadsheet formula execution. Protect text cells,
      // including formulas preceded by whitespace, without turning numeric coordinates into text.
      if (typeof value === 'string' && (/^[\s\uFEFF]*[=+\-@]/u.test(text) || /^[\t\r\n]/.test(text))) text = `'${text}`;
      return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
    };
    const rows = features.map(feature => columns.map(key => cell((feature.properties || {})[key])).join(','));
    return `\uFEFF${[columns.join(','), ...rows].join('\r\n')}\r\n`;
  }

  function perKmCoverage(bin = {}) {
    const nonnegative = value => typeof value === 'number' && Number.isFinite(value) ? Math.max(0, value) : 0;
    const surveyed = nonnegative(bin.surveyed_m);
    const notCovered = nonnegative(bin.not_covered_m);
    const values = Object.fromEntries(CLASS_NAMES.map(name => [name, nonnegative(bin[`${name}_m`])]));
    // Older servers may omit normal_m; infer only from classified_m, never from coverage.
    if (bin.normal_m == null && bin.classified_m != null) {
      values.normal = Math.max(0, nonnegative(bin.classified_m) - values.distress);
    }
    let remaining = surveyed;
    for (const name of CLASS_NAMES.filter(name => name !== 'not_covered')) {
      values[name] = Math.min(values[name], remaining);
      remaining -= values[name];
    }
    // Covered metres without a reported class remain unknown, never Normal.
    values.unknown += remaining;
    values.not_covered = notCovered;
    const total = surveyed + notCovered;
    return CLASS_NAMES.map(name => ({ name, metres: values[name], fraction: total > 0 ? values[name] / total : 0 }));
  }

  async function fetchJSON(url, { signal } = {}) {
    let response;
    try {
      response = await fetch(url, { signal, headers: { Accept: 'application/json, application/geo+json' } });
    } catch (cause) {
      if (cause.name === 'AbortError' || signal?.aborted) throw cause;
      const error = new Error('Survey data temporarily unavailable. Check the connection and retry.');
      error.status = 0;
      error.cause = cause;
      throw error;
    }
    let data;
    try { data = await response.json(); } catch (cause) {
      const error = new Error(response.ok ? 'The survey server returned an invalid response.'
        : `Survey request failed (${response.status}). Retry or check the API address.`);
      error.status = response.status;
      error.cause = cause;
      throw error;
    }
    if (!response.ok) {
      const detail = typeof data?.detail === 'string' ? data.detail : '';
      const error = new Error(response.status === 503 ? 'Survey data temporarily unavailable. Retrying in 30 seconds.'
        : detail || `Survey request failed (${response.status}). Please retry.`);
      error.status = response.status;
      error.detail = detail;
      throw error;
    }
    if (!data || data.schema !== SCHEMA) {
      const error = new Error('Unsupported survey data format. Expected geoai.gis.v1; check the API address.');
      error.status = response.status;
      throw error;
    }
    return data;
  }

  async function fetchPages(baseUrl, query = new URLSearchParams(), { signal, maxFeatures = 5000, pageSize = 1000 } = {}) {
    if (!Number.isInteger(maxFeatures) || maxFeatures < 1 || !Number.isInteger(pageSize) || pageSize < 1) {
      throw new RangeError('Page size and feature limit must be positive integers.');
    }
    const source = new URLSearchParams(query);
    const features = [];
    const seen = new Set();
    let total = 0, firstTotal = null;
    let offset = Number(source.get('offset') || 0);
    if (!Number.isInteger(offset) || offset < 0) throw new RangeError('Offset must be a nonnegative integer.');
    const start = offset;
    while (features.length < maxFeatures) {
      source.set('offset', String(offset));
      source.set('limit', String(Math.min(pageSize, maxFeatures - features.length, 20000)));
      const separator = String(baseUrl).includes('?') ? '&' : '?';
      const data = await fetchJSON(`${baseUrl}${separator}${source}`, { signal });
      if (!Array.isArray(data.features) || !Number.isSafeInteger(data.numberMatched) || data.numberMatched < 0) {
        throw new Error('The survey server returned an invalid feature page.');
      }
      total = data.numberMatched;
      if (firstTotal !== null && total !== firstTotal) throw new Error('Survey data changed during loading. Please refresh.');
      firstTotal = total;
      if (data.numberReturned !== undefined && data.numberReturned !== data.features.length) throw new Error('The survey server returned an invalid feature count.');
      if (data.features.length > Number(source.get('limit')) || (data.features.length > 0 && offset + data.features.length > total)) throw new Error('The survey server returned an invalid feature page.');
      if (data.features.some(feature => !object(feature) || feature.type !== 'Feature' || !object(feature.properties) ||
        (feature.geometry !== null && !object(feature.geometry)))) throw new Error('The survey server returned an invalid observation.');
      if (data.features.length === 0 && offset < total) {
        throw new Error('The survey server returned an incomplete page. Please retry.');
      }
      for (const feature of data.features.slice(0, maxFeatures - features.length)) {
        const id = featureId(feature);
        if (!id) throw new Error('The survey server returned an observation without an identifier.');
        if (id && seen.has(id)) throw new Error('Survey data changed during loading. Please refresh.');
        if (id) seen.add(id);
        features.push(feature);
      }
      offset += data.features.length;
      if (offset >= total || data.features.length === 0) break;
    }
    return { features, total, truncated: start + features.length < total };
  }

  return { validateMeta, matcher, matches, params, color, radius, featureId, formatDistance, localTime, citizenCSV, perKmCoverage, fetchJSON, fetchPages };
});
