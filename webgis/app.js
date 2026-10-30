/* Drive surveys: static, read-only client for geoai.gis.v1. */
(() => {
  'use strict';
  const D = window.SurveyData, $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number = value => Number.isFinite(Number(value)) && value !== null ? Number(value) : null;
  const n = value => Number(value || 0).toLocaleString('en-IN');
  const percent = value => value===null||value===undefined?'—':`${Number(value).toFixed(1)}%`;
  const cssColor = value => /^#[0-9a-f]{3,8}$/i.test(value || '') ? value : '#6b7280';
  const base = String(window.GIS_CONFIG?.apiBase || '').replace(/\/$/, '');
  const api = path => base + path;
  function mediaURL(value) {
    if (!value) return '';
    try { const url = new URL(value, base ? new URL(base, location.href) : location.origin); return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password ? url.href : ''; } catch { return ''; }
  }
  let defaultLabel='',urlHasLabels=false;
  const defaults = () => ({date_from:'',date_to:'',label:defaultLabel?[defaultLabel]:[],severity:[],review:'all',highConfidence:false,vlm_class:'',status:'',session:''});
  const state = {meta:null,sessions:[],video:[],coverage:[],summary:null,filters:defaults(),selected:null,stripSession:'',strip:[],playing:false,index:0,geom:'point',controller:null,loadVersion:0,retry:null,timer:null,loading:false,ready:false,videoTotal:0,layerOptions:{video:true,citizen:false,track:true,coverage:false,heat:false,wards:false},truncated:false,stale:true};
  let map;
  state.inspectSession=''; state.stripOpen=false; state.expanded=false; state.cache=[];
  let activeNav='video';
  const flowCards=new Map();
  const motionPreference=window.matchMedia('(prefers-reduced-motion: reduce)');
  let evidenceCloseTimer,evidenceEntryTimer,evidenceExpandTimer,evidenceClosing=false;
  function registerFlow(id,triggerId,{anchorId,hostId,align='start',above=false,modal=false,returnId}={}){
    const panel=$(id),trigger=$(triggerId),host=hostId?$(hostId):null;
    if(!panel||!trigger)return;
    const entry={panel,trigger,anchor:$(anchorId)||trigger,host,align,above,modal,returnTrigger:$(returnId)||trigger,isOpen:false,closing:false,timer:null};
    flowCards.set(id,entry);
    panel.classList.add('flow-card');panel.classList.toggle('flow-above',above);panel.inert=true;
    trigger.setAttribute('aria-expanded','false');
    if(host){
      trigger.addEventListener('click',event=>{event.preventDefault();toggleFlow(id);});
      host.addEventListener('toggle',()=>{if(host.open&&!entry.isOpen&&!entry.closing)openFlow(id);else if(!host.open&&entry.isOpen)closeFlow(id,{immediate:true,restore:false});});
    }
    if(panel.tagName==='DIALOG'){
      panel.addEventListener('cancel',event=>{event.preventDefault();event.stopPropagation();closeFlow(id);});
      panel.addEventListener('close',()=>{if(!panel.open&&entry.isOpen)closeFlow(id,{immediate:true,restore:false});});
    }
    panel.addEventListener('transitionend',event=>{if(event.target===panel&&event.propertyName==='transform'&&entry.closing&&!entry.isOpen)finishFlow(entry);});
  }
  function positionFlow(entry){
    if(entry.modal)return;
    const {panel,align,above}=entry,anchor=entry.anchor.getBoundingClientRect(),margin=12;
    const width=panel.offsetWidth||Math.min(640,innerWidth-margin*2);
    const left=Math.max(margin,Math.min(innerWidth-width-margin,align==='end'?anchor.right-width:anchor.left));
    const available=above?anchor.top-margin*2:innerHeight-anchor.bottom-margin*2;
    panel.style.maxHeight=Math.max(100,available)+'px';
    const top=above?Math.max(margin,anchor.top-12-panel.offsetHeight):Math.min(innerHeight-100-margin,anchor.bottom+12);
    panel.style.left=left+'px';panel.style.top=top+'px';panel.style.right='auto';panel.style.bottom='auto';
    panel.style.setProperty('--flow-origin-x',Math.max(20,Math.min(width-20,anchor.left+anchor.width/2-left))+'px');
  }
  function finishFlow(entry){
    clearTimeout(entry.timer);entry.timer=null;entry.closing=false;
    entry.panel.classList.remove('flow-open','flow-closing');entry.panel.inert=true;
    if(entry.panel.tagName==='DIALOG'){if(entry.panel.open)entry.panel.close();}else entry.panel.hidden=true;
    if(entry.host)entry.host.open=false;
    if(entry.panel.id==='drivePanel'){$('driveControl').classList.remove('flow-host-open');syncFloatingGeometry();}
    if(entry.restoreAfterClose&&!Array.from(flowCards.values()).some(card=>card.isOpen))entry.returnTrigger.focus({preventScroll:true});
    entry.restoreAfterClose=false;
  }
  function closeFlow(id,{restore=true,immediate=false}={}){
    const entry=flowCards.get(id);if(!entry||!entry.isOpen&&!entry.closing)return;
    entry.isOpen=false;entry.closing=true;clearTimeout(entry.timer);
    entry.trigger.setAttribute('aria-expanded','false');
    const returnFocus=restore&&entry.panel.contains(document.activeElement);
    entry.restoreAfterClose=entry.modal&&returnFocus;
    if(returnFocus&&!entry.modal)entry.returnTrigger.focus({preventScroll:true});
    entry.panel.inert=true;entry.panel.classList.remove('flow-open');entry.panel.classList.add('flow-closing');
    if(id==='overviewDialog'||id==='insightsDialog')setNav('video');
    const duration=parseFloat(getComputedStyle(entry.panel).getPropertyValue('--flow-duration'))||(['overviewDialog','filterDialog','aboutDialog'].includes(id)?300:250);
    if(immediate||motionPreference.matches)finishFlow(entry);else entry.timer=setTimeout(()=>{if(!entry.isOpen)finishFlow(entry);},duration+20);
  }
  function closeAllFlows({except,immediate=false}={}){
    for(const [id,entry] of flowCards)if(id!==except&&(entry.isOpen||entry.closing))closeFlow(id,{restore:false,immediate});
  }
  function openFlow(id){
    const entry=flowCards.get(id);if(!entry)return;
    closeAllFlows({except:id});pause();
    clearTimeout(entry.timer);entry.timer=null;entry.closing=false;entry.isOpen=true;entry.restoreAfterClose=false;
    const {panel,trigger,host}=entry;
    if(host)host.open=true;
    panel.hidden=false;panel.inert=false;panel.classList.remove('flow-closing');
    if(panel.tagName==='DIALOG'&&!panel.open){if(entry.modal)panel.showModal();else if(panel.show)panel.show();else panel.setAttribute('open','');}
    if(id==='drivePanel')$('driveControl').classList.add('flow-host-open');
    trigger.setAttribute('aria-expanded','true');positionFlow(entry);
    // Commit the collapsed pose before starting the main GIS's 250 ms downward reveal.
    void panel.offsetWidth;panel.classList.add('flow-open');
    if(id==='overviewDialog'||id==='insightsDialog')setNav('survey');
    const first=panel.querySelector('button:not(:disabled),input:not(:disabled),select:not(:disabled),a[href],[tabindex="0"]');
    first?.focus({preventScroll:true});
  }
  function toggleFlow(id){const entry=flowCards.get(id);if(entry?.isOpen)closeFlow(id);else openFlow(id);}
  function syncNavSlider(){
    const nav=document.querySelector('.nav-pill'),button=$(activeNav==='survey'?'surveyNav':'videoNav');
    if(!nav||!button)return;
    nav.style.setProperty('--nav-x',button.offsetLeft+'px');nav.style.setProperty('--nav-y',button.offsetTop+'px');nav.style.setProperty('--nav-width',button.offsetWidth+'px');
  }
  motionPreference.addEventListener?.('change',()=>{if(motionPreference.matches)for(const entry of flowCards.values())if(entry.closing)finishFlow(entry);});
  let viewTimer;
  let theme = 'light';
  try { theme = localStorage.getItem('geoaiSurveyTheme') || 'light'; } catch {}
  function notice(message, retry=false) { $('noticeText').textContent=message; $('notice').hidden=!message; $('retryButton').hidden=!retry; }
  function hideNotice() { notice(''); }
  function fail(error) {
    pause();
    if (error.name === 'AbortError') return;
    const unavailable = !error.status || error.status === 503 || error.name === 'TimeoutError';
    notice(unavailable ? 'Survey data temporarily unavailable. Retrying in 30 seconds. Displayed data may be out of date.' : error.message, true);
    $('dataStatus').textContent='Connection needs attention';
    clearTimeout(state.retry);
    if (unavailable) state.retry=setTimeout(() => state.ready ? load() : boot(),30000);
  }
  function setTheme(value) {
    theme=value==='dark'?'dark':'light'; document.body.classList.toggle('dark',theme==='dark');
    $('themeButton').setAttribute('aria-label',`Switch to ${theme==='dark'?'light':'dark'} theme`);
    map?.setTheme(theme); try { localStorage.setItem('geoaiSurveyTheme',theme); } catch {}
  }
  try {
    map = new window.SurveyMap('surveyMap', {onSelect:feature=>{pause();state.expanded=false;selectFeature(feature,true);}, onSelectionMove:()=>positionCard(), onSelectionInvalidated:()=>closeSelection(), onBasemapNotice:message=>{let el=$('basemapNotice');if(!el){el=document.createElement('div');el.id='basemapNotice';el.className='basemap-notice';el.setAttribute('role','status');document.body.append(el);}el.textContent=message;el.hidden=!message;}, onNotice:message=>notice(message), onViewChange:view=>{
      positionCard();
      if (!state.ready || state.loading) return;
      const geom=view.zoom>=16?'footprint':'point';
      if(geom!==state.geom) { clearTimeout(viewTimer); viewTimer=setTimeout(()=>load(),180); }
    }});
    setTheme(theme);
  } catch { notice('The map could not load. Reload this page to try again.',true); }
  function readURL() {
    const q=new URLSearchParams(location.hash.replace(/^#/,''));
    urlHasLabels=q.has('label');
    for(const key of ['session','date_from','date_to','review','vlm_class','status']) if(q.has(key)) state.filters[key]=q.get(key);
    for(const key of ['label','severity']) if(q.has(key)) state.filters[key]=q.get(key).split(',').filter(Boolean);
    state.filters.highConfidence=q.get('highConfidence')==='true';
  }
  function writeURL() {
    const q=new URLSearchParams();
    Object.entries(state.filters).forEach(([key,value])=>{ if(key==='threshold'||value===false||value===''||value==='all'||Array.isArray(value)&&!value.length&&key!=='label')return; q.set(key,Array.isArray(value)?value.join(','):value); });
    history.replaceState(null,'',location.pathname+location.search+(q.size?'#'+q:''));
  }
  async function boot() {
    clearTimeout(state.retry); state.controller?.abort();
    const controller=new AbortController(); state.controller=controller;
    const deadline=setTimeout(()=>controller.abort(),20000);
    try {
      const [meta,drives]=await Promise.all([D.fetchJSON(api('/gis/meta'),{signal:controller.signal}),D.fetchJSON(api('/gis/video/sessions?track=true&per_km=true'),{signal:controller.signal})]);
      if(controller.signal.aborted)return;
      D.validateMeta?.(meta);state.meta=meta;state.sessions=drives.sessions||[];state.filters.threshold=meta.confidence_threshold;
      defaultLabel=meta.labels.find(label=>label.name.toLowerCase()==='potholes')?.name||'';
      if(!urlHasLabels&&defaultLabel)state.filters.label=[defaultLabel];
      if(state.filters.session&&!state.sessions.some(s=>s.session_id===state.filters.session))state.filters.session='';
      if(Number(meta.video?.segments)>5000&&!state.filters.session&&!state.filters.date_from&&!state.filters.date_to&&state.sessions.length)state.filters.session=state.sessions[0].session_id;
      // Measure the floating controls for the first fit while the loader still covers the map.
      document.body.classList.remove('initial-loading');
      map?.setMeta(meta);map?.setSessions(state.sessions);map?.fitBBox(meta.video?.bbox);
      state.ready=true;renderFilterControls();renderLegend();renderSessions();
      await load();
      await map?.basemapsReady;
    } catch(error) { if(controller.signal.aborted) fail(Object.assign(new Error('Request timed out'),{name:'TimeoutError'})); else fail(error); }
    finally { clearTimeout(deadline);document.body.classList.remove('initial-loading');$('initialLoader')?.remove(); }
  }
  async function load() {
    if(!state.ready)return;
    clearTimeout(state.retry);pause();state.controller?.abort();
    const version=++state.loadVersion,controller=new AbortController();state.controller=controller;
    state.loading=true;state.stale=true;previewCached();updateExports(); $('dataStatus').textContent='Updating observations…';
    const deadline=setTimeout(()=>controller.abort(),25000);
    try {
      const geom=(map?.getView().zoom||12)>=16?'footprint':'point';
      const vq=D.params(state.filters,'video',{geom});
      const max=state.filters.session||state.filters.date_from||state.filters.date_to?20000:5000;
      const queries=[D.fetchPages(api('/gis/video/segments.geojson'),vq,{signal:controller.signal,maxFeatures:max}),D.fetchJSON(api('/gis/video/summary?')+D.params(state.filters,'video'),{signal:controller.signal})];
      if(state.layerOptions.coverage)queries.push(D.fetchPages(api('/gis/video/segments.geojson'),D.params(state.filters,'video',{geom:'line'}),{signal:controller.signal,maxFeatures:max}));
      const [video,summary,coverage]=await Promise.all(queries);
      if(version!==state.loadVersion)return;
      state.video=video.features.filter(feature=>feature.properties?.layer==='video');state.coverage=(coverage?.features||[]).filter(feature=>feature.properties?.layer==='video');state.summary=summary;
      state.videoTotal=video.total;state.truncated=video.truncated;state.geom=geom;
      map?.setSessions(state.sessions);
      map?.setData(state.video,[],state.coverage);map?.setLayers(state.layerOptions);
      if(state.selected)state.selected=state.video.find(f=>D.featureId(f)===D.featureId(state.selected))||null;
      if(state.stripOpen&&state.filters.session&&state.stripSession!==state.filters.session){state.stripSession=state.filters.session;state.inspectSession=state.filters.session;}
      cacheResults({...video,features:state.video},coverage,geom);
      state.stale=false;renderStats();renderSessions();renderLabels();renderActiveFilters();renderStrip();renderDetail();writeURL();
      $('mapEmpty').hidden=video.total!==0||!state.layerOptions.video; $('dataStatus').textContent=`${n(state.sessions.length)} drives · ${state.truncated?'partial map view':'data loaded'}`;
      if(state.truncated)notice(`Showing ${n(video.features.length)} of ${n(video.total)} drive-video segments. Choose a drive or narrower date range for a complete map. Statistics cover all matches.`);else hideNotice();
    } catch(error) {
      if(version!==state.loadVersion)return;
      if(controller.signal.aborted)fail(Object.assign(new Error('Request timed out'),{name:'TimeoutError'}));else fail(error);
    } finally { clearTimeout(deadline);if(version===state.loadVersion){state.loading=false;updateExports();if(!state.stale&&((map?.getView().zoom||12)>=16?'footprint':'point')!==state.geom){clearTimeout(viewTimer);viewTimer=setTimeout(()=>load(),0);}} }
  }
  function updateExports(){
    const blocked=state.loading||state.stale;
    for(const id of ['exportVideo','insightsButton','printButton','mobileInsights','driveReplay'])if($(id))$(id).disabled=blocked;
    for(const id of ['playButton','previousButton','nextButton'])$(id).disabled=blocked||!state.strip.length;
    $('stripViewport').setAttribute('aria-disabled',String(blocked));
    $('stripTrack').querySelectorAll('.frame-card').forEach(button=>button.disabled=blocked);
  }
  function renderStats() {
    const s=state.summary;
    $('surveyedStat').textContent=D.formatDistance(s.surveyed_m);
    $('segmentsStat').textContent=`${n(s.segments)} segments · ${D.formatDistance(s.not_covered_m)} not covered`;
    $('distressStat').textContent=percent(s.pct_distress_of_surveyed);
    $('distressDetail').textContent=`${D.formatDistance(s.distress_m)} of surveyed road`;
    $('reviewStat').textContent=n(s.needs_expert_review);
    $('classifiedStat').textContent=D.formatDistance(s.classified_m);
    $('classifiedDetail').textContent='With a model result · current filters';
  }
  function renderSessions() {
    $('sessionCount').textContent=String(state.sessions.length);
    $('sessions').innerHTML=state.sessions.length?state.sessions.map((s,i)=>{
      const complete=['stored','cleaned'].includes(s.state),active=state.inspectSession===s.session_id;
      return `<button class="session-card ${active?'active':''}" data-session="${esc(s.session_id)}" aria-pressed="${active}"><div class="session-top"><span class="session-title">${esc(s.road_name||s.road_id||'Unnamed road')}</span><span class="session-number">${String(i+1).padStart(2,'0')} ↗</span></div><div class="session-date">${esc(s.local_date||'Date unavailable')} · ${esc(s.road_id||'Survey')} · ${n(s.segments)} segments</div><div class="session-meta"><strong>${D.formatDistance(s.route_m)}</strong><span>${percent(s.pct_distress_of_surveyed)} distress</span><span class="mini-tag ${complete?'':'processing'}">${complete?'Complete':s.state==='failed'?'Failed':'Processing'}</span></div><div class="session-track"><span style="width:${Math.min(100,Math.max(0,s.pct_distress_of_surveyed||0))}%"></span></div></button>`;
    }).join(''):'<p class="empty">No drive surveys are available yet.</p>';
    renderDriveMenu();
    const session=state.sessions.find(s=>s.session_id===state.inspectSession);
    $('mapScope').textContent=session?`${session.road_name||session.road_id} · ${session.local_date}`:'All drive surveys';
  }
  function renderLabels() {
    const labels=(state.summary?.labels||[]).filter(l=>state.meta.labels.find(m=>m.name===l.label)?.is_distress&&l.length_m>0).sort((a,b)=>b.length_m-a.length_m);
    $('labelBars').innerHTML=labels.length?labels.slice(0,5).map(l=>{
      const color=cssColor(state.meta.labels.find(m=>m.name===l.label)?.color);
      return `<div class="label-row"><div class="label-row-head"><span class="label-row-name"><span class="swatch" style="background:${color}"></span>${esc(l.label)}</span><strong>${percent(l.pct_of_surveyed)}</strong></div><div class="bar-track"><div class="bar-fill" style="width:${Math.min(100,l.pct_of_surveyed||0)}%;background:${color}"></div></div></div>`;
    }).join(''):'<p class="empty">No distress in the current results.</p>';
    $('breakdownScope').textContent=(state.filters.session?'Selected drive':'All drives')+' · current filters';
  }
  function renderFilterControls() {
    const categories=[...new Set(state.meta.labels.map(l=>l.category))];
    $('labelFilters').innerHTML=categories.map(category=>`<div><p class="label-group-title">${esc(category)}</p>${state.meta.labels.filter(l=>l.category===category).map(l=>`<label class="label-option"><input type="checkbox" name="label" value="${esc(l.name)}">${esc(l.name)}</label>`).join('')}</div>`).join('');
    $('severityFilters').innerHTML=state.meta.severities.map(s=>`<label class="chip"><input type="checkbox" name="severity" value="${esc(s.name)}">${esc(s.name)}</label>`).join('');
    $('classFilter').innerHTML='<option value="">All results</option>'+state.meta.vlm_classes.map(c=>`<option value="${esc(c.name)}">${esc(className(c.name))}</option>`).join('');
    $('statusFilter').innerHTML='<option value="">All gate statuses</option>'+state.meta.statuses.map(s=>`<option value="${esc(s.name)}">${esc(s.name.replaceAll('_',' '))}</option>`).join('');
    $('confidenceFilterLabel').textContent=`High confidence only (≥ ${percent(state.meta.confidence_threshold*100)})`;
    $('aboutThreshold').textContent=percent(state.meta.confidence_threshold*100)+' confidence';
    syncFilterForm();
  }
  function syncFilterForm() {
    const form=$('filterForm');
    for(const input of form.querySelectorAll('[name]')) {
      const value=state.filters[input.name];
      if(input.type==='checkbox')input.checked=Array.isArray(value)?value.includes(input.value):Boolean(value);
      else input.value=value||'';
    }
  }
  function renderActiveFilters() {
    const f=state.filters,drive=state.sessions.find(s=>s.session_id===f.session),items=[f.session?'Drive: '+(drive?.road_name||drive?.road_id||f.session)+' · '+(drive?.local_date||''):'',...f.label,...f.severity,f.review!=='all'?f.review==='needs'?'Needs review':'Expert-reviewed':'',f.highConfidence?'High confidence':'',f.vlm_class?className(f.vlm_class):'',f.status?`Gate: ${f.status}`:'',f.date_from?`From ${f.date_from}`:'',f.date_to?`Through ${f.date_to}`:''].filter(Boolean);
    $('filterCount').textContent=String(items.length);$('filterCount').hidden=!items.length;$('activeFilters').hidden=!items.length;
    $('activeFilters').innerHTML=items.map(item=>`<span class="filter-tag">${esc(item)}</span>`).join('')+(items.length?'<button class="text-button" id="clearActive">Reset filters</button>':'');
    $('clearActive')?.addEventListener('click',clearFilters);
  }
  function renderLegend() {
    $('legend').innerHTML='<strong>Drive-video segments</strong><p>◇ One observation per road segment.<br>Footprints appear at street zoom.</p><strong>Size = final severity</strong><p>'+state.meta.severities.map(s=>`${esc(s.name)} ${s.radius_px}px`).join(' · ')+'</p><strong>Colour = final condition</strong>'+state.meta.labels.map(l=>`<div class="legend-row"><span class="swatch" style="background:${cssColor(l.color)}"></span>${esc(l.name)}</div>`).join('')+state.meta.vlm_classes.filter(c=>!['normal','distress'].includes(c.name)).map(c=>`<div class="legend-row"><span class="swatch" style="background:${cssColor(c.color)}"></span>${esc(className(c.name))}</div>`).join('')+'<p>Coverage: dashed dark = not covered; dashed grey = pending. A selection circle shows GNSS uncertainty (1σ).</p>';
  }
  function className(value) { return ({distress:'Distress',normal:'Normal',unknown:'Unknown result',pending:'Awaiting classification',not_sent:'Not sent to model',not_covered:'Not covered'})[value]||'Unknown result'; }
  const imageQueue=[],imageCache=new Set();let imageActive=0;
  function queueImage(node,url,onError,priority=false) {
    if(!url){onError?.();return;}
    const job={node,url,onError};priority?imageQueue.unshift(job):imageQueue.push(job);pumpImages();
  }
  function pumpImages() {
    while(imageActive<6&&imageQueue.length){
      const job=imageQueue.shift();
      if(!job.node.isConnected&&!job.node.dataset.preload)continue;
      imageActive++;let finished=false;
      const finish=error=>{if(finished)return;finished=true;clearTimeout(timeout);imageActive--;job.node.onload=job.node.onerror=null;if(error)job.onError?.();else imageCache.add(job.url);pumpImages();};
      const timeout=setTimeout(()=>{job.node.removeAttribute('src');finish(true);},15000);
      job.node.onload=()=>finish(false);job.node.onerror=()=>finish(true);job.node.src=job.url;
    }
  }
  function preloadNext() {
    if(!state.playing)return;
    for(const feature of state.strip.slice(state.index+1,state.index+3)){
      const url=mediaURL(feature.properties.preview_url);
      if(!url||imageCache.has(url)||imageQueue.some(j=>j.url===url))continue;
      const img=new Image();img.dataset.preload='true';queueImage(img,url);
    }
  }
  function imageFallback(container,text) {
    container.replaceChildren();const placeholder=document.createElement('div');placeholder.className='image-placeholder';placeholder.textContent=text;container.append(placeholder);
  }
  function selectFeature(feature,pan=false) {
    if(state.loading||state.stale||feature?.properties?.layer!=='video')return;
    state.selected=feature;
    if(feature?.properties.layer==='video'){state.inspectSession=feature.properties.session_id;if(state.stripSession!==feature.properties.session_id){state.stripSession=feature.properties.session_id;renderStrip();}renderDriveMenu();}
    map?.select(feature,{pan,focus:pan&&!state.playing&&!state.stripOpen});renderDetail();renderStripWindow(true);preloadNext();
  }
  function datum(label,value) { return `<div class="datum"><dt>${esc(label)}</dt><dd>${esc(value??'—')}</dd></div>`; }
  function renderDetail() {
    const panel=$('detailPanel'),feature=state.selected;
    const entering=Boolean(feature)&&(panel.hidden||!document.body.classList.contains('has-selection'));
    clearTimeout(evidenceCloseTimer);clearTimeout(evidenceExpandTimer);evidenceClosing=false;panel.classList.remove('evidence-closing','evidence-expanding');panel.inert=false;
    document.body.classList.toggle('has-selection',Boolean(feature));panel.classList.toggle('expanded',state.expanded);panel.classList.toggle('compact',!state.expanded);
    if(!feature){clearTimeout(evidenceEntryTimer);panel.classList.remove('evidence-enter');panel.hidden=true;const line=$('cardConnector');if(line)line.hidden=true;panel.innerHTML='<div class="empty detail-empty"><span class="empty-symbol">⌖</span><h2>A closer look at the road</h2><p>Select a segment on the map or choose a drive to inspect its frames.</p><span class="tag">5 m segments · full-frame evidence</span></div>';map?.select(null);return;}
    panel.hidden=false;
    if(entering&&!motionPreference.matches){panel.classList.add('evidence-enter');clearTimeout(evidenceEntryTimer);evidenceEntryTimer=setTimeout(()=>panel.classList.remove('evidence-enter'),600);}
    const p=feature.properties,confidence=number(p.confidence),labels=p.final_labels||[],threshold=state.meta.confidence_threshold;
    const main=p.final_primary_label||className(p.vlm_class),src=mediaURL(p.preview_url),full=mediaURL(p.image_url),hasImage=src&&(p.has_image!==false);
    const reviewed=p.expert_reviewed,review=p.needs_expert_review&&!reviewed;
    const badge=reviewed?'Expert-reviewed':review?'Model result · awaiting expert review':['pending','not_sent','not_covered','unknown'].includes(p.vlm_class)?className(p.vlm_class):'Model result · not expert-reviewed';
    const position=number(p.centre_lat??p.latitude)!==null&&number(p.centre_lon??p.longitude)!==null?`${Number(p.centre_lat??p.latitude).toFixed(5)}, ${Number(p.centre_lon??p.longitude).toFixed(5)}`:'Position unavailable';
    const corrected=p.expert_corrected_types||p.expert_corrected_severity;
    panel.innerHTML=`<div class="detail-top"><p class="eyebrow">DRIVE-VIDEO SEGMENT</p><button id="closeDetail" class="icon-button" aria-label="Close observation">×</button></div><div class="detail-frame" id="detailFrame">${hasImage?'<img id="detailImage" alt="Road surface captured for this observation" decoding="async">':'<div class="image-placeholder">'+esc(p.vlm_class==='not_covered'?'No usable frame · not covered':p.vlm_class==='not_sent'?'Frame not retained · not sent to model':'No image available')+'</div>'}${hasImage?`<span class="frame-id">FRAME ${String(p.k).padStart(5,'0')}</span>${full?`<a class="frame-full" href="${esc(full)}" target="_blank" rel="noopener noreferrer">Full resolution ↗</a>`:''}`:''}</div><div class="detail-content"><p class="detail-kicker">${esc((p.road_name||p.road_id||'Drive survey')+' / '+D.formatDistance(p.s0_m)+' from start')}</p><h2>${esc(main)}</h2><div class="tags">${labels.filter(l=>l!==main).map(l=>`<span class="tag">${esc(l)}</span>`).join('')}<span class="tag">${esc(p.final_severity||'Unknown')} severity</span>${p.partial?'<span class="tag">Partial segment</span>':''}</div><div class="tag ${review?'review':reviewed?'confirmed':''}">${esc(badge)}</div>${corrected&&reviewed?`<p class="model-original">Expert correction · model originally: <s>${esc((p.labels||[]).join(', ')||'No label')} · ${esc(p.severity||'Unknown')}</s></p>`:''}<p class="description">${esc(p.description||(['pending','not_sent','not_covered'].includes(p.vlm_class)?className(p.vlm_class)+'. No condition diagnosis is available.':'No description supplied.'))}</p><div class="confidence-head"><span>Model confidence</span><span>${confidence===null?'Not available':percent(confidence*100)}</span></div><div class="confidence-track"><div class="confidence-fill" style="width:${confidence===null?0:Math.min(100,Math.max(0,confidence*100))}%"></div><span class="confidence-threshold" style="left:${threshold*100}%"></span></div><p class="confidence-note">${percent(threshold*100)} expert-review threshold · model confidence is not certainty</p><dl class="detail-grid">${datum('Captured · local time',D.localTime(p.local_time||p.utc))}${datum('Position uncertainty',number(p.position_sigma_m)!==null?'±'+p.position_sigma_m+' m (1σ)':'Not supplied')}${datum('Coordinates · WGS84',position)}${datum('Distance from start',D.formatDistance(p.s0_m)+' – '+D.formatDistance(p.s1_m))+datum('Gate status',p.status?.replaceAll('_',' '))+datum('Speed',number(p.camera_speed_kmh)!==null?p.camera_speed_kmh+' km/h':'—')}${number(p.alt_msl_m??p.camera_alt_msl_m)!==null?datum('Height above mean sea level',(p.alt_msl_m??p.camera_alt_msl_m)+' m'):''}${number(p.chainage_start_km)!==null?datum('Road chainage',p.chainage_start_km+' – '+p.chainage_end_km+' km'):''}</dl><details class="detail-advanced"><summary>Frame & detector details</summary><dl>${datum('Device',p.device_model)}${datum('Lane / direction',(p.lane||'—')+' / '+(p.direction||'—').replaceAll('_',' '))}${datum('Frame selection',p.quality_reason)}${datum('Sharpness',p.quality_sharpness)}${datum('Blur',number(p.quality_blur_px)!==null?p.quality_blur_px+' px':'—')}${datum('Candidates',p.quality_n_candidates)}${datum('Gate mode',p.gate_mode)}${p.gate_mode==='all'?datum('Detector would have',p.gate_decision_if_gated):''}${datum('Probe fraction',p.gate_probe_frac)}${datum('Probe maximum logit',p.gate_probe_max_logit)}${datum('Gate evidence',p.gate_evidence)}${datum('Crack score (mm)',p.gate_crack_mm)}${datum('Gate reasons',(p.gate_reasons||[]).join(', '))}${datum('Session',p.session_id)}</dl></details></div>`;
    const compact=document.createElement('div'); compact.className='compact-info';
    compact.innerHTML=`<div><strong>${esc(main)}</strong><span>${esc(p.final_severity||className(p.vlm_class))} · ${D.formatDistance(p.s0_m)} from start</span></div><button id="expandDetail" class="card-open" aria-label="${state.expanded?'Collapse':'Open'} observation details">${state.expanded?'Collapse':'View details'} <span aria-hidden="true">↗</span></button>`;
    $('detailFrame').after(compact);
    $('expandDetail').addEventListener('click',()=>{pause();state.expanded=!state.expanded;panel.classList.toggle('expanded',state.expanded);panel.classList.toggle('compact',!state.expanded);clearTimeout(evidenceExpandTimer);panel.classList.toggle('evidence-expanding',state.expanded&&!motionPreference.matches);if(state.expanded)evidenceExpandTimer=setTimeout(()=>panel.classList.remove('evidence-expanding'),400);$('expandDetail').firstChild.textContent=state.expanded?'Collapse ':'View details ';$('expandDetail').setAttribute('aria-label',state.expanded?'Collapse observation details':'Open observation details');positionCard();});
    $('closeDetail').addEventListener('click',closeSelection);
    positionCard();
    if(hasImage){const frame=$('detailFrame'),img=$('detailImage');queueImage(img,src,()=>{if(frame.isConnected&&frame.contains(img))imageFallback(frame,'Preview unavailable. Select another frame or retry later.');},true);}
  }
  function renderStrip() {
    $('stripTrack').dataset.range='';
    state.strip=state.video.filter(f=>f.properties.session_id===state.stripSession).sort((a,b)=>a.properties.k-b.properties.k);
    $('stripPanel').hidden=!state.stripSession||!state.stripOpen;
    document.body.classList.toggle('has-strip',!$('stripPanel').hidden);
    if(!state.stripSession||!state.stripOpen)return;
    const session=state.sessions.find(s=>s.session_id===state.stripSession);
    $('stripTitle').textContent=`${session?.road_name||session?.road_id||'Drive'} · ${n(state.strip.length)} matching frames`;
    state.index=Math.max(0,state.strip.findIndex(f=>D.featureId(f)===D.featureId(state.selected||{})));
    $('stripTrack').style.width=`${Math.max(0,state.strip.length*136-8)}px`;
    $('playButton').disabled=!state.strip.length;$('previousButton').disabled=!state.strip.length;$('nextButton').disabled=!state.strip.length;
    renderStripWindow();syncFloatingGeometry();setTimeout(()=>map?.invalidateSize(),0);
  }
  function renderStripWindow(reveal=false) {
    if($('stripPanel').hidden)return;
    const viewport=$('stripViewport'),width=viewport.clientWidth||700;
    if(reveal){
      state.index=Math.max(0,state.strip.findIndex(f=>D.featureId(f)===D.featureId(state.selected||{})));
      const left=state.index*136;
      if(left<viewport.scrollLeft||left+128>viewport.scrollLeft+width)viewport.scrollLeft=Math.max(0,left-width/2+64);
    }
    const start=Math.max(0,Math.floor(viewport.scrollLeft/136)-1),end=Math.min(state.strip.length,Math.ceil((viewport.scrollLeft+width)/136)+1);
    const range=start+':'+end;
    if($('stripTrack').dataset.range!==range){
    $('stripTrack').dataset.range=range;$('stripTrack').replaceChildren();
    for(let i=start;i<end;i++){
      const f=state.strip[i],p=f.properties,active=D.featureId(f)===D.featureId(state.selected||{}),button=document.createElement('button');
      button.dataset.featureId=D.featureId(f);button.className='frame-card'+(active?' active':'');button.style.left=`${i*136}px`;button.setAttribute('role','option');button.setAttribute('aria-selected',String(active));button.setAttribute('aria-label',`Segment ${p.k+1}, ${D.formatDistance(p.s0_m)} from start, ${p.final_primary_label||className(p.vlm_class)}`);button.tabIndex=active?0:-1;
      const src=mediaURL(p.thumb_url),available=src&&p.has_image!==false;
      button.innerHTML=`${available?'<img alt="Road segment thumbnail" decoding="async">':`<span class="frame-thumb">${esc(className(p.vlm_class))}</span>`}<div class="frame-class-bar" style="background:${cssColor(D.color(p,state.meta))}"></div><div class="frame-caption"><span>${D.formatDistance(p.s0_m)}</span><span class="frame-status">${esc(p.final_primary_label||className(p.vlm_class))}</span></div>`;
      button.addEventListener('click',()=>{pause();selectFeature(f,true);});$('stripTrack').append(button);
      if(available)queueImage(button.querySelector('img'),src,()=>{const img=button.querySelector('img');if(img){const fallback=document.createElement('span');fallback.className='frame-thumb';fallback.textContent='Image unavailable';img.replaceWith(fallback);}});
    }
    }
    $('stripTrack').querySelectorAll('.frame-card').forEach(button=>{const active=button.dataset.featureId===D.featureId(state.selected||{});button.classList.toggle('active',active);button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;});
    $('stripPosition').textContent=state.strip.length?`${state.index+1} / ${n(state.strip.length)}`:'0 frames';
  }
  function step(delta) {
    if(!state.strip.length||state.loading||state.stale)return;
    state.index=Math.max(0,Math.min(state.strip.length-1,state.index+delta));selectFeature(state.strip[state.index],true);
  }
  function pause(){clearTimeout(state.timer);state.playing=false;$('playButton').innerHTML='▶ <span>Play route</span>';$('playButton').setAttribute('aria-label','Play along route');}
  function play(){
    if(!state.strip.length||state.loading||state.stale)return;
    ensurePlaybackLayer();
    state.playing=true;$('playButton').innerHTML='Ⅱ <span>Pause</span>';$('playButton').setAttribute('aria-label','Pause route');
    if(state.index>=state.strip.length-1)state.index=0;
    selectFeature(state.strip[state.index],true);
    function tick(){if(!state.playing)return;if(state.index>=state.strip.length-1){pause();return;}step(1);state.timer=setTimeout(tick,1000/Number($('speedSelect').value));}
    state.timer=setTimeout(tick,1000/Number($('speedSelect').value));
  }
  async function chooseSession(id) {
    closeAllFlows();
    pause();state.inspectSession=id;state.stripSession=id;state.selected=null;state.expanded=false;
    $('stripViewport').scrollLeft=0;
    const session=state.sessions.find(s=>s.session_id===id);
    map?.fitBBox(session?.bbox||state.meta?.video?.bbox);
    if(!id)state.stripOpen=false;
    renderDetail();renderSessions();renderStrip();
  }
  function ensurePlaybackLayer(){
    if(!state.layerOptions.video&&!state.layerOptions.coverage){state.layerOptions.video=true;const input=document.querySelector('[data-layer=video]');if(input)input.checked=true;map?.setLayers(state.layerOptions);}setNav('video');
  }
  async function openReplay() {
    const id=state.inspectSession||state.filters.session||state.sessions[0]?.session_id;
    if(!id||state.stale)return;
    closeAllFlows();
    state.inspectSession=id;state.stripSession=id;state.stripOpen=true;state.expanded=false;
    ensurePlaybackLayer();
    // A large catalogue is intentionally scoped; selecting a drive for replay fetches it explicitly.
    if(!state.video.some(f=>f.properties.session_id===id)){
      state.filters.session=id;syncFilterForm();await load();
      if(state.stale||state.loading)return;
    }
    renderDriveMenu();renderStrip();
    const session=state.sessions.find(s=>s.session_id===id);map?.fitBBox(session?.bbox);
    if(state.strip.length)selectFeature(state.strip[0],true);
  }
  function clearFilters(){state.filters={...defaults(),threshold:state.meta?.confidence_threshold};syncFilterForm();load();}
  function classColor(name){return cssColor(name==='distress'?'#a84235':state.meta.vlm_classes.find(c=>c.name===name)?.color);}
  function labelTable(labels){return `<div class="table-wrap"><table><thead><tr><th>Final label</th><th>Segments</th><th>Length</th><th>% surveyed</th></tr></thead><tbody>${(labels||[]).filter(l=>l.length_m>0).map(l=>`<tr><td>${esc(l.label)}</td><td>${n(l.segments)}</td><td>${D.formatDistance(l.length_m)}</td><td>${percent(l.pct_of_surveyed)}</td></tr>`).join('')}</tbody></table></div>`;}
  function openInsights(){
    if(!state.summary||state.loading||state.stale)return;
    const reportSession=state.inspectSession||state.filters.session;
    const sessions=reportSession?state.sessions.filter(s=>s.session_id===reportSession):state.sessions;
    $('insightsTitle').textContent='Road condition by kilometre';
    let html=`<section class="report-section"><h3>Current filter · ${D.formatDistance(state.summary.surveyed_m)} surveyed</h3><p>${percent(state.summary.pct_distress_of_surveyed)} distressed · ${D.formatDistance(state.summary.not_covered_m)} not covered · ${n(state.summary.needs_expert_review)} awaiting expert review</p><details class="report-labels"><summary>View all label percentages for the current filter</summary>${labelTable(state.summary.labels)}</details></section><p class="report-callout">The per-km profiles below describe each complete drive. Date, label, severity and review filters apply to the summary above. Distress labels can overlap; the coverage bars do not double-count road length.</p>`;
    for(const session of sessions){
      html+=`<section class="report-section"><h3>${esc(session.road_name||session.road_id)} · ${esc(session.local_date)}</h3><p class="report-subtitle">${D.formatDistance(session.surveyed_m)} surveyed · ${D.formatDistance(session.classified_m)} classified · ${percent(session.pct_distress_of_surveyed)} distressed</p>`;
      for(const bin of session.per_km||[]){const classes=D.perKmCoverage(bin);html+=`<div class="km-row"><div class="km-row-head"><strong>${D.formatDistance(bin.from_m)} – ${D.formatDistance(bin.to_m)}</strong><span>${percent(bin.pct_distress_of_surveyed)} distress · ${D.formatDistance(bin.surveyed_m)} surveyed</span></div><div class="km-stack" role="img" aria-label="${esc(classes.filter(c=>c.metres).map(c=>className(c.name)+': '+D.formatDistance(c.metres)).join(', '))}">${classes.map(c=>`<span style="width:${c.fraction*100}%;background:${classColor(c.name)}" title="${esc(className(c.name))}: ${D.formatDistance(c.metres)}"></span>`).join('')}</div></div>`;}
      const labels=state.meta.labels.filter(l=>l.is_distress&&(session.per_km||[]).some(b=>(b.labels_m?.[l.name]||0)>0));
      html+='<div class="km-legend">'+state.meta.vlm_classes.filter(c=>(session.per_km||[]).some(b=>D.perKmCoverage(b).some(v=>v.name===c.name&&v.metres>0))).map(c=>`<span><span class="swatch" style="background:${classColor(c.name)}"></span> ${esc(className(c.name))}</span>`).join('')+'</div>';
      html+=`<div class="table-wrap"><table><caption class="report-subtitle">Distress length by kilometre · metres (% of surveyed bin)</caption><thead><tr><th>Label</th>${(session.per_km||[]).map(b=>`<th>km ${b.km+1}</th>`).join('')}</tr></thead><tbody>${labels.map(l=>`<tr><td>${esc(l.name)}</td>${(session.per_km||[]).map(b=>`<td>${Number(b.labels_m?.[l.name]||0).toFixed(1)} (${percent(b.surveyed_m?(b.labels_m?.[l.name]||0)/b.surveyed_m*100:0)})</td>`).join('')}</tr>`).join('')}</tbody></table></div><details class="report-labels"><summary>Drive totals by label</summary>${labelTable(session.labels)}</details></section>`;
    }
    $('insightsBody').innerHTML=html;openFlow('insightsDialog');
  }
  $('sessions').addEventListener('click',event=>{const button=event.target.closest('[data-session]');if(button)chooseSession(button.dataset.session);});
  $('allDrives').addEventListener('click',()=>chooseSession(''));
  $('filterButton').addEventListener('click',()=>{if(!state.meta){notice('Survey filters will be available when the data connection returns.',true);return;}syncFilterForm();$('filterError')?.remove();toggleFlow('filterDialog');});
  $('filterForm').addEventListener('submit',event=>{event.preventDefault();const data=new FormData(event.currentTarget);const filters={...state.filters,label:data.getAll('label'),severity:data.getAll('severity'),date_from:data.get('date_from'),date_to:data.get('date_to'),review:data.get('review'),vlm_class:data.get('vlm_class'),status:data.get('status'),session:data.get('session')||'',highConfidence:data.has('highConfidence')};try{D.params(filters);state.filters=filters;closeFlow('filterDialog');renderActiveFilters();load();}catch(error){let el=$('filterError');if(!el){el=document.createElement('p');el.id='filterError';el.className='filter-error';el.setAttribute('role','alert');$('filterForm').querySelector('.dialog-footer').before(el);}el.textContent=error.message;}});
  $('resetFilters').addEventListener('click',()=>{state.filters={...defaults(),threshold:state.meta?.confidence_threshold};syncFilterForm();closeFlow('filterDialog');load();});
  $('clearEmpty').addEventListener('click',clearFilters);
  $('aboutButton').addEventListener('click',event=>{event.preventDefault();toggleFlow('aboutDialog');});
  document.querySelectorAll('[data-close]').forEach(button=>button.addEventListener('click',()=>closeFlow(button.dataset.close)));
  $('themeButton').addEventListener('click',()=>setTheme(theme==='dark'?'light':'dark'));
  $('retryButton').addEventListener('click',()=>state.ready?load():boot());
  $('fitButton').addEventListener('click',()=>{closeSelection();map?.fitBBox(state.meta?.video?.bbox);});
  document.querySelectorAll('[data-layer]').forEach(input=>input.addEventListener('change',()=>{if(!(input.dataset.layer in state.layerOptions)||input.dataset.layer==='citizen')return;state.layerOptions[input.dataset.layer]=input.checked;map?.setLayers(state.layerOptions);if(state.selected&&!state.layerOptions.video&&!state.layerOptions.coverage)closeSelection();if(input.dataset.layer==='coverage')load();}));
  $('stripViewport').addEventListener('scroll',()=>renderStripWindow());
  $('stripViewport').addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight'].includes(event.key))return;event.preventDefault();pause();step(event.key==='ArrowLeft'?-1:1);$('stripViewport').focus();});
  $('previousButton').addEventListener('click',()=>{pause();step(-1);});$('nextButton').addEventListener('click',()=>{pause();step(1);});
  $('playButton').addEventListener('click',()=>state.playing?pause():play());
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  $('insightsButton').addEventListener('click',openInsights);
  $('mobileInsights').addEventListener('click',openInsights);
  $('printButton').addEventListener('click',()=>{openInsights();if($('insightsDialog').open)window.print();});
  $('exportVideo').addEventListener('click',()=>{if(!state.meta||state.loading||state.stale)return;const url=api('/gis/video/segments.csv?')+D.params(state.filters,'video',{limit:20000});if(state.videoTotal>20000){notice('Narrow the drive or date filters to at most 20,000 segments before exporting a complete CSV.');return;}window.open(url,'_blank','noopener,noreferrer');});
  function syncFloatingGeometry(){
    const heading=document.querySelector('.dashboard-heading').getBoundingClientRect();
    document.body.style.setProperty('--overview-heading-bottom',Math.round(heading.bottom)+'px');
    const strip=$('stripPanel').getBoundingClientRect();
    if(!$('stripPanel').hidden&&strip.height){
      document.body.style.setProperty('--replay-height',Math.ceil(strip.height)+'px');
      document.body.style.setProperty('--replay-bottom',Math.max(0,innerHeight-strip.bottom)+'px');
    }
    positionCard();
    for(const entry of flowCards.values())if(entry.isOpen)positionFlow(entry);
    if(state.selected&&!state.expanded&&innerWidth<=820)map?.keepSelectionVisible?.();
  }
  window.addEventListener('resize',()=>{renderStripWindow();map?.invalidateSize();syncFloatingGeometry();syncNavSlider();});
  function setNav(name){
    activeNav=name;
    const nav=document.querySelector('.nav-pill');if(nav)nav.dataset.active=name;
    for(const [id,key] of [['videoNav','video'],['surveyNav','survey']]){
      $(id)?.classList.toggle('active',key===name);
      $(id)?.setAttribute('aria-pressed',String(key===name));
    }
    syncNavSlider();
  }
  function closeSelection(){
    pause();state.selected=null;state.expanded=false;map?.select(null);renderStripWindow();
    const panel=$('detailPanel');if(evidenceClosing)return;
    if(!panel.hidden&&document.body.classList.contains('has-selection')&&!state.loading&&!state.stale&&!motionPreference.matches){
      evidenceClosing=true;clearTimeout(evidenceEntryTimer);panel.classList.remove('evidence-enter');panel.classList.add('evidence-closing');panel.inert=true;
      if($('cardConnector'))$('cardConnector').hidden=true;
      evidenceCloseTimer=setTimeout(()=>{if(!state.selected)renderDetail();},270);
    }else renderDetail();
  }
  function renderDriveMenu(){
    const select=$('driveSelect');if(!select)return;
    const total=state.sessions.reduce((sum,drive)=>sum+(Number(drive.route_m)||0),0);
    const [distance,unit='']=D.formatDistance(total).split(' ');
    $('totalDistance').innerHTML=`<span class="distance-reading"><span class="distance-value">${esc(distance)}</span> <span class="distance-unit">${esc(unit)}</span></span> <span class="distance-label">surveyed routes</span>`;
    select.innerHTML='<option value="">All '+state.sessions.length+' drives</option>'+state.sessions.map(s=>`<option value="${esc(s.session_id)}">${esc(s.road_name||s.road_id||'Drive')} · ${esc(s.local_date)} · ${D.formatDistance(s.route_m)}</option>`).join('');
    select.value=state.inspectSession;select.disabled=!state.sessions.length;
    $('sessions').querySelectorAll('[data-session]').forEach(button=>{const active=button.dataset.session===state.inspectSession;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));});
    const session=state.sessions.find(s=>s.session_id===state.inspectSession);
    $('mapScope').textContent=session?`${session.road_name||session.road_id} · ${session.local_date}`:'All drive surveys';
    $('driveDetails').textContent=session?`${n(session.segments)} segments · ${percent(session.pct_distress_of_surveyed)} distressed · ${['stored','cleaned'].includes(session.state)?'Complete':session.state==='failed'?'Failed':'Processing'}${state.filters.session&&state.filters.session!==session.session_id?' · Not in current filter':''}`:`${n(state.sessions.length)} recorded drives · all routes visible`;
    $('driveFocus').textContent=session?'Focus drive':'Show all routes';
    $('driveFocus').disabled=!state.sessions.length;$('driveReplay').disabled=!state.sessions.length||state.stale;
    const filter=$('sessionFilter');if(filter){filter.innerHTML='<option value="">All drives</option>'+state.sessions.map(s=>`<option value="${esc(s.session_id)}">${esc(s.road_name||s.road_id||'Drive')} · ${esc(s.local_date)} · ${D.formatDistance(s.route_m)}</option>`).join('');filter.value=state.filters.session;}
  }
  function positionCard(){
    const panel=$('detailPanel'),connector=$('cardConnector');
    if(!state.selected||panel.hidden){if(connector)connector.hidden=true;return;}
    if(innerWidth<=820){panel.style.removeProperty('left');panel.style.removeProperty('top');panel.style.removeProperty('max-height');if(connector)connector.hidden=true;return;}
    const bounds=$('surveyMap').getBoundingClientRect();
    const anchor=map?.anchorFor?.(state.selected)||{x:bounds.left+bounds.width*.52,y:bounds.top+bounds.height*.5};
    const topLimit=bounds.top+16,bottomLimit=bounds.bottom-16;
    const stripTop=state.stripOpen&&!state.expanded&&!$('stripPanel').hidden?Math.min(bottomLimit,$('stripPanel').getBoundingClientRect().top-12):bottomLimit;
    const width=Math.min(parseFloat(getComputedStyle(panel).getPropertyValue('--evidence-target-width'))||panel.offsetWidth||320,bounds.width-32);
    const wantedHeight=state.expanded?panel.scrollHeight||480:panel.offsetHeight||120;
    const clampLeft=value=>Math.max(bounds.left+16,Math.min(bounds.right-width-16,value));
    const preferRight=anchor.x+42+width<bounds.right-16;
    const preferredLeft=clampLeft(preferRight?anchor.x+42:anchor.x-width-42);
    const desiredTop=anchor.y-wantedHeight*.4;
    // The map fills the viewport; evidence must use the gaps between floating controls.
    const obstacles=[...document.querySelectorAll('.topbar,.dashboard-heading,.dashboard-stats,#activeFilters,#driveControl,.toolbar,.legend-control,.map-tools,.map-status,.leaflet-control-zoom')].flatMap(element=>{
      const style=getComputedStyle(element),rect=element.getBoundingClientRect();
      return element.hidden||style.display==='none'||style.visibility==='hidden'||!rect.width||!rect.height?[]:[rect];
    });
    // Keep the selected vehicle visible beside its evidence, even on narrower desktops.
    obstacles.push({left:anchor.x-18,right:anchor.x+18,top:anchor.y-18,bottom:anchor.y+18});
    const placements=[];
    for(const left of new Set([preferredLeft,clampLeft(anchor.x+42),clampLeft(anchor.x-width-42),bounds.left+16,bounds.right-width-16])){
      let gaps=[[topLimit,stripTop]];
      for(const rect of obstacles){
        if(left+width<=rect.left-12||left>=rect.right+12)continue;
        gaps=gaps.flatMap(([start,end])=>rect.bottom+12<=start||rect.top-12>=end?[[start,end]]:[[start,Math.min(end,rect.top-12)],[Math.max(start,rect.bottom+12),end]].filter(([a,b])=>b>a));
      }
      for(const [start,end] of gaps){
        if(end-start<Math.min(wantedHeight,state.expanded?180:120))continue;
        const height=Math.min(wantedHeight,end-start),top=Math.max(start,Math.min(end-height,desiredTop));
        placements.push({left,top,height,maxHeight:end-start,score:(wantedHeight-height)*3+Math.abs(top-desiredTop)+Math.abs(left-preferredLeft)*.3});
      }
    }
    const best=placements.sort((a,b)=>a.score-b.score)[0]||{left:preferredLeft,top:Math.max(topLimit,Math.min(stripTop-wantedHeight,desiredTop)),height:Math.min(wantedHeight,stripTop-topLimit),maxHeight:Math.max(120,stripTop-topLimit)};
    const {left,top,height}=best,rightSide=left>=anchor.x;
    panel.style.maxHeight=best.maxHeight+'px';panel.style.left=left+'px';panel.style.top=top+'px';
    if(connector){const line=connector.querySelector('line');connector.hidden=false;line?.setAttribute('x1',anchor.x);line?.setAttribute('y1',anchor.y);line?.setAttribute('x2',rightSide?left:left+width);line?.setAttribute('y2',top+Math.min(height/2,130));}
  }
  function cacheResults(video,coverage,geom){
    const key=JSON.stringify({...state.filters,geom});
    state.cache=state.cache.filter(item=>item.key!==key);
    state.cache.push({key,filters:JSON.parse(JSON.stringify(state.filters)),video,coverage,geom});
    if(state.cache.length>8)state.cache.shift();
  }
  function previewCached(){
    if(!D.matches)return;
    // Only use a cache whose result set contains the new filter. A broadening request
    // keeps the old map until the authoritative response arrives.
    const compatible=(old,next)=>{
      for(const key of ['session','status','vlm_class','date_from','date_to'])if(old[key]&&old[key]!==next[key])return false;
      for(const key of ['label','severity'])if(old[key]?.length&&(!next[key]?.length||!next[key].every(v=>old[key].includes(v))))return false;
      if(old.review!=='all'&&old.review!==next.review)return false;
      if(old.highConfidence&&!next.highConfidence)return false;
      return true;
    };
    const cache=[...state.cache].reverse().find(c=>!c.video.truncated&&compatible(c.filters,state.filters));
    if(!cache)return;
    const video=cache.video.features.filter(D.matcher(state.filters,state.meta,'video'));
    const coverage=(cache.coverage?.features||[]).filter(D.matcher(state.filters,state.meta,'video'));
    map?.setData(video,[],coverage);
    if(state.selected&&!video.some(f=>D.featureId(f)===D.featureId(state.selected)))closeSelection();
  }
  $('videoNav')?.addEventListener('click',()=>{closeAllFlows();setNav('video');map?.invalidateSize();});
  $('surveyNav')?.addEventListener('click',()=>toggleFlow('overviewDialog'));
  $('driveSelect')?.addEventListener('change',event=>chooseSession(event.target.value));
  $('driveFocus')?.addEventListener('click',()=>{closeFlow('drivePanel');const s=state.sessions.find(s=>s.session_id===state.inspectSession);map?.fitBBox(s?.bbox||state.meta?.video?.bbox);});
  $('driveReplay')?.addEventListener('click',openReplay);
  $('closeStrip')?.addEventListener('click',()=>{pause();state.stripOpen=false;renderStrip();positionCard();});
  registerFlow('overviewDialog','surveyNav',{anchorId:'driveControl'});registerFlow('filterDialog','filterButton',{align:'end'});registerFlow('aboutDialog','aboutButton');
  registerFlow('settingsPanel','settingsButton',{hostId:'settingsControl',align:'end'});registerFlow('exportPanel','exportButton',{hostId:'exportControl',align:'end'});
  registerFlow('drivePanel','driveToggle');registerFlow('legend','legendButton',{hostId:'legendControl',align:'end',above:true});
  registerFlow('insightsDialog','insightsButton',{modal:true,returnId:'surveyNav'});
  $('driveToggle').addEventListener('click',()=>toggleFlow('drivePanel'));
  document.addEventListener('pointerdown',event=>{for(const [id,entry] of flowCards)if(entry.isOpen&&!entry.modal&&!entry.panel.contains(event.target)&&!entry.trigger.contains(event.target))closeFlow(id,{restore:false});});
  document.addEventListener('focusin',event=>{for(const [id,entry] of flowCards)if(entry.isOpen&&!entry.modal&&!entry.panel.contains(event.target)&&!entry.trigger.contains(event.target))closeFlow(id,{restore:false});});
  document.addEventListener('keydown',event=>{if(event.key!=='Escape'||event.defaultPrevented)return;const entry=[...flowCards.entries()].find(([,entry])=>entry.isOpen);if(entry){event.preventDefault();closeFlow(entry[0]);}else{closeSelection();document.querySelectorAll('.detail-advanced[open],.report-labels[open]').forEach(el=>el.open=false);}});
  document.fonts?.ready.then(()=>{syncNavSlider();syncFloatingGeometry();});
  if(window.ResizeObserver){const observer=new ResizeObserver(()=>{syncNavSlider();syncFloatingGeometry();});for(const element of [document.querySelector('.nav-pill'),document.querySelector('.dashboard-heading'),$('stripPanel')])observer.observe(element);}
  const loader=document.createElement('div');loader.id='initialLoader';loader.className='initial-loader';loader.setAttribute('role','status');loader.innerHTML='<div class="loader-brand"><span class="brand-geo">GEO</span><span class="brand-ai">AI</span></div><p>Preparing survey map…</p>';document.body.append(loader);document.body.classList.add('initial-loading');
  setNav('video');updateExports();readURL();boot();
})();
