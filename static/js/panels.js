// Result panels: summary metrics, room/space/opening tables, warnings, raw JSON.
// Everything rendered here is read directly from the /api/analyze response.

import { $, el, fmtNumber, fmtPercent, fmtSeconds } from './dom.js';
import { FUNCTION_COLORS, functionOf, isEstimateMethod } from './viewer.js';

const METHOD_LABELS = {
  'wall-region': 'Wall region',
  'independent-wall-region': 'Wall region (secondary pass)',
  'dimension-box-wall-snapped': 'Dimension box, wall-snapped',
  'dimension-box-estimate': 'Dimension box estimate',
  'dimension-only-estimate': 'Dimension-only estimate',
  'wall-ray-estimate': 'Wall-ray estimate',
  'passage-partition': 'Wall region, split at a passage',
  'open-plan-shared': 'Open-plan area (shared)',
  'merged-space': 'Shared space (rooms not separated)',
  'open-plan-zone': 'Open-plan zone (estimated extent)',
  'merged-space-zone': 'Zone of an unseparated space (estimate)',
};
const AREA_SOURCES = {
  'printed-dimensions': 'from printed dimensions',
  'pixel-scale-estimate': 'pixel-scale estimate',
  'polygon-pixels': 'polygon pixels',
  'zone-estimate': 'zone estimate',
  unavailable: 'unavailable',
};
const OPENING_TYPES = { door: 'Door', window: 'Window', opening: 'Unclassified' };
const RELATIONS = {
  between_rooms: 'between rooms',
  room_to_exterior: 'room ↔ exterior',
  same_room: 'same room',
  unknown: 'unknown',
};

function metric(label, value, { sub, wide, na, dot } = {}) {
  const dt = el('dt', {}, dot ? el('i', { class: 'dot', style: { background: dot } }) : null, label);
  const dd = el('dd', {}, String(value), sub ? el('small', { text: sub }) : null);
  return el('div', { class: `metric${wide ? ' is-wide' : ''}${na ? ' is-na' : ''}` }, dt, dd);
}

export function renderMetrics(data, elapsedMs) {
  const scale = data.pixel_scale;
  $('metrics-source').textContent = data.source_name || '';
  $('metrics').replaceChildren(
    metric('Rooms', data.room_count, { sub: 'labeled by OCR' }),
    metric('Unlabeled spaces', data.unlabeled_space_count),
    metric('Doors', data.door_count ?? 0, { dot: 'var(--door)' }),
    metric('Windows', data.window_count ?? 0, { dot: 'var(--window)' }),
    metric('Unclassified', data.unclassified_opening_count ?? 0, { dot: 'var(--opening)', sub: 'openings' }),
    metric('Openings total', data.opening_count ?? (data.openings || []).length),
    metric('Functional zones', data.zone_count ?? 0, { sub: 'inside structural spaces' }),
    metric('Objects', data.object_count ?? 0, { sub: 'furniture · fixtures' }),
    metric('Scale', scale ? `${fmtNumber(scale.pixels_per_unit)} px/${scale.unit}`
      : data.scale_estimate ? `≈ ${fmtNumber(data.scale_estimate.px_per_cm)} px/cm` : 'Not determined', {
      sub: scale ? `calibrated from ${scale.calibration_rooms} rooms`
        : data.scale_estimate ? `estimate: ${data.scale_estimate.source}` : undefined,
      na: !scale && !data.scale_estimate,
    }),
    metric('Processing', fmtSeconds(elapsedMs), { sub: 'round trip incl. upload' }),
    metric('Image', `${data.image.width} × ${data.image.height}`, { sub: 'pixels', wide: true }),
  );
  $('metrics-card').hidden = false;
}

function boundaryChip(boundary, kind) {
  if (!boundary) return [el('span', { class: 'chip chip-none', text: 'No boundary' })];
  const estimate = kind === 'space'
    ? boundary.method === 'dimension-only-estimate'
    : isEstimateMethod(boundary.method);
  const chipClass = estimate ? 'chip-estimate' : kind === 'space' ? 'chip-space' : 'chip-room';
  return [
    el('span', { class: `chip ${chipClass}`, text: METHOD_LABELS[boundary.method] || boundary.method }),
    el('span', { class: 'cell-note', text: `${fmtPercent(boundary.confidence)}${boundary.wall_edges_snapped ? ` · ${boundary.wall_edges_snapped} edges snapped` : ''}` }),
  ];
}

function dimensionsText(dimensions) {
  if (!dimensions) return null;
  return dimensions.text || dimensions.display || `${dimensions.width} × ${dimensions.height} ${dimensions.unit}`;
}

// Area (main) with the printed dimensions and the area source underneath.
function areaCell(area, dimensions) {
  const dims = dimensionsText(dimensions);
  const ocr = dimensions?.ocr_confidence !== undefined ? ` · OCR ${fmtNumber(dimensions.ocr_confidence, 0)}%` : '';
  const hasArea = area && area.value !== null && area.value !== undefined;
  return el('td', { class: 'num' },
    el('span', { class: 'cell-main', text: hasArea ? `${fmtNumber(area.value)} ${area.unit}` : '—' }),
    dims ? el('span', { class: 'cell-sub', text: `${dims}${ocr}` }) : null,
    hasArea ? el('span', { class: 'cell-sub', text: AREA_SOURCES[area.source] || area.source }) : null,
  );
}

function linkedRow(key, cells, handlers) {
  return el('tr', {
    'data-key': key,
    onmouseenter: () => handlers.onHover(key, 'table'),
    onmouseleave: () => handlers.onHover(null),
    onclick: () => handlers.onSelect(key, 'table'),
  }, cells);
}

export function renderRooms(data, handlers) {
  const rooms = data.rooms || [];
  $('rooms-count').textContent = rooms.length;
  $('rooms-table').tBodies[0].replaceChildren(...rooms.map((room) => linkedRow(`room:${room.id}`, [
    el('td', { class: 'mono id-cell', text: room.id }),
    el('td', {},
      el('span', { class: 'cell-main', text: room.name || 'Unnamed' }),
      el('span', { class: 'cell-sub', dir: 'auto', text: `“${room.label_text ?? ''}” · OCR ${fmtNumber(room.label_confidence, 0)}%` }),
      el('span', { class: 'cell-chips' }, ...boundaryChip(room.boundary, 'room')),
    ),
    areaCell(room.area, room.dimensions),
  ], handlers)));
  if (!rooms.length) {
    $('rooms-table').tBodies[0].append(el('tr', {}, el('td', { colspan: 3, class: 'cell-sub', text: 'No labeled rooms were identified.' })));
  }

  const spaces = data.unlabeled_spaces || [];
  $('spaces-section').hidden = spaces.length === 0;
  $('spaces-count').textContent = spaces.length;
  $('spaces-table').tBodies[0].replaceChildren(...spaces.map((space) => linkedRow(`space:${space.id}`, [
    el('td', { class: 'mono id-cell', text: space.id }),
    el('td', {},
      el('span', { class: 'cell-chips' }, ...boundaryChip(space.boundary, 'space')),
      spaceContents(space)),
    areaCell(space.area, space.dimensions),
  ], handlers)));
}

// What an unnamed structural space holds: functional zones, or one function inferred from its contents.
function spaceContents(space) {
  if (space.zone_ids?.length) {
    return el('span', { class: 'cell-sub', text: `open plan: ${space.zone_ids.length} functional zones${space.circulation_m2 ? ` · ${fmtNumber(space.circulation_m2, 1)} m² circulation` : ''}` });
  }
  const fn = space.function;
  if (!fn) return null;
  return el('span', { class: 'cell-sub', text: `probably ${fn.function} (${fmtPercent(fn.confidence)}) · ${fn.evidence.map((e) => e.kind).join(', ')}` });
}

const fnChip = (fn) => el('span', { class: 'chip chip-fn', style: `--fn:${FUNCTION_COLORS[fn] || '#666'}`, text: fn || 'unknown' });

export function renderZones(data, handlers) {
  const zones = data.zones || [];
  $('zones-section').hidden = zones.length === 0;
  $('zones-count').textContent = zones.length;
  $('zones-table').tBodies[0].replaceChildren(...zones.map((zone) => {
    const key = zone.room_ids?.length ? `room:${zone.room_ids[0]}` : `zone:${zone.id}`;
    const evidence = zone.evidence.map((e) => (e.source === 'label' ? `label “${e.id}”` : e.kind)).join(', ');
    const borders = (zone.boundaries || []).filter((b) => b.to !== 'circulation')
      .map((b) => `${b.to.split('.').pop()}: ${b.cue}`).join(' · ');
    return linkedRow(key, [
      el('td', { class: 'mono id-cell', text: zone.id }),
      el('td', {},
        el('span', { class: 'cell-chips' }, fnChip(zone.function),
          el('span', { class: 'cell-note', text: `${fmtPercent(zone.confidence)} · in ${zone.space_id}${zone.room_ids?.length ? ` · room ${zone.room_ids.join(', ')}` : ''}` })),
        el('span', { class: 'reason', title: evidence, text: evidence }),
        borders ? el('span', { class: 'cell-sub', text: `boundaries ${borders}` }) : null,
      ),
      el('td', { class: 'num' }, el('span', { class: 'cell-main', text: `${fmtNumber(zone.area_m2, 1)} m²` }),
        el('span', { class: 'cell-sub', text: 'zone estimate' })),
    ], handlers);
  }));
}

export function renderObjects(data, handlers) {
  const objects = data.objects || [];
  $('objects-section').hidden = objects.length === 0;
  $('objects-count').textContent = objects.length;
  $('objects-table').tBodies[0].replaceChildren(...objects.map((obj) => linkedRow(`object:${obj.id}`, [
    el('td', { class: 'mono id-cell', text: obj.id }),
    el('td', {},
      el('span', { class: 'cell-chips' }, el('span', { class: 'cell-main', text: obj.kind }), fnChip(functionOf(obj.functions)),
        el('span', { class: 'cell-note', text: fmtPercent(obj.confidence) })),
      el('span', { class: 'reason', text: (obj.evidence || []).join('; ') }),
    ),
    el('td', { class: 'num' }, el('span', { class: 'cell-sub', text: [obj.space_id, obj.zone_id?.split('.').pop()].filter(Boolean).join(' · ') || '—' })),
  ], handlers)));
}

export function renderOpenings(data, handlers) {
  const openings = data.openings || [];
  $('openings-count').textContent = openings.length;
  $('openings-empty').hidden = openings.length > 0;
  $('openings-table').hidden = openings.length === 0;
  $('openings-table').tBodies[0].replaceChildren(...openings.map((opening) => {
    const evidence = opening.evidence || {};
    const relation = RELATIONS[evidence.room_relation] || evidence.room_relation;
    return linkedRow(`opening:${opening.id}`, [
      el('td', { class: 'mono id-cell', text: opening.id }),
      el('td', {},
        el('span', { class: 'cell-chips' },
          el('span', { class: `chip chip-${opening.type}`, text: OPENING_TYPES[opening.type] || opening.type }),
          el('span', { class: 'cell-note', text: [opening.orientation, relation].filter(Boolean).join(' · ') }),
        ),
        el('span', { class: 'reason', dir: 'auto', title: evidence.reason || '', text: evidence.reason || '—' }),
        el('span', { class: 'cell-sub', text: `frame marks ${evidence.frame_marks ?? '—'} · swing ${fmtNumber(evidence.door_swing_score)}` }),
      ),
      el('td', { class: 'num' },
        el('span', { class: 'cell-main', text: opening.width_display || `${opening.width_pixels} px` }),
        opening.width_display ? el('span', { class: 'cell-sub', text: `${opening.width_pixels} px` }) : null,
        el('span', { class: 'cell-conf' },
          el('span', { class: 'bar' }, el('span', { style: { width: fmtPercent(opening.confidence) } })),
          fmtPercent(opening.confidence),
        ),
        el('span', { class: 'cell-sub', text: `geometry ${fmtPercent(opening.geometry_confidence)}` }),
      ),
    ], handlers);
  }));
}

export function renderWarnings(warnings) {
  const list = warnings || [];
  $('warnings-card').hidden = list.length === 0;
  $('warnings-count').textContent = list.length;
  $('warning-list').replaceChildren(...list.map((text) => el('li', { dir: 'auto', text })));
}

let jsonUrl = null;
export function renderRaw(data) {
  $('raw-json').textContent = JSON.stringify(data, null, 2);
  if (jsonUrl) URL.revokeObjectURL(jsonUrl);
  jsonUrl = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const base = (data.source_name || 'floorplan').replace(/\.[^.]+$/, '');
  $('download-json').href = jsonUrl;
  $('download-json').download = `${base}-analysis.json`;
  $('download-rooms-overlay').href = data.overlay_url;
  $('download-rooms-overlay').download = `${base}-rooms-overlay.png`;
  $('download-openings-overlay').href = data.openings_overlay_url;
  $('download-openings-overlay').download = `${base}-openings-overlay.png`;
}

// Reflect the viewer's highlighted entity in the tables.
export function markActiveRow(key, pinnedKey) {
  for (const row of document.querySelectorAll('.data-table tr[data-key]')) {
    row.classList.toggle('is-active', row.dataset.key === key);
    row.classList.toggle('is-pinned', row.dataset.key === pinnedKey);
  }
}

// Scroll the results panel (not the page) so the row for `key` is visible.
export function revealRow(key) {
  const row = document.querySelector(`.data-table tr[data-key="${CSS.escape(key)}"]`);
  const scroller = $('results-scroll');
  if (!row || !scroller) return;
  const top = row.getBoundingClientRect().top - scroller.getBoundingClientRect().top + scroller.scrollTop;
  const sticky = 40; // section header height
  if (top - sticky < scroller.scrollTop || top + row.offsetHeight > scroller.scrollTop + scroller.clientHeight) {
    const target = Math.max(0, top - sticky - scroller.clientHeight / 3);
    // Long jumps are instant; short ones glide so the eye can follow.
    const behavior = Math.abs(target - scroller.scrollTop) > 2 * scroller.clientHeight ? 'auto' : 'smooth';
    scroller.scrollTo({ top: target, behavior });
  }
}

export function setNavCounts(data) {
  $('nav-rooms-count').textContent = (data.rooms || []).length + (data.unlabeled_spaces || []).length;
  $('nav-openings-count').textContent = (data.openings || []).length;
  $('nav-zones').hidden = !(data.zones || []).length;
  $('nav-zones-count').textContent = (data.zones || []).length;
  $('nav-objects').hidden = !(data.objects || []).length;
  $('nav-objects-count').textContent = (data.objects || []).length;
  $('results-nav').hidden = false;
}
