// Floor-plan viewer. The original plan is always the base layer; detections are
// drawn on top as translucent SVG built from the /api/analyze JSON (rooms as
// filled outlines, openings as see-through bands over the wall), so walls, text
// and door/window symbols stay visible. The server-rendered overlay PNGs are
// only used when the browser cannot display the original (PDF, TIFF, size
// mismatch) and remain available as downloads.

import { el, svg } from './dom.js';

const VIEW_TITLES = {
  original: ['Original', 'uploaded image'],
  rooms: ['Rooms', 'detected rooms over the original'],
  openings: ['Openings', 'detected openings over the original'],
  combined: ['Combined', 'all detections over the original'],
  cleaned: ['Cleaned plan', 'architectural skeleton the recognition ran on, with the recognised rooms'],
  elements: ['Elements', 'kept: walls black · doors green · windows orange · suppressed: grey'],
};
const CLEANING_VIEWS = { cleaned: 'cleaned', elements: 'elements' };
const FALLBACK_SUBTITLES = {
  rooms: 'server overlay',
  openings: 'server overlay',
  combined: 'drawn over the rooms overlay',
};

const OPENING_TYPES = { door: 'Door', window: 'Window', opening: 'Unclassified' };

export function isEstimateMethod(method) {
  return method !== 'wall-region' && method !== 'independent-wall-region';
}

// Functional zones of a shared physical space (open plan, or rooms the walls did not separate):
// estimated extents inside the space, drawn dashed; the physical space itself is drawn once.
export function isZoneMethod(method) {
  return method === 'open-plan-zone' || method === 'merged-space-zone';
}

// Build a lookup of every drawable entity in the API response, keyed so that
// table rows and SVG shapes can refer to the same thing.
export function indexGeometry(data) {
  const entries = new Map();
  for (const room of data.rooms || []) {
    const space = room.physical_space;
    if (space?.polygon?.length && !entries.has(`physical:${space.id}`)) {
      entries.set(`physical:${space.id}`, {
        kind: 'physical',
        points: space.polygon.map((p) => [p.x, p.y]),
        merged: space.kind === 'merged',
        label: space.kind === 'merged' ? 'Rooms not separated by walls' : 'Open-plan space',
        tag: space.id,
      });
    }
  }
  for (const room of data.rooms || []) {
    if (!room.boundary?.polygon?.length) continue;
    entries.set(`room:${room.id}`, {
      kind: 'room',
      points: room.boundary.polygon.map((p) => [p.x, p.y]),
      estimate: isEstimateMethod(room.boundary.method),
      zone: isZoneMethod(room.boundary.method),
      label: room.name || room.id,
      tag: room.id,
    });
  }
  for (const space of data.unlabeled_spaces || []) {
    if (!space.boundary?.polygon?.length) continue;
    entries.set(`space:${space.id}`, {
      kind: 'space',
      points: space.boundary.polygon.map((p) => [p.x, p.y]),
      estimate: space.boundary.method === 'dimension-only-estimate',
      label: space.id,
      tag: space.id,
    });
  }
  for (const opening of data.openings || []) {
    entries.set(`opening:${opening.id}`, {
      kind: 'opening',
      type: opening.type,
      points: [opening.start, opening.end],
      thickness: Number(opening.evidence?.wall_thickness_px) || null,
      label: opening.id,
      tag: opening.id,
      anchor: opening.center,
    });
  }
  return entries;
}

const pointsAttr = (points) => points.map((p) => p.join(',')).join(' ');

function bounds(points) {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys) };
}

// Unit direction along the opening and a normal that points up/left (for labels).
function openingFrame([[x1, y1], [x2, y2]]) {
  const length = Math.hypot(x2 - x1, y2 - y1) || 1;
  const d = [(x2 - x1) / length, (y2 - y1) / length];
  let n = [-d[1], d[0]];
  if (n[1] > 0 || (n[1] === 0 && n[0] > 0)) n = [-n[0], -n[1]];
  return { d, n, length };
}

// Rectangle covering the opening's wall band, padded so the drawn jambs and
// symbol lines sit inside it rather than under its outline.
function openingBand(item, size, grow = 1) {
  const [[x1, y1], [x2, y2]] = item.points;
  const { d, n } = openingFrame(item.points);
  const t = item.thickness || size / 140;
  const half = Math.max(t / 2 + Math.max(2, t * 0.2), size / 320) * grow;
  const extend = Math.max(1.5, t * 0.08) * grow;
  const a = [x1 - d[0] * extend, y1 - d[1] * extend];
  const b = [x2 + d[0] * extend, y2 + d[1] * extend];
  return {
    half,
    points: [
      [a[0] + n[0] * half, a[1] + n[1] * half], [b[0] + n[0] * half, b[1] + n[1] * half],
      [b[0] - n[0] * half, b[1] - n[1] * half], [a[0] - n[0] * half, a[1] - n[1] * half],
    ],
  };
}

export class Viewer {
  constructor(elements, { onHover, onSelect }) {
    this.els = elements;
    this.onHover = onHover;
    this.onSelect = onSelect;
    this.preview = null; // { url, width, height, ok, reason }
    this.result = null;
    this.geometry = new Map();
    this.view = 'rooms';
    this.mode = 'original';
    this.zoom = 1;
    this.layers = { rooms: true, unlabeled: true, door: true, window: true, opening: true, labels: true };
    this.highlightKey = null;
    this.panes = [];
    this.bindControls();
  }

  bindControls() {
    this.els.viewTabs.addEventListener('click', (event) => {
      const button = event.target.closest('button[data-view]');
      if (!button || button.disabled) return;
      this.view = button.dataset.view;
      if (this.mode === 'original') this.mode = 'analysis';
      this.render();
    });
    this.els.modeTabs.addEventListener('click', (event) => {
      const button = event.target.closest('button[data-mode]');
      if (!button || button.disabled) return;
      this.mode = button.dataset.mode;
      this.render();
    });
    this.els.zoomTabs.addEventListener('click', (event) => {
      const button = event.target.closest('button[data-zoom]');
      if (!button) return;
      this.zoom = Number(button.dataset.zoom);
      this.els.stage.style.setProperty('--zoom', this.zoom);
      this.syncControls();
      if (this.highlightKey) requestAnimationFrame(() => this.reveal(this.highlightKey, { center: true }));
    });
    this.els.layerBar.addEventListener('change', (event) => {
      const input = event.target.closest('input[data-layer]');
      if (!input) return;
      this.layers[input.dataset.layer] = input.checked;
      this.applyLayerVisibility();
    });
  }

  setPreview(preview) {
    this.preview = preview;
    this.result = null;
    this.geometry = new Map();
    this.highlightKey = null;
    this.mode = 'original';
    this.render();
  }

  clear() {
    this.preview = null;
    this.result = null;
    this.geometry = new Map();
    this.highlightKey = null;
    this.mode = 'original';
    this.render();
  }

  setResult(data) {
    this.result = data;
    this.geometry = indexGeometry(data);
    this.highlightKey = null;
    this.view = 'rooms';
    this.mode = 'analysis';
    this.render();
  }

  get dims() {
    if (this.result) return [this.result.image.width, this.result.image.height];
    if (this.preview?.width) return [this.preview.width, this.preview.height];
    return [4, 3];
  }

  render() {
    const { stage, empty } = this.els;
    this.panes = [];
    stage.replaceChildren();
    if (!this.preview && !this.result) {
      stage.hidden = true;
      empty.hidden = false;
      this.syncControls();
      this.renderLegend();
      return;
    }
    empty.hidden = true;
    stage.hidden = false;
    const [width, height] = this.dims;
    stage.style.setProperty('--ar', `${width} / ${height}`);
    stage.style.setProperty('--zoom', this.zoom);

    const kinds = this.mode === 'original' || !this.result ? ['original']
      : this.mode === 'split' ? ['original', this.view]
      : [this.view];
    stage.classList.toggle('is-split', kinds.length === 2);
    for (const kind of kinds) stage.append(this.buildPane(kind, width, height));
    if (this.panes.length === 2) this.linkScroll(this.panes[0].scroller, this.panes[1].scroller);

    this.applyLayerVisibility();
    this.drawHighlight();
    this.syncControls();
    this.renderLegend();
  }

  buildPane(kind, width, height) {
    const previewOk = Boolean(this.preview?.ok);
    const [title, subtitle] = VIEW_TITLES[kind];
    const fallback = kind !== 'original' && !CLEANING_VIEWS[kind] && !previewOk;
    const head = el('div', { class: 'pane-head' }, el('strong', { text: title }),
      el('span', { text: `· ${fallback ? FALLBACK_SUBTITLES[kind] : subtitle}` }));
    const plane = el('div', { class: 'plane' });

    // Base layer: always the original when the browser can show it (the cleaning views show the
    // cleaned plan / typed elements produced by the server instead).
    let src = null;
    const cleaningUrl = CLEANING_VIEWS[kind] && this.result?.cleaning?.urls?.[CLEANING_VIEWS[kind]];
    if (cleaningUrl) src = cleaningUrl;
    else if (previewOk) src = this.preview.url;
    else if (kind === 'openings') src = this.result.openings_overlay_url;
    else if (kind !== 'original') src = this.result.overlay_url;

    if (fallback) {
      head.classList.add('has-note');
      head.append(el('span', { class: 'pane-note', text: 'Original not displayable — showing the server overlay' }));
    }
    if (src) {
      plane.append(el('img', { src, alt: `${title} view of the floor plan`, draggable: 'false' }));
    } else {
      plane.append(el('div', { class: 'plane-placeholder', text: this.preview?.reason || 'Preview not available.' }));
    }

    let svgRoot = null;
    let drawn = false;
    if (this.result) {
      svgRoot = svg('svg', { viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: 'none' });
      // Over the original, draw the view's detections. Over a server overlay the
      // detections are already in the image; only the combined view adds shapes.
      if (kind === 'cleaned') {
        this.drawGeometry(svgRoot, width, height, 'rooms');
        drawn = true;
      } else if (kind === 'elements') {
        // the typed elements image is the content: nothing drawn over it
      } else if ((kind !== 'original' && previewOk) || (kind === 'combined' && !previewOk)) {
        this.drawGeometry(svgRoot, width, height, kind);
        drawn = true;
      }
      svgRoot.append(svg('g', { class: 'hl-layer' }));
      plane.append(svgRoot);
    }

    const scroller = el('div', { class: 'pane-scroll' }, plane);
    const pane = el('div', { class: 'pane', 'data-kind': kind }, head, scroller);
    this.panes.push({ kind, svg: svgRoot, scroller, plane, drawn, fallback });
    return pane;
  }

  drawGeometry(root, width, height, kind) {
    const size = Math.max(width, height);
    const fontSize = size / 95;
    const showRooms = kind !== 'openings';
    const showOpenings = kind !== 'rooms';
    const groups = {
      physical: svg('g', { 'data-layer': 'rooms' }),
      rooms: svg('g', { 'data-layer': 'rooms' }),
      unlabeled: svg('g', { 'data-layer': 'unlabeled' }),
      door: svg('g', { 'data-layer': 'door' }),
      window: svg('g', { 'data-layer': 'window' }),
      opening: svg('g', { 'data-layer': 'opening' }),
      labels: svg('g', { 'data-layer': 'labels' }),
    };
    const interactive = (key) => ({
      'data-key': key,
      onmouseenter: () => this.onHover(key),
      onmouseleave: () => this.onHover(null),
      onclick: () => this.onSelect(key, 'plan'),
    });
    const tag = (x, y, text, extra = {}) => svg('text', {
      class: 'geo-label', x, y, 'font-size': fontSize, 'stroke-width': fontSize / 3.5, ...extra, text,
    });

    for (const [key, item] of this.geometry) {
      if (item.kind === 'physical' && showRooms) {
        const outline = svg('polygon', { class: `geo-physical${item.merged ? ' is-merged' : ''}`, points: pointsAttr(item.points) });
        outline.append(svg('title', { text: `${item.tag} · ${item.label}` }));
        groups.physical.append(outline);
      } else if ((item.kind === 'room' || item.kind === 'space') && showRooms) {
        const className = `geo geo-${item.kind}${item.zone ? ' is-zone' : item.estimate ? ' is-estimate' : ''}`;
        const polygon = svg('polygon', { class: className, points: pointsAttr(item.points), ...interactive(key) });
        polygon.append(svg('title', { text: item.label === item.tag ? item.tag : `${item.tag} · ${item.label}` }));
        (item.kind === 'room' ? groups.rooms : groups.unlabeled).append(polygon);
        // A small ID tag in the room's top-left corner, away from the printed room name.
        const box = bounds(item.points);
        groups.labels.append(tag(box.x0 + fontSize * 0.45, box.y0 + fontSize * 1.15, item.tag,
          { 'data-key': key, 'font-size': fontSize * 0.85 }));
      } else if (item.kind === 'opening' && showOpenings) {
        const group = groups[item.type] || groups.opening;
        const band = openingBand(item, size);
        const shape = svg('polygon', { class: `geo geo-opening ${item.type}`, points: pointsAttr(band.points), ...interactive(key) });
        shape.append(svg('title', { text: `${item.label} · ${OPENING_TYPES[item.type] || item.type}` }));
        group.append(shape);
        const { n } = openingFrame(item.points);
        const [cx, cy] = item.anchor;
        const offset = band.half + fontSize * 0.75;
        groups.labels.append(tag(cx + n[0] * offset, cy + n[1] * offset, item.label, {
          class: `geo-label ${item.type}`, 'font-size': fontSize * 0.8, 'text-anchor': 'middle',
          'dominant-baseline': 'middle', 'data-type': item.type, 'data-key': key,
        }));
      }
    }
    root.append(groups.physical, groups.rooms, groups.unlabeled, groups.door, groups.window, groups.opening, groups.labels);
  }

  applyLayerVisibility() {
    for (const { svg: root } of this.panes) {
      if (!root) continue;
      for (const group of root.querySelectorAll('g[data-layer]')) {
        group.style.display = this.layers[group.dataset.layer] ? '' : 'none';
      }
      // Opening labels follow their type's layer as well as the labels toggle.
      for (const label of root.querySelectorAll('text[data-type]')) {
        label.style.display = this.layers[label.dataset.type] ? '' : 'none';
      }
    }
  }

  setHighlight(key) {
    if (key === this.highlightKey) return;
    this.highlightKey = key;
    this.drawHighlight();
  }

  drawHighlight() {
    const key = this.highlightKey;
    const item = key ? this.geometry.get(key) : null;
    const [width, height] = this.dims;
    const size = Math.max(width, height);
    for (const { svg: root } of this.panes) {
      if (!root) continue;
      // Fade the other detections so the highlighted one reads clearly.
      root.classList.toggle('has-highlight', Boolean(item));
      for (const node of root.querySelectorAll('.is-hl')) node.classList.remove('is-hl');
      if (item) for (const node of root.querySelectorAll(`[data-key="${CSS.escape(key)}"]`)) node.classList.add('is-hl');

      const layer = root.querySelector('.hl-layer');
      layer.replaceChildren();
      if (!item) continue;
      const points = item.kind === 'opening' ? pointsAttr(openingBand(item, size, 1.35).points) : pointsAttr(item.points);
      const kindClass = item.kind === 'opening' ? ' hl-opening' : '';
      layer.append(
        svg('polygon', { class: `hl-glow${kindClass}`, points }),
        svg('polygon', { class: `hl-shape${kindClass}`, points }),
      );
    }
  }

  // Scroll each pane (only when zoomed in) so the entity is in view.
  reveal(key, { center = false } = {}) {
    const item = key ? this.geometry.get(key) : null;
    if (!item) return;
    const [width] = this.dims;
    const box = bounds(item.points);
    for (const { scroller, plane } of this.panes) {
      if (scroller.scrollWidth <= scroller.clientWidth && scroller.scrollHeight <= scroller.clientHeight) continue;
      const scale = plane.clientWidth / width;
      const x0 = plane.offsetLeft + box.x0 * scale;
      const x1 = plane.offsetLeft + box.x1 * scale;
      const y0 = plane.offsetTop + box.y0 * scale;
      const y1 = plane.offsetTop + box.y1 * scale;
      const visible = x0 >= scroller.scrollLeft && x1 <= scroller.scrollLeft + scroller.clientWidth
        && y0 >= scroller.scrollTop && y1 <= scroller.scrollTop + scroller.clientHeight;
      if (visible && !center) continue;
      scroller.scrollTo({
        left: (x0 + x1) / 2 - scroller.clientWidth / 2,
        top: (y0 + y1) / 2 - scroller.clientHeight / 2,
        behavior: 'smooth',
      });
    }
  }

  linkScroll(a, b) {
    let syncing = false;
    const follow = (source, target) => () => {
      if (syncing) return;
      syncing = true;
      target.scrollLeft = source.scrollLeft;
      target.scrollTop = source.scrollTop;
      requestAnimationFrame(() => { syncing = false; });
    };
    a.addEventListener('scroll', follow(a, b), { passive: true });
    b.addEventListener('scroll', follow(b, a), { passive: true });
  }

  syncControls() {
    const hasResult = Boolean(this.result);
    for (const button of this.els.viewTabs.querySelectorAll('button[data-view]')) {
      const view = button.dataset.view;
      const available = hasResult && view !== 'walls'
        && (!CLEANING_VIEWS[view] || Boolean(this.result?.cleaning?.urls?.[CLEANING_VIEWS[view]]));
      button.disabled = !available;
      button.classList.toggle('is-active', available && this.mode !== 'original' && button.dataset.view === this.view);
    }
    for (const button of this.els.modeTabs.querySelectorAll('button[data-mode]')) {
      button.disabled = button.dataset.mode !== 'original' && !hasResult;
      button.disabled ||= !this.preview && !hasResult;
      button.classList.toggle('is-active', button.dataset.mode === this.mode);
    }
    for (const button of this.els.zoomTabs.querySelectorAll('button[data-zoom]')) {
      button.disabled = !this.preview && !hasResult;
      button.classList.toggle('is-active', Number(button.dataset.zoom) === this.zoom);
    }
    // Layer toggles for whatever detections the visible panes draw.
    const drawnPane = this.panes.find((pane) => pane.drawn);
    this.els.layerBar.hidden = !drawnPane;
    if (drawnPane) {
      const present = new Set([...drawnPane.svg.querySelectorAll('g[data-layer]')]
        .filter((group) => group.childElementCount > 0).map((group) => group.dataset.layer));
      for (const input of this.els.layerBar.querySelectorAll('input[data-layer]')) {
        input.closest('label').hidden = !present.has(input.dataset.layer);
      }
    }
  }

  renderLegend() {
    const legend = this.els.legend;
    legend.replaceChildren();
    const analysis = this.panes.find((pane) => pane.kind !== 'original');
    if (!analysis) {
      legend.hidden = true;
      return;
    }
    const item = (swatch, text) => el('span', { class: 'legend-item' }, el('i', { class: `swatch ${swatch}` }), text);
    const roomItems = () => [
      item('swatch-room', 'Room · wall region'),
      item('swatch-estimate', 'Room · dimension estimate'),
      item('swatch-space', 'Unlabeled space'),
    ];
    const openingItems = () => [
      item('swatch-door', 'Door'),
      item('swatch-window', 'Window'),
      item('swatch-opening', 'Unclassified'),
    ];
    // When the layer toggles are shown they already act as the colour key; the
    // legend then only adds what they do not cover (dimension-estimate rooms).
    const showsRooms = analysis.kind !== 'openings' && analysis.kind !== 'elements';
    const items = analysis.drawn
      ? (showsRooms ? [item('swatch-estimate', 'Room · dimension estimate')] : [])
      : analysis.kind === 'rooms' ? roomItems()
        : analysis.kind === 'openings' ? openingItems()
          : [...roomItems(), ...openingItems()];
    const source = analysis.fallback && analysis.kind !== 'combined'
      ? 'Server overlay · IDs match the tables'
      : 'Drawn over the original · hover or click to link';
    legend.append(...items, el('span', { class: 'legend-source', text: source }));
    legend.hidden = false;
  }
}
