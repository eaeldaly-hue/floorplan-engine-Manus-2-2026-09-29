// Analysis Workbench entry point: upload → preview → /api/analyze → views.

import { $, fmtBytes, fmtSeconds } from './dom.js';
import { analyzeFile, analyzePdfPage, fetchHealth, fetchSampleFile, inspectPdf } from './api.js';
import { Viewer } from './viewer.js';
import {
  markActiveRow, renderMetrics, renderOpenings, renderRaw, renderRooms, renderWarnings, revealRow, setNavCounts,
} from './panels.js';

// Client-side mirror of the server's limits (app.py); the server still enforces them.
const ALLOWED = /\.(png|jpe?g|bmp|tiff?|pdf)$/i;
const MAX_BYTES = 20 * 1024 * 1024;
const MAX_PDF_BYTES = 100 * 1024 * 1024;
const isPdf = (file) => /\.pdf$/i.test(file?.name || '');

const state = {
  file: null,
  preview: null,
  running: false,
  pinnedKey: null,
  timer: null,
  // PDF input: the document's page list, the pages the user selected, and one result per page
  pdf: null,
  selectedPages: new Set(),
  pageResults: new Map(),
  activePage: null,
};

const viewer = new Viewer({
  stage: $('stage'),
  empty: $('viewer-empty'),
  legend: $('legend'),
  layerBar: $('layer-bar'),
  viewTabs: $('view-tabs'),
  modeTabs: $('mode-tabs'),
  zoomTabs: $('zoom-tabs'),
}, { onHover, onSelect });

// `source` is 'table' for result rows and 'plan' for shapes on the plan; the
// other side is scrolled (inside its own panel) so both stay in view together.
function onHover(key, source) {
  viewer.setHighlight(key ?? state.pinnedKey);
  markActiveRow(key, state.pinnedKey);
  if (key && source === 'table') viewer.reveal(key);
}

function onSelect(key, source) {
  state.pinnedKey = state.pinnedKey === key ? null : key;
  viewer.setHighlight(state.pinnedKey);
  markActiveRow(null, state.pinnedKey);
  if (!state.pinnedKey) return;
  if (source === 'plan') revealRow(key);
  else viewer.reveal(key, { center: true });
}

// ---------- File selection & preview ----------

function loadPreview(file) {
  const url = URL.createObjectURL(file);
  if (isPdf(file)) {
    return Promise.resolve({ url, ok: false, reason: 'Choose the floor-plan page(s) of the PDF above, then analyse them. Each selected page is shown here after its analysis.' });
  }
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => resolve({ url, ok: true, width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = () => resolve({ url, ok: false, reason: 'The browser could not display this file (a format it does not support, such as TIFF, or a damaged image). It can still be sent for analysis; the analysis views show the server-rendered result.' });
    image.src = url;
  });
}

function showError(message) {
  $('error-box').textContent = message;
  $('error-box').hidden = !message;
}

function resetResults() {
  state.pinnedKey = null;
  $('results').hidden = true;
  $('metrics-card').hidden = true;
  $('results-empty').hidden = false;
  $('results-nav').hidden = true;
  $('metrics-source').textContent = '';
}

async function chooseFile(file) {
  if (!file || state.running) return;
  showError('');
  if (!ALLOWED.test(file.name)) {
    showError('Unsupported file type. Choose a PNG, JPG, BMP, TIFF or PDF file.');
    return;
  }
  if (file.size > (isPdf(file) ? MAX_PDF_BYTES : MAX_BYTES)) {
    showError(isPdf(file) ? 'The PDF is larger than 100 MB.' : 'The file is larger than 20 MB.');
    return;
  }

  if (state.preview?.url) URL.revokeObjectURL(state.preview.url);
  state.file = file;
  state.preview = await loadPreview(file);
  resetResults();
  $('status-card').hidden = true;

  $('file-chip').hidden = false;
  $('file-thumb').hidden = !state.preview.ok;
  if (state.preview.ok) $('file-thumb').src = state.preview.url;
  $('file-name').textContent = file.name;
  $('file-meta').textContent = state.preview.ok
    ? `${fmtBytes(file.size)} · ${state.preview.width} × ${state.preview.height} px`
    : `${fmtBytes(file.size)} · no browser preview`;
  $('dropzone').hidden = true;
  $('analyze-label').textContent = 'Analyze floor plan';
  resetPdf();
  updateButtons();
  viewer.setPreview(state.preview);
  if (isPdf(file)) await loadPdf(file);
}

// ---------- PDF: page list, manual page selection ----------

function resetPdf() {
  state.pdf = null;
  state.selectedPages = new Set();
  state.pageResults = new Map();
  state.activePage = null;
  $('pdf-picker').hidden = true;
  $('pdf-pages').replaceChildren();
  $('page-tabs').hidden = true;
  $('page-tabs').replaceChildren();
}

async function loadPdf(file) {
  state.running = true;
  updateButtons();
  setStatus('running', 'Reading PDF', file.name);
  setProgress(0);
  try {
    const { data, elapsedMs } = await inspectPdf(file, { onUploadProgress: (f) => setProgress(f < 1 ? f : null) });
    state.pdf = data;
    $('file-meta').textContent = `${fmtBytes(file.size)} · PDF · ${data.page_count} page${data.page_count === 1 ? '' : 's'}`;
    if (data.page_count === 1) state.selectedPages.add(1);
    if (data.pages[0]?.thumbnail_url) {
      $('file-thumb').src = data.pages[0].thumbnail_url;
      $('file-thumb').hidden = false;
    }
    renderPdfPicker();
    setStatus('done', 'PDF ready', `${data.page_count} page${data.page_count === 1 ? '' : 's'} · read in ${fmtSeconds(elapsedMs)} · select the floor-plan page(s)`);
  } catch (error) {
    setStatus('error', 'PDF could not be read', error.message);
    showError(error.message);
  } finally {
    state.running = false;
    updateButtons();
  }
}

function describePage(page) {
  const size = page.sheet || `${page.width_in}×${page.height_in} in`;
  const kind = { vector: 'vector', raster: 'scanned', mixed: 'vector + images' }[page.kind] || page.kind;
  return `${size} · ${page.orientation} · ${kind}`;
}

function renderPdfPicker() {
  const { pdf } = state;
  $('pdf-picker').hidden = false;
  setPickerCollapsed(false);
  $('pdf-picker-meta').textContent = `${pdf.filename} · ${pdf.page_count} page${pdf.page_count === 1 ? '' : 's'} · the engine analyses each selected page on its own`;
  const cards = pdf.pages.map((page) => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'pdf-page';
    card.dataset.page = page.number;
    const thumb = document.createElement('div');
    thumb.className = 'pdf-page-thumb';
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.alt = `Page ${page.number}`;
    img.src = page.thumbnail_url;
    thumb.append(img);
    const label = document.createElement('div');
    label.className = 'pdf-page-label';
    const title = document.createElement('span');
    title.textContent = `Page ${page.number}`;
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.tabIndex = -1;
    box.setAttribute('aria-hidden', 'true');
    label.append(title, box);
    const meta = document.createElement('div');
    meta.className = 'pdf-page-meta';
    meta.textContent = describePage(page);
    meta.title = `${page.width_in} × ${page.height_in} in · analysed at ${page.analysis_dpi || '—'} DPI (${page.dpi_reason})`;
    card.append(thumb, label, meta);
    card.addEventListener('click', () => togglePage(page.number));
    return card;
  });
  $('pdf-pages').replaceChildren(...cards);
  syncPdfSelection();
}

// The page grid folds away while results are shown, so the viewer keeps its space.
function setPickerCollapsed(collapsed) {
  $('pdf-picker').classList.toggle('is-collapsed', collapsed);
  $('pdf-toggle').textContent = collapsed ? 'Change pages' : 'Hide pages';
  $('pdf-toggle').setAttribute('aria-expanded', String(!collapsed));
}

function togglePage(number) {
  if (state.running) return;
  if (state.selectedPages.has(number)) state.selectedPages.delete(number);
  else state.selectedPages.add(number);
  syncPdfSelection();
}

function syncPdfSelection() {
  const pages = [...state.selectedPages].sort((a, b) => a - b);
  for (const card of $('pdf-pages').children) {
    const on = state.selectedPages.has(Number(card.dataset.page));
    card.classList.toggle('is-selected', on);
    card.setAttribute('aria-pressed', String(on));
    card.querySelector('input').checked = on;
  }
  $('pdf-selected').textContent = pages.length ? `Selected page${pages.length === 1 ? '' : 's'}: ${pages.join(', ')}` : 'No page selected';
  $('analyze-label').textContent = pages.length > 1 ? `Analyze ${pages.length} selected pages`
    : pages.length === 1 ? `Analyze page ${pages[0]}` : 'Select page(s) to analyse';
  updateButtons();
}

function clearFile() {
  if (state.running) return;
  if (state.preview?.url) URL.revokeObjectURL(state.preview.url);
  state.file = null;
  state.preview = null;
  resetPdf();
  const cleaning = cleaningOn();
  $('upload-form').reset();
  $('cleaning-toggle').checked = cleaning;        // a preference, not part of the file form
  $('file-chip').hidden = true;
  $('dropzone').hidden = false;
  $('status-card').hidden = true;
  showError('');
  resetResults();
  updateButtons();
  viewer.clear();
}

function updateButtons() {
  $('analyze-button').disabled = state.running || !state.file || (isPdf(state.file) && (!state.pdf || !state.selectedPages.size));
  $('pdf-clear-selection').disabled = state.running || !state.selectedPages.size;
  $('sample-button').disabled = state.running;
  $('clear-file').disabled = state.running;
  $('replace-file').disabled = state.running;
  $('plan-file').disabled = state.running;
}

// ---------- Status ----------

function setStatus(stateName, title, detail = '') {
  const card = $('status-card');
  card.hidden = false;
  card.dataset.state = stateName;
  $('status-title').textContent = title;
  $('status-detail').textContent = detail;
}

function setProgress(fraction) {
  const bar = $('status-progress');
  bar.classList.toggle('is-indeterminate', fraction === null);
  bar.firstElementChild.style.width = fraction === null ? '' : `${Math.round(fraction * 100)}%`;
}

function startTimer() {
  const started = performance.now();
  const tick = () => { $('status-elapsed').textContent = fmtSeconds(performance.now() - started); };
  tick();
  state.timer = setInterval(tick, 100);
}

function stopTimer() {
  clearInterval(state.timer);
  state.timer = null;
}

// ---------- Analysis ----------

// Show one analysis result in the viewer and the panels (image upload or one PDF page).
function showResult(data, elapsedMs) {
  if (data.page_image_url) {
    // PDF page: draw over the exact page image the engine analysed
    state.preview = { url: data.page_image_url, ok: true, width: data.image.width, height: data.image.height };
  } else if (state.preview.ok && (state.preview.width !== data.image.width || state.preview.height !== data.image.height)) {
    // The SVG layers use the server's image coordinates; only draw them over the
    // browser preview when both decoded the file to the same size.
    state.preview = {
      ...state.preview,
      ok: false,
      reason: `Browser preview (${state.preview.width} × ${state.preview.height}) differs from the server-decoded image (${data.image.width} × ${data.image.height}).`,
    };
  }
  state.pinnedKey = null;
  viewer.preview = state.preview;
  viewer.setResult(data);
  renderMetrics(data, elapsedMs);
  renderWarnings(data.warnings);
  renderRooms(data, { onHover, onSelect });
  renderOpenings(data, { onHover, onSelect });
  renderRaw(data);
  setNavCounts(data);
  if (data.source?.type === 'pdf') {
    $('metrics-source').textContent = `${data.source.filename} · page ${data.source.page} of ${data.source.page_count}`;
  }
  $('results-empty').hidden = true;
  $('results').hidden = false;
  $('results-scroll').scrollTop = 0;
}

async function runAnalysis() {
  if (!state.file || state.running) return;
  if (isPdf(state.file)) {
    await runPdfAnalysis();
    return;
  }
  state.running = true;
  updateButtons();
  showError('');
  setStatus('running', 'Uploading', state.file.name);
  setProgress(0);
  startTimer();

  try {
    const { data, elapsedMs } = await analyzeFile(state.file, {
      cleaning: cleaningOn(),
      onUploadProgress: (fraction) => setProgress(fraction),
      onUploaded: () => {
        setStatus('running', 'Analyzing on server', 'OCR, walls, rooms and openings — this can take a while');
        setProgress(null);
      },
    });
    stopTimer();
    $('status-elapsed').textContent = fmtSeconds(elapsedMs);
    showResult(data, elapsedMs);
    setStatus('done', 'Analysis complete',
      `${data.room_count} rooms · ${data.opening_count ?? 0} openings · ${(data.warnings || []).length} warnings`);
    $('analyze-label').textContent = 'Analyze again';
    syncCleaningHint();
  } catch (error) {
    stopTimer();
    setStatus('error', 'Analysis failed', error.message);
    showError(error.message);
  } finally {
    state.running = false;
    updateButtons();
  }
}

// Each selected page is rendered and analysed on its own (one request per page, in page order).
async function runPdfAnalysis() {
  const pages = [...state.selectedPages].sort((a, b) => a - b);
  if (!state.pdf || !pages.length) return;
  state.running = true;
  state.pageResults = new Map();
  setPickerCollapsed(true);
  updateButtons();
  showError('');
  startTimer();
  let shown = false;
  for (const [i, page] of pages.entries()) {
    setStatus('running', `Analysing page ${page} (${i + 1} of ${pages.length})`,
      'Rendering the page, then OCR, walls, rooms and openings — large sheets can take a few minutes');
    setProgress(pages.length > 1 ? i / pages.length : null);
    try {
      const result = await analyzePdfPage(state.pdf.pdf_id, page, { cleaning: cleaningOn() });
      state.pageResults.set(page, result);
      if (!shown) {
        showPage(page);
        shown = true;
      }
    } catch (error) {
      state.pageResults.set(page, { error: error.message });
    }
    renderPageTabs();
  }
  stopTimer();
  const failed = pages.filter((p) => state.pageResults.get(p)?.error);
  if (failed.length === pages.length) {
    const message = state.pageResults.get(pages[0]).error;
    setStatus('error', 'Analysis failed', message);
    showError(message);
  } else {
    setStatus('done', 'Analysis complete',
      `${pages.length - failed.length} of ${pages.length} page${pages.length === 1 ? '' : 's'} analysed` +
      (failed.length ? ` · failed: page ${failed.join(', ')}` : ''));
    if (failed.length) showError(failed.map((p) => `Page ${p}: ${state.pageResults.get(p).error}`).join(' · '));
  }
  $('analyze-label').textContent = 'Analyze again';
  state.running = false;
  updateButtons();
  syncCleaningHint();
}

function showPage(page) {
  const result = state.pageResults.get(page);
  if (!result || result.error) return;
  state.activePage = page;
  showResult(result.data, result.elapsedMs);
  renderPageTabs();
  syncCleaningHint();
}

function renderPageTabs() {
  const pages = [...state.pageResults.keys()].sort((a, b) => a - b);
  const bar = $('page-tabs');
  bar.hidden = pages.length < 1 || !state.pdf;
  bar.replaceChildren(...pages.map((page) => {
    const result = state.pageResults.get(page);
    const tab = document.createElement('button');
    tab.type = 'button';
    tab.className = `page-tab${result.error ? ' is-failed' : ''}`;
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-selected', String(page === state.activePage));
    tab.textContent = `Page ${page}`;
    tab.title = result.error ? result.error : `${result.data.room_count} rooms · ${result.data.opening_count ?? 0} openings`;
    tab.disabled = Boolean(result.error);
    tab.addEventListener('click', () => showPage(page));
    return tab;
  }));
}

// ---------- Wiring ----------

$('upload-form').addEventListener('submit', (event) => {
  event.preventDefault();
  runAnalysis();
});
$('plan-file').addEventListener('change', (event) => chooseFile(event.target.files[0]));
$('clear-file').addEventListener('click', clearFile);
$('replace-file').addEventListener('click', () => $('plan-file').click());
$('pdf-toggle').addEventListener('click', () => setPickerCollapsed(!$('pdf-picker').classList.contains('is-collapsed')));
$('pdf-clear-selection').addEventListener('click', () => {
  state.selectedPages.clear();
  syncPdfSelection();
});
$('sample-button').addEventListener('click', async () => {
  try {
    await chooseFile(await fetchSampleFile());
  } catch (error) {
    showError(error.message);
  }
});

for (const target of [$('dropzone'), $('viewer')]) {
  target.addEventListener('dragover', (event) => {
    event.preventDefault();
    $('dropzone').classList.add('is-over');
  });
  target.addEventListener('dragleave', () => $('dropzone').classList.remove('is-over'));
  target.addEventListener('drop', (event) => {
    event.preventDefault();
    $('dropzone').classList.remove('is-over');
    chooseFile(event.dataTransfer.files[0]);
  });
}

$('results-nav').addEventListener('click', (event) => {
  const chip = event.target.closest('button[data-target]');
  if (!chip) return;
  const scroller = $('results-scroll');
  const target = $(chip.dataset.target);
  scroller.scrollTo({ top: target.offsetTop, behavior: 'smooth' }); // .results-scroll is the offset parent
});

fetchHealth().then((health) => {
  const status = $('health-status');
  status.dataset.state = health.ocr_available ? 'ok' : 'warn';
  status.querySelector('.health-text').textContent = health.ocr_available
    ? 'Engine ready · OCR available'
    : 'Engine ready · OCR unavailable';
  if (!health.ocr_available && health.ocr_error) status.title = health.ocr_error;
}).catch(() => {
  $('health-status').dataset.state = 'error';
  $('health-status').querySelector('.health-text').textContent = 'Engine unreachable';
});

viewer.render();

// Architectural Cleaning (experimental): clean the plan before recognition.
// The choice is remembered (per user, this app) and survives clearing the file and restarts; it
// applies to the NEXT analysis, so a result that does not match it says so on the Analyze button.
const CLEANING_KEY = 'floorplan.cleaningFirst';

function cleaningOn() {
  return Boolean(document.getElementById('cleaning-toggle')?.checked);
}

function saveCleaningChoice() {
  try { localStorage.setItem(CLEANING_KEY, cleaningOn() ? '1' : '0'); } catch { /* storage unavailable */ }
}

function restoreCleaningChoice() {
  const toggle = document.getElementById('cleaning-toggle');
  if (!toggle) return;
  try {
    const saved = localStorage.getItem(CLEANING_KEY);
    if (saved !== null) toggle.checked = saved === '1';
  } catch { /* storage unavailable */ }
}

// When the shown result was analysed with the other setting, the Analyze button says what the next
// run will do (the Cleaned / Elements views need a run with cleaning).
function syncCleaningHint() {
  const result = viewer.result;
  if (!result || state.running) return;
  const cleaned = Boolean(result.cleaning);
  if (cleaningOn() !== cleaned) {
    $('analyze-label').textContent = cleaningOn() ? 'Analyze again with cleaning' : 'Analyze again without cleaning';
  } else if (/with(out)? cleaning$/.test($('analyze-label').textContent)) {
    $('analyze-label').textContent = 'Analyze again';
  }
}

restoreCleaningChoice();
document.getElementById('cleaning-toggle')?.addEventListener('change', () => {
  saveCleaningChoice();
  syncCleaningHint();
});
