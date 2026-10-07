// Small DOM and formatting helpers shared by the workbench modules.

export const $ = (id) => document.getElementById(id);

const SVG_NS = 'http://www.w3.org/2000/svg';

// el('td', { class: 'num', dir: 'auto' }, 'text', childNode, ...)
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  setAttrs(node, attrs);
  appendChildren(node, children);
  return node;
}

export function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  setAttrs(node, attrs);
  appendChildren(node, children);
  return node;
}

function setAttrs(node, attrs) {
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'text') node.textContent = value;
    else if (key === 'style' && typeof value === 'object') Object.assign(node.style, value);
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : String(value));
  }
}

function appendChildren(node, children) {
  for (const child of children.flat()) {
    if (child === undefined || child === null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export function fmtNumber(value, digits = 2) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return '—';
  return Number(value).toLocaleString('en-US', { maximumFractionDigits: digits });
}

export function fmtPercent(fraction) {
  if (fraction === undefined || fraction === null) return '—';
  return `${Math.round(Number(fraction) * 100)}%`;
}

export function fmtSeconds(ms) {
  return `${(ms / 1000).toFixed(1)} s`;
}

export function fmtBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
