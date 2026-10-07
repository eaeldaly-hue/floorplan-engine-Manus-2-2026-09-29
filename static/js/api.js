// Thin client for the existing Flask endpoints. No analysis logic lives here.

export async function fetchHealth() {
  const response = await fetch('/health');
  if (!response.ok) throw new Error(`Health check failed (${response.status})`);
  return response.json();
}

export async function fetchSampleFile() {
  const response = await fetch('/api/sample.png');
  if (!response.ok) throw new Error('The bundled sample plan is not available on this server.');
  const blob = await response.blob();
  return new File([blob], 'test_floorplan.png', { type: blob.type || 'image/png' });
}

// POST the file to /api/analyze. XHR (rather than fetch) is used only to get
// real upload progress; the request itself is identical to a form upload.
export function analyzeFile(file, { onUploadProgress, onUploaded, cleaning = false } = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const started = performance.now();
    xhr.open('POST', '/api/analyze');
    xhr.responseType = 'text';

    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable && onUploadProgress) onUploadProgress(event.loaded / event.total);
    });
    xhr.upload.addEventListener('load', () => onUploaded && onUploaded());

    xhr.addEventListener('load', () => {
      const elapsedMs = performance.now() - started;
      let data = null;
      try {
        data = JSON.parse(xhr.responseText);
      } catch {
        reject(new Error(`The server returned an unexpected response (HTTP ${xhr.status}).`));
        return;
      }
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(data?.error || `Analysis failed (HTTP ${xhr.status}).`));
        return;
      }
      resolve({ data, elapsedMs });
    });
    xhr.addEventListener('error', () => reject(new Error('Could not reach the analysis server.')));
    xhr.addEventListener('abort', () => reject(new Error('The request was cancelled.')));

    const body = new FormData();
    body.append('file', file);
    if (cleaning) body.append('cleaning', 'on');
    xhr.send(body);
  });
}

// ---------- PDF documents ----------

function xhrJson(method, url, body, { onUploadProgress } = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const started = performance.now();
    xhr.open(method, url);
    xhr.responseType = 'text';
    if (onUploadProgress) {
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) onUploadProgress(event.loaded / event.total);
      });
    }
    xhr.addEventListener('load', () => {
      let data = null;
      try {
        data = JSON.parse(xhr.responseText);
      } catch {
        reject(new Error(`The server returned an unexpected response (HTTP ${xhr.status}).`));
        return;
      }
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(data?.error || `Request failed (HTTP ${xhr.status}).`));
        return;
      }
      resolve({ data, elapsedMs: performance.now() - started });
    });
    xhr.addEventListener('error', () => reject(new Error('Could not reach the analysis server.')));
    if (body instanceof FormData) {
      xhr.send(body);
    } else {
      xhr.setRequestHeader('Content-Type', 'application/json');
      xhr.send(JSON.stringify(body || {}));
    }
  });
}

// Upload a PDF and get its page list (count, sizes, thumbnails). Nothing is analysed yet.
export function inspectPdf(file, options) {
  const body = new FormData();
  body.append('file', file);
  return xhrJson('POST', '/api/pdf', body, options);
}

// Render one selected page and analyse it with the engine (one request per page).
export function analyzePdfPage(pdfId, page, { cleaning = false } = {}) {
  return xhrJson('POST', `/api/pdf/${pdfId}/pages/${page}/analyze`, cleaning ? { cleaning: 'on' } : {});
}
