"""Desktop GUI check of Architectural Cleaning (opens a real window; not part of pytest).

Drives the Workbench inside the native WKWebView exactly like a user: load a PDF, pick the page,
tick "Clean plan first", Analyze, then open the Cleaned and Elements views. Several click orders
are tried (toggle before / after loading the file, after an analysis, after clearing the file).
The PDF bytes are passed into the page in base64 chunks and set on the real file input.

    .venv/bin/python -m scripts.diagnostics.desktop_cleaning_smoke ["/path/to/22.pdf:6" "/path/to/3.pdf:4"]
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import desktop
from scripts.diagnostics.desktop_smoke import check, js, wait_for, RESULTS, STATE

TEST_CASES = Path(os.environ.get("FLOORPLAN_REAL_PLANS", Path(__file__).resolve().parents[3] / "Test Cases"))
PLANS = sys.argv[1:] or [f"{TEST_CASES / '22.pdf'}:6", f"{TEST_CASES / '3.pdf'}:4"]

HELPERS = r"""
window.__chunks = [];
window.__putPdf = (name) => {
  const bin = atob(window.__chunks.join('')); window.__chunks = [];
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const dt = new DataTransfer(); dt.items.add(new File([bytes], name, { type: 'application/pdf' }));
  const input = document.getElementById('plan-file'); input.files = dt.files;
  input.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
};
window.__selectOnly = (page) => {
  const cards = [...document.querySelectorAll('#pdf-pages input[type=checkbox]')];
  cards.forEach((b, i) => { if (b.checked !== (i === page - 1)) b.click(); });
  return cards.length;
};
window.__tabs = () => JSON.stringify(Object.fromEntries([...document.querySelectorAll('#view-tabs button[data-view]')]
  .map((b) => [b.dataset.view, { disabled: b.disabled, title: b.title }])));
window.__lastResult = () => { try { return JSON.parse(document.getElementById('raw-json').textContent); } catch { return null; } };
true;
"""


def put_pdf(window, path: Path) -> None:
    data = base64.b64encode(path.read_bytes()).decode()
    js(window, "window.__chunks = []; true")
    step = 200_000
    for i in range(0, len(data), step):
        js(window, f"window.__chunks.push('{data[i:i + step]}'); true")
    js(window, f"window.__putPdf({json.dumps(path.name)})")


def toggle(window, on: bool) -> bool:
    """Click the checkbox like a user (only if its state differs)."""
    return js(window, f"(() => {{ const t = document.getElementById('cleaning-toggle'); if (t.checked !== {str(on).lower()}) t.click(); return t.checked; }})()")


def analyze(window) -> str:
    js(window, "document.getElementById('analyze-button').click(); true")
    wait_for(window, "document.getElementById('status-card').dataset.state === 'running'", timeout=15)
    return wait_for(window, "['done','error'].includes(document.getElementById('status-card').dataset.state) && document.getElementById('status-card').dataset.state", timeout=600)


def load(window, path: Path, page: int) -> None:
    put_pdf(window, path)
    ok = wait_for(window, "document.querySelectorAll('#pdf-pages input[type=checkbox]').length > 0 && !document.getElementById('plan-file').disabled", timeout=120)
    check(f"{path.name}: PDF pages listed", ok)
    js(window, f"window.__selectOnly({page})")


def inspect(window, label: str, expect_cleaned: bool) -> None:
    tabs = json.loads(js(window, "window.__tabs()"))
    result = js(window, "JSON.stringify(window.__lastResult()?.cleaning || null)")
    cleaning = json.loads(result) if result else None
    check(f"{label}: response carries cleaning", (cleaning is not None) == expect_cleaned,
          f"source={cleaning and cleaning.get('source')} urls={sorted((cleaning or {}).get('urls', {}))}")
    for view in ("cleaned", "elements"):
        check(f"{label}: {view} tab enabled", (not tabs[view]["disabled"]) == expect_cleaned, tabs[view]["title"][:110])
    if expect_cleaned and not tabs["cleaned"]["disabled"]:
        js(window, "document.querySelector('#view-tabs button[data-view=cleaned]').click(); true")
        src = wait_for(window, "[...document.querySelectorAll('#stage img')].map(i => i.src).find(s => s.includes('/cleaning/cleaned.png')) || ''", timeout=10)
        loaded = wait_for(window, "[...document.querySelectorAll('#stage img')].some(i => i.src.includes('/cleaning/cleaned.png') && i.complete && i.naturalWidth > 0)", timeout=20)
        check(f"{label}: Cleaned view shows the cleaned plan", bool(src) and bool(loaded), src)
        js(window, "document.querySelector('#view-tabs button[data-view=elements]').click(); true")
        loaded = wait_for(window, "[...document.querySelectorAll('#stage img')].some(i => i.src.includes('/cleaning/elements.png') && i.complete && i.naturalWidth > 0)", timeout=20)
        check(f"{label}: Elements view shows the typed elements", bool(loaded))


def drive(window, server) -> None:
    STATE["server"] = server
    try:
        wait_for(window, f"location.href.startsWith('{server.url}') && document.readyState === 'complete' && !!document.getElementById('cleaning-toggle')", timeout=30)
        js(window, HELPERS)
        first = True
        for spec in PLANS:
            path, page = spec.rsplit(":", 1)
            path, page = Path(path), int(page)
            # order A (the natural one): tick the toggle first, then load the file, pick the page, analyze
            toggle(window, True)
            load(window, path, page)
            check(f"{path.name}: toggle still ticked after loading the file", js(window, "document.getElementById('cleaning-toggle').checked"))
            state = analyze(window)
            check(f"{path.name} p{page}: analysis finished", state == "done", str(state))
            inspect(window, f"{path.name} p{page} [toggle before load]", True)
            if first:
                first = False
                # order B: analysed WITHOUT cleaning, then the toggle is ticked
                toggle(window, False)
                state = analyze(window)
                inspect(window, f"{path.name} p{page} [no cleaning]", False)
                ticked = toggle(window, True)
                label = js(window, "document.getElementById('analyze-label').textContent")
                check(f"{path.name}: ticking the toggle after an analysis offers re-analysis", ticked and "clean" in label.lower(), label)
                state = analyze(window)
                inspect(window, f"{path.name} p{page} [ticked after analysis, re-analysed]", True)
                # order C: clear the file (X) - the toggle must keep its state
                js(window, "document.getElementById('clear-file').click(); true")
                check("toggle kept after clearing the file", js(window, "document.getElementById('cleaning-toggle').checked"))
    except Exception as error:
        check("cleaning smoke run completed without exceptions", False, repr(error))
    finally:
        window.destroy()


def main() -> int:
    desktop.run_desktop(after_load=drive)
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name in failed:
        print("FAILED:", name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
