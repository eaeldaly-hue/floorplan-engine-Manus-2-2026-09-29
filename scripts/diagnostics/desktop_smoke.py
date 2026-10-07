"""GUI smoke test for the desktop app (opens a real window; not part of pytest).

Launches the app exactly as ``desktop.py`` does, drives the Workbench inside
the native WKWebView via JavaScript (sample → analyze → views → zoom →
highlight → raw JSON → PNG/JPEG upload → replace → invalid and corrupt
files), closes the window, and checks the Flask child process is gone.

The native open/save panels cannot be scripted; files are injected into the
real <input type=file> instead. Click "Replace" and a download link once by
hand to check those panels.

    .venv/bin/python -m scripts.diagnostics.desktop_smoke
"""

from __future__ import annotations

import json
import os
import sys
import time

import desktop

RESULTS: list[tuple[str, bool, str]] = []
STATE: dict = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{' — ' + detail if detail else ''}", flush=True)


def js(window, code: str):
    return window.evaluate_js(code)


def wait_for(window, expression: str, timeout: float = 120.0, interval: float = 0.25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = js(window, expression)
        if value:
            return value
        time.sleep(interval)
    return None


# Async browser-side helpers; results are polled from window.__smoke.
HELPERS = r"""
window.__smoke = {};
window.__putFile = (file) => {
  const dt = new DataTransfer(); dt.items.add(file);
  const input = document.getElementById('plan-file'); input.files = dt.files;
  input.dispatchEvent(new Event('change', { bubbles: true }));
};
window.__makeJpeg = async () => {
  const blob = await (await fetch('/api/sample.png')).blob();
  const bmp = await createImageBitmap(blob);
  const w = 1800, h = Math.round(bmp.height * w / bmp.width);
  const canvas = document.createElement('canvas'); canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d'); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, w, h); ctx.drawImage(bmp, 0, 0, w, h);
  const jpg = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92));
  return new File([jpg], 'sample_1800.jpg', { type: 'image/jpeg' });
};
window.__makePng = async () => new File([await (await fetch('/api/sample.png')).blob()], 'plan_upload.png', { type: 'image/png' });
window.__consistency = () => {
  const data = JSON.parse(document.getElementById('raw-json').textContent);
  const roomRows = document.querySelectorAll('#rooms-table tbody tr[data-key]').length;
  const openingRows = document.querySelectorAll('#openings-table tbody tr[data-key]').length;
  const metric = (label) => [...document.querySelectorAll('#metrics .metric')].find((m) => m.querySelector('dt').textContent.trim() === label)?.querySelector('dd').childNodes[0].textContent.trim();
  return JSON.stringify({
    source: data.source_name, image: data.image, rooms: data.room_count, roomRows,
    openings: (data.openings || []).length, openingRows,
    doors: data.door_count, windows: data.window_count,
    uiRooms: metric('Rooms'), uiDoors: metric('Doors'), uiWindows: metric('Windows'), uiOpenings: metric('Openings total'),
  });
};
true;
"""


def analyze_and_wait(window) -> str:
    js(window, "document.getElementById('analyze-button').click(); true")
    wait_for(window, "document.getElementById('status-card').dataset.state === 'running'", timeout=10)
    return wait_for(window, "['done','error'].includes(document.getElementById('status-card').dataset.state) && document.getElementById('status-card').dataset.state", timeout=180)


def drive(window, server) -> None:
    STATE["server"] = server
    try:
        _drive(window, server)
    except Exception as error:  # report, then still close the window
        check("smoke run completed without exceptions", False, repr(error))
    finally:
        window.destroy()


def _drive(window, server) -> None:
    loaded = wait_for(window, f"location.href.startsWith('{server.url}') && document.readyState === 'complete' && !!document.getElementById('sample-button')", timeout=30)
    check("Workbench loads in the desktop window", loaded, server.url)
    check("window title", window.title == desktop.APP_NAME, window.title)
    check("window size ~1400×900", abs(window.width - 1400) <= 2 and abs(window.height - 900) <= 40, f"{window.width}×{window.height}")
    js(window, HELPERS)
    health = wait_for(window, "document.getElementById('health-status').dataset.state !== 'pending' && document.getElementById('health-status').dataset.state", timeout=15)
    check("engine health shown", health in ("ok", "warn"), str(health))

    # 1. Sample plan → analyze
    js(window, "document.getElementById('sample-button').click(); true")
    check("sample plan loads", wait_for(window, "document.getElementById('file-name').textContent === 'test_floorplan.png' && !document.getElementById('analyze-button').disabled", timeout=15))
    started = time.monotonic()
    state = analyze_and_wait(window)
    data = json.loads(js(window, "window.__consistency()"))
    check("sample analysis completes", state == "done", f"{time.monotonic() - started:.1f} s")
    check("results match API JSON", data["roomRows"] == data["rooms"] and data["openingRows"] == data["openings"]
          and data["uiRooms"] == str(data["rooms"]) and data["uiDoors"] == str(data["doors"]) and data["uiWindows"] == str(data["windows"]),
          f"rooms {data['rooms']}, openings {data['openings']}, doors {data['doors']}, windows {data['windows']}")

    # 2. Views, zoom, highlight
    for view in ("rooms", "openings", "combined"):
        js(window, f"document.querySelector('#view-tabs [data-view=\"{view}\"]').click(); true")
        ok = wait_for(window, f"[...document.querySelectorAll('.pane')].map(p => p.dataset.kind).includes('{view}') && [...document.querySelectorAll('.plane img')].every(i => i.complete && i.naturalWidth > 0)", timeout=15)
        check(f"{view} view renders", ok)
    check("combined view draws API geometry", js(window, "document.querySelectorAll('.pane[data-kind=\"combined\"] .geo').length") > 0)
    js(window, "document.querySelector('#zoom-tabs [data-zoom=\"2\"]').click(); true")
    zoomed = wait_for(window, "(() => { const s = document.querySelector('.pane-scroll'); return s.scrollWidth > s.clientWidth; })()", timeout=5)
    js(window, "document.querySelector('#zoom-tabs [data-zoom=\"1\"]').click(); true")
    check("zoom 200% makes the plan scrollable", zoomed)
    highlight = js(window, """(() => {
      const room = document.querySelector('#rooms-table tbody tr[data-key]');
      const opening = document.querySelector('#openings-table tbody tr[data-key]');
      room.dispatchEvent(new MouseEvent('mouseenter'));
      const r = [...document.querySelectorAll('.hl-layer')].map(l => l.children.length);
      room.dispatchEvent(new MouseEvent('mouseleave'));
      opening.dispatchEvent(new MouseEvent('mouseenter'));
      const o = [...document.querySelectorAll('.hl-layer')].map(l => l.children.length);
      opening.dispatchEvent(new MouseEvent('mouseleave'));
      return JSON.stringify({ r, o });
    })()""")
    hl = json.loads(highlight)
    check("room/opening highlight", all(n == 2 for n in hl["r"] + hl["o"]), highlight)
    raw = js(window, "document.querySelector('.raw-card details').open = true; document.getElementById('raw-json').textContent.length")
    check("raw JSON shown", raw > 1000, f"{raw} chars")
    links = json.loads(js(window, """JSON.stringify(['download-json','download-rooms-overlay','download-openings-overlay'].map(id => {
      const a = document.getElementById(id); return [id, a.getAttribute('download'), (a.getAttribute('href') || '').slice(0, 24)]; }))"""))
    check("download links carry download filenames", all(d and h for _, d, h in links), json.dumps(links))
    overlay_ok = js(window, "(() => { const x = new XMLHttpRequest(); x.open('GET', document.getElementById('download-rooms-overlay').href, false); x.send(); return x.status; })()")
    check("overlay download URL serves the PNG", overlay_ok == 200, str(overlay_ok))

    # 3. PNG upload, JPEG replace, sequential analyses
    js(window, "window.__makePng().then(f => { window.__putFile(f); window.__smoke.png = true; }); true")
    check("PNG upload selected", wait_for(window, "window.__smoke.png && document.getElementById('file-name').textContent === 'plan_upload.png' && document.getElementById('results').hidden", timeout=15))
    js(window, "window.__makeJpeg().then(f => { window.__putFile(f); window.__smoke.jpg = true; }); true")
    check("Replace with JPEG", wait_for(window, "window.__smoke.jpg && document.getElementById('file-name').textContent === 'sample_1800.jpg'", timeout=15))
    state = analyze_and_wait(window)
    data = json.loads(js(window, "window.__consistency()"))
    check("JPEG analysis completes (2nd analysis)", state == "done" and data["source"] == "sample_1800.jpg" and data["image"]["width"] == 1800,
          f"{data['image']['width']}×{data['image']['height']}, rooms {data['rooms']}, openings {data['openings']}")
    js(window, "window.__makePng().then(f => { window.__putFile(f); window.__smoke.png2 = true; }); true")
    wait_for(window, "window.__smoke.png2 && document.getElementById('file-name').textContent === 'plan_upload.png'", timeout=15)
    state = analyze_and_wait(window)
    data = json.loads(js(window, "window.__consistency()"))
    check("PNG analysis completes (3rd analysis)", state == "done" and data["source"] == "plan_upload.png" and data["rooms"] == data["roomRows"],
          f"rooms {data['rooms']}, openings {data['openings']}")

    # 4. Invalid and corrupt files
    js(window, "window.__putFile(new File(['not a plan'], 'notes.txt', { type: 'text/plain' })); true")
    invalid = wait_for(window, "!document.getElementById('error-box').hidden && document.getElementById('error-box').textContent", timeout=5)
    check("invalid file rejected", invalid and "Unsupported" in invalid, str(invalid))
    js(window, "window.__putFile(new File([new Uint8Array([1,2,3,4,5,6,7,8])], 'broken.png', { type: 'image/png' })); true")
    wait_for(window, "document.getElementById('file-name').textContent === 'broken.png'", timeout=10)
    state = analyze_and_wait(window)
    error = js(window, "document.getElementById('error-box').textContent")
    check("corrupt PNG reports server error", state == "error" and bool(error), str(error))
    check("controls re-enabled after error", js(window, "!document.getElementById('analyze-button').disabled && !document.getElementById('sample-button').disabled"))


def main() -> int:
    exit_code = desktop.run_desktop(after_load=drive)
    server = STATE.get("server")
    if server is not None:
        pid = server.process.pid
        try:
            os.kill(pid, 0)
            alive = server.process.poll() is None
        except ProcessLookupError:
            alive = False
        check("app closed cleanly", exit_code == 0)
        check("Flask child process stopped", not alive and not server.running, f"pid {pid}, exit {server.process.returncode}")
        check("server port released", desktop.port_is_free(server.port), str(server.port))
    else:
        check("server started", False)
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
