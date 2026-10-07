# Floor Plan Engine — macOS desktop app

A thin desktop window around the existing Workbench. It is a developer/testing
build, not a distributable product.

## Development

```bash
.venv/bin/python desktop.py
```

Starts the Flask app in a background process on `127.0.0.1`, waits until
`/health` answers, then opens the Workbench in a native window (1400×900,
resizable, minimum 1000×700). Closing the window (or ⌘Q) stops the server.

Headless check (no window — starts the server, confirms it answers, stops it):

```bash
.venv/bin/python desktop.py --check
```

## Browser mode (unchanged)

```bash
.venv/bin/python run.py
```

Then open <http://127.0.0.1:8000>. Both modes can run at the same time.

## Building `Floor Plan Engine.app`

```bash
.venv/bin/python scripts/build_macos_app.py
```

Writes `dist/Floor Plan Engine.app` (git-ignored). Double-click it, or move it to
`~/Applications` (`--output ~/Applications` builds it there directly).

The bundle is a launcher: it runs `desktop.py` with this project's `.venv`
Python, so it always uses the current source tree. It is tied to this project
folder — **rebuild it if the folder moves**. It does not bundle Python or the
dependencies. Output from the app goes to
`~/Library/Logs/Floor Plan Engine/desktop.log`.

## Architecture

```
Floor Plan Engine.app ─► desktop.py (pywebview window, WKWebView)
                              │ starts and stops
                              ▼
                   Flask child process (app.py, 127.0.0.1)
                              │ same /api/analyze as browser mode
                              ▼
                        engine/ (unchanged)
```

There is one analysis path. The window only displays the existing Workbench
(`templates/` + `static/`); `desktop.py` contains no analysis code.

- **Port:** 8000 if it is free, otherwise a free local port (override with
  `FLOORPLAN_DESKTOP_PORT=…`). The window loads whichever port the server got.
- **Server process:** `desktop.py --serve` runs `app.run(..., debug=False,
  use_reloader=False)`, so there is one server process and no reloader. It exits
  on its own if the desktop app dies, so it cannot be orphaned.
- **Downloads** (JSON, overlay PNGs) use the native save panel; uploads use the
  native open panel.

GUI smoke test (opens a real window, drives the Workbench, closes it):

```bash
.venv/bin/python -m scripts.diagnostics.desktop_smoke
```

## Troubleshooting

| Problem | What to do |
|---|---|
| Port 8000 already in use | Nothing — the app picks another free port. To force one: `FLOORPLAN_DESKTOP_PORT=8123 .venv/bin/python desktop.py`. |
| `ModuleNotFoundError: webview` | `.venv/bin/pip install -r requirements.txt` (installs `pywebview` and its PyObjC dependencies on macOS). |
| App shows "could not start" | The server failed to start. Run `.venv/bin/python desktop.py` in a terminal, or read `~/Library/Logs/Floor Plan Engine/desktop.log`. |
| `.app` does nothing / alert "Project or .venv not found" | The project folder moved or `.venv` is missing. Recreate `.venv`, then rebuild the app. |
| `incompatible architecture (have 'arm64', need 'x86_64')` in the log | The app was launched under Rosetta. Rebuild with the current script (it forces native arm64); don't tick "Open using Rosetta" in Finder's Get Info. |
| OCR unavailable in the app but fine in a terminal | Finder starts apps with a minimal `PATH`. The launcher adds `/opt/homebrew/bin` and `/usr/local/bin`; if Tesseract lives elsewhere, set `TESSERACT_CMD` in the launcher or install it with Homebrew. |
| macOS blocks opening the app | Locally built apps are not quarantined, so this is rare. If it happens: right-click the app → Open, or allow it in System Settings → Privacy & Security. No special permissions are needed (the server only listens on 127.0.0.1). |
| Menu bar/Dock show "Python" details | Expected for this developer build: the app runs the framework Python. The window title, menu name and Dock icon are set to Floor Plan Engine at runtime. |
