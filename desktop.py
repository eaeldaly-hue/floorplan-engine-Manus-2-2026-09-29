"""Floor Plan Engine — macOS desktop wrapper for the existing Workbench.

Starts the existing Flask app (``app.app``) in a child process bound to
127.0.0.1, waits until it answers ``/health``, and shows the Workbench in a
native window (pywebview → WKWebView). Closing the window stops the server.
There is no analysis code here: the window talks to the same ``/api/analyze``
as the browser workflow.

    .venv/bin/python desktop.py           # open the desktop app
    .venv/bin/python desktop.py --check   # start/stop the server headlessly

Browser mode is unchanged: ``.venv/bin/python run.py``.
"""

from __future__ import annotations

import argparse
import atexit
import html
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

APP_NAME = "Floor Plan Engine"
BUNDLE_ID = "com.floorplan.engine"
HOST = "127.0.0.1"
PREFERRED_PORT = 8000
READY_TIMEOUT_S = 60.0
PROJECT_ROOT = Path(__file__).resolve().parent
ICON_PATH = PROJECT_ROOT / "assets" / "app-icon.png"
LOG_HINT = "~/Library/Logs/Floor Plan Engine/desktop.log"


class ServerStartError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Port selection
# ---------------------------------------------------------------------------

def port_is_free(port: int, host: str = HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def free_port(host: str = HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


def choose_port(preferred: int | None = None) -> int:
    """Use the usual dev port when it is free, otherwise any free local port."""
    if preferred is None:
        preferred = int(os.environ.get("FLOORPLAN_DESKTOP_PORT", PREFERRED_PORT))
    return preferred if port_is_free(preferred) else free_port()


# ---------------------------------------------------------------------------
# Flask child process
# ---------------------------------------------------------------------------

class ServerProcess:
    """The existing Flask app running in a child process on 127.0.0.1."""

    def __init__(self, port: int):
        self.port = port
        self.process: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}"

    def start(self) -> None:
        command = [
            sys.executable, str(PROJECT_ROOT / "desktop.py"),
            "--serve", "--port", str(self.port), "--parent-pid", str(os.getpid()),
        ]
        self.process = subprocess.Popen(command, cwd=PROJECT_ROOT)

    def wait_until_ready(self, timeout: float = READY_TIMEOUT_S) -> None:
        """Poll /health until the server answers; fail fast if the child exits."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                code = None if self.process is None else self.process.returncode
                raise ServerStartError(f"The Flask server exited during startup (exit code {code}).")
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as response:
                    if response.status == 200 and json.load(response).get("status") == "ok":
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError, ValueError):
                pass
            time.sleep(0.1)
        raise ServerStartError(f"The Flask server did not respond on {self.url} within {timeout:.0f} s.")

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        if not self.running:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def start_server(preferred_port: int | None = None, attempts: int = 2) -> ServerProcess:
    """Start the server and wait for it; retry once on another port if startup fails."""
    port = choose_port(preferred_port)
    last_error: ServerStartError | None = None
    for _ in range(attempts):
        server = ServerProcess(port)
        server.start()
        try:
            server.wait_until_ready()
            return server
        except ServerStartError as error:
            server.stop()
            last_error = error
            port = free_port()  # e.g. the preferred port was taken between probe and bind
    raise last_error


def serve(port: int, parent_pid: int | None) -> None:
    """Child-process entry point: run the existing Flask app, no reloader."""
    if parent_pid:
        def exit_with_parent() -> None:
            while True:
                try:
                    os.kill(parent_pid, 0)
                except ProcessLookupError:
                    os._exit(0)  # never outlive the desktop app
                except PermissionError:
                    pass
                time.sleep(1.0)

        threading.Thread(target=exit_with_parent, name="parent-watchdog", daemon=True).start()

    from app import app

    app.run(host=HOST, port=port, debug=False, use_reloader=False, threaded=True)


# ---------------------------------------------------------------------------
# Desktop window
# ---------------------------------------------------------------------------

_PAGE_STYLE = """
<style>
  html, body { height: 100%; margin: 0; }
  body { display: grid; place-items: center; background: #f3f5f8; color: #0f1a2b;
         font: 14px/1.5 -apple-system, system-ui, sans-serif; -webkit-user-select: none; }
  main { text-align: center; max-width: 520px; padding: 24px; }
  h1 { font-size: 17px; margin: 14px 0 4px; font-weight: 650; }
  p { color: #66758a; margin: 0; }
  pre { text-align: left; white-space: pre-wrap; background: #fff; border: 1px solid #e2e7ee;
        border-radius: 8px; padding: 12px; font-size: 12px; color: #c4302b; }
  .spinner { width: 26px; height: 26px; margin: auto; border-radius: 50%;
             border: 3px solid #e2e7ee; border-top-color: #2459d6; animation: spin .8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
"""

SPLASH_HTML = f"""<!doctype html><html><head><meta charset="utf-8">{_PAGE_STYLE}</head>
<body><main><div class="spinner"></div><h1>Starting {APP_NAME}…</h1>
<p>Launching the local analysis server.</p></main></body></html>"""


def error_html(message: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">{_PAGE_STYLE}</head>
<body><main><h1>{APP_NAME} could not start</h1><pre>{html.escape(message)}</pre>
<p>Run <code>.venv/bin/python desktop.py</code> in a terminal to see the server output.
When launched from the .app, output is written to {html.escape(LOG_HINT)}.</p></main></body></html>"""


def _set_macos_app_name() -> None:
    """Show 'Floor Plan Engine' instead of 'Python' in the menu bar when run from source."""
    if sys.platform != "darwin":
        return
    try:
        from Foundation import NSBundle

        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None:
            info["CFBundleName"] = APP_NAME
    except Exception:
        pass


def run_desktop(after_load=None) -> int:
    """Open the app window. ``after_load(window, server)`` is an optional hook
    (used by scripts/diagnostics/desktop_smoke.py) run once the Workbench URL is loaded."""
    import webview

    webview.settings["ALLOW_DOWNLOADS"] = True  # JSON / overlay downloads use the native save panel
    _set_macos_app_name()

    state: dict[str, ServerProcess | None] = {"server": None}

    def shutdown(*_args) -> None:
        if state["server"] is not None:
            state["server"].stop()

    def on_signal(*_args) -> None:
        shutdown()
        os._exit(0)  # leave the Cocoa run loop immediately; the server is already stopped

    atexit.register(shutdown)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, on_signal)

    window = webview.create_window(
        APP_NAME,
        html=SPLASH_HTML,
        width=1400,
        height=900,
        min_size=(1000, 700),
        background_color="#f3f5f8",
    )

    def boot() -> None:
        try:
            state["server"] = start_server()
        except ServerStartError as error:
            window.load_html(error_html(str(error)))
            return
        print(f"{APP_NAME}: Workbench at {state['server'].url}", flush=True)
        window.load_url(state["server"].url)
        if after_load is not None:
            after_load(window, state["server"])

    try:
        webview.start(boot, icon=str(ICON_PATH) if ICON_PATH.is_file() else None)
    finally:
        shutdown()
    return 0


def run_check() -> int:
    """Headless self-check: start the server, confirm it answers, stop it."""
    server = start_server()
    try:
        print(f"Server ready at {server.url} (pid {server.process.pid})")
    finally:
        server.stop()
    print(f"Server stopped (exit code {server.process.returncode})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} desktop app")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--check", action="store_true", help="start and stop the server without a window")
    args = parser.parse_args(argv)

    if args.serve:
        serve(args.port, args.parent_pid)
        return 0
    if args.check:
        return run_check()
    return run_desktop()


if __name__ == "__main__":
    sys.exit(main())
