"""Tests for the desktop launcher (desktop.py). No GUI window is opened here."""

import plistlib
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

import desktop

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_module_imports_without_opening_a_window():
    assert desktop.APP_NAME == "Floor Plan Engine"
    assert desktop.BUNDLE_ID == "com.floorplan.engine"
    assert desktop.HOST == "127.0.0.1"
    # pywebview (and the GUI) is only loaded by run_desktop(), never on import.
    probe = subprocess.run(
        [sys.executable, "-c", "import sys, desktop; print('webview' in sys.modules)"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    assert probe.stdout.strip() == "False"


def test_choose_port_prefers_free_port_and_falls_back_when_taken():
    preferred = desktop.free_port()
    assert desktop.choose_port(preferred) == preferred

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind((desktop.HOST, 0))
        busy.listen()
        taken = busy.getsockname()[1]
        chosen = desktop.choose_port(taken)
        assert chosen != taken
        assert desktop.port_is_free(chosen)


def test_serve_runs_existing_app_on_localhost_without_reloader(monkeypatch):
    import app as app_module

    calls = []
    monkeypatch.setattr(app_module.app, "run", lambda **kwargs: calls.append(kwargs))
    desktop.serve(port=5123, parent_pid=None)
    assert calls == [{"host": "127.0.0.1", "port": 5123, "debug": False, "use_reloader": False, "threaded": True}]


def test_server_process_starts_answers_and_stops():
    server = desktop.ServerProcess(desktop.free_port())
    server.start()
    try:
        server.wait_until_ready(timeout=60)
        with urllib.request.urlopen(f"{server.url}/health", timeout=5) as response:
            assert response.status == 200
        with urllib.request.urlopen(f"{server.url}/", timeout=5) as response:
            assert "Analysis Workbench" in response.read().decode()
    finally:
        server.stop()
    assert not server.running
    assert server.process.returncode is not None
    assert desktop.port_is_free(server.port)


def test_wait_until_ready_fails_fast_when_server_exits():
    server = desktop.ServerProcess(desktop.free_port())
    server.process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    server.process.wait()
    with pytest.raises(desktop.ServerStartError, match="exit code 3"):
        server.wait_until_ready(timeout=5)


def test_server_exits_when_parent_process_dies():
    fake_parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    port = desktop.free_port()
    child = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "desktop.py"), "--serve", "--port", str(port), "--parent-pid", str(fake_parent.pid)],
        cwd=PROJECT_ROOT,
    )
    try:
        server = desktop.ServerProcess(port)
        server.process = child
        server.wait_until_ready(timeout=60)
        fake_parent.kill()
        fake_parent.wait()
        assert child.wait(timeout=10) == 0
    finally:
        fake_parent.kill()
        if child.poll() is None:
            child.kill()
            child.wait()


@pytest.mark.skipif(sys.platform != "darwin", reason="builds a macOS app bundle")
def test_build_script_produces_app_bundle(tmp_path):
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build_macos_app.py"), "--output", str(tmp_path)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    app = tmp_path / "Floor Plan Engine.app"
    assert "Built:" in result.stdout
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleName"] == "Floor Plan Engine"
    assert info["CFBundleIdentifier"] == "com.floorplan.engine"
    assert info["LSArchitecturePriority"][0] == "arm64"
    launcher = app / "Contents" / "MacOS" / info["CFBundleExecutable"]
    assert launcher.stat().st_mode & 0o111
    script = launcher.read_text()
    assert "desktop.py" in script and ".venv/bin/python" in script
    assert "arch -arm64" in script  # avoid launching the arm64 .venv under Rosetta
    subprocess.run(["bash", "-n", str(launcher)], check=True)
