"""Regression: Architectural Cleaning reachable from the Workbench.

Bug (2026-10-07): the Cleaned / Elements tabs stayed locked in the desktop app. The server worked;
the toggle only applies to the next analysis, so ticking it after an analysis changed nothing and
said nothing, and clearing the file (form reset) or restarting the app unticked it silently.

- the PDF page route returns working cleaning artifacts when asked, and none otherwise
- the tab tooltip states why a cleaning view is unavailable (viewer.js, run under Node)
- the toggle is a remembered preference (not reset with the file form) and a mismatch between the
  toggle and the shown result is announced on the Analyze button (main.js)

The full click-through in the real desktop window: scripts/diagnostics/desktop_cleaning_smoke.py.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import create_app
from tests.test_pdf_input import vector_pdf

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "static" / "js"


def hollow_wall_ops(w: float, h: float) -> str:
    """3 x 2 rooms; every wall drawn as its outline in a heavy pen (hollow walls), a door gap, and
    thin-pen furniture."""
    s = 2                                  # raster scale for building the outline
    t = 10                                 # wall thickness, points
    band = np.zeros((int(h * s), int(w * s)), np.uint8)
    xs = [0.1 * w, 0.37 * w, 0.63 * w, 0.9 * w]
    ys = [0.15 * h, 0.5 * h, 0.85 * h]
    for x in xs:
        cv2.rectangle(band, (int((x - t / 2) * s), int((ys[0] - t / 2) * s)), (int((x + t / 2) * s), int((ys[-1] + t / 2) * s)), 255, -1)
    for y in ys:
        cv2.rectangle(band, (int((xs[0] - t / 2) * s), int((y - t / 2) * s)), (int((xs[-1] + t / 2) * s), int((y + t / 2) * s)), 255, -1)
    gx = xs[1]
    band[int((0.27 * h) * s):int((0.38 * h) * s), int((gx - t) * s):int((gx + t) * s)] = 0     # door gap
    contours, _ = cv2.findContours(band, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ops = ["0 G 2.2 w"]
    for c in contours:
        pts = [(p[0][0] / s, h - p[0][1] / s) for p in c]
        ops.append(f"{pts[0][0]:.1f} {pts[0][1]:.1f} m " + " ".join(f"{x:.1f} {y:.1f} l" for x, y in pts[1:]) + " h S")
    ops.append("0.3 w")                    # furniture in a light pen
    ops.append(f"{0.2 * w:.1f} {0.3 * h:.1f} {0.08 * w:.1f} {0.06 * h:.1f} re S")
    return "\n".join(ops)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wb")
    app = create_app({"TESTING": True, "UPLOAD_FOLDER": str(tmp / "results"), "PDF_FOLDER": str(tmp / "pdfs")})
    return app.test_client()


@pytest.fixture(scope="module")
def pdf_id(client):
    payload = vector_pdf([{"w": 1100, "h": 760, "ops": hollow_wall_ops(1100, 760)}])
    res = client.post("/api/pdf", data={"file": (io.BytesIO(payload), "walls.pdf")}, content_type="multipart/form-data")
    assert res.status_code == 200, res.get_json()
    return res.get_json()["pdf_id"]


def test_pdf_page_with_cleaning_returns_working_cleaned_and_elements_views(client, pdf_id):
    res = client.post(f"/api/pdf/{pdf_id}/pages/1/analyze", json={"cleaning": "on"})
    data = res.get_json()
    assert res.status_code == 200, data
    cleaning = data["cleaning"]
    assert cleaning["applicable"] and cleaning["source"].startswith("vector"), cleaning["reason"]
    assert cleaning["mode_used"] == "cleaned"
    for view in ("cleaned", "elements"):                      # what the Workbench tabs need
        url = cleaning["urls"][view]
        img = client.get(url)
        assert img.status_code == 200 and img.data[:8] == b"\x89PNG\r\n\x1a\n", url


def test_pdf_page_without_cleaning_has_no_cleaning_block(client, pdf_id):
    data = client.post(f"/api/pdf/{pdf_id}/pages/1/analyze", json={}).get_json()
    assert "cleaning" not in data


# --- Workbench logic ------------------------------------------------------------------------

node = shutil.which("node")


@pytest.mark.skipif(node is None, reason="Node.js not installed")
def test_cleaning_tab_tooltip_explains_why_it_is_locked(tmp_path):
    for f in JS.glob("*.js"):
        shutil.copy(f, tmp_path / f.name)
    (tmp_path / "package.json").write_text('{"type": "module"}')
    script = """
      import { cleaningTabTitle } from './viewer.js';
      const help = 'HELP';
      console.log(JSON.stringify({
        none: cleaningTabTitle(null, false, help),
        noCleaning: cleaningTabTitle({ room_count: 3 }, false, help),
        notApplicable: cleaningTabTitle({ cleaning: { reason: 'no wall pen' } }, false, help),
        available: cleaningTabTitle({ cleaning: { urls: { cleaned: 'x' } } }, true, help),
      }));
    """
    (tmp_path / "check.mjs").write_text(script)
    out = subprocess.run([node, "check.mjs"], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    titles = json.loads(out.stdout)
    assert "analysed without cleaning" in titles["noCleaning"] and "Analyze again" in titles["noCleaning"]
    assert "not applicable" in titles["notApplicable"] and "no wall pen" in titles["notApplicable"]
    assert titles["available"] == "HELP" and "Clean plan first" in titles["none"]


def test_cleaning_toggle_is_a_remembered_preference_not_part_of_the_file_form():
    main = (JS / "main.js").read_text()
    clear = main[main.index("function clearFile()"):]
    clear = clear[:clear.index("\n}\n")]
    # the file form is reset, but the toggle is restored right after
    assert re.search(r"const cleaning = cleaningOn\(\);\s*\$\('upload-form'\)\.reset\(\);\s*\$\('cleaning-toggle'\)\.checked = cleaning;", clear)
    # remembered across restarts, and the next run is announced when it differs from the shown result
    assert "localStorage.setItem(CLEANING_KEY" in main and "restoreCleaningChoice();" in main
    listener = main[main.index("getElementById('cleaning-toggle')?.addEventListener('change'"):]
    assert "saveCleaningChoice()" in listener[:200] and "syncCleaningHint()" in listener[:200]
    assert "'Analyze again with cleaning'" in main
