"""DWG export: the DXF drawing of engine.export.dxf converted to DWG by a local converter.

No Python library writes DWG under a permissive licence, so the conversion runs a converter
installed on this computer; nothing is uploaded anywhere. Backends, in order of preference unless
FLOORPLAN_DWG_BACKEND names one (libredwg | oda | off):

  libredwg   GNU LibreDWG's dxf2dwg / dwg2dxf (GPL-3.0, run as a separate program, never linked or
             bundled; e.g. `brew install libredwg`). Writes R2000 DWG (AC1015).
  oda        the ODA File Converter through ezdxf.addons.odafc (free of charge, proprietary licence
             accepted by the user when installing it). Not verified in this project's environment.

Every conversion is checked before the file is returned: the DWG header names the requested
version, and the DWG read back by the same converter holds the same layers and the same entities
per layer as the DXF it was made from. That check uses the converter's own reader; it is not an
independent DWG validation. A failed check is an error: a DXF is never returned in place of a DWG.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import ezdxf

from .dxf import build_dxf

# R2000 only: LibreDWG 0.14 also writes R2004, but its own reader cannot read those files back
# completely, so they cannot be checked (measured on this project's exports)
VERSIONS = {"R2000": "AC1015"}
SEARCH = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")   # GUI apps on macOS do not inherit the shell PATH


class DwgUnavailable(RuntimeError):
    """No DWG converter is installed (or DWG export is switched off)."""


class DwgConversionError(RuntimeError):
    """The converter failed, or its DWG did not hold the drawing it was given."""


def _which(name: str) -> str | None:
    return shutil.which(name) or next((str(Path(d) / name) for d in SEARCH if (Path(d) / name).is_file()), None)


def _libredwg() -> tuple[str, str] | None:
    exe = os.environ.get("FLOORPLAN_DXF2DWG") or _which("dxf2dwg")
    back = _which("dwg2dxf") if not os.environ.get("FLOORPLAN_DXF2DWG") else str(Path(exe).with_name("dwg2dxf"))
    return (exe, back) if exe and back and Path(back).exists() else None


def _oda() -> bool:
    try:
        from ezdxf.addons import odafc
        return bool(odafc.is_installed())
    except Exception:
        return False


def backend() -> dict:
    """The DWG converter this computer offers: {available, backend, verified, reason}."""
    choice = os.environ.get("FLOORPLAN_DWG_BACKEND", "auto").lower()
    if choice == "off":
        return {"available": False, "backend": None, "verified": False, "reason": "DWG export is switched off."}
    if choice in ("auto", "libredwg") and _libredwg():
        return {"available": True, "backend": "libredwg", "verified": True,
                "reason": "GNU LibreDWG dxf2dwg (R2000); checked by a round trip through LibreDWG's own reader."}
    if choice in ("auto", "oda") and _oda():
        return {"available": True, "backend": "oda", "verified": False,
                "reason": "ODA File Converter via ezdxf.addons.odafc; not verified in this project's environment."}
    return {"available": False, "backend": None, "verified": False,
            "reason": "No DWG converter installed: install GNU LibreDWG (dxf2dwg) or the ODA File Converter. "
                      "DXF export does not need one."}


def _summary(doc) -> dict:
    out: dict = {}
    for e in doc.modelspace():
        key = f"{e.dxf.layer}/{e.dxftype()}"
        out[key] = out.get(key, 0) + 1
    return out


def _convert_libredwg(tools, src: Path, dst: Path, version: str, timeout: float):
    exe, back = tools
    run = subprocess.run([exe, "-y", "--as", version.lower(), "-o", str(dst), str(src)],
                         capture_output=True, text=True, timeout=timeout)
    if run.returncode != 0 or not dst.is_file():
        raise DwgConversionError(f"dxf2dwg failed (exit {run.returncode}): {(run.stderr or run.stdout)[-400:]}")
    _check_header(dst, version)
    readback = dst.with_suffix(".back.dxf")
    run = subprocess.run([back, "-y", "-o", str(readback), str(dst)], capture_output=True, text=True, timeout=timeout)
    if run.returncode != 0 or not readback.is_file():
        raise DwgConversionError(f"dwg2dxf could not read the DWG back (exit {run.returncode}).")
    try:
        return ezdxf.readfile(readback)
    except Exception as exc:
        raise DwgConversionError(f"The DWG read back by dwg2dxf is not a readable drawing: {exc}") from exc


def _check_header(dst: Path, version: str) -> None:
    head = dst.read_bytes()[:6] if dst.is_file() else b""
    if head.decode("ascii", "replace") != VERSIONS[version]:
        raise DwgConversionError(f"The converter's output is not a {version} DWG (header {head!r}).")


def _convert_oda(doc, dst: Path, version: str):
    from ezdxf.addons import odafc
    odafc.export_dwg(doc, str(dst), version=version, replace=True)
    return odafc.readfile(str(dst))


def dwg_bytes(result: dict, version: str = "R2000", timeout: float = 120.0) -> tuple[bytes, dict]:
    """The analysis `result` as DWG bytes, with the DXF export report and the conversion check."""
    if version not in VERSIONS:
        raise ValueError(f"Unsupported DWG version {version}; use one of {', '.join(VERSIONS)}.")
    info = backend()
    if not info["available"]:
        raise DwgUnavailable(info["reason"])
    doc, report = build_dxf(result)
    expected = _summary(doc)
    with tempfile.TemporaryDirectory(prefix="fpe-dwg-") as tmp:
        src, dst = Path(tmp) / "plan.dxf", Path(tmp) / "plan.dwg"
        doc.saveas(src)
        try:
            back = (_convert_libredwg(_libredwg(), src, dst, version, timeout) if info["backend"] == "libredwg"
                    else _convert_oda(doc, dst, version))
        except subprocess.TimeoutExpired as exc:
            raise DwgConversionError(f"The DWG converter did not finish within {timeout:.0f} s.") from exc
        _check_header(dst, version)
        data = dst.read_bytes()
        got = _summary(back)
        if got != expected:
            diff = {k: (expected.get(k), got.get(k)) for k in sorted(set(expected) | set(got)) if expected.get(k) != got.get(k)}
            raise DwgConversionError(f"The DWG does not hold the exported drawing (layer/entity: DXF, DWG): {diff}")
    return data, report.as_dict() | {"dwg": {"version": version, "backend": info["backend"], "bytes": len(data),
                                             "entities": sum(expected.values()),
                                             "check": "header + same layers and entities after a round trip through "
                                                      "the converter's own reader (not an independent validation)"}}
