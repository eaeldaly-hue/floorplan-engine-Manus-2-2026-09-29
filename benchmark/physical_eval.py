"""Physical-space evaluation (open space done right) with per-room regression tracking.

Ground truth: each plan's 'rooms' points plus 'physical' = {must_share: [[i, ...]], may_share: [[i, ...]]}
(see fixtures/real_plans.json notes). Two room points may lie in one physical space only when
they are in a common must_share or may_share set; a must_share set must be one space.

Per room point:
  ok       its space holds only rooms it may share with, and its whole must_share set
  split    its must_share set is spread over several spaces (false physical split)
  merged   its space also holds a room it must not share with (false physical merge)
  missed   no space at the point

    python -m benchmark.physical_eval [--api] [--only ...] [--save out.json] [--compare old.json]

--compare prints previously-correct / newly-fixed / newly-broken rooms against an earlier run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmark.real_plans import DEFAULT_DIR, FIXTURE, _inside, load_plan, space_at

STRUCTURAL = ("wall-region", "open-plan-shared", "merged-space", "passage-partition")


def allowed_pairs(spec: dict) -> tuple[list[set], set]:
    phys = spec.get("physical") or {}
    must = [set(s) for s in phys.get("must_share", [])]
    ok = set()
    for s in phys.get("must_share", []) + phys.get("may_share", []):
        for a in s:
            for b in s:
                ok.add((a, b))
    return must, ok


def evaluate_assignment(spec: dict, where: list) -> dict:
    """where[i]: the space key holding room i (None = no space)."""
    rooms = spec.get("rooms", [])
    must, ok = allowed_pairs(spec)
    n = len(rooms)
    by_key: dict = {}
    for i, k in enumerate(where):
        if k is not None:
            by_key.setdefault(k, []).append(i)
    status = []
    for i in range(n):
        k = where[i]
        if k is None:
            status.append("missed")
            continue
        if any(j != i and (i, j) not in ok for j in by_key[k]):
            status.append("merged")
            continue
        m = next((s for s in must if i in s), None)
        if m and len({where[j] for j in m}) > 1:
            status.append("split")
            continue
        status.append("ok")
    sets = []
    for m in must:
        keys = {where[j] for j in m}
        merged = any(status[j] == "merged" for j in m)
        sets.append("missed" if keys == {None} else "merged" if merged else "split" if len(keys) > 1 else "one_space")
    return {"status": status, "must_share": sets,
            "counts": {k: status.count(k) for k in ("ok", "split", "merged", "missed")}}


def structural_where(labels: np.ndarray, rooms: list[dict]) -> list:
    out = []
    for r in rooms:
        k = space_at(labels, *r["point"])
        out.append(k if k > 0 else None)
    return out


def physical_region(room: dict):
    """(key, polygon, structural) of the physical space a returned room lies in: its
    'physical_space' when it is a zone of a shared space, else its own boundary."""
    if room.get("physical_space"):
        return ("space", room["physical_space"]["id"]), room["physical_space"]["polygon"], True
    b = room.get("boundary")
    if not b:
        return None
    structural = b["method"] in STRUCTURAL
    return (("space", room["space_id"]) if structural and room.get("space_id") else ("room", room["id"])), b["polygon"], structural


def claims(response: dict, rooms: list[dict]) -> dict:
    """How many returned named-room extents claim each labelled room point (1 is right;
    several = stacked / overlapping rooms on screen; 0 = shown as no room)."""
    polys = [r["boundary"]["polygon"] for r in response["rooms"] if r.get("boundary")]
    counts = [sum(1 for poly in polys if _inside(poly, *r["point"])) for r in rooms]
    return {"one": counts.count(1), "several": sum(c > 1 for c in counts), "none": counts.count(0), "per_point": counts}


def api_where(response: dict, rooms: list[dict]) -> list:
    regions = []
    for r in response["rooms"]:
        reg = physical_region(r)
        if reg:
            regions.append(reg)
    for u in response.get("unlabeled_spaces", []):
        if u.get("boundary"):
            regions.append((("space", u["id"]), u["boundary"]["polygon"], True))
    out = []
    for r in rooms:
        hits = [reg for reg in regions if _inside(reg[1], *r["point"])]
        hits.sort(key=lambda reg: not reg[2])            # structural regions first
        out.append(hits[0][0] if hits else None)
    return out


def run(plans: dict, plans_dir: Path, with_api: bool, only=None) -> dict:
    from engine.structure import analyze_structure
    results = {}
    for name, spec in plans.items():
        if (only and name not in only) or spec.get("format_only") or spec.get("out_of_scope") or not spec.get("rooms"):
            continue
        image = load_plan(name, spec, plans_dir)
        s = analyze_structure(image)
        res = {"names": [r.get("name") for r in spec["rooms"]],
               "structure": evaluate_assignment(spec, structural_where(s.space_labels, spec["rooms"]))}
        if with_api:
            from engine.analyzer import FloorPlanAnalyzer
            response = FloorPlanAnalyzer().analyze(image, name)
            res["api"] = evaluate_assignment(spec, api_where(response, spec["rooms"]))
            res["api"]["claims"] = claims(response, spec["rooms"])
        results[name] = res
    return results


def summarize(results: dict, level: str) -> dict:
    tot = {"ok": 0, "split": 0, "merged": 0, "missed": 0}
    sets = {"one_space": 0, "split": 0, "merged": 0, "missed": 0}
    for res in results.values():
        if level not in res:
            continue
        for k, v in res[level]["counts"].items():
            tot[k] += v
        for st in res[level]["must_share"]:
            sets[st] += 1
    pts = sum(tot.values())
    out = {"rooms": pts, **tot, "ok_rate": round(tot["ok"] / max(1, pts), 3), "must_share_sets": sets}
    cl = [res[level]["claims"] for res in results.values() if level in res and "claims" in res[level]]
    if cl:
        out["claims"] = {k: sum(c[k] for c in cl) for k in ("one", "several", "none")}
    return out


def compare(old: dict, new: dict, level: str) -> dict:
    rows = {"previously_correct": 0, "still_correct": 0, "newly_fixed": [], "newly_broken": [], "still_wrong": 0}
    for name, res in new.items():
        if name not in old or level not in res or level not in old[name]:
            continue
        for i, (a, b) in enumerate(zip(old[name][level]["status"], res[level]["status"])):
            label = f"{name}:{res['names'][i]}#{i}"
            if a == "ok":
                rows["previously_correct"] += 1
                if b == "ok":
                    rows["still_correct"] += 1
                else:
                    rows["newly_broken"].append(f"{label} ok->{b}")
            elif b == "ok":
                rows["newly_fixed"].append(f"{label} {a}->ok")
            else:
                rows["still_wrong"] += 1
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--no-ocr-cache", action="store_true", help="run OCR afresh (default: benchmark.ocr_cache)")
    args = ap.parse_args()
    if args.api and not args.no_ocr_cache:
        from benchmark import ocr_cache
        ocr_cache.enable()
    plans = json.loads(FIXTURE.read_text())["plans"]
    t = time.perf_counter()
    results = run(plans, DEFAULT_DIR, args.api, set(args.only.split(",")) if args.only else None)
    levels = ["structure"] + (["api"] if args.api else [])
    for name, res in results.items():
        line = "  ".join(f"{lv}: {res[lv]['counts']} sets {res[lv]['must_share']}" for lv in levels)
        bad = [f"{res['names'][i]}={st}" for i, st in enumerate(res[levels[-1]]["status"]) if st != "ok"]
        print(f"{name:20s} {line}  {'; '.join(bad)}")
    for lv in levels:
        print(lv.upper(), json.dumps(summarize(results, lv)))
    if args.compare:
        old = json.loads(Path(args.compare).read_text())["plans"]
        for lv in levels:
            c = compare(old, results, lv)
            print(f"{lv.upper()} vs {args.compare}: previously correct {c['previously_correct']}, still correct {c['still_correct']}, "
                  f"still wrong {c['still_wrong']}\n  newly fixed {c['newly_fixed']}\n  newly broken {c['newly_broken']}")
    print(f"({time.perf_counter() - t:.0f} s)")
    if args.api and not args.no_ocr_cache:
        print(ocr_cache.summary())
    if args.save:
        Path(args.save).write_text(json.dumps({"plans": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
