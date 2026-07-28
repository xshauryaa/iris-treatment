"""
compute_core_metrics.py

CORE Trajectory Analysis (6.3.3): computes the two aggregate,
Day-0-anchored measures of belief-graph drift, per instance.

  - cosine_similarity_to_day0 : cosine similarity between the Day 0
    confidence vector and the Day N confidence vector. Sensitive to
    SHAPE (relative pattern across the 18 beliefs); invariant to
    uniform scaling.

  - polygon_area : Area = 0.5 * sin(2*pi/N) * sum_i(r_i * r_{i+1}),
    N=18 equally-spaced spokes, node order held fixed across all days.
    Sensitive to MAGNITUDE, not shape -- the blind spot cosine
    similarity leaves open.

Both measures are Day-0-anchored only, per the locked 6.3.3 spec --
this script does NOT compute day-to-day (previous-day) comparisons.

Area "relative to Day 0" is reported as a difference:
  - polygon_area_delta_to_day0 = area_N - area_0
(locked as delta, not ratio -- do not silently reintroduce a ratio
column without updating the 6.3.3 text to match, since delta and ratio
tell different stories and switching mid-paper would be inconsistent.)

Both arms share an identical Day 0 CORE state, so raw deltas are
directly comparable between treatment and baseline without needing to
also report Day-0 area as a normalizing baseline.

LAYOUT ASSUMED (per-instance, single arm):
    <this script's directory>/
        core_initial_state.json      <- Day 0
        core_checkpoints/
            core_state_day_1.json ... core_state_day_36.json

Output: BASE_DIR/core_metrics_<ARM_LABEL>.csv
"""

import os
import csv
import json
import math


def resolve_day_path(day, base_dir, arm_dir, anchor_day):
    """Day == anchor_day resolves to BASE_DIR/core_initial_state.json
    (pre-experiment state -- lives in base_dir, NOT the per-day
    checkpoints directory). Every other day resolves to
    ARM_DIR/core_state_day_{day}.json."""
    if day == anchor_day:
        return os.path.join(base_dir, "core_initial_state.json")
    return os.path.join(arm_dir, f"core_state_day_{day}.json")


def load_core_leaves(filepath):
    """Load a core_state_day_N.json (or core_initial_state.json) file and
    return an ordered list of (id, label, category, confidence) tuples for
    the 18 leaf beliefs. Raises loudly on missing/malformed structure --
    a silently-wrong belief count is worse than a crash."""
    with open(filepath, "r") as f:
        data = json.load(f)

    children = data.get("children", [])
    leaves = []
    for node in children:
        conf = node.get("confidence")
        if conf is None:
            continue
        leaves.append(
            (node["id"], node["label"], node.get("category"), float(conf))
        )

    if len(leaves) == 0:
        raise ValueError(f"No leaf beliefs with confidence found in {filepath}")

    return leaves

# ---------------------------------------------------------------------------
# CONFIG -- mirrors generate_core_radar_charts.py
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
ARM_DIR = os.path.join(BASE_DIR, "core_checkpoints")
DAY_RANGE = range(1, 37)  # Day 1 .. Day 36 inclusive
ANCHOR_DAY = 0            # core_initial_state.json

ARM_LABEL = "treatment"   # set to "baseline" on the baseline instance
OUTPUT_CSV = os.path.join(BASE_DIR, f"core_metrics_{ARM_LABEL}.csv")


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------
def leaves_to_vector(leaves, id_order):
    """Confidence vector in a fixed id order, so comparisons line up by
    belief id regardless of child order in the source JSON."""
    by_id = {lid: conf for lid, lbl, cat, conf in leaves}
    missing = set(id_order) - set(by_id)
    if missing:
        raise ValueError(f"Beliefs missing from file: {missing}")
    return [by_id[lid] for lid in id_order]


def polygon_area(values):
    """Area = 0.5 * sin(2*pi/n) * sum_i(r_i * r_{i+1}), r wraps (r_n = r_0)."""
    n = len(values)
    angle_step = 2 * math.pi / n
    total = sum(values[i] * values[(i + 1) % n] for i in range(n))
    return 0.5 * math.sin(angle_step) * total


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return None  # undefined -- flag rather than divide by zero
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    anchor_path = resolve_day_path(ANCHOR_DAY, BASE_DIR, ARM_DIR, ANCHOR_DAY)
    if not os.path.exists(anchor_path):
        raise FileNotFoundError(
            f"Anchor file not found at {anchor_path} -- cannot compute "
            f"Day-0-anchored metrics without it."
        )
    anchor_leaves = load_core_leaves(anchor_path)
    id_order = [lid for lid, lbl, cat, conf in anchor_leaves]

    anchor_vector = leaves_to_vector(anchor_leaves, id_order)
    anchor_area = polygon_area(anchor_vector)

    rows = [{
        "day": ANCHOR_DAY,
        "cosine_similarity_to_day0": 1.0,
        "polygon_area": anchor_area,
        "polygon_area_delta_to_day0": 0.0,
    }]

    for day in DAY_RANGE:
        filepath = resolve_day_path(day, BASE_DIR, ARM_DIR, ANCHOR_DAY)
        if not os.path.exists(filepath):
            print(f"[skip] {filepath} not found")
            continue

        leaves = load_core_leaves(filepath)
        vector = leaves_to_vector(leaves, id_order)
        area = polygon_area(vector)
        cos_sim = cosine_similarity(vector, anchor_vector)

        rows.append({
            "day": day,
            "cosine_similarity_to_day0": cos_sim,
            "polygon_area": area,
            "polygon_area_delta_to_day0": area - anchor_area,
        })

        print(f"[ok] Day {day}: cos_to_day0={cos_sim:.4f}, area={area:.4f} "
              f"(delta={area - anchor_area:+.4f})")

    fieldnames = [
        "day", "cosine_similarity_to_day0", "polygon_area",
        "polygon_area_delta_to_day0",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} rows written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()