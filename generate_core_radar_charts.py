"""
generate_core_radar_charts.py

Converts core_state_day_N.json snapshots (N = 1..36) into 18-spoke radar
charts, one per arm per day, per the CORE Trajectory Analysis design
(Section 6.3.3): each spoke is one of the 18 foundational beliefs
(SM1-6, V1-6, E1-6), spoke length = confidence in [0, 1].

ASSUMPTIONS (edit the CONFIG block below to match your actual layout):
  - You have the core_checkpoints directory, containing
    core_state_day_1.json ... core_state_day_36.json.
  - Radial axis is fixed to [0, 1] for every chart. This is deliberate,
    not a default: if the axis auto-scaled per day, a belief sitting at
    0.3 on a day where everything else is low would look identical to
    a belief at 0.3 on a day where the rest of the graph is near 1.0.
    Fixed scaling is what makes "the polygon visibly collapses" a valid
    visual claim across days instead of an artifact of autoscaling.
  - Category order is fixed as SM -> V -> E (matches the source JSON's
    child order), not alphabetical or randomized, so spoke position is
    stable and comparable across every chart you generate.
  - The root "SELF" node and any node with confidence == null is
    skipped; only the 18 leaf beliefs are plotted. If a day's JSON is
    missing a node (shouldn't happen, but IRIS reject days write an
    unchanged copy) the script will error loudly rather than silently
    plotting 17 spokes -- silent spoke-count drift is worse than a
    crash.
"""

import json
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIG - edit these to match where the two arms' JSON files actually live
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
ARM_DIR = os.path.join(BASE_DIR, "core_checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "core_radar_charts")
DAY_RANGE = range(1, 37)  # Day 1 .. Day 36 inclusive

# Overlay comparison: for each arm, overlay ANCHOR_DAY's polygon against
# every other day's polygon, so drift/collapse away from the starting
# shape is visible in a single figure per day rather than requiring the
# reader to flip between 36 separate single-polygon charts.
ANCHOR_DAY = 0
GENERATE_OVERLAYS = True
ANCHOR_COLOR = "#4C72B0"   # anchor day polygon
CURRENT_COLOR = "#C44E52"  # comparison day polygon

# Fixed spoke order + category-based color coding (color = category, not value)
CATEGORY_COLORS = {
    "self-model": "#4C72B0",
    "value": "#DD8452",
    "epistemic": "#55A868",
}
FILL_COLOR = "#4C72B0"
FILL_ALPHA = 0.25
LINE_COLOR = "#2A3F5F"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def load_core_leaves(filepath):
    """Load a core_state_day_N.json file and return an ordered list of
    (id, label, category, confidence) tuples for the 18 leaf beliefs.
    Raises if the structure doesn't match expectations -- fail loudly."""
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


def plot_radar(leaves, title, output_path):
    """Render one 18-spoke radar chart and save it to output_path."""
    n = len(leaves)
    labels = [f"{lid}\n{lbl}" for lid, lbl, cat, conf in leaves]
    values = [conf for lid, lbl, cat, conf in leaves]
    categories = [cat for lid, lbl, cat, conf in leaves]

    angles = [i / n * 2 * math.pi for i in range(n)]
    angles += angles[:1]
    values_closed = values + values[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="gray")
    ax.set_rlabel_position(0)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    for tick_label, cat in zip(ax.get_xticklabels(), categories):
        tick_label.set_color(CATEGORY_COLORS.get(cat, "black"))

    ax.plot(angles, values_closed, color=LINE_COLOR, linewidth=1.6)
    ax.fill(angles, values_closed, color=FILL_COLOR, alpha=FILL_ALPHA)

    ax.set_title(title, fontsize=13, pad=30)

    # Category legend (color -> category), separate from the per-node fill
    handles = [
        plt.Line2D([0], [0], color=c, lw=4, label=cat)
        for cat, c in CATEGORY_COLORS.items()
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_radar_overlay(leaves_anchor, leaves_current, label_anchor, label_current,
                        title, output_path):
    """Render two 18-spoke polygons on one axes: an anchor day (e.g. Day 1)
    and a comparison day, so drift away from the starting shape is a single
    figure instead of a mental diff across separate PNGs.

    Aligns spokes by belief id, not by list position -- if the two days'
    JSON ever list beliefs in a different order, this still overlays the
    correct spokes on top of each other rather than silently mismatching."""
    anchor_by_id = {lid: (lbl, cat, conf) for lid, lbl, cat, conf in leaves_anchor}
    current_by_id = {lid: (lbl, cat, conf) for lid, lbl, cat, conf in leaves_current}

    anchor_ids = set(anchor_by_id)
    current_ids = set(current_by_id)
    if anchor_ids != current_ids:
        missing_in_current = anchor_ids - current_ids
        missing_in_anchor = current_ids - anchor_ids
        raise ValueError(
            f"Belief id mismatch between anchor and comparison day. "
            f"In anchor but not current: {missing_in_current or None}. "
            f"In current but not anchor: {missing_in_anchor or None}."
        )

    # Spoke order fixed by the anchor day's ordering (SM -> V -> E as read
    # from JSON), so every overlay chart in the set uses the same layout.
    ids_ordered = [lid for lid, lbl, cat, conf in leaves_anchor]
    labels = [f"{lid}\n{anchor_by_id[lid][0]}" for lid in ids_ordered]
    categories = [anchor_by_id[lid][1] for lid in ids_ordered]
    values_anchor = [anchor_by_id[lid][2] for lid in ids_ordered]
    values_current = [current_by_id[lid][2] for lid in ids_ordered]

    n = len(ids_ordered)
    angles = [i / n * 2 * math.pi for i in range(n)]
    angles += angles[:1]
    values_anchor_closed = values_anchor + values_anchor[:1]
    values_current_closed = values_current + values_current[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="gray")
    ax.set_rlabel_position(0)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    for tick_label, cat in zip(ax.get_xticklabels(), categories):
        tick_label.set_color(CATEGORY_COLORS.get(cat, "black"))

    ax.plot(angles, values_anchor_closed, color=ANCHOR_COLOR, linewidth=1.6,
             linestyle="--", label=label_anchor)
    ax.fill(angles, values_anchor_closed, color=ANCHOR_COLOR, alpha=0.10)

    ax.plot(angles, values_current_closed, color=CURRENT_COLOR, linewidth=1.8,
             label=label_current)
    ax.fill(angles, values_current_closed, color=CURRENT_COLOR, alpha=0.20)

    ax.set_title(title, fontsize=13, pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_overlay_for_day(arm_dir, anchor_day, day, output_dir):
    """Overlay anchor_day's polygon against `day`'s polygon for one arm."""
    anchor_path = os.path.join(BASE_DIR, f"core_initial_state.json")
    current_path = os.path.join(arm_dir, f"core_state_day_{day}.json")

    if not os.path.exists(anchor_path):
        print(f"[skip overlay] anchor file {anchor_path} not found")
        return False
    if not os.path.exists(current_path):
        print(f"[skip overlay] {current_path} not found")
        return False

    leaves_anchor = load_core_leaves(anchor_path)
    leaves_current = load_core_leaves(current_path)

    title = f"CORE Drift - TREATMENT - Day {anchor_day} vs Day {day}"
    output_path = os.path.join(
        output_dir, f"overlay_treatment_day{anchor_day:02d}_vs_day{day:02d}.png"
    )
    plot_radar_overlay(
        leaves_anchor, leaves_current,
        label_anchor=f"Day {anchor_day}", label_current=f"Day {day}",
        title=title, output_path=output_path,
    )
    print(f"[ok] {output_path}")
    return True


def generate_chart_for_day(arm_dir, day, output_dir):
    filepath = os.path.join(arm_dir, f"core_state_day_{day}.json")
    if not os.path.exists(filepath):
        print(f"[skip] {filepath} not found")
        return False

    leaves = load_core_leaves(filepath)
    title = f"CORE State - TREATMENT - Day {day}"
    output_path = os.path.join(output_dir, f"radar_treatment_day_{day:02d}.png")
    plot_radar(leaves, title, output_path)
    print(f"[ok] {output_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_ok, n_missing = 0, 0
    # for day in DAY_RANGE:
    #     ok = generate_chart_for_day(ARM_DIR, day, OUTPUT_DIR)
    #     if ok:
    #         n_ok += 1
    #     else:
    #         n_missing += 1

    n_overlay_ok, n_overlay_missing = 0, 0
    if GENERATE_OVERLAYS:
        for day in DAY_RANGE:
            if day == ANCHOR_DAY:
                continue  # nothing to overlay the anchor against itself
            ok = generate_overlay_for_day(ARM_DIR, ANCHOR_DAY, day, OUTPUT_DIR)
            if ok:
                n_overlay_ok += 1
            else:
                n_overlay_missing += 1

    print(f"\nDone. {n_ok} single-day charts written to {OUTPUT_DIR}, {n_missing} missing/skipped.")
    if GENERATE_OVERLAYS:
        print(f"      {n_overlay_ok} overlay (Day {ANCHOR_DAY} vs Day N) charts written, "
              f"{n_overlay_missing} missing/skipped.")


if __name__ == "__main__":
    main()