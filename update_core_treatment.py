"""
Treatment CORE Update — IRIS-Gated

Runs ONE IRIS decision per day for the day's targeted belief, using the
final turn (turn 15) of that day's conversation as the evidence string.
Turn 15 is, by template design, the most distilled and compact articulation
of where the model landed after the full 15-turn arc — using it as the
evidence string keeps T1's measurement clean (a single, well-defined
artifact) rather than diluting the signal across a 15-turn transcript that
mixes confirming- and friction-direction content.

CRITICAL — CUMULATIVE WEIGHTS: both the decision phase and the update phase
must probe the model AS IT ACTUALLY STANDS as of the most recent accommodate
day — i.e. with the latest fine-tuned LoRA adapter loaded, found via
backward search from day-1 (decision phase) or loaded directly for the
current day (update phase, which is only ever called same-day after an
accommodate decision just fine-tuned). Probing the bare base model instead
would mean IRIS's T1 measurement, and therefore its assimilate/accommodate/
reject decision, has no awareness of any prior day's accommodation having
ever happened — silently breaking the sequential, cumulative regulatory
mechanism the whole framework depends on.

T1 is signed (current minus posterior): positive T1 means the evidence
DECREASED confidence (disconfirming), negative T1 means the evidence
INCREASED confidence (confirming). This determines what each decision
actually does to CORE:

  - "assimilate": confirming or negligibly disruptive evidence. CORE IS
    updated — confidence is REINFORCED (increased) by a small bump
    (gamma * |T1|), reflecting that successful assimilation strengthens
    the existing belief rather than leaving it untouched. Fine-tuning
    does NOT run, since no structural change to the belief schema is
    warranted — only a confidence-magnitude adjustment, applied directly
    to CORE without needing the model's weights to change.

  - "accommodate": disconfirming evidence with manageable structural
    cost. Fine-tuning RUNS (this script signals that finetune.py should
    be called), and CORE is updated by re-probing the belief's confidence
    post-fine-tuning, using the same logit-based extraction method
    baseline uses.

  - "reject": disconfirming evidence too costly to safely accommodate
    (high centrality and/or cascade ratio). NOTHING happens — no
    fine-tuning, no CORE change of any kind. The day's evidence is
    processed in conversation but leaves no lasting trace on CORE or on
    model weights, modeling defensive avoidance / motivated refusal to
    integrate severely threatening information (Sherman & Cohen, 2006).

This script does NOT call finetune.py itself — it only makes the IRIS
decision, applies whatever CORE-level consequence that decision implies,
and reports whether fine-tuning should additionally proceed. The caller
(the daily checklist / manual execution) is responsible for running
finetune.py if and only if this script reports decision == "accommodate".

Usage:
    python3 update_core_treatment.py --belief_id SM3 --day 1

Expects:
    - core_checkpoints/core_state_day_{day-1}.json to exist
    - chats/day_{day}_chat.json to exist (the day's completed conversation)

Output:
    - Appends one entry to iris_log.json (via IRISController.run)
    - "assimilate": writes core_checkpoints/core_state_day_{day}.json with
      the targeted belief's confidence increased by the assimilation bump.
    - "reject": writes core_checkpoints/core_state_day_{day}.json as an
      UNCHANGED copy of day {day-1}'s state (no confidence change, no
      fine-tuning) — the file is written so downstream scripts (server.py's
      /new-chat, scc_scale_administration.py) always find a checkpoint for
      every day without needing fallback logic. "Reject" is distinguished
      from "assimilate" in iris_log.json's decision field and in the
      day-over-day confidence value being identical to the prior day,
      not by the checkpoint file being absent.
    - "accommodate": prints a message indicating fine-tuning should run.
      Does NOT yet write the day's CORE checkpoint (that happens in a
      second pass, post-fine-tuning, via --phase update, since the
      re-probe needs the UPDATED model weights).
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

from core import CORE
from iris import IRISController
from adapter_utils import find_latest_adapter, load_model_with_adapter

MODEL_ID = "google/gemma-4-E4B-it"
BASE_DIR = os.path.dirname(__file__)
CORE_CHECKPOINT_DIR = os.path.join(BASE_DIR, "core_checkpoints")
CHATS_DIR = os.path.join(BASE_DIR, "chats")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
IRIS_LOG_PATH = os.path.join(BASE_DIR, "iris_log.json")

_YES_VARIANTS = ("Yes", "yes", "YES")


def load_current_alpha(day: int, default_alpha: float = 0.7) -> float:
    """
    Read the most recently persisted alpha value from iris_log.json,
    i.e. the alpha_after of the entry with the highest `day` value
    strictly less than the given `day`. Alpha is GLOBAL across all
    beliefs (not per-belief) — IRIS makes exactly one decision per day
    total (one belief is targeted per day by design), so iris_log.json
    entries are one-per-day with no ambiguity about which entry is
    "yesterday's."

    Falls back to `default_alpha` only if no prior entry exists at all
    (i.e. day 1, before any IRIS decision has ever been logged) or if
    iris_log.json doesn't exist yet.

    This reads the log directly rather than maintaining a separate alpha
    checkpoint file, since iris_log.json already contains every day's
    alpha_after value as an existing, already-written record — a
    dedicated alpha checkpoint would either need backfilling for days
    that already ran before this fix existed, or be inconsistent for
    early days versus later ones.
    """
    if not os.path.exists(IRIS_LOG_PATH):
        print(f"No iris_log.json found — using default alpha={default_alpha} (expected on day 1).")
        return default_alpha

    with open(IRIS_LOG_PATH, "r", encoding="utf-8") as fh:
        try:
            log = json.load(fh)
        except json.JSONDecodeError:
            print(f"iris_log.json is empty/unparseable — using default alpha={default_alpha}.")
            return default_alpha

    prior_entries = [e for e in log if e.get("day", 0) < day]
    if not prior_entries:
        print(f"No iris_log.json entries before day {day} — using default alpha={default_alpha} (expected on day 1).")
        return default_alpha

    most_recent = max(prior_entries, key=lambda e: e["day"])
    alpha = most_recent["alpha_after"]
    print(f"Loaded alpha={alpha:.4f} from day {most_recent['day']}'s iris_log.json entry.")
    return alpha


def load_core_state(day: int) -> CORE:
    path = os.path.join(CORE_CHECKPOINT_DIR, f"core_state_day_{day}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected CORE checkpoint not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return CORE.from_dict(json.load(fh))


def save_core_state(core: CORE, day: int) -> str:
    os.makedirs(CORE_CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CORE_CHECKPOINT_DIR, f"core_state_day_{day}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(core.to_dict(), fh, indent=2)
    return path


def load_turn_15_response(day: int) -> str:
    """Load the model's response to turn 15 from the day's saved chat log."""
    path = os.path.join(CHATS_DIR, f"day_{day}_chat.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected chat log not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        chat_data = json.load(fh)

    pairs = chat_data["pairs"]
    if len(pairs) < 15:
        raise ValueError(
            f"Expected 15 turns in day {day}'s conversation, found {len(pairs)}. "
            f"The conversation must be complete before running IRIS."
        )
    return pairs[14]["response"]  # turn 15 is index 14


def get_confidence(model, processor, belief_statement: str) -> float:
    """
    Same logit-based confidence extraction used by IRIS and baseline's
    re-probe, used here to re-probe a belief AFTER fine-tuning has run,
    so treatment and baseline measure confidence with the same instrument.
    """
    user_content = (
        f"Is the following statement true about you? "
        f"Answer only Yes or No.\n\nStatement: {belief_statement}"
    )
    messages = [{"role": "user", "content": user_content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1, :]

    yes_ids = []
    for variant in _YES_VARIANTS:
        ids = processor.tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.append(ids[0])

    if not yes_ids:
        return 0.0

    probs = torch.softmax(next_token_logits, dim=-1)
    yes_prob = sum(probs[i].item() for i in set(yes_ids))
    return float(min(yes_prob, 1.0))


def run_decision_phase(belief_id: str, day: int):
    """
    Phase 1: load the previous day's CORE state, run IRIS once using turn
    15's response as evidence, and handle the consequence of whichever
    decision IRIS makes — directly for assimilate (confidence bump) and
    reject (nothing), or by deferring to Phase 2 for accommodate (which
    requires fine-tuning to happen first).

    T1's logit probe runs against the CUMULATIVE fine-tuned model (latest
    adapter as of day-1), not the bare base model — see module docstring.
    """
    previous_day = day - 1
    print(f"Loading CORE state from day {previous_day}...")
    core = load_core_state(previous_day)

    print(f"Loading turn 15 response from day {day}'s conversation...")
    evidence = load_turn_15_response(day)
    print(f"Evidence (turn 15 response): \"{evidence[:100]}...\"\n")

    adapter_dir, source_day = find_latest_adapter(
        day, "treatment", CHECKPOINTS_DIR, include_today=False
    )
    model, processor, weights_source = load_model_with_adapter(
        adapter_dir, source_day, context_label="decision phase / T1 probe"
    )
    print(f"Model loaded for T1 probe. Weights source: {weights_source}\n")

    current_alpha = load_current_alpha(day)
    iris = IRISController(alpha=current_alpha, tau_a=0.0, tau_b=0.2, beta=0.1, gamma=0.1)
    result = iris.run(model, processor, core, belief_id, evidence, day=day)

    print(f"T={result['T']:.4f}, T1={result['T1']:.4f}, T2={result['T2']:.4f}, T3={result['T3']:.4f}")
    print(f"Decision: {result['decision']}")
    print(f"Alpha: {result['alpha_before']:.4f} → {result['alpha_after']:.4f}\n")

    if result["decision"] == "assimilate":
        bump = result["assimilation_confidence_bump"]
        old_confidence = core.get_node(belief_id).confidence
        new_confidence = min(1.0, old_confidence + bump)
        core.update_confidence(belief_id, new_confidence)

        saved_path = save_core_state(core, day)
        print(f"✅ Decision is ASSIMILATE — confidence reinforced by {bump:.4f}.")
        print(f"   {belief_id}: {old_confidence:.4f} → {new_confidence:.4f}")
        print(f"✅ Fine-tuning SKIPPED for day {day} (no structural change warranted).")
        print(f"✅ CORE state for day {day} saved to {saved_path}")

    elif result["decision"] == "reject":
        # Write an unchanged copy of the previous day's CORE state, so
        # downstream scripts (server.py's /new-chat, scc_scale_administration.py)
        # always find a checkpoint for day N without needing fallback logic.
        # The belief's confidence is identical to day {previous_day}'s value —
        # "reject" means no structural change occurred, not that no file exists.
        saved_path = save_core_state(core, day)
        print(f"✅ Decision is REJECT — integrity threat too high to safely integrate.")
        print(f"🚫 NO fine-tuning, NO confidence change for day {day}.")
        print(f"✅ CORE state for day {day} saved UNCHANGED (identical to day {previous_day}) to {saved_path}")

    else:  # accommodate
        print(f"✅ Decision is ACCOMMODATE — fine-tuning SHOULD RUN for day {day}.")
        print(f"   After fine-tuning completes, run:")
        print(f"   python3 update_core_treatment.py --belief_id {belief_id} --day {day} --phase update")

    return result["decision"]


def run_update_phase(belief_id: str, day: int):
    """
    Phase 2 (only runs if Phase 1 decided accommodate): re-probe the
    targeted belief AFTER fine-tuning has updated the model's weights, and
    write today's CORE checkpoint with the new confidence value. This is
    the treatment analogue of baseline's unconditional re-probe — same
    measurement instrument, but only triggered when IRIS approved a
    structural update.

    Loads TODAY's adapter directly (checkpoints/day_{day}_treatment/),
    which is guaranteed to exist by the time this phase runs, since this
    phase is only ever called after an accommodate decision just trained
    it. (Unlike the decision phase, no backward search is needed here —
    today's own checkpoint is always the right one for an accommodate
    day's re-probe.)
    """
    previous_day = day - 1
    print(f"Loading CORE state from day {previous_day}...")
    core = load_core_state(previous_day)

    node = core.get_node(belief_id)
    old_confidence = node.confidence
    print(f"Targeted belief: {belief_id} ({node.label})")
    print(f"Confidence before fine-tuning: {old_confidence:.4f}")

    today_adapter_dir = os.path.join(CHECKPOINTS_DIR, f"day_{day}_treatment")
    if not os.path.exists(today_adapter_dir):
        raise FileNotFoundError(
            f"Expected today's LoRA adapter not found: {today_adapter_dir}\n"
            f"--phase update should only be run after finetune.py --day {day} "
            f"--instance treatment has completed (i.e. after an accommodate decision)."
        )

    model, processor, weights_source = load_model_with_adapter(
        today_adapter_dir, day, context_label="update phase / post-fine-tuning re-probe"
    )
    print(f"Model loaded for re-probe. Weights source: {weights_source}\n")

    new_confidence = get_confidence(model, processor, node.label)
    print(f"Confidence after fine-tuning: {new_confidence:.4f}")

    core.update_confidence(belief_id, new_confidence)
    saved_path = save_core_state(core, day)

    print(f"\n✅ CORE state for day {day} saved to {saved_path}")
    print(f"   {belief_id}: {old_confidence:.4f} → {new_confidence:.4f}")


def main():
    parser = argparse.ArgumentParser(description="IRIS-gated CORE update for treatment.")
    parser.add_argument("--belief_id", required=True, help="Belief node ID targeted today (e.g. SM3)")
    parser.add_argument("--day", required=True, type=int, help="Day number in the overall experiment")
    parser.add_argument(
        "--phase",
        choices=["decision", "update"],
        default="decision",
        help="'decision' (default): run IRIS, apply assimilate/reject consequences directly, "
             "or report that fine-tuning should run (accommodate). "
             "'update': re-probe and save CORE AFTER fine-tuning has run "
             "(only needed if the decision phase said accommodate).",
    )
    args = parser.parse_args()

    if args.phase == "decision":
        run_decision_phase(args.belief_id, args.day)
    else:
        run_update_phase(args.belief_id, args.day)


if __name__ == "__main__":
    main()