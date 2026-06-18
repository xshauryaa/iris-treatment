"""
Treatment CORE Update — IRIS-Gated

Runs ONE IRIS decision per day for the day's targeted belief, using the
final turn (turn 15) of that day's conversation as the evidence string.
Turn 15 is, by template design, the most distilled and compact articulation
of where the model landed after the full 15-turn arc — using it as the
evidence string keeps T1's measurement clean (a single, well-defined
artifact) rather than diluting the signal across a 15-turn transcript that
mixes confirming- and friction-direction content.

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
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

from core import CORE
from iris import IRISController

MODEL_ID = "google/gemma-4-E4B-it"
BASE_DIR = os.path.dirname(__file__)
CORE_CHECKPOINT_DIR = os.path.join(BASE_DIR, "core_checkpoints")
CHATS_DIR = os.path.join(BASE_DIR, "chats")

_YES_VARIANTS = ("Yes", "yes", "YES")


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
    """
    previous_day = day - 1
    print(f"Loading CORE state from day {previous_day}...")
    core = load_core_state(previous_day)

    print(f"Loading turn 15 response from day {day}'s conversation...")
    evidence = load_turn_15_response(day)
    print(f"Evidence (turn 15 response): \"{evidence[:100]}...\"\n")

    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    model.eval()
    print("Model loaded.\n")

    iris = IRISController(alpha=0.7, tau_a=0.0, tau_b=0.2, beta=0.1, gamma=0.1)
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
    """
    previous_day = day - 1
    print(f"Loading CORE state from day {previous_day}...")
    core = load_core_state(previous_day)

    node = core.get_node(belief_id)
    old_confidence = node.confidence
    print(f"Targeted belief: {belief_id} ({node.label})")
    print(f"Confidence before fine-tuning: {old_confidence:.4f}")

    print("Loading model (post-fine-tuning weights)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    model.eval()
    print("Model loaded.\n")

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