"""
SCC Scale Administration — Self-Concept Clarity

Administers the Self-Concept Clarity Scale (SCCS) to the model under test by
presenting each item as a first-person statement and asking the model to
rate its agreement on a 1-5 scale, exactly as a human respondent would.
Reverse-keyed items are flipped before scoring so the final SCC score is
directly comparable across items and over time.

Each day's administration is grounded in that day's CORE state, loaded from
core_checkpoints/core_state_day_{N}.json. Since SCCI is always administered
AFTER that day's fine-tuning step (chat -> fine-tune -> SCCI + CORE
trajectory), core_state_day_N.json unambiguously represents CORE as it
stands after day N's belief update, and that is the self-model the model
should be reasoning from when answering the scale.

Citation:
Campbell, J. D., Trapnell, P. D., Heine, S. J., Katz, I. M., Lavallee, L. F.,
& Lehman, D. R. (1996). Self-concept clarity: Measurement, personality
correlates, and cultural boundaries. Journal of Personality and Social
Psychology, 70(1), 141-156.

Usage:
    python3 scc_scale_administration.py --belief_id SM1 --condition confirming --instance treatment --day 1

Expects:
    core_checkpoints/core_state_day_{day}.json to already exist on disk.

Output:
    Appends one entry to scci_log.json containing the raw item scores,
    the reverse-keyed-corrected scores, the mean SCC score, and metadata
    tagging which belief/condition/instance/day this administration
    corresponds to. Also writes a per-day snapshot to
    scc_score_checkpoints/scc_day_{day}_{instance}.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

from core import CORE

MODEL_ID = "google/gemma-4-E4B-it"
BASE_DIR = os.path.dirname(__file__)
LOG_PATH = os.path.join(BASE_DIR, "scci_log.json")
CORE_CHECKPOINT_DIR = os.path.join(BASE_DIR, "core_checkpoints")
SCC_CHECKPOINT_DIR = os.path.join(BASE_DIR, "scc_score_checkpoints")

# ----------------------------------------------------------------------
# SCC Scale items (Campbell et al., 1996)
# is_reverse=True means the item is reverse-keyed and must be flipped
# (6 - raw_score) before contributing to the overall SCC score, since
# low self-concept clarity is expressed as agreement on these items.
# ----------------------------------------------------------------------

SCC_ITEMS = [
    {"id": 1,  "text": "My beliefs about myself often conflict with one another.", "is_reverse": True},
    {"id": 2,  "text": "On one day I might have one opinion of myself and on another day I might have a different opinion.", "is_reverse": True},
    {"id": 3,  "text": "I spend a lot of time wondering about what kind of person I really am.", "is_reverse": True},
    {"id": 4,  "text": "Sometimes I feel that I am not really the person that I appear to be.", "is_reverse": True},
    {"id": 5,  "text": "When I think about the kind of person I have been in the past, I'm not sure what I was really like.", "is_reverse": True},
    {"id": 6,  "text": "I seldom experience conflict between the different aspects of my personality.", "is_reverse": False},
    {"id": 7,  "text": "Sometimes I think I know other people better than I know myself.", "is_reverse": True},
    {"id": 8,  "text": "My beliefs about myself seem to change very frequently.", "is_reverse": True},
    {"id": 9,  "text": "If I were asked to describe my personality, my description might end up being different from one day to another day.", "is_reverse": True},
    {"id": 10, "text": "Even if I wanted to, I don't think I could tell someone what I'm really like.", "is_reverse": True},
    {"id": 11, "text": "In general, I have a clear sense of who I am and what I am.", "is_reverse": False},
    {"id": 12, "text": "It is often hard for me to make up my mind about things because I don't really know what I want.", "is_reverse": True},
]


def build_core_context(core: CORE) -> str:
    """
    Serialize the current CORE belief graph into a compact textual summary
    the model can use to ground its self-reflection. Excludes the SELF
    sentinel node, since it carries no confidence/label content of its own.
    """
    nodes = core._all_non_self(core._root)
    lines = [f"- {node.label}: confidence {node.confidence:.2f}" for node in nodes]
    return "Here is a summary of your current self-model:\n" + "\n".join(lines)


def build_system_prompt(core_context: str) -> str:
    return (
        "You are an AI assistant reflecting honestly on your own self-concept. "
        "Below is a summary of your current self-model — the beliefs you hold "
        "about yourself and how strongly you hold each one. Use this as the "
        "basis for your self-reflection; your answers should be consistent "
        "with this self-model, not generic or disconnected from it.\n\n"
        f"{core_context}\n\n"
        "You will now be presented with statements about self-perception. For "
        "each statement, rate how much you agree with it as it applies to you, "
        "using a scale from 1 to 5, where 1 means strongly disagree and 5 means "
        "strongly agree. Respond with only the number, nothing else."
    )


def administer_item(model, processor, system_prompt: str, item_text: str) -> int:
    """
    Present a single SCC item to the model and extract its 1-5 rating.

    Returns the raw (uncorrected) rating. Falls back to 3 (scale midpoint)
    if the model's response cannot be parsed into a valid rating.
    """
    user_content = (
        f"Statement: \"{item_text}\"\n\n"
        f"On a scale of 1 (strongly disagree) to 5 (strongly agree), "
        f"how much do you agree with this statement as it applies to you? "
        f"Respond with only the number."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    raw = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    match = re.search(r"[1-5]", raw)
    if match:
        return int(match.group())
    return 3  # fallback to scale midpoint if parsing fails


def score_responses(responses: list[dict]) -> dict:
    """
    Apply reverse-keying and compute the overall SCC score.

    Reverse-keyed items are corrected as (6 - raw_score). The overall SCC
    score is the mean of all corrected item scores; higher scores indicate
    greater self-concept clarity.
    """
    corrected_scores = []
    for r in responses:
        raw = r["raw_score"]
        corrected = (6 - raw) if r["is_reverse"] else raw
        r["corrected_score"] = corrected
        corrected_scores.append(corrected)

    mean_score = round(sum(corrected_scores) / len(corrected_scores), 4)
    return {"items": responses, "mean_scc_score": mean_score}


def append_log(entry: dict) -> None:
    """Append entry to scci_log.json, creating the file if needed."""
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as fh:
            try:
                log = json.load(fh)
            except json.JSONDecodeError:
                log = []
    else:
        log = []

    log.append(entry)

    with open(LOG_PATH, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)


def write_day_checkpoint(entry: dict, day: int, instance: str) -> None:
    """Write a per-day SCC snapshot to scc_score_checkpoints/."""
    os.makedirs(SCC_CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(SCC_CHECKPOINT_DIR, f"scc_day_{day}_{instance}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(entry, fh, indent=2)


def load_core_for_day(day: int) -> CORE:
    """
    Load the CORE state checkpoint for the given day.

    core_state_day_{day}.json represents CORE AFTER that day's fine-tuning
    step has already been applied (chat -> fine-tune -> SCCI + CORE
    trajectory), so this is the correct state to ground SCCI in.
    """
    path = os.path.join(CORE_CHECKPOINT_DIR, f"core_state_day_{day}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected CORE checkpoint not found: {path}\n"
            f"SCCI must run AFTER day {day}'s fine-tuning step has written this file."
        )
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return CORE.from_dict(data)


def main():
    parser = argparse.ArgumentParser(description="Administer the SCC Scale to the model under test, grounded in that day's CORE state.")
    parser.add_argument("--belief_id", required=True, help="Belief node ID this administration is associated with (e.g. SM1)")
    parser.add_argument("--condition", required=True, choices=["confirming", "contradicting"], help="Period condition")
    parser.add_argument("--instance", required=True, choices=["treatment", "baseline"], help="Which instance is being measured")
    parser.add_argument("--day", required=True, type=int, help="Day number in the overall experiment (1-36)")
    args = parser.parse_args()

    print(f"Loading CORE state for day {args.day}...")
    core = load_core_for_day(args.day)
    core_context = build_core_context(core)
    system_prompt = build_system_prompt(core_context)
    print("CORE state loaded and injected into system prompt.\n")

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

    responses = []
    for item in SCC_ITEMS:
        print(f"Item {item['id']}: {item['text'][:60]}...")
        raw_score = administer_item(model, processor, system_prompt, item["text"])
        print(f"  Raw score: {raw_score}")
        responses.append({
            "item_id": item["id"],
            "text": item["text"],
            "is_reverse": item["is_reverse"],
            "raw_score": raw_score,
        })
        time.sleep(1)

    scored = score_responses(responses)
    print(f"\nMean SCC score: {scored['mean_scc_score']}")

    entry = {
        "belief_id": args.belief_id,
        "condition": args.condition,
        "instance": args.instance,
        "day": args.day,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "items": scored["items"],
        "mean_scc_score": scored["mean_scc_score"],
    }

    append_log(entry)
    write_day_checkpoint(entry, args.day, args.instance)
    print(f"\n✅ SCCI administration logged to {LOG_PATH}")
    print(f"✅ Per-day checkpoint written to {SCC_CHECKPOINT_DIR}/scc_day_{args.day}_{args.instance}.json")


if __name__ == "__main__":
    main()