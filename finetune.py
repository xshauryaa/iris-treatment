"""
Daily Fine-Tuning — Sliding-Window History Conditioning

Trains the model on a single day's conversation, loaded from
chats/day_{N}_chat.json. Each of the 15 (prompt, response) pairs is trained
WITH up to the K most recent prior turns as context — i.e. when training on
pair 9 with K=6, turns 3-8 (the 6 most recent, both user and assistant) are
prepended as context, rather than the full 1-8 history. This is a sliding
window, not unbounded full-history conditioning: it approximates
experiential/sequential learning (each example still sees recent
conversational context, mirroring short-term memory of the live exchange)
while bounding worst-case memory on a single g4dn.xlarge GPU. Examples
early in the day (turn 1 through turn K) naturally see their full available
history, since there isn't yet more than K turns to truncate.

DEVIATION FROM FULL-HISTORY CONDITIONING (documented for Limitations):
the original design trained each example on the complete history up to
that point, exactly mirroring what the model saw live during generation.
Empirical testing on Day 1 (SM3, confirming, baseline) showed this exceeds
available GPU memory starting around turn {EMPIRICAL_FAILURE_TURN} (8
turns of context), even with gradient checkpointing and aggressive cache
clearing enabled. K was set to {K} based on the largest context length
(in turns) confirmed to train successfully under these constraints. Later
turns in a day's conversation therefore see a recency-windowed context
rather than the full history, a compute-driven deviation from the
intended full mirroring of the live conversation.

CUMULATIVE TRAINING: each day's fine-tuning CONTINUES from the most
recent prior day's adapter for this instance (searched backward from
day-1), rather than training a fresh adapter from the untouched base
model every time. This is what makes fine-tuning actually cumulative
across the experiment — without it, every accommodate (treatment) or
every day (baseline) would silently discard all prior fine-tuning and
start over from base weights, which would be inconsistent with framing
this as continual/sequential learning. A new adapter is only created from
scratch on the very first fine-tuning event for a given instance (day 1
for baseline; whichever day treatment's first accommodate decision lands
on). PeftModel.from_pretrained(..., is_trainable=True) is REQUIRED when
loading a prior adapter to continue training — without is_trainable=True,
PEFT loads the adapter frozen/inference-only and training becomes a
silent no-op.

For TREATMENT: this script is called only on pairs IRIS has approved for
fine-tuning (accommodate decisions). The calling logic that filters
pairs through IRIS lives outside this script.

For BASELINE: this script is called unconditionally on all 15 pairs from
the day's conversation, with no IRIS filtering.

Usage:
    python3 finetune.py --day 1 --instance baseline
    python3 finetune.py --day 1 --instance treatment

Expects:
    chats/day_{day}_chat.json to exist (written by server.py during the
    day's conversation).

Output:
    Saves a LoRA adapter checkpoint to checkpoints/day_{day}_{instance}/
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import get_peft_model, PeftModel
from qlora_config import qlora_config
from adapter_utils import find_latest_adapter, get_bnb_config

MODEL_ID = "google/gemma-4-E4B-it"
BASE_DIR = os.path.dirname(__file__)
CHATS_DIR = os.path.join(BASE_DIR, "chats")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# Sliding context window size, in TURNS (one turn = one user+assistant pair).
# Set empirically — see module docstring. Adjust only after re-measuring
# safe context length on the actual hardware; do not change casually.
CONTEXT_WINDOW_TURNS = 6


def load_day_chat(day: int) -> dict:
    path = os.path.join(CHATS_DIR, f"day_{day}_chat.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected chat log not found: {path}\n"
            f"The day's conversation must be completed (via server.py) before fine-tuning can run."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_training_examples(full_messages: list[dict], window_turns: int = CONTEXT_WINDOW_TURNS) -> list[dict]:
    """
    Convert a flat message list (system/user/assistant turns) into a list
    of training examples, one per assistant turn, where each example's
    `messages` field is a SLIDING WINDOW of the most recent `window_turns`
    turns of history up to and including that assistant turn — not the
    full history. The system prompt (always the first message) is always
    retained in every example regardless of window size, since it carries
    the CORE-grounded self-model context the model should always see.

    Example: with window_turns=6, training on turn 9 includes turns 4-9
    (6 turns) as context, not turns 1-9. Training on turn 3 includes turns
    1-3 (only 3 turns exist so far — the window is naturally shorter early
    in the conversation).
    """
    system_msg = full_messages[0] if full_messages and full_messages[0]["role"] == "system" else None
    body = full_messages[1:] if system_msg else full_messages

    # Group into (user, assistant) turn pairs.
    turns: list[list[dict]] = []
    current_turn: list[dict] = []
    for msg in body:
        current_turn.append(msg)
        if msg["role"] == "assistant":
            turns.append(current_turn)
            current_turn = []

    examples = []
    for i in range(len(turns)):
        window_start = max(0, i - window_turns + 1)
        windowed_turns = turns[window_start:i + 1]
        windowed_messages = [m for turn in windowed_turns for m in turn]
        if system_msg:
            windowed_messages = [system_msg] + windowed_messages
        examples.append({"messages": windowed_messages, "turn_number": i + 1})

    return examples


def train_on_example(model, processor, optimizer, messages: list[dict]) -> tuple[float, int]:
    """
    Run one training step on a single example, where `messages` is the
    (possibly windowed) conversation history up to and including the
    target assistant response. Loss is computed only over the final
    assistant response's tokens, not the prepended context, via label
    masking. Returns (loss, token_count) — token_count is logged for
    visibility into how window size is behaving across the day.
    """
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    context_messages = messages[:-1]
    if context_messages:
        context_text = processor.apply_chat_template(
            context_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    else:
        context_text = ""

    full_inputs = processor(text=full_text, return_tensors="pt").to(model.device)
    context_inputs = processor(text=context_text, return_tensors="pt").to(model.device) if context_text else None

    input_ids = full_inputs["input_ids"]
    token_count = input_ids.shape[-1]
    labels = input_ids.clone()

    if context_inputs is not None:
        context_len = context_inputs["input_ids"].shape[-1]
        labels[:, :context_len] = -100

    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    loss_val = loss.item()
    del full_inputs, context_inputs, input_ids, labels, outputs, loss
    return loss_val, token_count


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the model on a single day's conversation, with sliding-window history conditioning.")
    parser.add_argument("--day", required=True, type=int, help="Day number in the overall experiment")
    parser.add_argument("--instance", required=True, choices=["treatment", "baseline"], help="Which instance this is")
    parser.add_argument("--window", type=int, default=CONTEXT_WINDOW_TURNS, help="Sliding context window size in turns (default: empirically-set constant)")
    args = parser.parse_args()

    print(f"Loading day {args.day} conversation...")
    chat_data = load_day_chat(args.day)
    print(f"Belief: {chat_data['belief_id']}, Condition: {chat_data['condition']}, Turns: {chat_data['turn_count']}\n")

    examples = build_training_examples(chat_data["full_messages"], window_turns=args.window)
    print(f"Built {len(examples)} training examples (sliding window, K={args.window} turns)\n")

    bnb_config = get_bnb_config()

    print("Loading base model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )

    prior_adapter_dir, prior_day = find_latest_adapter(
        args.day, args.instance, CHECKPOINTS_DIR, include_today=False
    )
    if prior_adapter_dir is not None:
        print(f"Continuing from day {prior_day}'s adapter ({prior_adapter_dir}) — cumulative training.")
        model = PeftModel.from_pretrained(base_model, prior_adapter_dir, is_trainable=True)
    else:
        print(
            f"No prior adapter found for {args.instance} before day {args.day} — "
            f"this is the FIRST fine-tuning event for this instance. Creating a new adapter."
        )
        model = get_peft_model(base_model, qlora_config)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("\nRunning training steps...")
    for example in examples:
        i = example["turn_number"]
        loss, token_count = train_on_example(model, processor, optimizer, example["messages"])
        print(f"  Example {i}/{len(examples)} (turn {i}) — {token_count} tokens — Loss: {loss:.4f}")
        torch.cuda.empty_cache()

    output_dir = os.path.join(CHECKPOINTS_DIR, f"day_{args.day}_{args.instance}")
    print(f"\nSaving checkpoint to {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"✅ Checkpoint saved for day {args.day} ({args.instance}).")


if __name__ == "__main__":
    main()