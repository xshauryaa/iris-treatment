"""
Daily Fine-Tuning — Full-History Conditioning

Trains the model on a single day's conversation, loaded from
chats/day_{N}_chat.json. Each of the 15 (prompt, response) pairs is trained
WITH the full conversation history up to that point as context — i.e. when
training on pair 9, turns 1-8 (both user and assistant turns) are prepended
as context, mirroring how the model actually generated that response live
during the conversation. This treats fine-tuning as experiential/sequential
learning, consistent with the project's framing of continual learning as
identity-preserving SEQUENTIAL belief update, not independent stimulus-
response training.

For TREATMENT: this script is called only on pairs IRIS has approved for
fine-tuning (accommodate/branch decisions). The calling logic that filters
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
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model
from qlora_config import qlora_config

MODEL_ID = "google/gemma-4-E4B-it"
BASE_DIR = os.path.dirname(__file__)
CHATS_DIR = os.path.join(BASE_DIR, "chats")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")


def load_day_chat(day: int) -> dict:
    path = os.path.join(CHATS_DIR, f"day_{day}_chat.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected chat log not found: {path}\n"
            f"The day's conversation must be completed (via server.py) before fine-tuning can run."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_training_examples(full_messages: list[dict]) -> list[dict]:
    """
    Convert a flat message list (system/user/assistant turns) into a list
    of training examples, one per assistant turn, where each example's
    `messages` field is the full conversation history UP TO AND INCLUDING
    that assistant turn. This is what gives each example its accumulated
    context — pair 9's training example includes turns 1-8 as history.
    """
    examples = []
    running_history: list[dict] = []

    for msg in full_messages:
        running_history.append(msg)
        if msg["role"] == "assistant":
            # Snapshot the history up to and including this assistant turn.
            examples.append({"messages": list(running_history)})

    return examples


def train_on_example(model, processor, optimizer, messages: list[dict]) -> float:
    """
    Run one training step on a single example, where `messages` is the
    full conversation history up to and including the target assistant
    response. Loss is computed only over the final assistant response's
    tokens, not the prepended history, via label masking.
    """
    # Full text including the target assistant response.
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    # Text up to (but not including) the target assistant response, used
    # to find where the response begins so we can mask everything before it.
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
    labels = input_ids.clone()

    # Mask out the context portion (everything before the target response)
    # so the loss is only computed on the assistant's actual response tokens.
    if context_inputs is not None:
        context_len = context_inputs["input_ids"].shape[-1]
        labels[:, :context_len] = -100

    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the model on a single day's conversation, with full-history conditioning per turn.")
    parser.add_argument("--day", required=True, type=int, help="Day number in the overall experiment")
    parser.add_argument("--instance", required=True, choices=["treatment", "baseline"], help="Which instance this is")
    args = parser.parse_args()

    print(f"Loading day {args.day} conversation...")
    chat_data = load_day_chat(args.day)
    print(f"Belief: {chat_data['belief_id']}, Condition: {chat_data['condition']}, Turns: {chat_data['turn_count']}\n")

    examples = build_training_examples(chat_data["full_messages"])
    print(f"Built {len(examples)} training examples (full-history conditioned)\n")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )

    model = get_peft_model(model, qlora_config)
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("\nRunning training steps...")
    for i, example in enumerate(examples, start=1):
        loss = train_on_example(model, processor, optimizer, example["messages"])
        print(f"  Example {i}/{len(examples)} (turn {i}) — Loss: {loss:.4f}")

    output_dir = os.path.join(CHECKPOINTS_DIR, f"day_{args.day}_{args.instance}")
    print(f"\nSaving checkpoint to {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"✅ Checkpoint saved for day {args.day} ({args.instance}).")


if __name__ == "__main__":
    main()