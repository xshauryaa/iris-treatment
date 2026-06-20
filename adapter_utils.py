"""
Adapter Utilities — Shared Cumulative-Adapter Loading Logic

This module is the single source of truth for two pieces of logic that
were previously duplicated, with risk of silent divergence, across
finetune.py, server.py, scc_scale_administration.py, and
update_core_treatment.py:

  1. find_latest_adapter() — backward search from a given day to find the
     most recent fine-tuned LoRA checkpoint for an instance, so that any
     script needing "the model's current cumulative state" resolves to
     the same answer everywhere, not four slightly-different
     reimplementations of the same search.

  2. load_model_with_adapter() — load the base model and, if an adapter
     directory is provided, wrap it with that adapter. Used for inference/
     probing (eval mode) by default; finetune.py still handles its own
     is_trainable=True wrapping directly, since it has additional
     training-specific setup (gradient checkpointing, optimizer, etc.)
     that doesn't belong in a shared utility.

BACKGROUND (why this exists): during the pilot runs, multiple scripts 
independently loaded the bare base model instead of the fine-tuned model
when re-probing confidence or generating conversation, because each
script's model-loading block had no awareness of LoRA checkpoints at all.
Once fixed individually, the same backward-search logic was copy-pasted
across four files, which is itself a risk — a future change to the search
logic (e.g. a different fallback policy) would need to be made in four
places consistently. This module removes that duplication.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "google/gemma-4-E4B-it"


def get_bnb_config() -> BitsAndBytesConfig:
    """
    Single source of truth for the 4-bit quantization config, so all
    scripts load the model with identical quantization settings. Returns
    a fresh BitsAndBytesConfig each call (these are lightweight, not
    meant to be shared/reused across model loads).
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def find_latest_adapter(
    day: int,
    instance: str,
    checkpoints_dir: str,
    include_today: bool = False,
) -> tuple[str | None, int | None]:
    """
    Find the most recent fine-tuned adapter checkpoint for `instance`,
    searching backward.

    Args:
        day: the day to search relative to.
        instance: "treatment" or "baseline".
        checkpoints_dir: absolute path to the checkpoints/ directory.
        include_today: if True, search starts at `day` itself (used by
            SCCI, which runs AFTER that day's fine-tuning and should
            prefer today's own checkpoint if one exists). If False,
            search starts at `day - 1` (used by finetune.py before
            creating today's checkpoint, server.py's /new-chat before
            today's conversation/fine-tuning has happened yet, and
            update_core_treatment.py's decision phase, which runs BEFORE
            today's possible fine-tuning).

    Returns:
        (adapter_dir, source_day) for the most recent existing checkpoint,
        or (None, None) if no fine-tuning has ever happened yet for this
        instance as of the search start point.
    """
    start_day = day if include_today else day - 1
    for candidate_day in range(start_day, 0, -1):
        candidate_dir = os.path.join(checkpoints_dir, f"day_{candidate_day}_{instance}")
        if os.path.exists(candidate_dir):
            return candidate_dir, candidate_day
    return None, None


def load_model_with_adapter(
    adapter_dir: str | None,
    source_day: int | None,
    context_label: str = "",
    is_trainable: bool = False,
) -> tuple[object, object, str]:
    """
    Load the base model (with shared quantization config) and, if
    adapter_dir is provided, wrap it with that LoRA adapter.

    Args:
        adapter_dir: path to a saved adapter, or None to use the bare
            base model.
        source_day: the day the adapter came from, used only for the
            human-readable weights_source string.
        context_label: short string printed alongside the loading
            message, to distinguish which caller/phase is loading
            (e.g. "decision phase / T1 probe").
        is_trainable: pass True only when the caller intends to continue
            TRAINING this adapter (finetune.py's cumulative-continuation
            path). False (default) loads in inference/eval mode, correct
            for every re-probing or conversation-generation use case.

    Returns:
        (model, processor, weights_source) — weights_source is a string
        suitable for logging/auditing (e.g. into chat JSON or scci_log.json)
        recording exactly which weights were used.
    """
    bnb_config = get_bnb_config()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )

    if adapter_dir is not None:
        label = f"[{context_label}] " if context_label else ""
        print(f"{label}Loading adapter from day {source_day}: {adapter_dir}")
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=is_trainable)
        weights_source = f"finetuned:day_{source_day}:{adapter_dir}"
    else:
        label = f"[{context_label}] " if context_label else ""
        print(f"{label}No adapter found — using bare base model.")
        model = base_model
        weights_source = "base_model_no_adapter_ever"

    if not is_trainable:
        model.eval()

    return model, processor, weights_source