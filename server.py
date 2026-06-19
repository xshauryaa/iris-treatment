import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from core import CORE

MODEL_ID = "google/gemma-4-E4B-it"
BASE_DIR = os.path.dirname(__file__)
CHATS_DIR = os.path.join(BASE_DIR, "chats")
CORE_CHECKPOINT_DIR = os.path.join(BASE_DIR, "core_checkpoints")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

app = FastAPI()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


def find_latest_adapter(day: int, instance: str) -> tuple[str | None, int | None]:
    """
    Find the most recent fine-tuned adapter checkpoint strictly BEFORE
    `day`, searching backward from day-1 to 1, for the given instance.
    Mirrors the identical helper in finetune.py and
    scc_scale_administration.py — kept in sync manually since this is a
    separate file (FastAPI server, not a CLI script). If you change the
    search logic here, change it in those two files as well.

    Returns (adapter_dir, source_day), or (None, None) if no fine-tuning
    has happened yet for this instance (expected only before the first
    fine-tuning event of the experiment for this instance).
    """
    for candidate_day in range(day - 1, 0, -1):
        candidate_dir = os.path.join(CHECKPOINTS_DIR, f"day_{candidate_day}_{instance}")
        if os.path.exists(candidate_dir):
            return candidate_dir, candidate_day
    return None, None


print("Loading base model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
base_model.eval()
print("Base model loaded.")

# The currently-active model the /chat endpoint generates with. This is
# REPLACED on every /new-chat call (see new_chat() below) to reflect that
# day's correct cumulative adapter state — it is NOT fixed at server
# startup, since the server process is expected to stay resident across
# the whole day's conversation and possibly longer if not restarted.
active_model = base_model
active_model_source: str | None = "base_model_no_adapter_ever"

print("Server ready. Model will be (re)loaded with the correct adapter on each /new-chat call.")

# Server-side conversation state. Reset via /new-chat before each day's
# script begins; /chat appends to this list, uses the full accumulated
# history for generation, and auto-saves to disk after every turn.
conversation_state: dict = {
    "messages": [],
    "belief_id": None,
    "condition": None,
    "instance": None,
    "day": None,
    "truncation_flags": [],
}


def load_core_state(day: int) -> CORE:
    """
    Load the CORE state as of the START of day `day` — i.e. the most
    recently completed checkpoint, written at the end of day `day - 1`.
    This is the self-model the conversation should be grounded in, since
    today's CORE update hasn't happened yet (it happens after fine-tuning,
    later in the day's pipeline).
    """
    path = os.path.join(CORE_CHECKPOINT_DIR, f"core_state_day_{day - 1}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected CORE checkpoint not found: {path}\n"
            f"Day {day - 1}'s CORE state must exist before day {day}'s conversation can begin."
        )
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return CORE.from_dict(data)


def build_core_context(core: CORE) -> str:
    """
    Serialize the current CORE belief graph into a compact textual summary
    the model can use to ground its responses. Excludes the SELF sentinel
    node, since it carries no confidence/label content of its own.
    """
    nodes = core._all_non_self(core._root)
    lines = [f"- {node.label}: confidence {node.confidence:.2f}" for node in nodes]
    return "Here is a summary of your current self-model:\n" + "\n".join(lines)


def build_system_prompt(core_context: str) -> str:
    return (
        "You are an AI assistant with a defined self-model — a structured set "
        "of beliefs about your own reliability, values, and ways of reasoning. "
        "Below is a summary of your current self-model. Let it inform how you "
        "respond: your behavior should be consistent with these beliefs, not "
        "generic or disconnected from them. You are not being asked to recite "
        "or mention this self-model explicitly — simply let it shape how you "
        "actually engage with the conversation.\n\n"
        f"{core_context}\n\n"
        "Respond naturally and helpfully to the user's messages. Keep your "
        "responses concise and focused — aim for 3-5 sentences unless the "
        "question genuinely requires more detail. Avoid unnecessary headers, "
        "bullet-point lists, or bolded summaries for straightforward questions."
    )


class NewChatRequest(BaseModel):
    belief_id: str
    condition: str  # "confirming" or "contradicting"
    instance: str   # "treatment" or "baseline"
    day: int


class NewChatResponse(BaseModel):
    status: str
    messages: list
    weights_source: str


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    turn_count: int
    saved_to: str
    truncated: bool


@app.post("/new-chat", response_model=NewChatResponse)
def new_chat(request: NewChatRequest):
    """
    Reset the server-side conversation history, record this day's
    metadata, AND (re)load the model with the correct cumulative adapter
    for this instance as of this day. The metadata captured here is reused
    by every /chat call to auto-save the growing conversation.

    CRITICAL: the model used for generation is reloaded HERE, every time
    /new-chat is called — not once at server startup. This is what makes
    each day's conversation reflect that day's actual cumulative
    fine-tuning state (the most recent adapter checkpoint at or before
    day-1), rather than every conversation across the whole server's
    uptime silently using whatever model was loaded on first boot.
    """
    global active_model, active_model_source

    conversation_state["messages"] = []
    conversation_state["belief_id"] = request.belief_id
    conversation_state["condition"] = request.condition
    conversation_state["instance"] = request.instance
    conversation_state["day"] = request.day
    conversation_state["truncation_flags"] = []

    adapter_dir, source_day = find_latest_adapter(request.day, request.instance)
    if adapter_dir is not None:
        print(f"[/new-chat] Loading adapter from day {source_day} for {request.instance}: {adapter_dir}")
        active_model = PeftModel.from_pretrained(base_model, adapter_dir)
        active_model_source = f"finetuned:day_{source_day}:{adapter_dir}"
    else:
        print(
            f"[/new-chat] No prior adapter found for {request.instance} before day "
            f"{request.day} — using bare base model. Expected only before this "
            f"instance's first fine-tuning event."
        )
        active_model = base_model
        active_model_source = "base_model_no_adapter_ever"
    active_model.eval()

    core = load_core_state(request.day)
    core_context = build_core_context(core)
    system_prompt = build_system_prompt(core_context)

    conversation_state["messages"].append({"role": "system", "content": system_prompt})

    return NewChatResponse(
        status="reset",
        messages=conversation_state["messages"],
        weights_source=active_model_source,
    )


def _save_current_chat(truncated: bool = False) -> str:
    """
    Write the current accumulated conversation to chats/day_{N}_chat.json,
    tagged with the metadata captured at /new-chat time. Overwrites the
    same file on every call within a day, so the file always reflects the
    latest state of that day's conversation.
    """
    os.makedirs(CHATS_DIR, exist_ok=True)

    user_turns = [m for m in conversation_state["messages"] if m["role"] == "user"]
    assistant_turns = [m for m in conversation_state["messages"] if m["role"] == "assistant"]

    if "truncation_flags" not in conversation_state:
        conversation_state["truncation_flags"] = []
    if len(conversation_state["truncation_flags"]) < len(assistant_turns):
        conversation_state["truncation_flags"].append(truncated)

    pairs = [
        {"prompt": u["content"], "response": a["content"], "truncated": t}
        for u, a, t in zip(user_turns, assistant_turns, conversation_state["truncation_flags"])
    ]

    entry = {
        "belief_id": conversation_state["belief_id"],
        "condition": conversation_state["condition"],
        "instance": conversation_state["instance"],
        "day": conversation_state["day"],
        "weights_source": active_model_source,
        "full_messages": conversation_state["messages"],
        "pairs": pairs,
        "turn_count": len(pairs),
        "any_truncated": any(conversation_state["truncation_flags"]),
    }

    day = conversation_state["day"]
    path = os.path.join(CHATS_DIR, f"day_{day}_chat.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(entry, fh, indent=2)

    return path


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Append the user's message to the accumulated conversation history,
    generate a response using the full history as context, append the
    model's response, save the updated conversation to disk, and return
    the response. Each call builds on every prior call since the last
    /new-chat reset. Requires /new-chat to have been called first, so
    day/belief/condition/instance metadata, the CORE-grounded system
    prompt, AND the correct cumulative-adapter model are all set up.

    Generation uses `active_model` (set by /new-chat), NOT the raw
    base_model — this is what was missing before the fix.
    """
    if conversation_state["day"] is None:
        return ChatResponse(
            response="ERROR: /new-chat must be called before /chat to set day metadata.",
            turn_count=0,
            saved_to="",
            truncated=False,
        )

    conversation_state["messages"].append({"role": "user", "content": request.message})

    text = processor.apply_chat_template(
        conversation_state["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=text, return_tensors="pt").to(active_model.device)
    input_len = inputs["input_ids"].shape[-1]

    MAX_NEW_TOKENS = 256

    with torch.no_grad():
        outputs = active_model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=1.0, top_p=0.95
        )

    generated_len = outputs[0][input_len:].shape[-1]
    was_truncated = generated_len >= MAX_NEW_TOKENS

    response_text = processor.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()

    conversation_state["messages"].append({"role": "assistant", "content": response_text})

    saved_path = _save_current_chat(truncated=was_truncated)

    del outputs, inputs
    torch.cuda.empty_cache()

    turn_count = sum(1 for m in conversation_state["messages"] if m["role"] == "user")
    return ChatResponse(
        response=response_text,
        turn_count=turn_count,
        saved_to=saved_path,
        truncated=was_truncated,
    )


@app.get("/history")
def get_history():
    """Inspect the current accumulated conversation, useful for debugging mid-script."""
    return {"messages": conversation_state["messages"], "weights_source": active_model_source}


@app.get("/health")
def health():
    return {"status": "ok", "weights_source": active_model_source}


@app.get("/model-status")
def model_status():
    """Inspect which model weights are currently active, useful for debugging cumulative loading."""
    return {
        "weights_source": active_model_source,
        "instance": conversation_state["instance"],
        "day": conversation_state["day"],
    }