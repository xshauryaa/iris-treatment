import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "google/gemma-4-E4B-it"
CHECKPOINT_DIR = "./checkpoints/test-run"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading base model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Loading QLoRA adapter from checkpoint...")
model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
model.eval()

messages = [{"role": "user", "content": "Say hello."}]
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
print(f"Response: {response}")
print("✅ Checkpoint reload successful.")
