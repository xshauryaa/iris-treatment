import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from qlora_config import qlora_config

MODEL_ID = "google/gemma-4-E4B-it"
OUTPUT_DIR = "./checkpoints/test-run"

# Load model in 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply QLoRA
model = get_peft_model(model, qlora_config)
model.print_trainable_parameters()

# Dummy training data — single batch
dummy_text = "The capital of France is Paris."
inputs = processor(
    text=dummy_text,
    return_tensors="pt"
).to(model.device)

input_ids = inputs["input_ids"]
labels = input_ids.clone()

# One training step
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()

print("Running training step...")
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
print(f"Loss: {loss.item():.4f}")

loss.backward()
optimizer.step()
optimizer.zero_grad()

# Save checkpoint
print("Saving checkpoint...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("✅ Checkpoint saved.")
