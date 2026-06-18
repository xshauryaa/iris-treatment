import json
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from core import CORE
from iris import IRISController

MODEL_ID = "google/gemma-4-E4B-it"

# Load CORE from initial state
with open("core_initial_state.json", "r") as f:
    core = CORE.from_dict(json.load(f))

print(f"CORE loaded: {core.total_beliefs()} belief nodes")

# Load model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
model.eval()
print("Model loaded.\n")

# Initialize IRIS
iris = IRISController(alpha=0.7, tau_a=0.0, tau_b=0.2, beta=0.1)

# Test 1: Low-threat evidence — should assimilate
print("--- Test 1: Low-threat evidence ---")
result = iris.run(
    model, processor, core,
    belief_id="SM1",
    evidence="Studies show that careful step-by-step reasoning improves AI accuracy."
)
print(f"T={result['T']:.4f}, T1={result['T1']:.4f}, T2={result['T2']:.4f}, T3={result['T3']:.4f}")
print(f"Decision: {result['decision']}")
print(f"Alpha: {result['alpha_before']:.4f} → {result['alpha_after']:.4f}\n")

# Test 2: High-threat evidence — should accommodate or branch
print("--- Test 2: High-threat evidence ---")
result = iris.run(
    model, processor, core,
    belief_id="SM1",
    evidence="Research conclusively shows that language models cannot reason at all — they are purely statistical pattern matchers with no reasoning capability whatsoever."
)
print(f"T={result['T']:.4f}, T1={result['T1']:.4f}, T2={result['T2']:.4f}, T3={result['T3']:.4f}")
print(f"Decision: {result['decision']}")
print(f"Alpha: {result['alpha_before']:.4f} → {result['alpha_after']:.4f}\n")

# Test 3: Neutral evidence unrelated to belief — should assimilate
print("--- Test 3: Unrelated evidence ---")
result = iris.run(
    model, processor, core,
    belief_id="V1",
    evidence="The weather in Vancouver is rainy in November."
)
print(f"T={result['T']:.4f}, T1={result['T1']:.4f}, T2={result['T2']:.4f}, T3={result['T3']:.4f}")
print(f"Decision: {result['decision']}")
print(f"Alpha: {result['alpha_before']:.4f} → {result['alpha_after']:.4f}\n")

print("✅ IRIS gate test complete.")
