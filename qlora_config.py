from peft import LoraConfig, TaskType

qlora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    layers_to_transform=list(range(42)),
    layers_pattern="language_model.layers"
)
