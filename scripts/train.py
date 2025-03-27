from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import LoraConfig, prepare_model_for_kbit_training

model_id = "codellama/CodeLlama-7b-hf"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)  # Quantize for efficiency

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False  # Required for gradient checkpointing
)


# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Define LoRA layers to train
peft_config = LoraConfig(
    r=8,              # Rank of LoRA matrices (smaller = less memory)
    lora_alpha=16,    # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Layers to apply LoRA to
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="codellama-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Reduce if OOM errors occur
    gradient_accumulation_steps=4,   # Simulate larger batch size
    optim="paged_adamw_8bit",        # Memory-efficient optimizer
    save_steps=500,
    logging_steps=100,
    learning_rate=2e-4,
    fp16=True,                       # Use mixed precision
    report_to="none",
    lr_scheduler_type="cosine"
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    data_collator=lambda data: {"input_ids": torch.stack([d["input_ids"] for d in data])},
    peft_config=peft_config,
)

trainer.train()

import wandb

# Initialize tracking
wandb.init(project="code-finetune")

# Train on subsets
sizes = [0.1, 0.3, 0.5, 1.0]
for size in sizes:
    subset = tokenized_data["train"].train_test_split(train_size=size)["train"]
    trainer.train_dataset = subset
    trainer.train()
    eval_results = trainer.evaluate(test_data)
    wandb.log({"CodeBLEU": eval_results["codebleu"], "subset_size": size})


    # Merge LoRA weights with the base model
model = model.merge_and_unload()

# Save the model
model.save_pretrained("fine-tuned-model")
tokenizer.save_pretrained("fine-tuned-model")

# Test on a sample
input_text = "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        # [MASKED]"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))