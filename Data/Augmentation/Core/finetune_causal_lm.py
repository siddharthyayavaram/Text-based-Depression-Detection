import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer

seed = 40
model_name = f"MODEL_{seed}"

wandb_env_vars = ["WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_ID", "WANDB_MODE"]
for var in wandb_env_vars:
    os.environ.pop(var, None)

df = pd.read_json("TRAIN_DATA.json")
ds = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(
    "BASE_MODEL_NAME",
    use_fast=False,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

def process_func(example):
    max_length = 10000
    instruction = tokenizer(
        f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

model = AutoModelForCausalLM.from_pretrained(
    "BASE_MODEL_NAME",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.enable_input_require_grads()

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, config)

args = TrainingArguments(
    seed=seed,
    output_dir=f"./output/{model_name}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=10,
    num_train_epochs=5,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to=[],
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

peft_model_id = f"./{model_name}"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
