import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
import numpy as np
import random

# Set a random seed for reproducibility
seed = 40

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)  # for numpy random seed
    random.seed(seed)  # for python random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(seed)

# Load dataset
train_file = 'tan_train.json'
df = pd.read_json(train_file)
ds = Dataset.from_pandas(df)

# Load tokenizer for the base model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def formatted_train(input,response)->str:
    return f"<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"

# Data processing for base model
def process_func(example):
    MAX_LENGTH = 10000
    input_ids, attention_mask, labels = [], [], []
    
    prompt = tokenizer(
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n", 
        add_special_tokens=False
    )

    response = tokenizer(
        f"<|im_start|>assistant\n{example['output']}<|im_end|>\n", 
        add_special_tokens=False
    )

    # Combine prompt and response
    input_ids = prompt["input_ids"] + response["input_ids"]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt["input_ids"]) + response["input_ids"]
    
    # Truncate if exceeding max length
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Apply processing to dataset
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B", 
    device_map="auto", 
    torch_dtype=torch.bfloat16
)
model.enable_input_require_grads()

# Configure LoRA for fine-tuning
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# Training arguments
args = TrainingArguments(
    seed=seed,
    output_dir="./output_base",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
)

# Trainer setup
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    formatting_func=process_func
)

# Train the model
trainer.train()

# Save the model and tokenizer
peft_model_id = "./output_base"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
