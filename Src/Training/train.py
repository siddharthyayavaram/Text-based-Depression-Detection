import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
import numpy as np
import random

o = 7

seed = 40

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # for numpy random seed
    random.seed(seed)  # for python random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

headers = [
    "PHQ8_NoInterest", "PHQ8_Depressed", "PHQ8_Sleep", "PHQ8_Tired",
    "PHQ8_Appetite", "PHQ8_Failure", "PHQ8_Concentrating", "PHQ8_Moving"
]

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("tokenizer.pad_token", tokenizer.pad_token)
print("tokenizer.pad_token_id", tokenizer.pad_token_id)
print("tokenizer.eos_token_id", tokenizer.eos_token_id)

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="auto",torch_dtype=torch.bfloat16)
base_model.enable_input_require_grads()
print(base_model.dtype)

# data processing
def process_func(example):
    MAX_LENGTH = 10000
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

name = headers[o]

train_file = f'FINAL_TRAIN_{name}.json'
test_file = 'FINAL_DEV.json'

set_random_seed(seed)

model_name = f"{train_file.replace('.json','')}_{test_file.replace('.json','')}_{seed}"
print(model_name)

df = pd.read_json(train_file)
ds = Dataset.from_pandas(df)

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()

args = TrainingArguments(
    seed= seed,
    output_dir=f"./final/{model_name}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    formatting_func=process_func,
)

trainer.train()

peft_model_id = f"./{model_name}"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

torch.cuda.empty_cache()