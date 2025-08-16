import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
import numpy as np

seed = 45
train_file = 'data.json'
test_file = 'phq8_d_test.json'
torch.manual_seed(seed)

model_name = f'NEW_{train_file}_{test_file}_{seed}'
print(model_name)

df = pd.read_json(train_file)
ds = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("tokenizer.pad_token", tokenizer.pad_token)
print("tokenizer.pad_token_id", tokenizer.pad_token_id)
print("tokenizer.eos_token_id", tokenizer.eos_token_id)

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

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="auto",torch_dtype=torch.bfloat16)
model.enable_input_require_grads()
print(model.dtype)

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

args = TrainingArguments(
    seed= seed,
    output_dir=f"./output/{model_name}",
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

#INFERENCE

import torch
from peft import PeftModel, LoraConfig, TaskType

mode_path = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
lora_path = model_name

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

tokenizer = AutoTokenizer.from_pretrained(mode_path)
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", torch_dtype=torch.bfloat16)

model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

# model.to('cuda')

all_responses = []

import json

def read_json_file(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {filename}.")
        return None

data = read_json_file(test_file)

for i in data:
    dialogue = i['input']
    prompt = f"You are an experienced clinician specializing in Major Depressive Disorder (MDD). Please predict each participant\u2019s scores for each PHQ 8 question based on the following dialogue.\n{dialogue}"

    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.5,
        repetition_penalty=1.1,
        eos_token_id=terminators,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    all_responses.append(f'Participant_ID: \nResponse: {response}\n')

with open(f'{model_name}.txt', 'w', encoding='utf-8') as file:
    for response in all_responses:
        file.write(response + '\n')