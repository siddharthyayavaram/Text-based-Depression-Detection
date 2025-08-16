import torch
from peft import PeftModel, LoraConfig, TaskType
import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
import numpy as np
import random

seed = 40

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # for numpy random seed
    random.seed(seed)  # for python random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

train_file = 'tan_train.json'
test_file = 'tan_test.json'

set_random_seed(seed)

# model_name = f'{train_file.replace('.json','')}_{test_file.replace('.json','')}_{seed}'
model_name = 'output_base'
print(model_name)

mode_path = 'meta-llama/Meta-Llama-3.1-8B'
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

model.to('cuda')

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

def formatted_prompt(question)-> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant:"

for i in data:
    dialogue = i['input']
    prompt = f"You are an experienced clinician specializing in Major Depressive Disorder (MDD). Please predict each participant\u2019s scores for each PHQ 8 question based on the following dialogue.\n{dialogue}"

    # messages = [{"role": "user", "content": prompt}]

    # text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    text = formatted_prompt(prompt)

    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    # terminators = [
    #     tokenizer.eos_token_id,
    #     tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # ]

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.5,
        repetition_penalty=1.1,
        # eos_token_id=terminators,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    all_responses.append(f'Participant_ID: \nResponse: {response}\n')

with open(f'{model_name}.txt', 'w', encoding='utf-8') as file:
    for response in all_responses:
        file.write(response + '\n')