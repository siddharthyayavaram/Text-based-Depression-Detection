import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer,DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig, TrainerCallback
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
import numpy as np

import random

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # for numpy random seed
    random.seed(seed)  # for python random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(33)

# Unset wandb related environment variables
wandb_env_vars = ["WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_ID", "WANDB_MODE"]
for var in wandb_env_vars:
    os.environ.pop(var, None)  # Remove the environment variable if it exists

# load data
df = pd.read_json('aug_data_mlm.json')
ds = Dataset.from_pandas(df)

# load model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("tokenizer.pad_token",tokenizer.pad_token)
print("tokenizer.pad_token_id", tokenizer.pad_token_id)
print("tokenizer.eos_token_id",tokenizer.eos_token_id)

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


# create model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto",torch_dtype=torch.bfloat16)
model.enable_input_require_grads()
print(model.dtype)

# lora

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

# training param

args = TrainingArguments(
    output_dir="./output/L_mlm",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=10,
    num_train_epochs=10,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to= []
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)   

trainer.train()

# save result

peft_model_id="./L_mlm"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, LoraConfig, TaskType
import pandas as pd
import os

csv_path = 'full_test_split.csv'
txt_folder_path = 'Text_LLM_test'
mode_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
lora_path = 'L_mlm'
from peft import LoraConfig, TaskType, get_peft_model

df = pd.read_csv(csv_path)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)


tokenizer = AutoTokenizer.from_pretrained(mode_path)
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16)

from peft import PeftModel

model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

all_responses = []

for participant_id in df['Participant_ID']:
    txt_file_path = os.path.join(txt_folder_path, f'{participant_id}_combined.txt')
    
    # 检查txt文件是否存在
    if os.path.exists(txt_file_path):
        with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
            dialogue = txt_file.read()

        prompt = f"You are an experienced clinician specializing in Major Depressive Disorder (MDD). Please predict each participant\u2019s scores for each PHQ 8 question based on the following dialogue.\n{dialogue}"
        messages = [
         {"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

        # input_ids = tokenizer.apply_chat_template(
        #     messages,
        #     add_generation_prompt=True,
        #     return_tensors="pt"
        # ).to(model.device)
    
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=False,
            top_p=1.0, 
            temperature=0.0, 
            repetition_penalty=1.0,
            eos_token_id=terminators,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        all_responses.append(f'Participant_ID: {participant_id}\nResponse: {response}\n')


#print(response)    
with open('L_mlm_op.txt', 'w', encoding='utf-8') as file:
    # 将response写入文件
    for response in all_responses:
        file.write(response + '\n')
