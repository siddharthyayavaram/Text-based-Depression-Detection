import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
import numpy as np
import random
import argparse

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_lora_config(r=32):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=r,
        lora_alpha=32,
        lora_dropout=0.1
    )

def process_func(example, tokenizer):
    MAX_LENGTH = 10000
    instruction = tokenizer(
        f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 
        add_special_tokens=False
    )
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

def main():
    parser = argparse.ArgumentParser(description='Train 4-class depression severity model with validation')
    parser.add_argument('--train_file', type=str, default='C_train.json', help='Training data file')
    parser.add_argument('--val_file', type=str, default='C_val.json', help='Validation data file')
    parser.add_argument('--seed', type=int, default=40, help='Random seed')
    parser.add_argument('--model_path', type=str, default='meta-llama/Meta-Llama-3.1-8B', help='Base model path')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA rank')
    
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    
    # Disable wandb logging
    wandb_env_vars = ["WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_ID", "WANDB_MODE"]
    for var in wandb_env_vars:
        os.environ.pop(var, None)
    
    model_name = f'multiclass_{args.seed}'
    
    # Load training data
    df = pd.read_json(args.train_file)
    ds = Dataset.from_pandas(df)
    
    # Load validation data
    df_v = pd.read_json(args.val_file)
    ds_v = Dataset.from_pandas(df_v)
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Process datasets
    tokenized_id = ds.map(lambda x: process_func(x, tokenizer), remove_columns=ds.column_names)
    tokenized_id_v = ds_v.map(lambda x: process_func(x, tokenizer), remove_columns=ds_v.column_names)
    
    # Load and configure model
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto', torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()
    
    config = create_lora_config(r=args.lora_r)
    model = get_peft_model(model, config)
    
    # Training arguments with validation
    training_args = TrainingArguments(
        seed=args.seed,
        output_dir=f"./models/{model_name}",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_strategy="epoch",
        gradient_accumulation_steps=1,
        logging_steps=10,
        num_train_epochs=args.epochs,
        save_steps=100,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to=[]
    )
    
    # Train model with validation
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_id,
        eval_dataset=tokenized_id_v,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    trainer.train()
    
    # Save model
    peft_model_id = f"./{model_name}"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

if __name__ == "__main__":
    main()