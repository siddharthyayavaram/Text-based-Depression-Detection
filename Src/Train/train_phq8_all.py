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

def create_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

def process_func(example, tokenizer):
    MAX_LENGTH = 10000
    
    prompt = tokenizer(
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n", 
        add_special_tokens=False
    )
    response = tokenizer(
        f"<|im_start|>assistant\n{example['output']}<|im_end|>\n", 
        add_special_tokens=False
    )
    
    input_ids = prompt["input_ids"] + response["input_ids"]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt["input_ids"]) + response["input_ids"]
    
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
    parser = argparse.ArgumentParser(description='Train model to predict all PHQ-8 scores simultaneously')
    parser.add_argument('--train_file', type=str, default='tan_train.json', help='Training data file')
    parser.add_argument('--seed', type=int, default=40, help='Random seed')
    parser.add_argument('--model_path', type=str, default='meta-llama/Meta-Llama-3.1-8B', help='Base model path')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./output_base', help='Output directory')
    
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    
    # Load dataset
    df = pd.read_json(args.train_file)
    ds = Dataset.from_pandas(df)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Apply processing to dataset
    tokenized_id = ds.map(lambda x: process_func(x, tokenizer), remove_columns=ds.column_names)
    
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    model.enable_input_require_grads()
    
    # Configure LoRA
    config = create_lora_config()
    model = get_peft_model(model, config)
    
    # Training arguments
    training_args = TrainingArguments(
        seed=args.seed,
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=10,
        num_train_epochs=args.epochs,
        save_steps=100,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to=[]
    )
    
    # Trainer setup
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        formatting_func=lambda x: process_func(x, tokenizer)
    )
    
    # Train the model
    trainer.train()
    
    # Save the model and tokenizer
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()