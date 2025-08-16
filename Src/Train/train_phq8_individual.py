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
    parser = argparse.ArgumentParser(description='Train individual PHQ-8 question models')
    parser.add_argument('--question_index', type=int, required=True, help='PHQ-8 question index (0-7)')
    parser.add_argument('--seed', type=int, default=40, help='Random seed')
    parser.add_argument('--model_path', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Base model path')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    headers = [
        "PHQ8_NoInterest", "PHQ8_Depressed", "PHQ8_Sleep", "PHQ8_Tired",
        "PHQ8_Appetite", "PHQ8_Failure", "PHQ8_Concentrating", "PHQ8_Moving"
    ]
    
    questions = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself, or that you are a failure, or have let yourself or your family down",
        "Trouble concentrating on things, such as reading the newspaper or watching TV",
        "Moving or speaking so slowly that other people could have noticed, or the opposite - being so fidgety or restless that you have been moving around a lot more than usual"
    ]
    
    set_random_seed(args.seed)
    
    # Disable wandb logging
    os.environ.pop("WANDB_API_KEY", None)
    os.environ.pop("WANDB_PROJECT", None)
    os.environ.pop("WANDB_ENTITY", None)
    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_MODE", None)
    
    name = headers[args.question_index]
    train_file = f'FINAL_TRAIN_{name}.json'
    test_file = 'FINAL_DEV.json'
    model_name = f"{train_file.replace('.json','')}_{test_file.replace('.json','')}_{args.seed}"
    
    # Load and process data
    df = pd.read_json(train_file)
    ds = Dataset.from_pandas(df)
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Process dataset
    tokenized_id = ds.map(lambda x: process_func(x, tokenizer), remove_columns=ds.column_names)
    
    # Load and configure model
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16)
    base_model.enable_input_require_grads()
    
    config = create_lora_config()
    model = get_peft_model(base_model, config)
    
    # Training arguments
    training_args = TrainingArguments(
        seed=args.seed,
        output_dir=f"./models/{model_name}",
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
    
    # Train model
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        formatting_func=lambda x: process_func(x, tokenizer),
    )
    
    trainer.train()
    
    # Save model
    peft_model_id = f"./{model_name}"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()