import torch
from peft import PeftModel, LoraConfig, TaskType
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

def load_json_data(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filename} was not found.")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from the file {filename}.")

def main():
    parser = argparse.ArgumentParser(description='Test individual PHQ-8 question models')
    parser.add_argument('--question_index', type=int, required=True, help='PHQ-8 question index (0-7)')
    parser.add_argument('--model_path', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Base model path')
    parser.add_argument('--test_file', type=str, default='FINAL_DEV.json', help='Test data file')
    parser.add_argument('--seed', type=int, default=40, help='Random seed')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum new tokens to generate')
    
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
    
    name = headers[args.question_index]
    train_file = f'FINAL_TRAIN_{name}.json'
    model_name = f"{train_file.replace('.json','')}_{args.test_file.replace('.json','')}_{args.seed}"
    
    # Load model and tokenizer
    config = create_lora_config()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, model_id=model_name, config=config)
    model.to('cuda')
    
    # Load test data
    data = load_json_data(args.test_file)
    
    all_responses = []
    
    for i, item in enumerate(data):
        dialogue = item['input']
        prompt = f"You are an experienced clinician specializing in Major Depressive Disorder (MDD). Please predict the participant's score for this PHQ 8 question: {questions[args.question_index]} based on the following dialogue.\n{dialogue}"
        
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=args.max_tokens,
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
        all_responses.append(f'Participant_ID: {i}\nResponse: {response}\n')
    
    # Save results
    output_file = f'{model_name}_results.txt'
    with open(output_file, 'w', encoding='utf-8') as file:
        for response in all_responses:
            file.write(response + '\n')
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()