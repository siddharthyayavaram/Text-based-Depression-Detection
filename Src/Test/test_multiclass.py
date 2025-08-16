import torch
from peft import PeftModel, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_lora_config(r=32):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,
        r=r,
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
    parser = argparse.ArgumentParser(description='Test 4-class depression severity model')
    parser.add_argument('--model_path', type=str, default='meta-llama/Meta-Llama-3.1-8B', help='Base model path')
    parser.add_argument('--lora_path', type=str, required=True, help='LoRA adapter path')
    parser.add_argument('--test_file', type=str, default='C_test.json', help='Test data file')
    parser.add_argument('--seed', type=int, default=40, help='Random seed')
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA rank')
    parser.add_argument('--max_tokens', type=int, default=6000, help='Maximum new tokens to generate')
    
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    
    # Load model and tokenizer
    config = create_lora_config(r=args.lora_r)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, model_id=args.lora_path, config=config)
    
    # Load test data
    data = load_json_data(args.test_file)
    
    all_responses = []
    
    for i, item in enumerate(data):
        dialogue = item['input']
        prompt = f"You are an experienced clinician specializing in Major Depressive Disorder (MDD). Please predict each participant's overall depression class based on the following dialogue out of the possible classes of 'Non depressed', 'Mild depression', 'Moderate depression' and 'Severe depression'.\n{dialogue}"
        
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
            do_sample=False,
            top_p=1.0, 
            temperature=0.0, 
            repetition_penalty=1.0,
            eos_token_id=terminators,
        )
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        all_responses.append(f'Participant_ID: {i}\nResponse: {response}\n')
    
    # Save results
    output_file = f'{args.lora_path}_results.txt'
    with open(output_file, 'w', encoding='utf-8') as file:
        for response in all_responses:
            file.write(response + '\n')
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()