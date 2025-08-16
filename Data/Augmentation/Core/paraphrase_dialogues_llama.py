import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
from tqdm import tqdm
import json

model_path = 'meta-llama/Meta-Llama-3-8B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

data = read_json_file('C_train_3.json')

def aug(ip,label):
    aug_sentences = []
    for _ in range(4):
        prompt = f"Paraphrase the given dialogue without changing the labels 'Ellie' and 'Participant' and immediately return the output!. Input: {ip} Paraphrased Output: "
        messages = [
            {"role": "user", "content": prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        temperature = random.uniform(0.5, 1.0)
        top_p = random.uniform(0.8, 1.0)
        outputs = model.generate(
            input_ids,
            max_new_tokens=6000,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        aug_sentences.append(response)
    return aug_sentences

aug_data = []

for i in tqdm(data[:]):
    aug_data.append(i)
    label = i['output']
    res = aug(i['input'],label)
    for j in range(4):
        aug_data.append({'instruction':i['instruction'], 'input': res[j], 'output' : i['output']})

def list_to_json_file(list_of_dicts, file_path):
    with open(file_path, 'w') as file:
        json.dump(list_of_dicts, file, indent=4)

list_to_json_file(aug_data,'C_train_3_fulpar.json')
