import time
import torch
import warnings
warnings.filterwarnings("ignore")

import random
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def random_state(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

random_state(42)

model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

def read_json_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

FILENAME = ""
OUTPUT_FILE = ""

data = read_json_file(FILENAME)

def aug(s, num_aug=4):
    print(time.time())
    outputs = [""] * num_aug
    lines = s.split("\n")

    for i in range(len(lines[6:])):
        boo = False
        line = lines[i]
        ind = line.find(":")
        if line[0] == "E":
            boo = True
        sentence = line[ind + 1 :]
        aug_sentences = []

        if not boo:
            instr = (
                f"Rephrase this sentence with siimilar meaning using different vocabulary. "
                f"Directly write down the result after the prompt without any explanations. "
                f"Adhere to these 5 principles during the generation. "
                f"1) Integrity of Content: Retain the original meaning and sentiment. "
                f"2) Conversational Authenticity: Use natural, casual language. "
                f"3) Respectful Communication: Maintain a respectful tone. "
                f"4) Consistency in Length: Keep similar length to the original. "
                f"5) Tolerance for Informality: Tolerate some irregularities (omissions, repetitions, filler words). "
                f"These are the sentences before the one you have to rephrase. "
                f"{' '.join(lines[i-6:i])}. Here is the input sentence: {sentence}."
            )

            messages = [{"role": "user", "content": instr}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            for _ in range(num_aug):
                temperature = random.uniform(0.5, 1.0)
                top_p = random.uniform(0.8, 1.0)
                outputs_ids = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
                response = tokenizer.decode(
                    outputs_ids[0][input_ids.shape[-1]:],
                    skip_special_tokens=True
                )
                aug_sentences.append(response)

        for j in range(num_aug):
            if boo:
                outputs[j] += line + "\n"
            else:
                outputs[j] += line[: ind + 1] + aug_sentences[j] + "\n"

    return outputs

aug_data = []
num_aug = 4

for i in tqdm(data[:]):
    aug_data.append(i)
    variants = aug(i["input"])
    for j in range(num_aug):
        aug_data.append(
            {"instruction": i["instruction"], "input": variants[j], "output": i["output"]}
        )

def list_to_json_file(list_of_dicts, file_path):
    with open(file_path, "w") as file:
        json.dump(list_of_dicts, file, indent=4)

list_to_json_file(aug_data, OUTPUT_FILE)
