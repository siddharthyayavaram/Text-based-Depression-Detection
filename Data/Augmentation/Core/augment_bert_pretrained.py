import torch
import warnings
warnings.filterwarnings("ignore")

import random
import json
from tqdm import tqdm
from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased", device="cuda")

def random_state(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

random_state(42)

def read_json_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

FILENAME = ""
data = read_json_file(FILENAME)

def augment(input_text, unmasker=unmasker):
    words = input_text.split()
    rand_idx = random.randint(1, len(words) - 1)
    orig_word = words[rand_idx]
    words[rand_idx] = "[MASK]"
    masked_text = " ".join(words)
    candidates = unmasker(masked_text)
    results = [res["sequence"] for res in candidates if res["token_str"] != orig_word]
    return results[:4]

def aug(s, num=4, unmasker=unmasker):
    outputs = [""] * num
    lines = s.split("\n")

    for line in lines:
        ind = line.find(":")
        sentence = line[ind + 1 :]
        valid = len(sentence.split()) >= 5
        aug_sentences = augment(sentence, unmasker) if valid else []
        if not aug_sentences or len(aug_sentences) < num:
            valid = False
        for j in range(num):
            if not valid:
                outputs[j] += line + "\n"
            else:
                outputs[j] += line[: ind + 1] + aug_sentences[j][0] + "\n"
    return outputs

aug_data = []
num = 4

for i in tqdm(data):
    aug_data.append(i)
    variants = aug(i["input"], num, unmasker)
    for j in range(num):
        aug_data.append(
            {"instruction": i["instruction"], "input": variants[j], "output": i["output"]}
        )

def list_to_json_file(list_of_dicts, file_path):
    with open(file_path, "w") as file:
        json.dump(list_of_dicts, file, indent=4)

OUTPUT_FILE = ""
list_to_json_file(aug_data, OUTPUT_FILE)
