import json
import pandas as pd
from eda_library import eda

INPUT_JSON_PATH = ''
OUTPUT_JSON_PATH = ''

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def list_to_json_file(list_of_dicts, file_path):
    with open(file_path, 'w') as file:
        json.dump(list_of_dicts, file, indent=4)

def augment_text(s, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, num_aug=9):
    augmented_output = [''] * num_aug
    for line in s.split('\n'):
        ind = line.find(':')
        sentence = line[ind+1:]
        do_augment = len(sentence.split()) >= 5

        if do_augment:
            aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri,
                                alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)

        for j in range(num_aug):
            if not do_augment:
                augmented_output[j] += line + '\n'
            else:
                augmented_output[j] += line[:ind+1] + aug_sentences[j] + '\n'

    return augmented_output

data = read_json_file(INPUT_JSON_PATH)

augmented_data = []
NUM_AUG = 4

for item in data:
    augmented_data.append(item)
    augmented_versions = augment_text(item['input'])
    for j in range(NUM_AUG):
        augmented_data.append({
            'instruction': item['instruction'],
            'input': augmented_versions[j],
            'output': item['output']
        })

list_to_json_file(augmented_data, OUTPUT_JSON_PATH)
