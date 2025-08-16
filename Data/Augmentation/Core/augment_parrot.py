import json
import torch
from parrot import Parrot
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def set_random_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_random_seed(42)

# Initialize Parrot model
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=True)

def read_json_file(file_path: str):
    with open(file_path, 'r') as file:
        return json.load(file)

def list_to_json_file(list_of_dicts: list, file_path: str):
    with open(file_path, 'w') as file:
        json.dump(list_of_dicts, file, indent=4)

def augment_text(text: str, num_aug: int = 4):
    augmented_outputs = [''] * num_aug
    lines = text.split('\n')

    for line in lines:
        ind = line.find(':') if line[0] != 'E' else line.find('(')
        sentence = line[ind + 1:]
        should_augment = len(sentence.split()) >= 10

        augmented_sentences = []
        if should_augment:
            augmented_sentences = parrot.augment(
                input_phrase=sentence,
                diversity_ranker="levenshtein",
                do_diverse=False,
                max_return_phrases=10,
                max_length=32,
                adequacy_threshold=0.6,
                fluency_threshold=0.6
            )

        if not augmented_sentences or len(augmented_sentences) < num_aug:
            should_augment = False

        for j in range(num_aug):
            if not should_augment:
                augmented_outputs[j] += line + '\n'
            else:
                if line[0] == 'E':
                    augmented_outputs[j] += line[:ind + 1] + augmented_sentences[j][0] + ')\n'
                else:
                    augmented_outputs[j] += line[:ind + 1] + augmented_sentences[j][0] + '\n'

    return augmented_outputs

# Main augmentation process
data = read_json_file('data.json')
augmented_data = []

num_aug = 4
for item in tqdm(data):
    augmented_data.append(item)
    augmented_inputs = augment_text(item['input'], num_aug=num_aug)
    for aug_input in augmented_inputs:
        augmented_data.append({
            'instruction': item['instruction'],
            'input': aug_input,
            'output': item['output']
        })

list_to_json_file(augmented_data, 'aug_data_par.json')
print('Data augmentation complete.')
