import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType

seed = 40
model_name = f"MODEL_{seed}"
base_model_path = "BASE_MODEL_NAME"
lora_path = f"./{model_name}"

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
)

tokenizer = AutoTokenizer.from_pretrained(lora_path)
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

all_responses = []

labels = ['Mild depression', 'Moderate depression', 'Non depressed', 'Severe depression']
labels = labels * 200

for label in tqdm(labels):
    prompt = (
        f"You are a dialogue generator, generate a dialogue between 'Ellie'—a clinical chatbot "
        f"specialized in Major Depressive Disorder and a 'Participant' based on an input label. "
        f"Input: {label}"
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=6000,
        do_sample=True,
        top_p=0.8,
        temperature=1.0,
        repetition_penalty=1.0,
        eos_token_id=terminators,
    )

    # Remove prompt tokens from output
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    all_responses.append(f"\nDialogue: {response}\n")

with open("LLM_OUTPUT.txt", "w", encoding="utf-8") as file:
    for response in all_responses:
        file.write(response + "\n")
