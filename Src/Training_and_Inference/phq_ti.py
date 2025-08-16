import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer,DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig, TrainerCallback
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
import numpy as np

import random

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # for numpy random seed
    random.seed(seed)  # for python random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(33)

wandb_env_vars = ["WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_ID", "WANDB_MODE"]
for var in wandb_env_vars:
    os.environ.pop(var, None)  # Remove the environment variable if it exists

df = pd.read_json('MLM.json')
ds = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("tokenizer.pad_token",tokenizer.pad_token)
print("tokenizer.pad_token_id", tokenizer.pad_token_id)
print("tokenizer.eos_token_id",tokenizer.eos_token_id)

# data processing
def process_func(example):
    MAX_LENGTH = 10000    
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False) 
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

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

# create model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto",torch_dtype=torch.bfloat16)
model.enable_input_require_grads()
print(model.dtype)

# lora

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)


model = get_peft_model(model, config)

model.print_trainable_parameters()

# training param

args = TrainingArguments(
    output_dir="./output/MLM",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=10,
    num_train_epochs=10,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to= []
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)   

trainer.train()

# save result

peft_model_id="./MLM"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, LoraConfig, TaskType
import pandas as pd
import os

csv_path = 'full_test_split.csv'
txt_folder_path = 'Text_LLM_test'
mode_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
lora_path = 'MLM'
from peft import LoraConfig, TaskType, get_peft_model

df = pd.read_csv(csv_path)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)


tokenizer = AutoTokenizer.from_pretrained(mode_path)
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16)

from peft import PeftModel

model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

all_responses = []

for participant_id in df['Participant_ID']:
    txt_file_path = os.path.join(txt_folder_path, f'{participant_id}_combined.txt')
    
    # 检查txt文件是否存在
    if os.path.exists(txt_file_path):
        with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
            dialogue = txt_file.read()

#dialogue = "Participant: <synch>\nEllie: IntroV4Confirmation (hi i'm ellie thanks for coming in today i was created to talk to people in a safe and secure environment i'm not a therapist but i'm here to learn about people and would love to learn about you i'll ask a few questions to get us started and please feel free to tell me anything your answers are totally confidential are you ok with this)\nParticipant: yes\nEllie: okay_confirm (okay)\nEllie: how_doingV (so how are you doing today)\nParticipant: i'm not bad i'm a little tired but okay\nEllie: thats_good (that's good)\nEllie: where_originally (where are you from originally)\nParticipant: uh from saint louis missouri\nEllie: really (really)\nParticipant: yep\nEllie: when_LA (when did you move to l_a)\nParticipant: uh two months ago\nEllie: often_backB (how often do you go back to your hometown)\nParticipant: uh not often i think i'm out here for good now\nEllie: compares_LA (how does it compare to l_a)\nParticipant: uh smaller town less people everyone kinda acts different the pace of life is slower out there\nEllie: mhm (mhm)\nEllie: why_LA (why did you move to l_a)\nParticipant: uh looking for work as a game designer or game developer just\nParticipant: try something new too new place new people\nEllie: adapted_LA (how easy was it for you to get used to living in l_a)\nParticipant: uh i guess i'm still getting used to it\nEllie: tell_about_that (can you tell me about that)\nParticipant: um\nParticipant: i guess\nParticipant: people are different you know there's little shops everywhere just\nParticipant: everything's different\nEllie: yeah_downer (yeah)\nEllie: like_about_LA (what are some things you really like about l_a)\nParticipant: uh man the weather's great out here\nParticipant: the people are really nice\nParticipant: like to look at the mountains the beach is nice\nParticipant: just\nEllie: uh_huh (uh huh)\nParticipant: better geography\nEllie: dont_like_LA (what are some things you don't really like about l_a)\nParticipant: um traffic i guess the commute\nParticipant: and sometimes there's maybe too many people\nEllie: see_mean (i see what you mean)\nEllie: travel_shoes (i'm sure you can tell by my shoes i'm not much of a world explorer do you travel a lot)\nParticipant: <laughter>\nParticipant: not really never been outside of the u_s before\nEllie: why2 (why)\nParticipant: uh just never had the opportunity i'd love to go though\nEllie: okay_confirm (okay)\nEllie: study (what did you study at school)\nParticipant: uh game design and graphic design sort of\nEllie: sounds_interesting (that sounds interesting)\nParticipant: oh i love it\nEllie: hard_decision (how did you decide to do that)\nParticipant: uh\nParticipant: mm been playing video games since i was a kid like most guys my age so\nParticipant: just like making people happy and it's kinda easy to do that with games\nEllie: thats_great (that's great)\nParticipant: <laughter>\nEllie: dream_job (what's your dream job)\nParticipant: uh work for myself make my own games and write my own paychecks i guess\nEllie: difficult (how hard is that)\nParticipant: uh pretty hard i've been trying to do it for the past couple years it's a little difficult\nEllie: isee_downer (i see)\nEllie: shyoutgoing (do you consider yourself more shy or outgoing)\nParticipant: i guess i'm kinda shy it depends on the people and the environment i guess\nEllie: understand (i understand)\nEllie: relax_fishtank (sometimes when i'm feeling tense i turn on the fish tank screensaver hey i know it's not hawaii but it's the best i've got what do you do to relax)\nParticipant: <laughter>\nParticipant: uh play guitar listen to some music talk to friends\nEllie: cool3 (cool)\nEllie: control_temper (how are you at controlling your temper)\nParticipant: i'd say i'm pretty good at controlling my temper i'm not the type of guy that gets angry really easily\nEllie: thats_good (that's good)\nEllie: last_argument (when was the last time you argued with someone and what was it about)\nParticipant: uh probably with my girlfriend over something silly i can't even remember\nEllie: feelguilty (what's something you feel guilty about)\nParticipant: uh <sigh>\nParticipant: don't know i don't think i'm really guilty about anything i try to avoid being guilty\nEllie: okay_confirm (okay)\nEllie: situation_handled (tell me about a situation that you wish you had handled differently)\nParticipant: hmm\nParticipant: don't know\nParticipant: can't really think of anything\nEllie: hmm_downer (hmm)\nEllie: hard_decisionB (tell me about the hardest decision you've ever had to make)\nParticipant: leaving saint louis was one of them 'cause all my friends and family were there\nEllie: change_directions (what made you decide to do that)\nParticipant: uh i guess the benefits outweighed\nParticipant: you know the cons\nEllie: mhm (mhm)\nEllie: happy_didthat (are you happy you did that)\nParticipant: definitely\nEllie: thats_good (that's good)\nEllie: memory_erase (tell me about an event or something that you wish you could erase from your memory)\nParticipant: huh <sigh> maybe getting kicked out of high school the first time\nEllie: tell_about_that (can you tell me about that)\nParticipant: uh just i ended up getting myself and and my friends in trouble and\nParticipant: it was just no good you kinda look back on it and\nParticipant: i guess that's something from my past i wish i would've done different\nEllie: avoid (could you have done anything to avoid it)\nParticipant: uh\nParticipant: followed the <laughter> rules i guess just been a good student would've helped\nEllie: okay_confirm (okay)\nEllie: family_relationship (tell me about your relationship with your family)\nParticipant: uh\nParticipant: not really that close with my family i've got a small family anyway and\nParticipant: when i was in saint louis we'd see each other a couple times a year but that's really it so no strong ties\nEllie: uh_huh (uh huh)\nEllie: influence_positive (who's someone that's been a positive influence in your life)\nParticipant: uh i guess most of my friends they're always there to support me and encourage me and help me so\nEllie: good_hear (that's so good to hear)\nEllie: easy_sleep (how easy is it for you to get a good night's sleep)\nParticipant: say pretty easy every once in awhile there's a night i can't sleep but it's pretty rare\nEllie: sleep_affects (what are you like when you don't sleep well)\nParticipant: mm just tired groggy\nParticipant: i guess my mind's unfocused and\nParticipant: day goes by heck of a lot longer\nEllie: feel_lately (how have you been feeling lately)\nParticipant: uh pretty good\nParticipant: not really much to say\nEllie: behavior_changes (have you noticed any changes in your behavior or thoughts lately)\nParticipant: uh maybe a little bit more depressed 'cause i'm away from most of my friends and family but\nParticipant: nothing unmanageable\nEllie: symptoms_cope (how do you cope with them)\nParticipant: uh just try to talk to 'em when i can think about something else think about happy times\nEllie: ptsd_diagnosed (have you ever been diagnosed with p_t_s_d)\nParticipant: no\nEllie: depression_diagnosed (have you been diagnosed with depression)\nParticipant: no\nEllie: happy_lasttime (tell me about the last time you felt really happy)\nParticipant: uh <sigh> i guess on the drive out here\nParticipant: 'cause it was such a new experience and so exhilarating\nEllie: memorableB (what's one of your most memorable experiences)\nParticipant: um i think graduating college 'cause i was really proud of myself and really looking forward to the future\nEllie: describe_felt (how did you feel in that moment)\nParticipant: just i guess kind of euphoria standing there it seemed kinda surreal <su> surreal\nEllie: yeah3 (yeah)\nEllie: BF_describe (how would your best friend describe you)\nParticipant: uh\nParticipant: funny caring nice guy\nParticipant: easy to get along with\nEllie: self_change (what are some things you wish you could change about yourself)\nParticipant: uh <sigh>\nParticipant: pretty happy being me i guess\nEllie: feelbadly (tell me about a time when someone made you feel really badly about yourself)\nParticipant: hmm\nParticipant: guess when i broke up with my last girlfriend for obvious reasons you tend to fight and say things you wish you'd you'd forget i guess\nEllie: yeah_downer (yeah)\nEllie: regret (is there anything you regret)\nParticipant: no i don't like to live in the past so there's nothing to really regret\nEllie: advice_back (what advice would you give to yourself ten or twenty years ago)\nParticipant: uh <sigh>\nParticipant: keep working hard and just\nParticipant: keep thinking about what you do before you do it\nEllie: ideal_weekendC (tell me how you spend your ideal weekend)\nParticipant: on the computer playing video games with friends\nEllie: Ellie17Dec2012_09 (what are some things that usually put you in a good mood)\nParticipant: uh\nParticipant: music uh\nParticipant: and my friends i guess <laughter>\nEllie: okay_confirm (okay)\nEllie: Ellie17Dec2012_08 (what are you most proud of in your life)\nParticipant: mm\nParticipant: i guess my determination and my xxx my passion\nParticipant: for my creativity\nEllie: asked_everything (okay i think i have asked everything i need to)\nEllie: appreciate_open (thanks for sharing your thoughts with me)\nEllie: bye (goodbye)\nParticipant: bye"
        prompt = f"You are an experienced clinician specializing in Major Depressive Disorder (MDD). Please predict each participant\u2019s overall depression class based on the following dialogue.\n{dialogue}"
        messages = [
         {"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

        # input_ids = tokenizer.apply_chat_template(
        #     messages,
        #     add_generation_prompt=True,
        #     return_tensors="pt"
        # ).to(model.device)
    
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=False,
            top_p=1.0, 
            temperature=0.0, 
            repetition_penalty=1.0,
            eos_token_id=terminators,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        all_responses.append(f'Participant_ID: {participant_id}\nResponse: {response}\n')


#print(response)    
with open('MLM.txt', 'w', encoding='utf-8') as file:
    # 将response写入文件
    for response in all_responses:
        file.write(response + '\n')
