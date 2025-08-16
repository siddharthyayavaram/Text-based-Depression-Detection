import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import random
import argparse
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
import time
import datetime

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_daic_data():
    dataset1 = np.array(pd.read_csv('NTU_datasets/dev_split_Depression_AVEC2017.csv', delimiter=',', encoding='utf-8'))[:, 0:2]
    dataset2 = np.array(pd.read_csv('NTU_datasets/full_test_split.csv', delimiter=',', encoding='utf-8'))[:, 0:2]
    dataset3 = np.array(pd.read_csv('NTU_datasets/train_split_Depression_AVEC2017.csv', delimiter=',', encoding='utf-8'))[:, 0:2]
    
    dataset = np.concatenate((dataset1, np.concatenate((dataset2, dataset3))))
    
    def checkPosNeg(dataset, index):
        for i in range(len(dataset)):
            if dataset[i][0] == index:
                return dataset[i][1]
        return 0
    
    # Load training data
    Data, Y = [], []
    for i in range(len(dataset3)):
        val = checkPosNeg(dataset, dataset3[i][0])
        Y.append(val)
        try:
            fileName = f"NTU_datasets/transcript/{int(dataset3[i][0])}_TRANSCRIPT.csv"
            Data.append(np.array(pd.read_csv(fileName, delimiter='\t', encoding='utf-8', engine='python'))[:, 2:4])
        except Exception as e:
            print(f"Error loading {fileName}: {e}")
    
    # Load validation data  
    for i in range(len(dataset1)):
        val = checkPosNeg(dataset, dataset1[i][0])
        Y.append(val)
        try:
            fileName = f"NTU_datasets/transcript/{int(dataset1[i][0])}_TRANSCRIPT.csv"
            Data.append(np.array(pd.read_csv(fileName, delimiter='\t', encoding='utf-8', engine='python'))[:, 2:4])
        except Exception as e:
            print(f"Error loading {fileName}: {e}")
    
    # Load test data
    Data_test, Y_test = [], []
    for i in range(len(dataset2)):
        Y_test.append(checkPosNeg(dataset, dataset2[i][0]))
        try:
            fileName = f"NTU_datasets/transcript/{int(dataset2[i][0])}_TRANSCRIPT.csv"
            Data_test.append(np.array(pd.read_csv(fileName, delimiter='\t', encoding='utf-8', engine='python'))[:, 2:4])
        except Exception as e:
            print(f"Error loading {fileName}: {e}")
    
    return Data, np.array(Y), Data_test, np.array(Y_test)

def extract_participant_text(Data):
    Data_processed = []
    for i in range(len(Data)):
        script = []
        for k in range(1, len(Data[i])):
            if Data[i][k][0] == "Participant":
                script.append(Data[i][k][1])
        Data_processed.append(script)
    return Data_processed

def join_sentences(Data, min_words=15):
    joined_sentences = []
    for i in range(len(Data)):
        sentences = Data[i]
        filtered_sentences = [
            sentence for sentence in sentences 
            if not isinstance(sentence, float) and len(sentence.split()) > min_words
        ]
        joined_sentences.append(" ".join(filtered_sentences))
    return joined_sentences

def batch_summarize(sentences_list, batch_size=4, max_length=115, min_length=80, device='cuda'):
    """Summarize text using BART model"""
    model_name = "philschmid/bart-large-cnn-samsum"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model.to(device)
    
    summaries = []
    for i in tqdm(range(0, len(sentences_list), batch_size), desc="Batch Summarization"):
        batch = sentences_list[i:i + batch_size]
        batch_input_ids = tokenizer.batch_encode_plus(
            batch, max_length=1024, truncation=True, 
            return_tensors='pt', pad_to_max_length=True
        )['input_ids'].to(device)
        
        with torch.no_grad():
            summary_ids = model.generate(batch_input_ids, max_length=max_length, min_length=min_length)
        
        batch_summaries = [
            tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) 
            for g in summary_ids
        ]
        summaries.extend(batch_summaries)
    
    return summaries

def prepare_bert_inputs(sentences, tokenizer, max_len=128):
    input_ids = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        input_ids.append(encoded_sent)
    
    input_ids = pad_sequences(input_ids, maxlen=max_len, truncating="post", padding="post")
    
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    
    return input_ids, attention_masks

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def main():
    parser = argparse.ArgumentParser(description='Train BERT model for depression classification')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--summarize', action='store_true', help='Use text summarization')
    
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    
    Data, Y, Data_test, Y_test = load_daic_data()
    
    Data2 = extract_participant_text(Data)
    Data2_test = extract_participant_text(Data_test)
    
    joined_sen = join_sentences(Data2)
    joined_sen_test = join_sentences(Data2_test)
    
    if args.summarize:
        sum_train = batch_summarize(joined_sen, batch_size=4, device='cuda')
        sum_test = batch_summarize(joined_sen_test, batch_size=4, device='cuda')
    else:
        sum_train = joined_sen
        sum_test = joined_sen_test
    
    Y = Y.astype(int)
    df = pd.DataFrame({'sentence': sum_train, 'label': Y})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_class_0 = df[df['label'] == 0]
    df_class_1 = df[df['label'] == 1]
    df_class_1_upsampled = resample(df_class_1, replace=True, n_samples=len(df_class_0), random_state=42)
    df_balanced = pd.concat([df_class_0, df_class_1_upsampled])
    df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)
    
    sentences = df_balanced.sentence.values
    labels = df_balanced.label.values
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    input_ids, attention_masks = prepare_bert_inputs(sentences, tokenizer, args.max_len)
    
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        input_ids, labels, test_size=0.2, random_state=42
    )
    train_masks, validation_masks, _, _ = train_test_split(
        attention_masks, labels, test_size=0.2, random_state=42
    )

    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.batch_size)
    
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.cuda()
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    device = torch.device("cuda")
    loss_values = []
    
    for epoch_i in range(args.epochs):
        print(f'\n======== Epoch {epoch_i + 1} / {args.epochs} ========')
        print('Training...')
        
        t0 = time.time()
        total_loss = 0
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(f'  Batch {step:>5,}  of  {len(train_dataloader):>5,}.    Elapsed: {elapsed}.')
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            
            loss = outputs[0]
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)
        
        print(f"\n  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training epoch took: {format_time(time.time() - t0)}")
        
        # Validation
        print("\nRunning Validation...")
        t0 = time.time()
        model.eval()
        
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
        
        print(f"  Accuracy: {eval_accuracy/nb_eval_steps:.2f}")
        print(f"  Validation took: {format_time(time.time() - t0)}")
    
    print("\nTraining complete!")

    Y_test = Y_test.astype(int)
    df_test = pd.DataFrame({'sentence': sum_test, 'label': Y_test})
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    
    df_class_0_test = df_test[df_test['label'] == 0]
    df_class_1_test = df_test[df_test['label'] == 1]
    df_class_1_upsampled_test = resample(df_class_1_test, replace=True, n_samples=len(df_class_0_test), random_state=42)
    df_test_balanced = pd.concat([df_class_0_test, df_class_1_upsampled_test])
    df_test_balanced = df_test_balanced.sample(frac=1).reset_index(drop=True)
    
    test_sentences = df_test_balanced.sentence.values
    test_labels = df_test_balanced.label.values
    
    test_input_ids, test_attention_masks = prepare_bert_inputs(test_sentences, tokenizer, args.max_len)

    prediction_inputs = torch.tensor(test_input_ids)
    prediction_masks = torch.tensor(test_attention_masks)
    prediction_labels = torch.tensor(test_labels)
    
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=args.batch_size)

    print(f'Predicting labels for {len(prediction_inputs):,} test sentences...')
    
    model.eval()
    predictions, true_labels = [], []
    
    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        predictions.append(logits)
        true_labels.append(label_ids)
    
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    
    accuracy = accuracy_score(flat_true_labels, flat_predictions)
    print(f'Test Accuracy: {accuracy:.4f}')
    print('\nClassification Report:')
    print(classification_report(flat_true_labels, flat_predictions, digits=4))
    
    model.save_pretrained('./bert_depression_model')
    tokenizer.save_pretrained('./bert_depression_model')
    print("Model saved to ./bert_depression_model")

if __name__ == "__main__":
    main()