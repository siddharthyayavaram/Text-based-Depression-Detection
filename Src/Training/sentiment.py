import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

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

dataset1 = np.array(pd.read_csv('NTU_datasets/dev_split_Depression_AVEC2017.csv',delimiter=',',encoding='utf-8'))[:, 0:2]
dataset2 = np.array(pd.read_csv('NTU_datasets/full_test_split.csv',delimiter=',',encoding='utf-8'))[:, 0:2]
dataset3 = np.array(pd.read_csv('NTU_datasets/train_split_Depression_AVEC2017.csv',delimiter=',',encoding='utf-8'))[:, 0:2]

dataset = np.concatenate((dataset1, np.concatenate((dataset2, dataset3))))
countPos = 0

def checkPosNeg(dataset, index):
    for i in range(0, len(dataset)):
        if(dataset[i][0] == index):
            return dataset[i][1]
    return 0

Data = []
Y = []

countPos = 0
index = -1
Data_test = []
Y_test = []

text_train_ids = [int(item[0]) for item in dataset3]
text_dev_ids = [int(item[0]) for item in dataset1]
text_test_ids = [int(item[0]) for item in dataset2]

for i in range(0, len(dataset3)):
    val = checkPosNeg(dataset, dataset3[i][0])
    Y.append(val)
    try:
        fileName = "NTU_datasets/transcript/" + str(int(dataset3[i][0])) + "_TRANSCRIPT.csv"
        Data.append(np.array(pd.read_csv(fileName,delimiter='\t',encoding='utf-8', engine='python'))[:, 2:4])
    except Exception as e:
        print(e)

for i in range(0, len(dataset1)):
    val = checkPosNeg(dataset, dataset1[i][0])
    Y.append(val)
    try:
        fileName = "NTU_datasets/transcript/" + str(int(dataset1[i][0])) + "_TRANSCRIPT.csv"
        Data.append(np.array(pd.read_csv(fileName, delimiter='\t', encoding='utf-8', engine='python'))[:, 2:4])
    except Exception as e:
        print(e)

for i in range(0, len(dataset2)):
    Y_test.append(checkPosNeg(dataset, dataset2[i][0]))
    try:
        fileName = "NTU_datasets/transcript/" + str(int(dataset2[i][0])) + "_TRANSCRIPT.csv"
        Data_test.append(np.array(pd.read_csv(fileName,delimiter='\t',encoding='utf-8', engine='python'))[:, 2:4])
    except Exception as e:
        print(e)


Y = np.array(Y)
Data2 = []
Data2_test = []
Y_test = np.array(Y_test)

for i in range(0, len(Data)):
    script = []
    for k in range(1, len(Data[i])):
        if(Data[i][k][0] == "Participant"):
            script.append(Data[i][k][1])
    Data2.append(script)

for i in range(0, len(Data_test)):
    script = []
    for k in range(1, len(Data_test[i])):
        if(Data_test[i][k][0] == "Participant"):
            script.append(Data_test[i][k][1])
    Data2_test.append(script)

from tqdm import tqdm

joined_sen = []

for i in range(len(Data2)):
      a = Data2[i]
      b = [sentence for sentence in a if(not isinstance(sentence, float) and len(sentence.split()) > 15)]
      joined_sen.append(" ".join(b[:]))

joined_sen_test = []

for i in range(len(Data2_test)):
      a = Data2_test[i]
      b = [sentence for sentence in a if(not isinstance(sentence, float) and len(sentence.split()) > 15)]
      joined_sen_test.append(" ".join(b[:]))

from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm
import torch

def batch_summarize(sentences_list, batch_size=4, max_length=115, min_length=80, device='cuda'):
    model_name = "philschmid/bart-large-cnn-samsum"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model.to(device)
    summaries = []
    for i in tqdm(range(0, len(sentences_list), batch_size), desc="Batch Summarization"):
        batch = sentences_list[i:i + batch_size]
        batch_input_ids = tokenizer.batch_encode_plus(batch, max_length=1024, truncation=True, return_tensors='pt', pad_to_max_length=True)['input_ids'].to(device)
        with torch.no_grad():
            summary_ids = model.generate(batch_input_ids, max_length=max_length, min_length=min_length)
        batch_summaries = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        summaries.extend(batch_summaries)
    return summaries

batch_size = 4
device = 'cuda'
sum_train= batch_summarize(joined_sen, batch_size=batch_size, device=device)
sum_test =  batch_summarize(joined_sen_test, batch_size=batch_size, device=device)

Y = Y.astype(int)
df = pd.DataFrame({'sentence': sum_train, 'label': Y})

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
result_df_train = df

from sklearn.utils import resample

df_class_0 = result_df_train[result_df_train['label'] == 0]
df_class_1 = result_df_train[result_df_train['label'] == 1]

# Upsample the minority class (class 1)
df_class_1_upsampled = resample(df_class_1, replace=True, n_samples=len(df_class_0), random_state=42)
result_df_train = pd.concat([df_class_0, df_class_1_upsampled])
result_df_train = result_df_train.sample(frac=1).reset_index(drop=True)

sentences=result_df_train.sentence.values
result_df_train['label'] = result_df_train['label'].replace({1.0: 1, 0.0: 0})
labels = result_df_train.label.values

from transformers import BertTokenizer
# using the low level BERT for our task.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Printing the original sentence.
print(' Original: ', sentences[0])

# Printing the tokenized sentence in form of list.
print('Tokenized: ', tokenizer.tokenize(sentences[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

input_ids = []
for sent in sentences:
    # so basically encode tokenizing , mapping sentences to thier token ids after adding special tokens.
    encoded_sent = tokenizer.encode(
                        sent,                      # Sentence which are encoding.
                        add_special_tokens = True, # Adding special tokens '[CLS]' and '[SEP]'

                         )
    input_ids.append(encoded_sent)

from keras.preprocessing.sequence import pad_sequences

MAX_LEN = 128

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN , truncating="post", padding="post")

input_ids

attention_masks = []

for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)

from sklearn.model_selection import train_test_split

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


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

batch_size = 32

# DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# DataLoader for our validation(test) set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# Check the sizes of the first dimensions
print(train_inputs.size(0))
print(train_masks.size(0))
print(train_labels.size(0))

from transformers import BertForSequenceClassification, AdamW, BertConfig

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)

import torch
print(torch.cuda.is_available())

model.cuda()

optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )

from transformers import get_linear_schedule_with_warmup

# Number of training epochs (authors recommend between 2 and 4)
epochs = 20

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

device = torch.device("cuda")

import random
seed_val = 33

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []

# For each epoch...
for epoch_i in range(0, epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()
        outputs = model(b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels)

        loss = outputs[0]
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():

            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")

# Create DataFrames for each list
Y_test = Y_test.astype(int)
df = pd.DataFrame({'sentence': sum_test, 'label': Y_test})

# Shuffle the DataFrame
df = df.sample(frac=1).reset_index(drop=True)
df.head()

result_df = df

# result_df_train = result_df[:600]

# Separate the DataFrame into classes
df_class_0_test = result_df[result_df['label'] == 0]
df_class_1_test = result_df[result_df['label'] == 1]

# Upsample the minority class (class 1)
df_class_1_upsampled_test = resample(df_class_1_test, replace=True, n_samples=len(df_class_0_test), random_state=42)

# Concatenate the upsampled minority class with the original majority class
result_df = pd.concat([df_class_0_test, df_class_1_upsampled_test])

# Shuffle the upsampled DataFrame
result_df = result_df.sample(frac=1).reset_index(drop=True)

# Display the head of the upsampled DataFrame

df = result_df
sentences = df.sentence.values
labels = df.label.values

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []

# For every sentence...
for sent in sentences:
    encoded_sent = tokenizer.encode(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                   )

    input_ids.append(encoded_sent)

# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                          dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

# Convert to tensors.
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels)

# Set the batch size.
batch_size = 32

# Create the DataLoader.
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

# Put model in evaluation mode
model.eval()

# Tracking variables
predictions , true_labels = [], []

# Predict
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)

  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch

  # Telling the model not to compute or store gradients, saving memory and
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None,
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()

  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')

print('Positive samples: %d of %d (%.2f%%)' % (df.label.sum(), len(df.label), (df.label.sum() / len(df.label) * 100.0)))

from sklearn.metrics import accuracy_score, classification_report

flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# Combine the correct labels for each batch into a single list.
flat_true_labels = [item for sublist in true_labels for item in sublist]

# Calculate accuracy
accuracy = accuracy_score(flat_true_labels, flat_predictions)
print('Accuracy: %.3f' % accuracy)

# Show classification report
print('Classification Report:')
print(classification_report(flat_true_labels, flat_predictions,digits = 4))
