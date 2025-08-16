import pandas as pd
from fuzzywuzzy import fuzz
import os

def filter_ellie_responses(transcript_df, questions, threshold=70):
    indices_to_remove = []
    for index, row in transcript_df.iterrows():
        if row['speaker'] == 'Ellie':
            if all(fuzz.token_sort_ratio(row['value'], question) < threshold for question in questions):
                indices_to_remove.append(index)
    return transcript_df.drop(indices_to_remove)

# Load questions
questions = list(pd.read_csv("questions.csv")['Questions'])

# Load participant data
train_df = pd.read_csv('train_split_Depression_AVEC2017.csv')
test_df = pd.read_csv('full_test_split.csv')
dev_df = pd.read_csv('dev_split_Depression_AVEC2017.csv')
participant_ids = list(train_df.Participant_ID.values) + list(dev_df.Participant_ID.values) + list(test_df.Participant_ID.values)

# Create output directory
os.makedirs('filtered_transcripts', exist_ok=True)

# Process each participant
for participant_id in participant_ids:
    participant_transcript = pd.read_csv(f'transcript/{participant_id}_TRANSCRIPT.csv', sep='\t')
    filtered_transcript = filter_ellie_responses(participant_transcript, questions)
    filtered_transcript.to_csv(f'filtered_transcripts/{participant_id}_TRANSCRIPT.csv', index=False)