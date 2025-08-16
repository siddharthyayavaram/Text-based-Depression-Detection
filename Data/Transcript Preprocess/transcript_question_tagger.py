import pandas as pd
from fuzzywuzzy import fuzz
import re
import os

def filter_ellie_responses(transcript_df, questions, threshold=70):
    indices_to_remove = []
    matched_questions = [None] * len(transcript_df)

    for index, row in transcript_df.iterrows():
        if row['speaker'] == 'Ellie' and isinstance(row['value'], str):
            # Extract content from parentheses if present
            match = re.search(r'\((.*?)\)', row['value'])
            if match:
                transcript_df.at[index, 'value'] = match.group(1)
            
            # Find best matching question
            best_match = None
            best_score = 0
            for question in questions:
                score = fuzz.token_sort_ratio(row['value'], question)
                if score > best_score:
                    best_match = question
                    best_score = score

            if best_score < threshold:
                indices_to_remove.append(index)
            else:
                matched_questions[index] = best_match

    # Remove low-similarity responses and add question tags
    transcript_df = transcript_df.drop(indices_to_remove).reset_index(drop=True)
    matched_questions = [q for i, q in enumerate(matched_questions) if i not in indices_to_remove]
    transcript_df['matched_question'] = matched_questions
    
    return transcript_df

# Load questions
questions = list(pd.read_csv("questions.csv")['Questions'])

# Load participant data
train_df = pd.read_csv('train_split_Depression_AVEC2017.csv')
test_df = pd.read_csv('full_test_split.csv')
dev_df = pd.read_csv('dev_split_Depression_AVEC2017.csv')
participant_ids = list(train_df.Participant_ID.values) + list(dev_df.Participant_ID.values) + list(test_df.Participant_ID.values)

# Create output directory
os.makedirs('tagged_transcripts', exist_ok=True)

# Process each participant
for participant_id in participant_ids:
    participant_transcript = pd.read_csv(f'transcript/{participant_id}_TRANSCRIPT.csv', sep='\t')
    tagged_transcript = filter_ellie_responses(participant_transcript, questions)
    tagged_transcript.to_csv(f'tagged_transcripts/{participant_id}_TRANSCRIPT.csv', index=False)