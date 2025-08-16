# LLM-based Depression-Detection

# Data

## Augmentation

### Core Scripts

This directory contains scripts for data augmentation and generation using various language models.

- **`augment_bert_pretrained.py`**: Augments text data by masking and filling with a pre-trained BERT model.
- **`augment_bert_finetuned.py`**: Augments text data using a fine-tuned Deproberta model for more domain-specific paraphrasing.
- **`augment_llama3.py`**: Paraphrases sentences in a dialogue using the Llama 3.1 model.
- **`generate_dialogues_from_finetuned_model.py`**: Generates dialogues between a chatbot and a participant using a fine-tuned causal language model.
- **`finetune_causal_lm.py`**: Fine-tunes a causal language model on a dialogue dataset using LoRA.
- **`paraphrase_dialogues_llama.py`**: Paraphrases entire dialogues using the Llama 3 model to generate augmented data.
- **`augment_parrot.py`**: Augments text data by paraphrasing sentences using the Parrot T5 model.

### EDA Augmentation Scripts

This directory contains scripts for data augmentation using the Easy Data Augmentation (EDA) techniques.

- **`eda_library.py`**: A library implementing Easy Data Augmentation (EDA) techniques for text.
- **`augment_dialogues_eda.py`**: Augments dialogues from a JSON file using the EDA library.


### Analysis

This directory contains a jupyter notebook for analyzing predicted and actual DAIC-WOz phq-8 scores and plots a few graphs

### Transcript Preprocess

- **`interview_transcript_filter.py`** - Filters interview transcripts to remove Ellie's responses that don't match predefined questions using fuzzy string matching
- **`transcript_question_tagger.py`** - Enhanced version that filters transcripts and tags matched questions, extracting content from parentheses and adding matched question metadata
- **`questions.csv`** - Contains the predefined list of interview questions used for matching and filtering transcripts

### Usage

Both scripts process interview transcripts by comparing Ellie's responses against a standard question set, filtering out responses below a similarity threshold (default 70%) and organizing the output for further analysis.

## Eval

- **`binary_classification_eval.py`** - Evaluates binary depression classification results across multiple random seeds and computes averaged metrics
- **`phq8_score_eval.py`** - Evaluates PHQ-8 score predictions by parsing individual question scores and converting to binary classification
- **`comp_results.py`** - Comprehensive evaluation script that processes PHQ-8 predictions, handles JSON data, and provides detailed statistical analysis with standard deviations

### Usage

These evaluation scripts analyze model performance on depression classification tasks using different approaches (direct binary classification, PHQ-8 scoring, and comprehensive analysis with statistics).
