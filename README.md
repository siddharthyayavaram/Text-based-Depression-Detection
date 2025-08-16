# Text-based Depression Detection

This project explores **depression detection from text** using transcript preprocessing, data augmentation, fine-tuning LLMs (BERT, LLaMA, etc.), and multiple evaluation strategies.

---

## Workflow

### 1. Preprocess & Analyze Data

* **Transcript Preprocess**

  * `interview_transcript_filter.py` – filters transcripts, removes irrelevant responses
  * `transcript_question_tagger.py` – tags responses with matched interview questions
  * `questions.csv` – predefined set of interview questions
* **Analysis**

  * `DAIC-WOZ-analysis.ipynb` – notebook with data exploration and PHQ-8 score visualization
---

### 2. Augment Data

#### Core (model-based)

Scripts for paraphrasing and generating new dialogue data:

* `augment_bert_pretrained.py`, `augment_bert_finetuned.py` – augment text using BERT (general vs. fine-tuned)
* `augment_llama3.py`, `paraphrase_dialogues_llama.py` – LLaMA-based augmentation
* `augment_parrot.py` – Parrot T5 paraphrasing
* `finetune_causal_lm.py`, `generate_dialogues_from_finetuned_model.py` – fine-tune/generate synthetic dialogues

#### EDA (rule-based)

* `eda_library.py`, `augment_dialogues_eda.py` – synonym replacement, random insertion/deletion, etc.

---

### 3. Train Models

Located in `Src/Train/` — these scripts accept command-line arguments.

* **Binary classification** → `train_binary.py`
* **Multiclass severity (4-class)** → `train_multiclass.py`
* **PHQ-8 prediction** → `train_phq8_all.py`, `train_phq8_individual.py`
* **BERT sentiment baseline** → `train_bert_sentiment.py`

```bash
# Example: Train binary classifier
python Src/Train/train_binary.py --train_file clean_train.json --epochs 5
```

---

### 4. Run Inference

Located in `Src/Test/` — all take command-line arguments.

* **Binary:** `test_binary.py`
* **Multiclass:** `test_multiclass.py`
* **PHQ-8 (all/individual):** `test_phq8_all.py`, `test_phq8_individual.py`

```bash
# Example: Run inference for binary classifier
python Src/Test/test_binary.py --test_file clean_test.json
```

---

### 5. Evaluate Results

Located in `Eval/`.
* `binary_classification_eval.py` – evaluates binary classification across seeds
* `phq_score_eval.py` – evaluates PHQ-8 score predictions (individual + binary conversion)
* `comp_results.py` – aggregates and compares results with detailed stats