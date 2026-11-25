# Dialogue Summarizer using T5

Fine-tuned T5-Small model to summarize long dialogues/conversations into short, meaningful summaries.

Built using Python, PyTorch, and Hugging Face Transformers.

## Problem
Long WhatsApp / meeting/chat conversations are hard to read. This model reads the full dialogue and gives a short summary in 1-2 lines.

**Example:**
> **Input Dialogue:**
> Amanda: I baked cookies. Do you want some?
> Jerry: Sure! Can I have two?
> Amanda: I'll bring them tomorrow.
>
> **Output Summary:**
> Amanda baked cookies and will bring some for Jerry tomorrow.

## Dataset
SAMSum Dialogue Dataset

- `samsum-train.csv` - Training data
- `samsum-validation.csv` - Validation data
- `samsum-test.csv` - Testing data

Each row has 2 columns:
- `dialogue` - Full conversation
- `summary` - Human-written summary

## Files in this Repo
- `text_summarizer.ipynb` - Full code: data loading, preprocessing, tokenization, training, evaluation, and inference
- `samsum-train.csv` / `samsum-validation.csv` / `samsum-test.csv` - Dataset files

## How it Works
1. Load the SAMSum dataset
2. Clean and preprocess dialogues
3. Tokenize dialogue and summary using T5 Tokenizer
4. Fine-tune the `t5-small` model for the summarization task
5. Evaluate on test data using ROUGE score
6. Generate summary for new dialogues

## Tech Stack
- Python
- PyTorch
- Hugging Face Transformers - T5-Small
- Pandas, NumPy
- Jupyter Notebook

## How to Run
```bash
pip install transformers torch pandas datasets
