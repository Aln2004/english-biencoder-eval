# Cross-Lingual Retrieval Experiments

## Overview
This folder contains the code, scripts, and results for evaluating zero-shot cross-lingual information retrieval using Bi-Encoder models. Each language pair tests whether a model trained on one language's queries can effectively retrieve relevant articles from a corpus in a different language.

## Folder Structure

### `common_code/` – Shared Evaluation Codebase
This directory contains the Python source code used by all cross-lingual evaluation pairs.

| File | Description |
| :--- | :--- |
| `data/lleqa.py` | LLeQA dataset loader: reads articles and questions JSONs, constructs IR evaluation datasets |
| `data/annotation.py` | Annotation utility for LLM-based labeling tasks |
| `data/text_processor.py` | Text preprocessing utilities |
| `retriever/biencoder_training.py` | **Main evaluation script**: loads a pre-trained Bi-Encoder, runs IR evaluation on dev/test sets using dynamic dataset paths via argparse |
| `retriever/biencoder_inference.py` | Inference-only script for running retrieval on new queries |
| `retriever/bm25.py` | BM25 baseline retrieval implementation |
| `utils/SentenceTransformer.py` | Custom SentenceTransformer wrapper with evaluation hooks |
| `utils/common.py` | Common utility functions |
| `utils/loggers.py` | Logging configuration |
| `utils/metrics.py` | IR evaluation metrics computation |
| `utils/FastChat.py` | FastChat model integration utilities |
| `utils/shampoo.py` | Shampoo optimizer implementation |

### Per-Pair Folders
Each language pair has its own folder containing:

| File | Description |
| :--- | :--- |
| `run_eval.sh` | Standalone bash script to run the evaluation for this specific pair. Sets model path, corpus path, and query paths. |
| `results/` | CSV files with IR evaluation metrics for this pair (dev and test sets) |

### Language Pairs Evaluated

| Folder | Model/Queries Language | Corpus Language | Language Family Relationship |
| :--- | :--- | :--- | :--- |
| `french_to_italian/` | French | Italian | Romance → Romance |
| `italian_to_french/` | Italian | French | Romance → Romance |
| `spanish_to_french/` | Spanish | French | Romance → Romance |
| `dutch_to_french/` | Dutch | French | Germanic → Romance |
| `french_to_dutch/` | French | Dutch | Romance → Germanic |
| `finnish_to_italian/` | Finnish | Italian | Uralic → Romance |
| `finnish_to_dutch/` | Finnish | Dutch | Uralic → Germanic |

## How to Run a Single Pair

```bash
# Activate the environment
conda activate rachitlegalnlp

# Navigate to the pair folder
cd cross-lingual/french_to_italian

# Run the evaluation
bash run_eval.sh
```

## How to Run All 7 Pairs Sequentially

The orchestration script `scripts/run_cross_lingual_eval.sh` (in the repo root) iterates through all 7 pairs:

```bash
cd ~/cbot_bleu/english
nohup bash scripts/run_cross_lingual_eval.sh > experiment_outs/cross_lingual_7pairs.txt 2>&1 &
```
