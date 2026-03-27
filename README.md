# mLLeQA: Multilingual Legal Question Answering – Back-Translation & Cross-Lingual Experiments

## Experiment Description

This repository contains all the code, data, and results for the **back-translation** and **cross-lingual retrieval** experiments conducted as part of the mLLeQA (Multilingual Legal Legislative Question Answering) project. The back-translation pipeline uses Meta's SeamlessM4T model to translate the French LLeQA dataset into five target languages—Dutch, English, Finnish, Italian, and Spanish—to produce a multilingual legal QA benchmark. The cross-lingual retrieval experiments evaluate how well a Bi-Encoder model trained on one language's dataset can retrieve relevant legal articles from a corpus in a different language, testing zero-shot cross-lingual transferability across 7 language pairs. All experiments use the translated LLeQA datasets and were run on a single NVIDIA RTX 6000 Ada Generation GPU (48 GB VRAM) using the `rachitlegalnlp` conda environment with Python 3.8.

## GPU Information

All experiments were run on:
- **GPU**: NVIDIA RTX 6000 Ada Generation (48 GB VRAM)
- **Driver Version**: 590.48.01
- **CUDA Version**: 13.1
- **Server**: `btech@10.240.166.7`

The full `nvidia-smi` output is available in `environment/gpu_info.txt`.

## Repository Structure

```
├── back-translation/           # SeamlessM4T translation pipeline
│   ├── seamless1/              # First translation run
│   ├── seamless2/              # Second translation run
│   ├── seamless3/              # Third translation run
│   ├── pickle_results/         # Evaluation result pickle files
│   ├── translated_data/        # Translated datasets per language
│   │   ├── dutch/
│   │   ├── english/
│   │   ├── finnish/
│   │   ├── italian/
│   │   └── spanish/
│   └── README.md               # Detailed description of back-translation files
│
├── cross-lingual/              # Cross-lingual retrieval experiments
│   ├── common_code/            # Shared evaluation codebase
│   │   ├── data/               # Data loaders (lleqa.py, annotation.py)
│   │   ├── retriever/          # Bi-Encoder training & evaluation scripts
│   │   └── utils/              # Utilities (SentenceTransformer, metrics, etc.)
│   ├── french_to_italian/      # French model → Italian corpus
│   ├── italian_to_french/      # Italian model → French corpus
│   ├── spanish_to_french/      # Spanish model → French corpus
│   ├── dutch_to_french/        # Dutch model → French corpus
│   ├── french_to_dutch/        # French model → Dutch corpus
│   ├── finnish_to_italian/     # Finnish model → Italian corpus
│   ├── finnish_to_dutch/       # Finnish model → Dutch corpus
│   └── README.md               # Detailed description of cross-lingual files
│
├── environment/                # Conda and pip environment exports
│   ├── environment.yml
│   ├── condalistexport.txt
│   ├── requirementspip3.txt
│   ├── requirementspip.txt
│   └── gpu_info.txt
│
├── evaluation_results/         # Raw evaluation logs and CSV outputs
├── results.md                  # Summary table of cross-lingual metrics
└── README.md                   # This file
```

## Step-by-Step Instructions to Run the Experiments

### A. Back-Translation Experiments

#### Prerequisites
1. SSH into the GPU server:
   ```bash
   ssh btech@10.240.166.7
   ```
2. Activate the conda environment:
   ```bash
   conda activate rachitlegalnlp
   ```
3. Obtain the French LLeQA dataset from https://huggingface.co/datasets/maastrichtlawtech/lleqa and place the articles and questions JSONs in the appropriate directory.

#### Running the Translation
1. Navigate to the back-translation directory:
   ```bash
   cd ~/cbot_bleu/back-translation/seamless1
   ```
2. Run the translation script:
   ```bash
   bash runseamless.sh
   ```
3. The script calls `multilingual_lleqa.py`, which uses the SeamlessM4T model to translate the French LLeQA articles and questions into Dutch, English, Finnish, Italian, and Spanish.
4. The output logs are saved to `multilleqa.txt`.
5. The translation was run in three successive attempts (`seamless1/`, `seamless2/`, `seamless3/`) to handle intermediate failures and ensure all languages were fully translated.
6. The final translated datasets are in `translated_data/{language}/`:
   - `{language}_articles.json` – translated legal articles corpus
   - `{language}_questions_train.json` – training question set
   - `{language}_questions_val.json` – validation question set
   - `{language}_questions_test.json` – test question set

### B. Cross-Lingual Retrieval Experiments

#### Prerequisites
1. SSH into the GPU server:
   ```bash
   ssh btech@10.240.166.7
   ```
2. Activate the conda environment:
   ```bash
   conda activate rachitlegalnlp
   ```
3. Ensure the translated datasets for each language are placed at:
   ```
   ~/cbot_bleu/{language}/data/lleqa/{language}_articles.json
   ~/cbot_bleu/{language}/data/lleqa/{language}_questions_train.json
   ~/cbot_bleu/{language}/data/lleqa/{language}_questions_val.json
   ~/cbot_bleu/{language}/data/lleqa/{language}_questions_test.json
   ```
4. Ensure trained Bi-Encoder model checkpoints exist at:
   ```
   /shared/checkpoints/{language}/final
   ```

#### Running All 7 Pairs at Once
1. Navigate to the English directory:
   ```bash
   cd ~/cbot_bleu/english
   ```
2. Run the orchestration script:
   ```bash
   nohup bash scripts/run_cross_lingual_eval.sh > experiment_outs/cross_lingual_7pairs.txt 2>&1 &
   ```
3. Monitor progress:
   ```bash
   tail -f experiment_outs/cross_lingual_7pairs.txt
   ```

#### Running a Single Language Pair
1. Navigate to the specific pair directory, e.g., for French → Italian:
   ```bash
   cd cross-lingual/french_to_italian
   ```
2. Run the pair-specific script:
   ```bash
   bash run_eval.sh
   ```
3. Results (CSV files) will be saved in the `results/` subdirectory.

#### Understanding the Results
- Each pair folder contains CSV files with Information Retrieval metrics:
  - `Information-Retrieval_evaluation_lleqa_dev_results.csv` – Dev set metrics
  - `Information-Retrieval_evaluation_lleqa_test_results.csv` – Test set metrics
- Key metrics: **Recall@100**, **MRR@10**, **NDCG@10**, **MAP@10**
- The summary of all results is in `results.md` at the root of this repository.

### C. Reproducing the Environment
1. To recreate the exact conda environment:
   ```bash
   conda env create -f environment/environment.yml
   ```
2. Or install pip packages directly:
   ```bash
   pip install -r environment/requirementspip3.txt
   ```

## Cross-Lingual Results Summary

| Source Model / Queries | Target Corpus | Recall@100 | MRR@10 | NDCG@10 | MAP@10 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **French** | **Italian** | 13.46% | 0.0340 | 0.0164 | 0.0087 |
| **Italian** | **French** | 0.43% | 0.0000 | 0.0000 | 0.0000 |
| **Spanish** | **French** | 9.49% | 0.0116 | 0.0111 | 0.0058 |
| **Dutch** | **French** | 2.31% | 0.0000 | 0.0000 | 0.0000 |
| **French** | **Dutch** | 4.02% | 0.0393 | 0.0192 | 0.0104 |
| **Finnish** | **Italian** | 0.34% | 0.0017 | 0.0012 | 0.0006 |
| **Finnish** | **Dutch** | 1.54% | 0.0039 | 0.0025 | 0.0011 |
