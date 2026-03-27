# Back-Translation Experiments

## Overview
This folder contains the code, logs, and output data from translating the French LLeQA dataset into Dutch, English, Finnish, Italian, and Spanish using Meta's SeamlessM4T large model. The translation was performed in three successive runs, each in a separate subfolder.

## Folder Structure

### `seamless1/`, `seamless2/`, `seamless3/` – Translation Runs
Each folder represents one run of the SeamlessM4T translation pipeline.

| File | Description |
| :--- | :--- |
| `multilingual_lleqa.py` | Main Python script that loads the French LLeQA dataset and translates articles and questions into multiple target languages using SeamlessM4T. |
| `runseamless.sh` | Bash script to launch `multilingual_lleqa.py` with the correct conda environment and parameters. |
| `multilleqa.txt` / `multilleqa.out` | Full stdout/stderr log of the translation run, including progress, errors, and completion status. |
| `history_firstrun.txt` / `historysecondrunfinal` / `historythirdrunfinal` | Bash history log for the respective run, showing all commands executed during that session. |

**Note**: The translation was split across three runs because the SeamlessM4T model processing large corpora required restarts due to memory and timeout constraints. Each run picks up where the previous left off.

### `pickle_results/` – Evaluation Result Files
Contains serialized Python pickle files with the evaluation results for each language pair's translation quality assessment.

| File | Description |
| :--- | :--- |
| `dutch_fr_evaluation_results.pickle` | Translation quality evaluation: Dutch ↔ French |
| `en_fr_evaluation_results.pickle` | Translation quality evaluation: English ↔ French |
| `english_fr_evaluation_results.pickle` | Translation quality evaluation: English ↔ French (alternate) |
| `finnish_fr_evaluation_results.pickle` | Translation quality evaluation: Finnish ↔ French |
| `italian_fr_evaluation_results.pickle` | Translation quality evaluation: Italian ↔ French |
| `spanish_fr_evaluation_results.pickle` | Translation quality evaluation: Spanish ↔ French |

### `translated_data/` – Translated Datasets
Contains the final translated LLeQA datasets for each language. **French data is NOT included** as it is the source dataset and must be obtained separately from HuggingFace.

Each language subfolder contains:
| File | Description |
| :--- | :--- |
| `{language}_articles.json` | Translated legal articles corpus (legislative provisions) |
| `{language}_questions_train.json` | Training set: question-article pairs for retrieval training |
| `{language}_questions_val.json` | Validation set: question-article pairs for dev evaluation |
| `{language}_questions_test.json` | Test set: question-article pairs for final evaluation |

Languages available: Dutch, English, Finnish, Italian, Spanish.
