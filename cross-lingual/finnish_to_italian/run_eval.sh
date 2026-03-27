#!/bin/bash
# Cross-Lingual Evaluation: finnish model/queries -> italian corpus
# This script evaluates the finnish-trained Bi-Encoder on the italian corpus
# using finnish queries for zero-shot cross-lingual retrieval.
#
# Prerequisites:
#   - Activate conda environment: conda activate rachitlegalnlp
#   - Ensure the common_code/ directory is in the parent folder
#   - Ensure datasets are available at the specified paths
#   - Ensure model checkpoint exists at /shared/checkpoints/finnish/final

export WANDB_MODE=disabled

MODEL="/shared/checkpoints/finnish/final"
CORPUS_PATH="/home/btech/cbot_bleu/italian/data/lleqa/italian_articles.json"
TRAIN_PATH="/home/btech/cbot_bleu/finnish/data/lleqa/finnish_questions_train.json"
DEV_PATH="/home/btech/cbot_bleu/finnish/data/lleqa/finnish_questions_val.json"
TEST_PATH="/home/btech/cbot_bleu/finnish/data/lleqa/finnish_questions_test.json"

if [ ! -f "$TEST_PATH" ]; then
    TEST_PATH=$DEV_PATH
fi

cd ../..
python cross-lingual/common_code/retriever/biencoder_training.py \
    --model_name "$MODEL" \
    --corpus_path "$CORPUS_PATH" \
    --train_path "$TRAIN_PATH" \
    --dev_path "$DEV_PATH" \
    --test_path "$TEST_PATH" \
    --max_seq_length 384 \
    --pooling "mean" \
    --sim "cos_sim" \
    --epochs 20 \
    --train_batch_size 32 \
    --scheduler "warmuplinear" \
    --lr 2e-5 \
    --optimizer "AdamW" \
    --wd 0.01 \
    --warmup_steps 60 \
    --use_fp16 \
    --seed 42 \
    --eval_before_training \
    --eval_during_training \
    --log_steps 1 \
    --do_save \
    --output_dir "cross-lingual/finnish_to_italian/results" \
    --do_test
