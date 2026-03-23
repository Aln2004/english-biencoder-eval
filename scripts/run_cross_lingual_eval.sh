#!/bin/bash
export WANDB_MODE=disabled
MAX_SEQ_LEN=384
POOL="mean"
SIM="cos_sim"
EPOCHS=20
BATCH_SIZE=32
SCHEDULER="warmuplinear"
LR=2e-5
WARMUP_STEPS=60
OPTIMIZER="AdamW"
WD=0.01
FP16="--use_fp16"

EVAL_BEFORE_TRAINING="--eval_before_training"
EVAL_DURING_TRAINING="--eval_during_training"
LOG_STEPS=1
DO_TEST="--do_test"
DO_SAVE="--do_save"
SEED=42

# Define pairs as Source:Target
PAIRS=(
    "french:italian"
    "italian:french"
    "spanish:french"
    "dutch:french"
    "french:dutch"
    "finnish:italian"
    "finnish:dutch"
)

for PAIR in "${PAIRS[@]}"; do
    SRC=${PAIR%%:*}
    TGT=${PAIR##*:}
    
    echo "================================================================"
    echo "Running Cross-Lingual Pair: Model/Queries=${SRC} -> Corpus=${TGT}"
    echo "================================================================"
    
    MODEL="/shared/checkpoints/${SRC}/final"
    CORPUS_PATH="/home/btech/cbot_bleu/${TGT}/data/lleqa/${TGT}_articles.json"
    
    TRAIN_PATH="/home/btech/cbot_bleu/${SRC}/data/lleqa/${SRC}_questions_train.json"
    DEV_PATH="/home/btech/cbot_bleu/${SRC}/data/lleqa/${SRC}_questions_val.json"
    TEST_PATH="/home/btech/cbot_bleu/${SRC}/data/lleqa/${SRC}_questions_test.json"
    
    # If the explicit test JSON doesn't exist natively, safely fallback to val for metrics
    if [ ! -f "$TEST_PATH" ]; then
        TEST_PATH=$DEV_PATH
    fi
    
    python src/retriever/biencoder_training.py \
        --model_name "$MODEL" \
        --corpus_path "$CORPUS_PATH" \
        --train_path "$TRAIN_PATH" \
        --dev_path "$DEV_PATH" \
        --test_path "$TEST_PATH" \
        --max_seq_length $MAX_SEQ_LEN \
        --pooling "$POOL" \
        --sim "$SIM" \
        --epochs $EPOCHS \
        --train_batch_size $BATCH_SIZE \
        --scheduler "$SCHEDULER" \
        --lr $LR \
        --optimizer "$OPTIMIZER" \
        --wd $WD \
        --warmup_steps $WARMUP_STEPS \
        $FP16 \
        --seed $SEED \
        $EVAL_BEFORE_TRAINING \
        $EVAL_DURING_TRAINING \
        --log_steps $LOG_STEPS \
        $DO_SAVE \
        --output_dir "output/training/retriever/cross_lingual_${SRC}_to_${TGT}" \
        $DO_TEST
done
