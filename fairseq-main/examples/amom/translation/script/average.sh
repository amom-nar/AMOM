#!/usr/bin/env bash
set -x
set -e
#---------------------------------------

MODEL_PATH=[Path to save checkpoints]
python AMOM/fairseq-main/scripts/average_checkpoints.py \
        --inputs $MODEL_PATH/checkpoint.pt \
        $MODEL_PATH/checkpoint.pt \
        $MODEL_PATH/checkpoint.pt \
        $MODEL_PATH/checkpoint.pt \
        $MODEL_PATH/checkpoint.pt \
        --output $MODEL_PATH1/checkpoint_average_bleu.pt 