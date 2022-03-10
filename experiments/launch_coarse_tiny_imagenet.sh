#!/bin/bash

set -Eeuxo pipefail

EXP_ROOT=../experiments/2022-03-10-tiny-imagenet-coarsen/

for FORM in flat hier ; do
    for DIST in "beta-1-1" "beta-1-2" "beta-2-1" ; do
        EXP_NAME="coarsen-${DIST}-${FORM}"
        ../env/bin/python main.py \
            --experiment_dir=$EXP_ROOT/$EXP_NAME/ \
            --tensorboard_dir=$EXP_ROOT/tensorboard/$EXP_NAME \
            --config=configs/tiny_imagenet.py \
            --config.dataset_root=/home/jack/data/manual/tiny_imagenet/ \
            --noskip_initial_eval \
            --config.predict=${FORM}_softmax \
            --config.train_with_leaf_targets=false \
            --config.train_labels="tiny_imagenet-train-${DIST}-seed-0" &
    done
done
