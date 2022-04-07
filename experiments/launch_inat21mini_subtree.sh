#!/bin/bash

EXP_ROOT=/mnt/ssd1/projects/2022-01-hierarchical/experiments/2022-04-07-inat21-trunc/

for DEPTH in 3 4 5 6 ; do EXP_NAME=flat-depth-$DEPTH ; ../env/bin/python main.py --experiment_dir=${EXP_ROOT}/${EXP_NAME} --tensorboard_dir=${EXP_ROOT}/tensorboard/${EXP_NAME} --config=configs/inaturalist2021mini.py --config.dataset_root=/home/jack/data/manual/inaturalist2021/ --config.predict=flat_softmax --config.keep_examples=True --config.train_with_leaf_targets=True --config.train_subtree=inat21_max_depth_$DEPTH --loader_num_workers=8 --save_freq=20 ; done
