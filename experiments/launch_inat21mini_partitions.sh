#!/bin/bash

EXP_ROOT=/mnt/ssd1/projects/2022-01-hierarchical/experiments/2022-04-11-inat21mini-partition/

# ( for DEPTH in 7 6 5 4 ; do for INDEX in 0 1 ; do echo d${DEPTH}_n2_i${INDEX}_c ; done ; done ) | \
# xargs -I{} -P 2 ../env/bin/python main.py --experiment_dir=${EXP_ROOT}/{} --tensorboard_dir=${EXP_ROOT}/tensorboard/{} --config=configs/inaturalist2021mini.py --config.dataset_root=/home/jack/data/manual/inaturalist2021/ --config.train_subtree=inat21_partition_{} --config.keep_examples=False --config.train_with_leaf_targets=true --loader_num_workers=8 --save_freq=5

( for INDEX in 0 1 ; do for DEPTH in 7 6 5 4 ; do echo d${DEPTH}_n2_i${INDEX}_c ; done ; done ) | \
xargs -t -I{} -P 1 ../env/bin/python main.py --experiment_dir=${EXP_ROOT}/{} --tensorboard_dir=${EXP_ROOT}/tensorboard/{} --config=configs/inaturalist2021mini.py --config.dataset_root=/home/jack/data/manual/inaturalist2021/ --config.train_subtree=inat21_partition_{} --config.keep_examples=False --config.train_with_leaf_targets=true --loader_num_workers=8 --save_freq=5

( for INDEX in 0 1 ; do for DEPTH in 7 6 5 4 ; do echo d${DEPTH}_n2_i${INDEX}_c ; done ; done ) | \
xargs -t -I{} -P 1 ../env/bin/python main.py --experiment_dir=${EXP_ROOT}/{}-hier --tensorboard_dir=${EXP_ROOT}/tensorboard/{}-hier --config=configs/inaturalist2021mini.py --config.dataset_root=/home/jack/data/manual/inaturalist2021/ --config.train_subtree=inat21_partition_{} --config.keep_examples=False --config.train_with_leaf_targets=true --config.predict=hier_softmax --loader_num_workers=8 --save_freq=5


( PARTITION=d7_n2_i0_c ; EXP_NAME=$PARTITION ; ../env/bin/python main.py --experiment_dir=${EXP_ROOT}/${EXP_NAME} --tensorboard_dir=${EXP_ROOT}/tensorboard/${EXP_NAME} --config=configs/inaturalist2021mini.py --config.dataset_root=/home/jack/data/manual/inaturalist2021/ --config.train_subtree=inat21_partition_$PARTITION --config.keep_examples=False --config.train_with_leaf_targets=true --loader_num_workers=8 --save_freq=5 )
