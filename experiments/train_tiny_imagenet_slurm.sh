#!/bin/bash

#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

module load arch/haswell
module load Python/3.8.6
module load CUDA/11.2.0
module load cuDNN/CUDA-11.2

PYTHON=$HOME/projects/2022-01-hierarchical/env/bin/python
SRC=$HOME/projects/2022-01-hierarchical/hier-class

EXP_ROOT=/hpcfs/users/$USER/exp/hier-class/2022-04-05-tiny-imagenet
EXP_NAME=flat

$PYTHON ${SRC}/main.py \
    --experiment_dir=$EXP_ROOT/$EXP_NAME/ \
    --tensorboard_dir=$EXP_ROOT/tensorboard/$EXP_NAME \
    --config=${SRC}/configs/tiny_imagenet.py \
    --config.dataset_root=/hpcfs/groups/phoenix-hpc-acvt/data/tiny_imagenet \
    --config.predict=flat_softmax
