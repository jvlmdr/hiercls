EXP_ROOT=/mnt/ssd1/projects/2022-01-hierarchical/experiments/2022-04-23-imagenet-features/

( LOSS=flat_softmax ; MODEL=linear ; FEAT=vitb32 ; EXP_NAME=${FEAT}-${MODEL}-${LOSS} ; python main.py --experiment_dir=${EXP_ROOT}/${EXP_NAME} --tensorboard_dir=${EXP_ROOT}/tensorboard/${EXP_NAME} --config=configs/imagenet_linear.py --config.dataset_root=resources/features/imagenet_vitb32/ --config.train_with_leaf_targets=true --loader_num_workers=8 --save_freq=5 --noskip_initial_eval )
