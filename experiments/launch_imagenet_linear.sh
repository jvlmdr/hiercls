EXP_ROOT=/mnt/ssd1/projects/2022-01-hierarchical/experiments/2023-01-26-imagenet-features/

( LOSS=flat_softmax ; MODEL=linear ; FEAT=vitb32 ; EXP_NAME=${FEAT}-${MODEL}-${LOSS} ; python main.py --experiment_dir=${EXP_ROOT}/${EXP_NAME} --tensorboard_dir=${EXP_ROOT}/tensorboard/${EXP_NAME} --config=configs/imagenet_linear.py --config.dataset_root=resources/features/imagenet_vitb32/ --config.train_with_leaf_targets=true --loader_num_workers=8 --save_freq=5 --noskip_initial_eval )

for LOSS in flat_softmax hier_softmax multilabel_focal share_flat_softmax 

( LOSS=flat_softmax ; MODEL=linear ; FEAT=vitb32 ; EXP_NAME=${FEAT}-${MODEL}-${LOSS} ; python main.py --experiment_dir=${EXP_ROOT}/${EXP_NAME} --tensorboard_dir=${EXP_ROOT}/tensorboard/${EXP_NAME} --config=configs/imagenet_linear.py --config.dataset_root=resources/features/imagenet_vitb32/ --config.train_with_leaf_targets=true --loader_num_workers=8 --save_freq=5 --noskip_initial_eval )
