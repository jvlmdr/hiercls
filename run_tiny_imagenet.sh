EXP_ROOT=../experiments/2022-02-28-tiny-imagenet/

EXP_NAME=flat ; python main.py --experiment_dir=$EXP_ROOT/$EXP_NAME/ --tensorboard_dir=$EXP_ROOT/tensorboard/$EXP_NAME --config=configs/tiny_imagenet.py --config.dataset_root=$HOME/data/manual/tiny_imagenet/ --config.predict=flat_softmax --noskip_initial_eval

EXP_NAME=hier ; python main.py --experiment_dir=$EXP_ROOT/$EXP_NAME/ --tensorboard_dir=$EXP_ROOT/tensorboard/$EXP_NAME --config=configs/tiny_imagenet.py --config.dataset_root=$HOME/data/manual/tiny_imagenet/ --config.predict=hier_softmax --noskip_initial_eval

ALPHA=0.1 ; EXP_NAME=hxe-$ALPHA ; python main.py --experiment_dir=$EXP_ROOT/$EXP_NAME/ --tensorboard_dir=$EXP_ROOT/tensorboard/$EXP_NAME --config=configs/tiny_imagenet.py --config.dataset_root=$HOME/data/manual/tiny_imagenet/ --config.predict=bertinetto_hxe --config.train.hxe_alpha=$ALPHA --noskip_initial_eval

ALPHA=0.2 ; EXP_NAME=hxe-$ALPHA ; python main.py --experiment_dir=$EXP_ROOT/$EXP_NAME/ --tensorboard_dir=$EXP_ROOT/tensorboard/$EXP_NAME --config=configs/tiny_imagenet.py --config.dataset_root=$HOME/data/manual/tiny_imagenet/ --config.predict=bertinetto_hxe --config.train.hxe_alpha=$ALPHA --noskip_initial_eval

ALPHA=0.5 ; EXP_NAME=hxe-$ALPHA ; python main.py --experiment_dir=$EXP_ROOT/$EXP_NAME/ --tensorboard_dir=$EXP_ROOT/tensorboard/$EXP_NAME --config=configs/tiny_imagenet.py --config.dataset_root=$HOME/data/manual/tiny_imagenet/ --config.predict=bertinetto_hxe --config.train.hxe_alpha=$ALPHA --noskip_initial_eval
