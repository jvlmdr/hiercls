import ml_collections


def get_config():
    return ml_collections.ConfigDict({
        'dataset': 'imagenet',
        'dataset_root': '/data/torchvision/imagenet/',
        'model': 'torch_resnet50',
        'train_split': 'train',
        'eval_split': 'val',
        'hierarchy': 'imagenet_fiveai',
        'predict': 'flat_softmax',
        'train_transform': 'rand_resizedcrop224_hflip',
        'eval_transform': 'resize256_crop224',

        'train_with_leaf_targets': True,
        'train_subtree': '',
        'filter_subtree': '',
        'train_labels': '',

        # Config for training algorithm.
        'train': ml_collections.ConfigDict({
            'batch_size': 256,
            'num_epochs': 50,
            'warmup_epochs': 0,
            'learning_rate': 0.1,
            'momentum': 0.9,
            'weight_decay': 3e-4,
            'label_smoothing': 0.0,
            'margin': 'incorrect',
            'margin_tau': 5.0,
            'hxe_alpha': 0.0,
            'hier_normalize': '',
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'hier_focal_power': 0.0,
            'random_cut_prob': 0.1,
            'loss_weighting': 'none',
        }),
    })
