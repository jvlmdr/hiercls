import ml_collections


def get_config():
    return ml_collections.ConfigDict({
        'dataset': 'npz',
        'dataset_root': '/data/torchvision/imagenet_vit32b/',
        'model': 'linear',
        'train_split': 'train',
        'eval_split': 'val',
        'hierarchy': 'imagenet_fiveai',
        'predict': 'flat_softmax',
        'train_transform': 'none',
        'eval_transform': 'none',

        'train_with_leaf_targets': True,
        'train_subtree': '',
        'filter_subtree': '',
        'train_labels': '',

        # Config for training algorithm.
        'train': ml_collections.ConfigDict({
            'batch_size': 256,
            'num_epochs': 100,
            'warmup_epochs': 0,
            'learning_rate': 0.1,
            'momentum': 0.9,
            'weight_decay': 3e-4,
            'label_smoothing': 0.0,
            'hxe_alpha': 0.0,
            'hier_normalize': '',
        }),
    })
