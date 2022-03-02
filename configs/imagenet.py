import ml_collections


def get_config():
    return ml_collections.ConfigDict({
        'dataset': 'imagenet',
        'dataset_root': '/data/torchvision/imagenet/',
        'model': 'torch_resnet18',
        'train_split': 'train',
        'eval_split': 'val',
        'hierarchy': 'imagenet_fiveai',
        'train_subset': '',
        'predict': 'flat_softmax',
        'train_transform': 'rand_resizedcrop224_hflip',
        'eval_transform': 'resize256_crop224',

        # Config for training algorithm.
        'train': ml_collections.ConfigDict({
            'batch_size': 256,
            'num_epochs': 100,
            'warmup_epochs': 0,
            'learning_rate': 0.1,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'hxe_alpha': 0.0,
        }),
    })
