import ml_collections


def get_config():
    return ml_collections.ConfigDict({
        'dataset': 'imagenet',
        'dataset_root': '/data/torchvision/imagenet/',
        'train_split': 'train',
        'eval_split': 'val',
        'model': 'torch_resnet18',
        'train_transform': 'rand_resizedcrop224_hflip',
        'eval_transform': 'resize256_crop224',
        'hierarchy': 'imagenet_fiveai',
        'predict': 'flat_softmax',

        # Config for training algorithm.
        'train': ml_collections.ConfigDict({
            'batch_size': 64,
            'num_epochs': 100,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 5e-4,
        }),
    })
