import ml_collections


def get_config():
    return ml_collections.ConfigDict({
        'dataset': 'tiny_imagenet',
        'dataset_root': '/data/manual/tiny_imagenet/',
        'model': 'torch_resnet18',
        'train_split': 'train',
        'eval_split': 'val',
        'hierarchy': 'tiny_imagenet_fiveai',
        'predict': 'flat_softmax',
        'train_transform': 'rand_crop56_hflip',
        'eval_transform': 'crop56',

        # Config for training algorithm.
        'train': ml_collections.ConfigDict({
            'batch_size': 64,
            'num_epochs': 100,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 5e-4,
        }),
    })
