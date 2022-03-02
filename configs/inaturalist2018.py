import ml_collections


def get_config():
    return ml_collections.ConfigDict({
        'dataset': 'inaturalist2018',
        'dataset_root': '/data/manual/inaturalist2018',
        'model': 'torch_resnet18_pretrain',
        'train_split': 'train',
        'eval_split': 'val',
        'hierarchy': 'inat18',
        'train_subset': '',
        'predict': 'flat_softmax',
        'train_transform': 'rand_resizedcrop224_hflip',
        'eval_transform': 'resize256_crop224',

        # Config for training algorithm.
        'train': ml_collections.ConfigDict({
            'batch_size': 256,
            'num_epochs': 50,
            'warmup_epochs': 0,
            'learning_rate': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'label_smoothing': 0.0,
            'hxe_alpha': 0.0,
        }),
    })
