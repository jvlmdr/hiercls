import ml_collections


def get_config():
    return ml_collections.ConfigDict({
        'dataset': 'tiny_imagenet',
        'dataset_root': '/data/manual/tiny_imagenet/',
        # Is the label set equal to the leaf nodes?
        # This makes some losses (e.g. flat softmax) simpler.
        # If not, the labels are assumed to be the node index
        # unless a label order is specified.
        'leaf_labels': False,
        'label_order': '',
        'model': 'torch_resnet18',
        'train_split': 'train',
        'train_labels': '',  # Override labels in dataset.
        'train_subset': '',  # Subset of tree to use during training.
        'eval_split': 'val',
        'hierarchy': 'tiny_imagenet_fiveai',
        'predict': 'flat_softmax',
        'train_transform': 'rand_crop56_hflip',
        'eval_transform': 'crop56',

        # Config for training algorithm.
        'train': ml_collections.ConfigDict({
            'batch_size': 256,
            'num_epochs': 100,
            'warmup_epochs': 0,
            'learning_rate': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'label_smoothing': 0.0,
            'hxe_alpha': 0.0,
        }),
    })
