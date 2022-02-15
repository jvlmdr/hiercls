import pathlib

from absl import app
from absl import flags
from absl import logging
import ml_collections
from ml_collections import config_flags
import torch
from torch import _pin_memory, nn
from torch import optim
import torchvision
from torchvision import transforms
import tqdm

import datasets
import hier
import hier_torch
import models.kuangliu_cifar.resnet
import models.moit.preact_resnet

EVAL_BATCH_SIZE = 256
SOURCE_DIR = pathlib.Path(__file__).parent

config_flags.DEFINE_config_file('config')

FLAGS = flags.FLAGS


def main(_):
    train(FLAGS.config)


def train(config):
    device = torch.device('cuda')
    num_classes = get_num_classes(config.dataset)

    train_dataset = make_dataset(
        config.dataset,
        config.dataset_root,
        config.train_split,
        mode='train')
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        pin_memory=True)

    eval_dataset = make_dataset(
        config.dataset,
        config.dataset_root,
        config.eval_split,
        mode='eval')
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=True)

    hierarchy_file = SOURCE_DIR / f'resources/hierarchy/{config.hierarchy}.csv'
    with open(hierarchy_file) as f:
        edges = hier.load_edges(f)
    tree, node_names = hier.make_hierarchy_from_edges(edges)

    if config.predict == 'flat_softmax':
        num_outputs = tree.num_leaf_nodes()  # num_classes
    elif config.predict == 'hier_softmax':
        num_outputs = tree.num_nodes() - 1
    else:
        raise ValueError('unknown predict method', config.predict)

    net = torchvision.models.resnet18(pretrained=False, num_classes=num_outputs)
    # net = models.kuangliu_cifar.resnet.ResNet18(num_outputs)
    # net = models.moit.preact_resnet.PreActResNet18(num_outputs, mode='')
    net.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.train.learning_rate,
        momentum=config.train.momentum,
        weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.num_epochs)

    leaf_subset = torch.from_numpy(tree.leaf_subset()).to(device)

    for epoch in range(config.train.num_epochs):
        total_loss = 0.
        step_count = 0
        for i, data in enumerate(tqdm.tqdm(train_loader)):
            inputs, labels = map(lambda x: x.to(device), data)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            step_count += 1
        mean_loss = total_loss / step_count
        logging.info('mean train loss:%11.4e', mean_loss)

        with torch.inference_mode():
            total_correct = 0
            example_count = 0
            for i, data in enumerate(tqdm.tqdm(eval_loader)):
                inputs, labels = map(lambda x: x.to(device), data)
                theta = net(inputs)
                if config.predict == 'flat_softmax':
                    prob = hier_torch.sum_leaf_descendants(tree, theta.softmax(dim=-1), dim=-1)
                elif config.predict == 'hier_softmax':
                    prob = hier_torch.hier_log_softmax(theta).exp()
                pred_nodes = torch.argmax(prob, axis=-1)
                label_nodes = torch.take(leaf_subset, labels)
                is_correct = (pred_nodes == label_nodes)
                total_correct += is_correct.sum().item()
                example_count += is_correct.numel()
            mean_correct = total_correct / example_count
            logging.info('test acc: %0.4f', mean_correct)

        scheduler.step()


def get_num_classes(dataset) -> int:
    if dataset == 'imagenet':
        return 1000
    if dataset == 'tiny_imagenet':
        return 200
    if dataset == 'cifar10':
        return 10
    raise ValueError('unknown dataset')


DATASET_FNS = {
    'imagenet': torchvision.datasets.ImageNet,
    'tiny_imagenet': datasets.TinyImageNet,
    'cifar10': torchvision.datasets.CIFAR10,
}


def make_dataset(dataset, root, split, mode):
    try:
        dataset_fn = DATASET_FNS[dataset]
    except KeyError:
        raise ValueError('unknown dataset', dataset)

    if dataset == 'imagenet':
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif mode == 'eval':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])

    elif dataset == 'tiny_imagenet':
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomCrop(56),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif mode == 'eval':
            transform = transforms.Compose([
                transforms.CenterCrop(56),
                transforms.ToTensor(),
            ])

    elif 'cifar' in dataset:
        if mode == 'train':
            transform = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif mode == 'eval':
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    else:
        raise ValueError('no transform configured for dataset', dataset)

    return dataset_fn(root, split, transform=transform)


if __name__ == '__main__':
    app.run(main)
