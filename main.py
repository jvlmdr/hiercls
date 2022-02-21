from functools import partial
import pathlib
import pprint

from absl import app
from absl import flags
from absl import logging
import ml_collections
from ml_collections import config_flags
import numpy as np
import torch
from torch import _pin_memory, nn
from torch import optim
import torchvision
from torchvision import transforms
import tqdm

import datasets
import hier
import hier_torch
import metrics
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
        loss_fn = nn.CrossEntropyLoss()
        pred_fn = partial(
            lambda sum_fn, theta: sum_fn(theta.softmax(dim=-1), dim=-1),
            hier_torch.SumLeafDescendants(tree, strict=False).to(device))
    elif config.predict == 'hier_softmax':
        num_outputs = tree.num_nodes() - 1
        # loss_fn = partial(hier_torch.hier_softmax_nll_with_leaf, tree)
        loss_fn = hier_torch.HierSoftmaxNLL(tree, with_leaf_targets=True).to(device)
        pred_fn = partial(
            lambda log_softmax_fn, theta: log_softmax_fn(theta).exp(),
            # partial(hier_torch.hier_log_softmax, tree)
            hier_torch.HierLogSoftmax(tree).to(device))
    if config.predict == 'multilabel':
        num_outputs = tree.num_nodes() - 1
        loss_fn = hier_torch.MultiLabelNLL(tree, with_leaf_targets=True).to(device)
        pred_fn = partial(hier_torch.multilabel_likelihood, tree)
    else:
        raise ValueError('unknown predict method', config.predict)

    net = torchvision.models.resnet18(pretrained=False, num_classes=num_outputs)
    # net = models.kuangliu_cifar.resnet.ResNet18(num_outputs)
    # net = models.moit.preact_resnet.PreActResNet18(num_outputs, mode='')
    net.to(device)

    optimizer = optim.SGD(
        net.parameters(),
        lr=config.train.learning_rate,
        momentum=config.train.momentum,
        weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.num_epochs)

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
            totals = {
                method: {
                    'exact': 0,
                    'correct': 0,
                    'info_excess': 0,
                    'info_deficient': 0,
                    'depth_excess': 0,
                    'depth_deficient': 0,
                    'edge_dist': 0,
                } for method in ['leaf', 'threshold']
            }
            example_count = 0
            leaf_nodes = tree.leaf_subset()
            for i, data in enumerate(tqdm.tqdm(eval_loader)):
                # inputs, labels = map(lambda x: x.to(device), data)
                inputs, labels = data
                inputs = inputs.to(device)
                theta = net(inputs)
                prob = pred_fn(theta).cpu().numpy()
                preds = {}
                preds['leaf'] = argmax_leaf(tree, prob)
                preds['threshold'] = argmax_value_with_threshold(tree, tree.depths(), prob)
                gt = leaf_nodes[labels]
                for method, pr in preds.items():
                    totals[method]['exact'] += (pr == gt).sum()
                    totals[method]['correct'] += metrics.correct(tree, gt, pr).sum()
                    totals[method]['info_excess'] += metrics.excess_info(tree, gt, pr).sum()
                    totals[method]['info_deficient'] += metrics.deficient_info(tree, gt, pr).sum()
                    totals[method]['depth_excess'] += metrics.excess_depth(tree, gt, pr).sum()
                    totals[method]['depth_deficient'] += metrics.deficient_depth(tree, gt, pr).sum()
                    totals[method]['edge_dist'] += metrics.edge_dist(tree, gt, pr).sum()
                example_count += len(labels)
            means = {method: {field: totals[method][field] / example_count
                              for field in totals[method]} for method in preds}
            pprint.pprint(means)
            # mean_exact = total_exact / example_count
            # logging.info('test acc: %0.4f', mean_exact)

        scheduler.step()


def argmax_leaf(tree: hier.Hierarchy, prob: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.nanargmax(np.where(tree.leaf_mask(), prob, np.nan), axis=axis)


def argmax_value_with_threshold(
        tree: hier.Hierarchy,
        value: np.ndarray,
        prob: np.ndarray,
        threshold: float = 0.5,
        axis: int = -1) -> np.ndarray:
    assert axis == -1
    return np.nanargmax(np.where(prob > threshold, value, np.nan), axis=-1)


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
