from functools import partial
import json
import pathlib
import pickle
from typing import Optional

from absl import app
from absl import flags
from absl import logging
from jax import tree_util
import ml_collections
from ml_collections import config_flags
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import tensorboard
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
SAVE_FREQ_EPOCHS = 10
EVAL_FREQ_EPOCHS = 1
TENSBOARD_FLUSH_SECS = 5

PREDICT_METHODS = ['leaf', 'majority']

config_flags.DEFINE_config_file('config')
flags.DEFINE_string('experiment_dir', None,
                    'Where to write experiment data.')

FLAGS = flags.FLAGS


def main(_):
    config = FLAGS.config
    experiment_dir = FLAGS.experiment_dir
    if experiment_dir is not None:
        experiment_dir = pathlib.Path(FLAGS.experiment_dir)
        logging.info('write experiment data to: %s', str(experiment_dir))
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config_str = json.dumps(config.to_dict())
        with open(experiment_dir / 'config.json', 'w') as f:
            f.write(config_str)
    else:
        logging.warning('no experiment_dir; not saving experiment data')

    train(config, experiment_dir)


def train(config, experiment_dir: Optional[pathlib.Path]):
    device = torch.device('cuda')
    num_classes = get_num_classes(config.dataset)

    if experiment_dir is not None:
        checkpoint_dir = experiment_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if experiment_dir is not None:
        checkpoint_dir = experiment_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
    elif config.predict == 'multilabel':
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

    # Metric functions map gt, pr -> scalar (arrays).
    # TODO: Could avoid recomputation of LCA for each metric.
    info_metric = metrics.UniformLeafInfoMetric(tree)
    depth_metric = metrics.DepthMetric(tree)
    metric_fns = {
        'exact': lambda gt, pr: pr == gt,
        'correct': metrics.IsCorrect(tree),
        'info_excess': info_metric.excess,
        'info_deficient': info_metric.deficient,
        'info_dist': info_metric.dist,
        'info_lca': info_metric.at,
        'info_gt': lambda gt, _: info_metric.value[gt],
        'info_pr': lambda _, pr: info_metric.value[pr],
        'depth_excess': depth_metric.excess,
        'depth_deficient': depth_metric.deficient,
        'depth_dist': depth_metric.dist,
        'depth_lca': depth_metric.at,
        'depth_gt': lambda gt, _: depth_metric.value[gt],
        'depth_pr': lambda _, pr: depth_metric.value[pr],
    }

    writer = tensorboard.SummaryWriter(flush_secs=TENSBOARD_FLUSH_SECS)

    # Loop for one extra epoch to save and evaluate model.
    for epoch in range(config.train.num_epochs + 1):
        epoch_str = f'{epoch:04d}'

        if experiment_dir is not None and epoch % SAVE_FREQ_EPOCHS == 0:
            # Save model and results to filesystem.
            checkpoint_file = experiment_dir / f'checkpoints/epoch-{epoch_str}.pth'
            logging.info('write checkpoint: %s', checkpoint_file)
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), checkpoint_file)

        if epoch % EVAL_FREQ_EPOCHS == 0:
            leaf_nodes = tree.leaf_subset()
            depths = tree.depths()
            find_lca = hier.FindLCA(tree)

            # Counts and totals to obtain means.
            example_count = 0
            totals = {method: {field: 0 for field in metric_fns} for method in PREDICT_METHODS}
            # Per-example predictions to write to filesystem.
            outputs = {
                'gt': [],  # Node in hierarchy.
                'label': [],  # Index in label set (for debug).
                'pred': {method: [] for method in PREDICT_METHODS},
                'max_leaf_prob': [],
                'metric': {method: {field: [] for field in metric_fns} for method in PREDICT_METHODS},
            }
            # Sequence-per-example predictions. Cannot be concatenated due to ragged shape.
            seq_outputs = {
                'pred': [],
                'prob': [],
                'metric': {field: [] for field in metric_fns},
            }
            # Voluminous per-example predictions. To be written to a separate file.
            full_outputs = {'prob': []}

            with torch.inference_mode():
                for i, minibatch in enumerate(tqdm.tqdm(eval_loader, f'eval (epoch {epoch})')):
                    inputs, labels = map(lambda x: x.to(device), minibatch)
                    theta = net(inputs)
                    loss = loss_fn(theta, labels)
                    prob = pred_fn(theta)
                    prob = prob.cpu().numpy()
                    labels = labels.cpu().numpy()
                    batch_len = len(labels)
                    gt = leaf_nodes[labels]
                    pred = {}
                    pred['leaf'] = argmax_leaf(tree, prob)
                    pred['majority'] = argmax_value_with_threshold(tree, depths, prob, threshold=0.5)
                    # Truncate predictions where more specific than ground-truth.
                    # TODO: Need to do for range sequences as well.
                    # pred = {k: hier.truncate_given_lca(gt, pr, find_lca(gt, pr))
                    #         for k, pr in pred.items()}
                    max_leaf_prob = max_leaf(tree, prob)

                    example_count += batch_len
                    outputs['gt'].append(gt)
                    outputs['label'].append(labels)
                    outputs['max_leaf_prob'].append(max_leaf_prob)
                    full_outputs['prob'].append(prob)
                    for method in PREDICT_METHODS:
                        outputs['pred'][method].append(pred[method])
                        for field in metric_fns:
                            metric_fn = metric_fns[field]
                            metric_value = metric_fn(gt, pred[method])
                            outputs['metric'][method][field].append(metric_value)
                            totals[method][field] += metric_value.sum()

                    pred_seqs = [sorted_above_threshold(p, threshold=0.5) for p in prob]
                    # TODO: Could vectorize if necessary.
                    prob_seqs = [prob[i, pred_i] for i, pred_i in enumerate(pred_seqs)]
                    seq_outputs['pred'].extend(pred_seqs)
                    seq_outputs['prob'].extend(prob_seqs)
                    for field in metric_fns:
                        metric_fn = metric_fns[field]
                        # TODO: Could vectorize if necessary.
                        metric_seqs = [metric_fn(gt[i], pred_seqs[i]) for i in range(batch_len)]
                        seq_outputs['metric'][field].extend(metric_seqs)

            means = {method: {field: totals[method][field] / example_count
                              for field in totals[method]} for method in pred}
            writer.add_scalar('loss/eval', loss, epoch)
            for method in pred:
                for field in metric_fns:
                    writer.add_scalar(f'{field}/{method}/eval', means[method][field], epoch)

            # Concatenate minibatches into large array.
            leaf_predicate = lambda x: not isinstance(x, dict)  # Lists are values not containers.
            outputs = tree_util.tree_map(np.concatenate, outputs, is_leaf=leaf_predicate)
            full_outputs = tree_util.tree_map(np.concatenate, full_outputs, is_leaf=leaf_predicate)

            # TODO: Avoid memory consumption when not writing outputs to filesystem.
            if experiment_dir is not None:
                # Write data to filesystem.
                # Scalar outputs per example.
                path = experiment_dir / f'predictions/output-epoch-{epoch_str}.pkl'
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'wb') as f:
                    pickle.dump(outputs, f)
                # Sequence of operating points per example.
                path = experiment_dir / f'predictions/operating-points-epoch-{epoch_str}.pkl'
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'wb') as f:
                    pickle.dump(seq_outputs, f)
                # Write raw predictions to filesystem.
                path = experiment_dir / f'predictions/full-output-epoch-{epoch_str}.pkl'
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'wb') as f:
                    pickle.dump(full_outputs, f)

            # step_scores = np.concatenate([seq[1:] for seq in prob_seqs])
            # print('number of operating points:', len(step_scores))
            # step_order = np.argsort(-step_scores)
            # step_scores = step_scores[step_order]
            # total_init = {}
            # total_deltas = {}
            # for field in metric_fns:
            #     metric_seqs = metric_seqs[field]
            #     metric_seqs = [seq.astype(float) for seq in metric_seqs]
            #     total_init[field] = np.asarray([seq[0] for seq in metric_seqs]).sum()

        if not epoch < config.train.num_epochs:
            break

        # Train one epoch.
        total_loss = 0.
        step_count = 0
        for i, minibatch in enumerate(tqdm.tqdm(train_loader, f'train (epoch {epoch})')):
            inputs, labels = map(lambda x: x.to(device), minibatch)
            optimizer.zero_grad()
            theta = net(inputs)
            loss = loss_fn(theta, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            step_count += 1
        mean_loss = total_loss / step_count
        logging.info('mean train loss:%11.4e', mean_loss)
        writer.add_scalar('loss/train', mean_loss, epoch)

        scheduler.step()


def sorted_above_threshold(p: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    assert p.ndim == 1
    subset, = np.nonzero(p > threshold)
    order = np.argsort(-p[subset])
    return subset[order]


def argmax_leaf(tree: hier.Hierarchy, prob: np.ndarray, axis: int = -1) -> np.ndarray:
    # TODO: Avoid excessive calls to leaf_mask()?
    return np.nanargmax(np.where(tree.leaf_mask(), prob, np.nan), axis=axis)


def max_leaf(tree: hier.Hierarchy, prob: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.nanmax(np.where(tree.leaf_mask(), prob, np.nan), axis=axis)


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
