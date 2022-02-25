from functools import partial
import json
import pathlib
import pickle
import pprint
from typing import Optional, Tuple

from absl import app
from absl import flags
from absl import logging
from jax import tree_util
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
LOADER_NUM_WORKERS = 16
LOADER_PREFETCH_FACTOR = 2
TENSORBOARD_FLUSH_SECS = 5

PREDICT_METHODS = ['leaf', 'majority']

config_flags.DEFINE_config_file('config')
flags.DEFINE_string('experiment_dir', None,
                    'Where to write experiment data.')
flags.DEFINE_bool('resume', False, 'Resume from previous checkpoint.')
flags.DEFINE_bool('skip_initial_eval', True, 'Skip eval for epoch 0.')

FLAGS = flags.FLAGS


MODEL_FNS = {
    'torch_resnet18': lambda num_outputs: (
        torchvision.models.resnet18(pretrained=False, num_classes=num_outputs)),
    'torch_resnet50': lambda num_outputs: (
        torchvision.models.resnet50(pretrained=False, num_classes=num_outputs)),
    'kuangliu_resnet18': models.kuangliu_cifar.resnet.ResNet18,
    'moit_preact_resnet18': lambda num_outputs: (
        models.moit.preact_resnet.PreActResNet18(num_outputs, mode='')),
}


def make_model(name: str, num_outputs: int) -> nn.Module:
    try:
        model_fn = MODEL_FNS[name]
    except KeyError:
        raise ValueError('unknown model', name)
    return model_fn(num_outputs)


# Functions with signature:
# dataset_fn(root, split, transform=transform)
DATASET_FNS = {
    'imagenet': torchvision.datasets.ImageNet,
    'tiny_imagenet': datasets.TinyImageNet,
    'inaturalist2018': datasets.INaturalist2018,
}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Non-deterministic transforms have 'rand_' prefix.
TRANSFORMS = {
    # CIFAR train
    'rand_pad4_crop32_hflip': transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    # CIFAR eval
    'none': transforms.Compose([
        transforms.ToTensor(),
    ]),

    # ImageNet train
    'rand_resizedcrop224_hflip': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]),
    # ImageNet eval
    'resize256_crop224': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]),

    # Tiny ImageNet train (64px)
    'rand_crop56_hflip': transforms.Compose([
        transforms.RandomCrop(56),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    # Tiny ImageNet eval (64px)
    'crop56': transforms.Compose([
        transforms.CenterCrop(56),
        transforms.ToTensor(),
    ]),
}


def make_dataset(name, root, split, transform_name):
    try:
        dataset_fn = DATASET_FNS[name]
    except KeyError:
        raise ValueError('unknown dataset', name)

    try:
        transform = TRANSFORMS[transform_name]
    except KeyError:
        raise ValueError('unknown transform', transform_name)

    return dataset_fn(root, split, transform=transform)


def main(_):
    config = FLAGS.config
    experiment_dir = FLAGS.experiment_dir

    if not experiment_dir:
        logging.warning('no experiment_dir; not saving experiment data')
    else:
        # Ensure that experiment directory exists.
        # TODO: Check experiment_dir.exists() xor FLAGS.resume.
        experiment_dir = pathlib.Path(FLAGS.experiment_dir).absolute()
        if experiment_dir.exists() and not FLAGS.resume:
            raise ValueError('dir exists but resume is not set', str(experiment_dir))
        logging.info('write experiment data to: %s', str(experiment_dir))
        experiment_dir.mkdir(parents=True, exist_ok=True)
        # Dump config to file for future reference.
        config_str = json.dumps(config.to_dict())
        with open(experiment_dir / 'config.json', 'w') as f:
            f.write(config_str)

    train(config, experiment_dir)


def train(config, experiment_dir: Optional[pathlib.Path]):
    device = torch.device('cuda')

    train_dataset = make_dataset(
        config.dataset,
        root=config.dataset_root,
        split=config.train_split,
        transform_name=config.train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=LOADER_NUM_WORKERS,
        prefetch_factor=LOADER_PREFETCH_FACTOR)

    eval_dataset = make_dataset(
        config.dataset,
        root=config.dataset_root,
        split=config.eval_split,
        transform_name=config.eval_transform)
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=LOADER_NUM_WORKERS,
        prefetch_factor=LOADER_PREFETCH_FACTOR)

    hierarchy_file = SOURCE_DIR / f'resources/hierarchy/{config.hierarchy}.csv'
    with open(hierarchy_file) as f:
        edges = hier.load_edges(f)
    tree, node_names = hier.make_hierarchy_from_edges(edges)

    # TODO: Create a factory for this?
    if config.predict == 'flat_softmax':
        num_outputs = tree.num_leaf_nodes()
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
    elif config.predict == 'bertinetto_hxe':
        num_outputs = tree.num_nodes() - 1
        loss_fn = hier_torch.BertinettoHXE(tree, config.train.hxe_alpha, with_leaf_targets=True).to(device)
        pred_fn = partial(
            lambda log_softmax_fn, theta: log_softmax_fn(theta).exp(),
            hier_torch.HierLogSoftmax(tree).to(device))
    elif config.predict == 'multilabel':
        num_outputs = tree.num_nodes() - 1
        loss_fn = hier_torch.MultiLabelNLL(tree, with_leaf_targets=True).to(device)
        pred_fn = partial(hier_torch.multilabel_likelihood, tree)
    else:
        raise ValueError('unknown predict method', config.predict)

    net = make_model(config.model, num_outputs)
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
        'info_lca': info_metric.value_at_lca,
        'info_gt': info_metric.value_at_gt,
        'info_pr': info_metric.value_at_pr,
        'depth_excess': depth_metric.excess,
        'depth_deficient': depth_metric.deficient,
        'depth_dist': depth_metric.dist,
        'depth_lca': depth_metric.value_at_lca,
        'depth_gt': depth_metric.value_at_gt,
        'depth_pr': depth_metric.value_at_pr,
    }

    # TODO: Decide where to write logs. experiment_dir / 'tensorboard'?
    writer = tensorboard.SummaryWriter(flush_secs=TENSORBOARD_FLUSH_SECS)
    leaf_nodes = tree.leaf_subset()
    depths = tree.depths()

    # Loop for one extra epoch to save and evaluate model.
    for epoch in range(config.train.num_epochs + 1):
        epoch_str = f'{epoch:04d}'

        if experiment_dir is not None and epoch % SAVE_FREQ_EPOCHS == 0:
            # Save model and results to filesystem.
            checkpoint_file = experiment_dir / f'checkpoints/epoch-{epoch_str}.pth'
            logging.info('write checkpoint: %s', checkpoint_file)
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), checkpoint_file)

        if (epoch > 0 and epoch % EVAL_FREQ_EPOCHS == 0) or (epoch == 0 and not FLAGS.skip_initial_eval):
            find_lca = hier.FindLCA(tree)

            # Counts and totals to obtain means.
            example_count = 0
            total_loss = 0.
            metric_totals = {method: {field: 0 for field in metric_fns} for method in PREDICT_METHODS}
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

            net.eval()
            with torch.inference_mode():
                for minibatch in tqdm.tqdm(eval_loader, f'eval (epoch {epoch})'):
                    inputs, labels = map(lambda x: x.to(device), minibatch)
                    theta = net(inputs)
                    loss = loss_fn(theta, labels)
                    prob = pred_fn(theta).cpu().numpy()
                    labels = labels.cpu().numpy()
                    batch_len = len(labels)
                    gt = leaf_nodes[labels]
                    pred = {}
                    pred['leaf'] = argmax_leaf(tree, prob)
                    pred['majority'] = argmax_value_with_threshold(tree, depths, prob, threshold=0.5)
                    # Truncate predictions where more specific than ground-truth.
                    # (Only necessary if some labels are not leaf nodes.)
                    # TODO: Need to do for range sequences as well.
                    # pred = {k: hier.truncate_given_lca(gt, pr, find_lca(gt, pr))
                    #         for k, pr in pred.items()}
                    max_leaf_prob = max_leaf(tree, prob)

                    total_loss += batch_len * loss.item()  # TODO: Avoid assuming mean?
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
                            metric_totals[method][field] += metric_value.sum()

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

            mean_loss = total_loss / example_count
            writer.add_scalar('loss/eval', mean_loss, epoch)
            logging.info('eval loss: %.5g', mean_loss)
            metric_means = {method: {field: metric_totals[method][field] / example_count
                                     for field in metric_totals[method]} for method in pred}
            for method in pred:
                for field in metric_fns:
                    writer.add_scalar(f'{field}/{method}/eval', metric_means[method][field], epoch)
            logging.info(
                'leaf: exact %.3f, correct %.3f, depth_dist %.4g, info_dist %.4g',
                metric_means['leaf']['exact'], metric_means['leaf']['correct'],
                metric_means['leaf']['depth_dist'], metric_means['leaf']['info_dist'])
            logging.info(
                'majority: exact %.3f, correct %.3f, depth_dist %.4g, info_dist %.4g',
                metric_means['majority']['exact'], metric_means['majority']['correct'],
                metric_means['majority']['depth_dist'], metric_means['majority']['info_dist'])

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
        metric_totals = {method: {field: 0 for field in metric_fns}
                         for method in PREDICT_METHODS}
        step_count = 0
        example_count = 0
        net.train()
        for minibatch in tqdm.tqdm(train_loader, f'train (epoch {epoch})'):
            inputs, labels = map(lambda x: x.to(device), minibatch)
            optimizer.zero_grad()
            theta = net(inputs)
            loss = loss_fn(theta, labels)
            loss.backward()
            optimizer.step()
            if loss.isnan():
                raise ValueError('loss is nan')
            if loss.isinf():
                raise ValueError('loss is inf')
            loss = loss.item()
            logging.debug('loss: %g', loss)
            total_loss += loss
            step_count += 1

            # Evaluate metrics for batch predictions and add to totals.
            with torch.no_grad():
                prob = pred_fn(theta)
            prob = prob.cpu().numpy()
            labels = labels.cpu().numpy()
            batch_len = len(labels)
            gt = leaf_nodes[labels]  # Labels are leaf nodes.
            pred = {}
            pred['leaf'] = argmax_leaf(tree, prob)
            pred['majority'] = argmax_value_with_threshold(tree, depths, prob, threshold=0.5)
            for method in PREDICT_METHODS:
                for field in metric_fns:
                    metric_fn = metric_fns[field]
                    metric_totals[method][field] += metric_fn(gt, pred[method]).sum()
            example_count += batch_len

        mean_loss = total_loss / step_count
        writer.add_scalar('loss/train', mean_loss, epoch)
        logging.info('train loss: %.5g', mean_loss)
        metric_means = {method: {field: metric_totals[method][field] / example_count
                                 for field in metric_totals[method]} for method in pred}
        for method in pred:
            for field in metric_fns:
                writer.add_scalar(f'{field}/{method}/train', metric_means[method][field], epoch)
        logging.info(
            'leaf: exact %.3f, correct %.3f, depth_dist %.4g, info_dist %.4g',
            metric_means['leaf']['exact'], metric_means['leaf']['correct'],
            metric_means['leaf']['depth_dist'], metric_means['leaf']['info_dist'])
        logging.info(
            'majority: exact %.3f, correct %.3f, depth_dist %.4g, info_dist %.4g',
            metric_means['majority']['exact'], metric_means['majority']['correct'],
            metric_means['majority']['depth_dist'], metric_means['majority']['info_dist'])

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


if __name__ == '__main__':
    app.run(main)
