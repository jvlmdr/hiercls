from absl import app
from absl import flags
from absl import logging
import ml_collections
from ml_collections import config_flags
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import transforms
import tqdm

import models.kuangliu_cifar.resnet
import models.moit.preact_resnet

CIFAR10_ROOT = '/media/jack/data1/data/torchvision/cifar10'
NUM_CLASSES = 10
EVAL_BATCH_SIZE = 100

default_config = ml_collections.ConfigDict({
    'train': ml_collections.ConfigDict({
        'batch_size': 64,
        'num_epochs': 100,
        'learning_rate': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0,
    }),
})
config_flags.DEFINE_config_dict('config', default_config)

FLAGS = flags.FLAGS


def main(_):
    train(FLAGS.config)


def train(config):
    device = torch.device('cuda')

    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        CIFAR10_ROOT,
        train=True,
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True)

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        CIFAR10_ROOT,
        train=False,
        transform=eval_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False)

    # net = torchvision.models.resnet18(pretrained=False, num_classes=NUM_CLASSES)
    net = models.kuangliu_cifar.resnet.ResNet18(NUM_CLASSES)
    # net = models.moit.preact_resnet.PreActResNet18(NUM_CLASSES, mode='')
    net.to(device)

    loss_fn = nn.CrossEntropyLoss()
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
            total_correct = 0
            example_count = 0
            for i, data in enumerate(tqdm.tqdm(test_loader)):
                inputs, labels = map(lambda x: x.to(device), data)
                outputs = net(inputs)
                preds = torch.argmax(outputs, axis=-1)
                is_correct = (preds == labels)
                total_correct += is_correct.sum().item()
                example_count += is_correct.numel()
            mean_correct = total_correct / example_count
            logging.info('test acc: %0.4f', mean_correct)

        scheduler.step()


if __name__ == '__main__':
    app.run(main)
