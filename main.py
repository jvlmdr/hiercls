from absl import app
from absl import flags
import ml_collections
import torch
from torch import nn
import torchvision
from torchvision import transforms

import models.kuangliu_cifar.resnet
import models.moit.preact_resnet

CIFAR10_ROOT = '/media/jack/data1/data/torchvision/cifar10'
NUM_CLASSES = 10
EVAL_BATCH_SIZE = 100

train_config = ml_collections.ConfigDict({
    'batch_size': 64,
})


def main(_):
    device = torch.device('cuda')
    train()


def train():
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
        batch_size=train_config.batch_size,
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

    model = torchvision.models.resnet18(pretrained=False)
    model = models.kuangliu_cifar.resnet.ResNet18(NUM_CLASSES)
    model = models.moit.preact_resnet.PreActResNet18(
        NUM_CLASSES, mode='', head=None, low_dim=None)


if __name__ == '__main__':
    app.run(main)
