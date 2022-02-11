from absl import app
from absl import logging
import ml_collections
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

train_config = ml_collections.ConfigDict({
    'batch_size': 64,
    'num_epochs': 100,
    'learning_rate': 0.01,
})


def main(_):
    train()


def train():
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

    # model = torchvision.models.resnet18(pretrained=False, num_classes=NUM_CLASSES)
    model = models.kuangliu_cifar.resnet.ResNet18(NUM_CLASSES)
    # model = models.moit.preact_resnet.PreActResNet18(NUM_CLASSES, mode='')
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=train_config.learning_rate,
                          momentum=0.9)

    for epoch in range(train_config.num_epochs):
        total_loss = 0.
        step_count = 0
        for i, data in enumerate(tqdm.tqdm(train_loader)):
            inputs, labels = map(lambda x: x.to(device), data)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            step_count += 1
        mean_loss = total_loss / step_count
        logging.info('mean train loss: %+11.4e', mean_loss)

        with torch.inference_mode():
            total_correct = 0
            example_count = 0
            for i, data in enumerate(tqdm.tqdm(test_loader)):
                inputs, labels = map(lambda x: x.to(device), data)
                outputs = model(inputs)
                preds = torch.argmax(outputs, axis=-1)
                is_correct = (preds == labels)
                total_correct += is_correct.sum().item()
                example_count += is_correct.numel()
            mean_correct = total_correct / example_count
            logging.info('test acc: %0.4f', mean_correct)


if __name__ == '__main__':
    app.run(main)
