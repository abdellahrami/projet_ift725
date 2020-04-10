#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License: Opensource, free to use
Other: Suggestions are welcome
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from CNNTrainTestManager import CNNTrainTestManager, optimizer_setup
from models.AlexNet import AlexNet
from models.CNNVanilla import CnnVanilla
from models.ResNet import ResNet
from torchvision import datasets
from random import sample
import matplotlib.pyplot as plt
import os
from meth_naive import Naive_meth
from meth_aleat import Aleat_meth


def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [hyper_parameters]'
                                           '\n python3 train.py --model UNet [hyper_parameters]'
                                           '\n python3 train.py --model UNet --predict',
                                     description="This program allows to train different models of classification on different datasets.")
    parser.add_argument('--model', type=str, default="CnnVanilla",
                        choices=["CnnVanilla", "VggNet", "AlexNet", "ResNet"])
    parser.add_argument('--dataset', type=str,
                        default="cifar10", choices=["cifar10", "svhn", "Fashion-MNIST"])
    parser.add_argument('--batch_size', type=int, default=20,
                        help='The size of the training batch')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD"],
                        help="The optimizer to use for training the model")
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='The number of epochs')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--naive_meth', type=str, default="entropy", choices=["entropy", "diff", "max"],
                        help="choosing the type of the naive method")
    return parser.parse_args()


if __name__ == "__main__":

    args = argument_parser()
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    val_set = args.validation
    learning_rate = args.lr

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if args.dataset == 'cifar10':
        # Download the train and test set and apply transform on it
        train_set = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=base_transform)
        test_set = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=base_transform)
        in_channels = 3
    elif args.dataset == 'svhn':
        # Download the train and test set and apply transform on it
        train_set = datasets.SVHN(
            root='./data', split='train', download=True, transform=base_transform)
        test_set = datasets.SVHN(
            root='./data', split='test', download=True, transform=base_transform)
        in_channels = 3
    elif args.dataset == 'Fashion-MNIST':
        # Download the train and test set and apply transform on it
        train_set = datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transforms.ToTensor())
        in_channels = 1

    if args.optimizer == 'SGD':
        optimizer_factory = optimizer_setup(
            torch.optim.SGD, lr=learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)

    # The random choice method
    aleat_met = Aleat_meth(train_set=train_set,
                            test_set=test_set,
                            model_name=args.model,
                            optimizer_factory=optimizer_factory,
                            num_epochs=num_epochs,
                            val_set=val_set,
                            in_channels=in_channels,
                            batch_size=batch_size)

    aleat_met.run()
    aleat_acc = aleat_met.get_accuracies()
    print(aleat_acc)

    # The naive method
    naive_meth = Naive_meth(train_set=train_set,
                            test_set=test_set,
                            model_name=args.model,
                            optimizer_factory=optimizer_factory,
                            num_epochs=num_epochs,
                            val_set=val_set,
                            in_channels=in_channels,
                            batch_size=batch_size,
                            naive_meth=args.naive_meth)

    naive_meth.run()
    naive_acc = naive_meth.get_accuracies()
    print(naive_acc)




    data_sizes = list(range(10,101,10))

    f = plt.figure(figsize=(10, 5))
    ax = f.add_subplot()

    ax.plot(data_sizes, aleat_acc, '-o', label='random method')
    ax.plot(data_sizes, naive_acc, '-o', label='naive method')
    # ax.plot(data_sizes, naive_acc, '-o', label='naive method')
    # ax.plot(data_sizes, naive_acc, '-o', label='naive method')
    ax.set_title('Accuracy on test set')
    ax.set_xlabel('Data size')
    ax.set_ylabel('Accuracy')
    ax.legend()
    f.savefig('plot_{}_{}.png'.format(args.model, args.dataset))
    plt.show()
