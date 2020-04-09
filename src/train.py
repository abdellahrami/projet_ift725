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
import torchvision.transforms as transforms
from CNNTrainTestManager import CNNTrainTestManager, optimizer_setup
from models.AlexNet import AlexNet
from models.CNNVanilla import CnnVanilla
from models.ResNet import ResNet
from models.VggNet import VggNet
from torchvision import datasets
import os

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
    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "svhn"])
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
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=base_transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=base_transform)
    elif args.dataset == 'svhn':
        # Download the train and test set and apply transform on it             
        train_set = datasets.SVHN(root='./data', split='train', download=True, transform=base_transform)
        test_set = datasets.SVHN(root='./data', split='test', download=True, transform=base_transform)

    if args.optimizer == 'SGD':
        optimizer_factory = optimizer_setup(torch.optim.SGD, lr=learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)

    if args.model == 'CnnVanilla':
        model = CnnVanilla(num_classes=10)
    elif args.model == 'AlexNet':
        model = AlexNet(num_classes=10)
    elif args.model == 'VggNet':
        model = VggNet(num_classes=10)
    elif args.model == 'ResNet':
        model = ResNet(num_classes=10)
    

    model_trainer = CNNTrainTestManager(model=model,
                                        trainset=train_set,
                                        testset=test_set,
                                        batch_size=batch_size,
                                        loss_fn=nn.CrossEntropyLoss(),
                                        optimizer_factory=optimizer_factory,
                                        validation=val_set,
                                        use_cuda=True,
                                        train_index_list = list( range( 0, int( len(train_set)*0.2))))



    print("Training {} on {} for {} epochs".format(model.__class__.__name__, args.dataset, args.num_epochs))
    model_trainer.train(num_epochs)
    model_trainer.evaluate_on_test_set()
    model_trainer.plot_metrics()


