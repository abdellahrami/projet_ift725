
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from CNNTrainTestManager import CNNTrainTestManager, optimizer_setup
from models.AlexNet import AlexNet
from models.CNNVanilla import CnnVanilla
from models.ResNet import ResNet
from models.Encoder import Encoder
from torchvision import datasets
from random import sample
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.cluster import KMeans
import numpy as np


class MyDataset(Dataset):

        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, index):
            data, target = self.dataset[index]

            return data, target, index

        def __len__(self):
            return len(self.dataset)


class Cluster_meth(object):

    def __init__(self, train_set, test_set, model_name, optimizer_factory, num_epochs=1, val_set=0.1, in_channels=3, batch_size=1, naive_meth='entropy'):

        self.train_set = MyDataset(train_set)
        self.test_set = MyDataset(test_set)
        self.model_name = model_name
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.optimizer_factory = optimizer_factory
        self.val_set = val_set
        self.num_epochs = num_epochs
        self.naive_meth = naive_meth

    def run(self):

        print("Training using the naive active learning method with type {}".format(
            self.naive_meth))

        num_train = len(self.train_set)
        indexs = list(range(num_train))
        accuracies = []
        train_index = []

        clustering_model = Encoder(in_channels=self.in_channels, encoder_classes=10, out_channels=self.in_channels, init_weights=True)

        for i in range(10):
            print("training on the number <{}> 10% of the dataSet.".format(i+1))
            if i+1 < 10:
                clustering_indxs = sample(indexs, min(len(indexs),int(num_train*0.5)))
                cluster_trainer = CNNTrainTestManager(model=clustering_model,
                                                trainset=self.train_set,
                                                testset=self.test_set,
                                                batch_size=self.batch_size,
                                                loss_fn=nn.MSELoss(reduction='sum'),
                                                optimizer_factory=self.optimizer_factory,
                                                validation=0.1,
                                                use_cuda=True,
                                                train_index_list=clustering_indxs,
                                                tqdm_disable=True)
                cluster_trainer.train(num_epochs=5)
                outputs,indexs = cluster_trainer.encode(self.train_set,self.test_set,clustering_indxs)
                clf = KMeans(n_clusters=10, random_state=0)
                predict = clf.fit_predict(outputs)
                dist = clf.transform(outputs)
                dist = np.min(dist,axis=1)

                df = pd.DataFrame(zip(indexs,predict,dist),columns=['indx','cluster','dist'])
                df.sort_values(['cluster', 'dist'], ascending=[True, True],inplace=True)
                size_per_class = int(num_train*0.01)
                for i in set(df.cluster):
                    train_index += list(df.indx[df.cluster == i])[:size_per_class]
                indexs = [ l for l in indexs if l not in train_index]

            else:
                train_index += indexs

            if self.model_name == 'CnnVanilla':
                model = CnnVanilla(num_classes=10, in_channels=self.in_channels)
            elif self.model_name == 'AlexNet':
                model = AlexNet(num_classes=10, in_channels=self.in_channels)
            elif self.model_name == 'ResNet':
                model = ResNet(num_classes=10, in_dim=self.in_channels)

            model_trainer = CNNTrainTestManager(model=model,
                                                trainset=self.train_set,
                                                testset=self.test_set,
                                                batch_size=self.batch_size,
                                                loss_fn=nn.CrossEntropyLoss(),
                                                optimizer_factory=self.optimizer_factory,
                                                validation=self.val_set,
                                                use_cuda=True,
                                                train_index_list=train_index)
            
            model_trainer.train(self.num_epochs)
            accuracies.append(model_trainer.evaluate_on_test_set())

        self.accuracies = accuracies
    
    def get_accuracies(self):
        return self.accuracies
