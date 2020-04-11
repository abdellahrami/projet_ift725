# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License: ...
Other: Suggestions are welcome
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from typing import Union


class DataManager(object):
    """
    class that yields dataloaders for train, test, and validation data
    """

    def __init__(self, train_dataset: torch.utils.data.Dataset,
                 test_dataset: torch.utils.data.Dataset = None,
                 batch_size: int = 1,
                 num_classes: int = None,
                 input_shape: tuple = None,
                 validation: Union[Dataset, float] = 0.1,
                 seed: int = 0,
                 train_index_list = None,
                 **kwargs):
        """
        Args:
            train_dataset: pytorch dataset used for training
            test_dataset: pytorch dataset used for testing
            batch_size: int, size of batches
            num_classes: int number of classes
            input_shape: tuple, shape of the input image
            validation: float, proportion of the train dataset used for the validation set
            seed: int, random seed for splitting train and validation set
            **kwargs: dict with keywords to be used
        """

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.train_set = train_dataset
        self.test_set = test_dataset
    
        if type(validation) is float:
            train_sampler, val_sampler = self.train_validation_split(len(train_dataset), validation, seed,train_index_list)
            self.train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler, **kwargs)
            self.validation_loader = DataLoader(train_dataset, batch_size, sampler=val_sampler, **kwargs)
            self.test_loader = DataLoader(test_dataset, batch_size, shuffle=True, **kwargs)
        else:
            if train_index_list is not None :
                train_idx = torch.tensor(train_index_list).long()
                train_idx = train_idx[torch.randperm(len(train_idx))]
                train_sampler = SubsetRandomSampler(train_idx)
                self.train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler, **kwargs)
            else:
                self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True, **kwargs)
            self.validation_loader = DataLoader(validation, batch_size, shuffle=True, **kwargs)
            self.test_loader = DataLoader(test_dataset, batch_size, shuffle=True, **kwargs)

    @staticmethod
    def train_validation_split(num_samples, validation_ratio, seed=0, train_index_list=None):
        """
        Returns two torch Samplers, one for training and the other for validation.
        Both samplers are used with the training dataset (see __init__).

        Args:
            num_samples: number of samples to split between train and val set
            validation_ratio: proportion of the train dataset used for validation set
            seed: random seed for splitting train and validation set
        Returns:
            train and validation samplers
        """
        torch.manual_seed(seed)
        if train_index_list is not None :
            num_val = int(len(train_index_list) * validation_ratio)
            shuffled_idx = torch.tensor(train_index_list).long()
            shuffled_idx = shuffled_idx[torch.randperm(len(shuffled_idx))]
        else :
            num_val = int(num_samples * validation_ratio)
            shuffled_idx = torch.randperm(num_samples).long()
        train_idx = shuffled_idx[num_val:]
        val_idx = shuffled_idx[:num_val]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        return train_sampler, val_sampler

    def get_train_set(self):
        return self.train_loader

    def get_validation_set(self):
        return self.validation_loader

    def get_test_set(self):
        return self.test_loader

    def get_classes(self):
        return range(self.num_classes)

    def get_input_shape(self):
        return self.input_shape

    def get_batch_size(self):
        return self.batch_size

    def get_random_sample_from_test_set(self):
        indice = np.random.randint(0, len(self.test_set))
        return self.test_set[indice]
