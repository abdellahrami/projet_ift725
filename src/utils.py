# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import os
import tarfile
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn.functional as F
from skimage.util import crop
from torch import Tensor

CIFAR_SIZE = 170498071  # the expected bytes of cifar datasets

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_DATA_PATH = "data/cifar-10-batches-py"


def num_flat_features(x):
    """
    This function allows to flat features of a tensor before feeding it into FC layer
    @input : tensor x
    @output: int num_features
    """
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def make_dir(dir_name):
    """
    Create a directory safely
    Args:
        directory name
    """
    current_dir = os.getcwd()
    try:
        os.mkdir(os.path.join(current_dir, dir_name))
    except OSError:
        print("Failed to create {}".format(dir_name))


def one_hot(vec, vals=10):
    """
    Transforms a vector in one_hot
    Args:
        vec: numpy vector
        vals: int number of class, default 10 (cifar10)
    """
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


def download(download_url, local_destination, expected_bytes=None):
    """
    Download a file from download_url into local_destination if
    the file doesn't already exists.
    if expected_bytes is provided check if the downloaded file has the same
    number of bytes.
    """
    if os.path.exists(local_destination):
        print("{} already exists".format(local_destination))
    else:
        print("Downloading {}...".format(download_url))
        local_file, headers = urlretrieve(download_url, local_destination)
        file_stat = os.stat(local_destination)
        if expected_bytes:
            if file_stat.st_size == expected_bytes:
                print("Successfully downloaded {}".format(local_destination))


def download_and_extract_cifar10():
    """
    Download and extract cifar10 dataset. You can either use this function or
    use the API provided by pytorch.
    """
    cifar_dir = os.path.join(os.getcwd(), "data")
    make_dir("data")
    local_file = os.path.join(cifar_dir, "tmp_file")
    download(CIFAR_URL, local_file, CIFAR_SIZE)
    tf = tarfile.open(local_file)
    tf.extractall(path=cifar_dir)
    os.remove(local_file)
    tf.close()


def centered_padding(image, pad_size, c_val=0):
    """ Pad the image given in parameters to have a size of self.image_size.

    Args:
        image: ndarray (3d or 4d), Numpy array of data to be padded.
        pad_size: list or tuple, Size of the image after padding.
        c_val: int or float, Value used for padding.

    Returns:
        A ndarray (3D or 4D) padded with a size of pad_size.
    """
    im_size = np.array(pad_size)

    if image.ndim == 4:
        to_pad = (im_size - image.shape[1:3]) // 2
        to_pad = np.array(to_pad).astype(np.int)
        to_pad = ((0, 0), (to_pad[0], to_pad[0]), (to_pad[1], to_pad[1]), (0, 0))
    else:
        to_pad = (im_size - image.shape[:2]) // 2
        to_pad = np.array(to_pad).astype(np.int)
        to_pad = ((to_pad[0], to_pad[0]), (to_pad[1], to_pad[1]), (0, 0))

    return np.pad(image, to_pad, mode='constant', constant_values=c_val)


def centered_crop(image, crop_size):
    """ Crop the image given in parameters to have a size of crop_size.

    Args:
        image: ndarray (4D), Numpy array of data to be padded.
        crop_size: list or tuple, Define the new dimension of the image.

    Returns:
        A ndarray (4D) cropped of the size of crop_size.
    """

    if image.ndim == 4:
        to_crop = (np.array(image.shape[1:3]) - crop_size) // 2
        to_crop = np.array(to_crop, dtype=np.int)
        to_crop = ((0, 0), (to_crop[0], to_crop[0]), (to_crop[1], to_crop[1]), (0, 0))
    else:
        to_crop = (np.array(image.shape[:2]) - crop_size) // 2
        to_crop = np.array(to_crop, dtype=np.int)
        to_crop = ((to_crop[0], to_crop[0]), (to_crop[1], to_crop[1]), (0, 0))
    return crop(image, to_crop)


def centered_resize(image, size, c_val=0):
    """ Centered image resize using crop or padding with c_val.

    Args:
        image: ndarray, A 3d or 4d numpy array of the image.
        size: iterable of int, The output size of the input image.
        c_val: int or float, The value used for the padding.

    Returns:
        A numpy array with the needed output size.
    """

    if image.ndim == 4:
        isize = image.shape[1:3]
    else:
        isize = image.shape[:2]

    # Check the first dimension to select if we crop of pad
    if size[0] - isize[0] < 0:
        image = centered_crop(image, [size[0], isize[1]])
    elif size[0] - isize[0] > 0:
        image = centered_padding(image, [size[0], isize[1]], c_val)

    # Check if we crop or pad along the second dim of the image
    if size[1] - isize[1] < 0:
        image = centered_crop(image, size)
    elif size[1] - isize[1] > 0:
        image = centered_padding(image, size, c_val)

    return image


def dice(input: Tensor, target: Tensor, label: Tensor, reduction: str = 'mean') -> Tensor:
    """ Computes the dice score for a specific class.

    Args:
        input: (N, C, H, W), raw, unnormalized scores for each class.
        target: (N, H, W), where each value is 0 <= targets[i] <= C-1.
        label: class for which to compute the dice score.
        reduction: specifies the reduction to apply to the output:
                   ``'none'``: no reduction will be applied,
                   ``'mean'``: the sum of the output will be divided by the number of elements in the output.

    Returns:
        (1,) or (N,), the dice score for the requested class, reduced or for each sample.
    """
    if reduction == 'mean':
        reduce_axis = (0, 1)
    else:  # reduction == 'none'
        reduce_axis = 1

    input = F.softmax(input, dim=1)[:, label, ...]  # For the input, extract the probabilities of the requested label
    target = torch.eq(target, label)  # For the target, extract the boolean mask of the requested label

    # Flatten the tensors to facilitate broadcasting
    input = torch.flatten(input, start_dim=1)
    target = torch.flatten(target, start_dim=1)

    # Compute dice score
    intersect = torch.sum(input * target, 1, keepdim=True)
    sum_input = torch.sum(input, 1, keepdim=True)
    sum_target = torch.sum(target, 1, keepdim=True)
    dice = torch.mean((2 * intersect + 1) / (sum_input + sum_target + 1), dim=reduce_axis)
    return dice


def mean_dice(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """ Computes the mean dice score for all classes present in the target.

    Args:
        input: (N, C, H, W), raw, unnormalized scores for each class.
        target: (N, H, W), where each value is 0 <= targets[i] <= C-1.
        reduction: specifies the reduction to apply to the output:
                   ``'none'``: no reduction will be applied,
                   ``'mean'``: the sum of the output will be divided by the number of elements in the output.

    Returns:
        (1,) or (N,), the mean dice score for the classes in the target, reduced or for each sample.
    """
    labels = torch.unique(target[target.nonzero(as_tuple=True)])  # Identify classes (that are not background)
    # labels = Tensor([1])
    # Compute the dice score for each individual class
    dices = torch.stack([dice(input, target, label, reduction=reduction)
                         for label in labels])

    mean_dice = torch.mean(dices, dim=0)  # Compute the mean dice over all classes
    return mean_dice

