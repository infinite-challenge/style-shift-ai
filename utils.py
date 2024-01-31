import collections.abc as container_abcs
from itertools import repeat
import torch
from collections import OrderedDict

import torchvision.transforms as transforms

def load_dict(module, file_path : str):
    new_file_dict = OrderedDict()
    file_dict = torch.load(file_path)
    for key, value in file_dict.items():
        new_file_dict[key] = value
    return module.load_state_dict(new_file_dict)

# compute features' mean and standard deviation
def compute_mean_std(feature : list, eps : float = 1e-5):
    size = feature.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feature_var = feature.view(N, C, -1).var(dim=2) + eps
    feature_std = feature_var.sqrt().view(N,C,1,1)
    feature_mean = feature.view(N, C, -1).mean(dim=2).view(N,C,1,1)
    return feature_mean, feature_std

# normalized feature by using mean std
def normalize(feature : list, eps : float = 1e-5):
    feature_mean, feature_std = compute_mean_std(feature, eps)
    normalized = (feature-feature_mean)/feature_std
    return normalized

def data_transform(size : tuple, crop : int = 0):
    transform_list = []

    transform_list.append(transforms.Resize(size))

    if crop != 0:
        transform_list.append(transforms.CenterCrop(crop))

    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform(size : tuple):
    return transforms.Compose([
        transforms.Resize(size, antialias=True),
    ])


def _ntuple(n : int):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)