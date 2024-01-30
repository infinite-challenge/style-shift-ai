
import collections.abc as container_abcs
from itertools import repeat

def compute_mean_std(feature, eps=1e-6):
    """
    compute features' mean and standard deviation
    """
    size = feature.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feature_var = feature.view(N, C, -1).var(dim=2) + eps
    feature_std = feature_var.sqrt().view(N,C,1,1)
    feature_mean = feature.view(N, C, -1).mean(dim=2).view(N,C,1,1)
    return feature_mean, feature_std

def normalize(feature, eps=1e-5):
    """
    normalized feature by using mean std
    """
    feature_mean, feature_std = compute_mean_std(feature, eps)
    normalized = (feature-feature_mean)/feature_std
    return normalized

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)