import numpy as np
from torch.utils import data

def InfiniteSampler(n):
    # i = 0

    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

def FiniteSampler(n):
    order = np.random.permutation(n)
    for i in range(n):
        yield order[i]

    # if num_samples is 10, then length of the sampler is 10


class InfiniteSampleWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31