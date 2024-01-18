import random

from ._sampler import Sampler


class RandomSampler(Sampler):
    def __init__(self, dataset, ratio=1.0, **kwargs):
        super().__init__(dataset, ratio)

    def calculate_indices(self, ratio=None):
        ratio = ratio or self.ratio_

        num_samples = int(len(self.dataset_) * ratio)
        self.indices = random.sample(range(len(self.dataset_)), num_samples)
