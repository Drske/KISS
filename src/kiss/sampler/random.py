from .base import Sampler
import random

class RandomSampler(Sampler):
    def __init__(self, dataset, ratio=1.0):
        super().__init__(dataset, ratio)
        
        num_samples = int(len(dataset) * ratio)
        self.indices = random.sample(range(len(dataset)), num_samples)
