from torchvision.datasets.vision import VisionDataset
from abc import abstractmethod

class Sampler:
    indices : list
    
    def __init__(self, dataset: VisionDataset, ratio: float=1.0):
        self.dataset_ = dataset
        self.ratio_ = ratio
        
    @abstractmethod
    def calculate_indices(self, ratio: float = None):
        pass

    def __getitem__(self, index):
        original_index = self.indices[index]
        return self.dataset_[original_index]

    def __len__(self):
        return len(self.indices)
