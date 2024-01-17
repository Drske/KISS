from torchvision.datasets.vision import VisionDataset

class Sampler:
    indices : list
    
    def __init__(self, dataset: VisionDataset, ratio: float=1.0):
        self.dataset_ = dataset
        self.ratio = ratio

    def __getitem__(self, index):
        original_index = self.indices[index]
        return self.dataset_[original_index]

    def __len__(self):
        return len(self.indices)
