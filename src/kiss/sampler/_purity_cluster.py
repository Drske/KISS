import os
import cv2
import random
import pickle
import scipy.stats
import numpy as np

from skimage import feature
from abc import abstractmethod
from torchvision.datasets.vision import VisionDataset
from collections import defaultdict

from ._sampler import Sampler


class PurityClusterSampler(Sampler):
    def __init__(self, dataset: VisionDataset, ratio: float = 1.0, eqsize=True, min_purity: float = 0.5, load_clusters: str = None, save_clusters: str = None, **kwargs):
        super().__init__(dataset, ratio)
        self.eqsize = eqsize
        self.min_purity_ = min_purity
        
        if load_clusters is not None:
            with open(os.path.join(load_clusters, "class_data.pickle"), 'rb') as file:
                self.class_data_ = pickle.load(file)
                
            with open(os.path.join(load_clusters, "cluster_data.pickle"), 'rb') as file:
                self.cluster_data_ = pickle.load(file)
                
            with open(os.path.join(load_clusters, "purity_data.pickle"), 'rb') as file:
                self.purity_data_ = pickle.load(file)
        else:
            self.class_data_ = self._get_class_data()
            self.cluster_data_ = self._get_cluster_data(self.class_data_)
            self.purity_data_ = self._get_purity_data(self.cluster_data_)
        
        if save_clusters is not None:
            if not os.path.exists(save_clusters):
                os.makedirs(save_clusters)
                
            with open(os.path.join(save_clusters, "class_data.pickle"), 'wb+') as file:
                pickle.dump(self.class_data_, file)
                
            with open(os.path.join(save_clusters, "cluster_data.pickle"), 'wb+') as file:
                pickle.dump(self.cluster_data_, file)
                
            with open(os.path.join(save_clusters, "purity_data.pickle"), 'wb+') as file:
                pickle.dump(self.purity_data_, file)

    def calculate_indices(self, ratio: float = None):
        self.ratio_ = ratio or self.ratio_
        
        selected_indices = self._select_indices()
                        
        self.indices = []
        
        for _, indices in selected_indices.items():
            for idx in indices:
                self.indices.append(idx)
                    
        num_samples = int(len(self.dataset_) * ratio)
        print(num_samples, len(self.indices))
        if len(self.indices) < num_samples and self.eqsize:
            selected = set(self.indices)
            remained = set(range(len(self.dataset_)))
            available = remained - selected
            additional = random.sample(list(available), num_samples - len(self.indices))
            self.indices += list(additional)
    
    @abstractmethod
    def _select_indices(self):
        pass
    
    def _get_class_data(self):
        class_data = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset_):
            class_data[label].append(idx)
            
        return class_data
    
    @abstractmethod
    def _get_cluster_data(self, class_data):
        pass
    
    def _get_purity_data(self, cluster_data):
        cluster_sizes = defaultdict(lambda: 0)
        purity_data = {}
        
        for label, clusters in cluster_data.items():
            purity_data[label] = {}
            
            for cluster in clusters:
                if cluster not in purity_data[label]: purity_data[label][cluster] = 0
                purity_data[label][cluster] += 1
                cluster_sizes[cluster] += 1
                
        for label, clusters in purity_data.items():
            for cluster in clusters:
                purity_data[label][cluster] /= cluster_sizes[cluster]
                        
        return purity_data