import numpy as np
import random
from collections import defaultdict
from sklearn.cluster import OPTICS

from .base import Sampler

class OpticsSampler(Sampler):
    def __init__(self, dataset, ratio=1.0, eqsize=True, **kwargs):
        super().__init__(dataset, ratio)
        self.eqsize = eqsize
                
        self.class_data_ = self._get_class_data()
        self.cluster_data_ = self._get_cluster_data(self.class_data_)
        
    def calculate_indices(self, ratio: float = None):
        ratio = ratio or self.ratio_
        
        num_samples = int(len(self.dataset_) * ratio)
        mean_num_clustes = np.mean([len(set(clusters)) for _, clusters in self.cluster_data_.items()])
        print("mean_num_clusters", mean_num_clustes)
        num_samples_per_cluster = max(2, int(len(self.dataset_) * ratio / len(self.class_data_) / mean_num_clustes))

        selected_indices = {}
        for (label, indices), (_, clusters) in zip(self.class_data_.items(), self.cluster_data_.items()):
            selected_indices[label] = {}
            
            combined = list(zip(indices, clusters))
            random.shuffle(combined)
            indices, clusters = zip(*combined)

            for indice, cluster in zip(indices, clusters):
                if cluster not in selected_indices[label]:
                    selected_indices[label][cluster] = []
                    
                if len(selected_indices[label][cluster]) == num_samples_per_cluster:
                    continue
                
                selected_indices[label][cluster].append(indice)
                
        self.indices = []
        
        for label, clusters in selected_indices.items():
            for cluster, indices in clusters.items():
                for idx in indices:
                    self.indices.append(idx)
        
        if len(self.indices) < num_samples and self.eqsize:
            selected = set(self.indices)
            remained = set(range(len(self.dataset_)))
            available = remained - selected
            additional = random.sample(list(available), num_samples - len(self.indices))
            self.indices += list(additional)
        
        
    def _get_class_data(self):
        class_data = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset_):
            class_data[label].append(idx)
            
        return class_data
    
            
    def _get_cluster_data(self, class_data):
        cluster_data = defaultdict(list)
        
        for label, indices in class_data.items():
            print("Predicting next optics")
            data = [self.dataset_[i][0].numpy().flatten() for i in indices]
            optics = OPTICS(min_samples=3, n_jobs=-1)
            labels = optics.fit_predict(data)
            cluster_data[label] = labels
            print("labels", len(set(labels)))
            
        return cluster_data