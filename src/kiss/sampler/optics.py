import random
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import OPTICS

from ._cluster import ClusterSampler


class OpticsSampler(ClusterSampler):
    def __init__(self, dataset, ratio=1.0, eqsize=True, **kwargs):
        super().__init__(dataset, ratio, eqsize, **kwargs)
            
    def _select_indices(self):        
        selected_indices = {}
        mean_num_clustes = np.mean([len(set(clusters)) for _, clusters in self.cluster_data_.items()])
        num_samples_per_cluster = max(2, int(len(self.dataset_) * self.ratio_ / len(self.class_data_) / mean_num_clustes))
        
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
                
        return selected_indices
    
    def _get_cluster_data(self, class_data):
        cluster_data = defaultdict(list)
        
        for label, indices in tqdm(class_data.items(), desc="Clustering"):
            data = [self._extract_features(self.dataset_[i][0]) for i in indices]
            optics = OPTICS(min_samples=3, n_jobs=-1)
            labels = optics.fit_predict(data)
            cluster_data[label] = labels
            
        return cluster_data