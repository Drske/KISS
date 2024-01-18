import numpy as np
import random
from collections import defaultdict
from sklearn.cluster import KMeans

from .base import Sampler

class KMeansSampler(Sampler):
    def __init__(self, dataset, ratio=1.0):
        super().__init__(dataset, ratio)
                
        class_data = self._get_class_data()
        cluster_data = self._get_cluster_data(class_data)
        
        num_clusters = 20
        num_samples_per_cluster = max(2, int(len(dataset) * ratio / len(class_data) / num_clusters))
        print(f"Number of samples per cluster: {num_samples_per_cluster}")

        selected_indices = {}
        for (label, indices), (_, clusters) in zip(class_data.items(), cluster_data.items()):
            selected_indices[label] = {}
            
            for c in range(num_clusters):
                cs = [x for x in clusters if x == c]
                if len(cs) < num_samples_per_cluster:
                    print(f"Label {label} will be underrepresented by {num_samples_per_cluster - len(cs)} in cluster {c}")
            
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
                    
    def _get_class_data(self):
        class_data = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset_):
            class_data[label].append(idx)
            
        return class_data
            
    def _get_cluster_data(self, class_data):
        num_clusters = 20
        cluster_data = defaultdict(list)
        
        for label, indices in class_data.items():
            data = [self.dataset_[i][0].numpy().flatten() for i in indices]
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            cluster_data[label] = labels
            
        return cluster_data