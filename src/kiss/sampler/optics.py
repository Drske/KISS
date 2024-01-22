import random
import numpy as np

from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.cluster import OPTICS, SpectralClustering, Birch, MeanShift
from sklearn.preprocessing import MinMaxScaler


from ._cluster import ClusterSampler
from ..feature_extractor import ClassicFeatureExtractor


class OpticsSampler(ClusterSampler):
    def __init__(self, dataset, ratio=1.0, eqsize=True, numerous_ratio = 0.5, **kwargs):
        self.numerous_ratio_ = numerous_ratio
        self.feature_extractor = ClassicFeatureExtractor()
        super().__init__(dataset, ratio, eqsize, **kwargs)
            
    def _select_indices(self):        
        selected_indices = {}
        print("len clusters", [len(set(clusters)) for _, clusters in self.cluster_data_.items()])
        
        for (label, indices), (_, clusters) in zip(self.class_data_.items(), self.cluster_data_.items()):
            selected_indices[label] = {}
            cluster_sizes = dict(Counter(clusters))
            cluster_sizes = dict(sorted(cluster_sizes.items(), key=lambda item: item[1], reverse=True))
            
            num_samples_per_cluster = max(1, int(len(indices) * self.ratio_ / len(cluster_sizes)))

            print("general", label, len(indices), self.ratio_, len(clusters), num_samples_per_cluster)
            print("cluster sizes", label, cluster_sizes)
            
            keep_clusters = list(cluster_sizes.keys())[:int(self.numerous_ratio_ * len(clusters))]
            
            if -1 in keep_clusters: keep_clusters.remove(-1)
            
            print("keep clusters:", keep_clusters)
            
            combined = list(zip(indices, clusters))
            random.shuffle(combined)
            indices, clusters = zip(*combined)

            for indice, cluster in zip(indices, clusters):
                if cluster not in keep_clusters: continue
                
                if cluster not in selected_indices[label]:
                    selected_indices[label][cluster] = []
                    
                if len(selected_indices[label][cluster]) == num_samples_per_cluster:
                    continue
                
                selected_indices[label][cluster].append(indice)
                
        return selected_indices
    
    def _get_cluster_data(self, class_data):
        cluster_data = defaultdict(list)
        
        for label, indices in tqdm(class_data.items(), desc="Clustering"):
            imgs = [self.dataset_[i][0] for i in indices]
            data = self.feature_extractor(imgs)
            optics = OPTICS(min_samples=2)
            labels = optics.fit_predict(data)
            cluster_data[label] = labels
            
        return cluster_data