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
        mean_num_clusters = np.mean([len(set(clusters)) for _, clusters in self.cluster_data_.items()]) * self.numerous_ratio_
        num_samples_per_cluster = max(1, int(len(self.dataset_) * self.ratio_ / len(self.class_data_) / mean_num_clusters))
        
        print(len(self.dataset_), self.ratio_, mean_num_clusters, num_samples_per_cluster)
        
        for (label, indices), (_, clusters) in zip(self.class_data_.items(), self.cluster_data_.items()):
            selected_indices[label] = {}
            
            cluster_sizes = dict(Counter(clusters))
            cluster_sizes = dict(sorted(cluster_sizes.items(), key=lambda item: item[1], reverse=True))
            print(label, cluster_sizes)
            keep_clusters = list(cluster_sizes.keys())[:int(mean_num_clusters)]
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
            optics = OPTICS()
            labels = optics.fit_predict(data)
            cluster_data[label] = labels
            
        return cluster_data