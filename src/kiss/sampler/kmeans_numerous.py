import random

from collections import Counter

from .kmeans import KMeansSampler


class KMeansNumerousSampler(KMeansSampler):
    def __init__(self, dataset, ratio=1.0, num_clusters: int = 10, eqsize=True, numerous_ratio = 0.5, **kwargs):
        self.numerous_ratio_ = numerous_ratio
        super().__init__(dataset, ratio, num_clusters, eqsize, **kwargs)
        
    def _select_indices(self):
        selected_indices = {}
        num_samples_per_cluster = max(2, int(len(self.dataset_) * self.ratio_ / len(self.class_data_) / self.num_clusters_ / self.numerous_ratio_))
        for (label, indices), (_, clusters) in zip(self.class_data_.items(), self.cluster_data_.items()):
            selected_indices[label] = {}
            
            cluster_sizes = dict(Counter(clusters))
            cluster_sizes = dict(sorted(cluster_sizes.items(), key=lambda item: item[1], reverse=True))
            print(label, cluster_sizes)
            keep_clusters = list(cluster_sizes.keys())[:int(self.num_clusters_ * self.numerous_ratio_)]
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