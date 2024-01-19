import random

from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from ._cluster import ClusterSampler


class KMeansSampler(ClusterSampler):
    def __init__(self, dataset, ratio=1.0, num_clusters: int = 10, eqsize=True, **kwargs):
        self.num_clusters_ = num_clusters
        super().__init__(dataset, ratio, eqsize, **kwargs)

    def _select_indices(self):
        selected_indices = {}
        num_samples_per_cluster = max(2, int(len(self.dataset_) * self.ratio_ / len(self.class_data_) / self.num_clusters_))
        for (label, indices), (_, clusters) in zip(self.class_data_.items(), self.cluster_data_.items()):
            selected_indices[label] = {}
            
            cluster_sizes = dict(Counter(clusters))
            cluster_sizes = dict(sorted(cluster_sizes.items(), key=lambda item: item[1], reverse=True))
            print(label, cluster_sizes)
        
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
            # data = [self._extract_features(self.dataset_[i][0]) for i in indices]
            data = [self.dataset_[i][0].numpy().flatten() for i in indices]
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data)
            kmeans = KMeans(n_clusters=self.num_clusters_, n_init=10)
            labels = kmeans.fit_predict(data)
            cluster_data[label] = labels
            
        return cluster_data