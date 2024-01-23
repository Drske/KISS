import random

from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.cluster import KMeans

from ._cluster import ClusterSampler
from ..feature_extractor import ClassicFeatureExtractor


class KMeansSampler(ClusterSampler):
    def __init__(self, dataset, ratio=1.0, num_clusters: int = 10, eqsize=True, numerous_ratio = 0.95, **kwargs):
        self.num_clusters_ = num_clusters
        self.numerous_ratio_ = numerous_ratio
        if not hasattr(self, "feature_extractor"):
            self.feature_extractor = ClassicFeatureExtractor()
        super().__init__(dataset, ratio, eqsize, **kwargs)

    def _select_indices(self):
        selected_indices = {}
        
        # for (label, indices), (_, clusters) in zip(self.class_data_.items(), self.cluster_data_.items()):
        #     selected_indices[label] = {}
        #     num_samples_per_cluster = dict(Counter(clusters))
        #     num_samples_per_cluster = {cluster: max(1, int(size * self.ratio_)) for cluster, size in num_samples_per_cluster.items()}
            
        #     combined = list(zip(indices, clusters))
        #     random.shuffle(combined)
        #     indices, clusters = zip(*combined)

        #     for indice, cluster in zip(indices, clusters):
        #         if cluster not in selected_indices[label]:
        #             selected_indices[label][cluster] = []
                    
        #         if len(selected_indices[label][cluster]) == num_samples_per_cluster[cluster]:
        #             continue
                
        #         selected_indices[label][cluster].append(indice)
                
        for (label, indices), (_, clusters) in zip(self.class_data_.items(), self.cluster_data_.items()):
            selected_indices[label] = []
            num_samples_per_label = max(2, int(len(indices) * self.ratio_))
            
            cluster_sizes = dict(Counter(clusters))
            cluster_sizes = dict(sorted(cluster_sizes.items(), key=lambda item: item[1], reverse=True))
            
            keep_clusters = list(cluster_sizes.keys())[:max(1, int(self.num_clusters_ * self.ratio_ * self.numerous_ratio_))]
            if -1 in keep_clusters: keep_clusters.remove(-1)
            
            combined = list(zip(indices, clusters))
            random.shuffle(combined)
            indices, clusters = zip(*combined)
            
            for indice, cluster in zip(indices, clusters):
                if cluster not in keep_clusters: continue
                    
                if len(selected_indices[label]) == num_samples_per_label: continue
                
                selected_indices[label].append(indice)
            
        return selected_indices
            
    def _get_cluster_data(self, class_data):
        cluster_data = defaultdict(list)
        
        for label, indices in tqdm(class_data.items(), desc="Clustering"):
            imgs = [self.dataset_[i][0] for i in indices]
            data = self.feature_extractor(imgs)
            kmeans = KMeans(n_clusters=self.num_clusters_, n_init=10)
            labels = kmeans.fit_predict(data)
            cluster_data[label] = labels
            
        return cluster_data