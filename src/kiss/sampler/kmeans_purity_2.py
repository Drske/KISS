import random

from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from ._purity_cluster import PurityClusterSampler
from ..feature_extractor import ClassicFeatureExtractor


class KMeansPurity2Sampler(PurityClusterSampler):
    def __init__(self, dataset, ratio=1.0, num_clusters: int = 10, eqsize=True, min_purity=0.5, **kwargs):
        self.num_clusters_ = num_clusters
        if not hasattr(self, "feature_extractor"):
            self.feature_extractor = ClassicFeatureExtractor()
        super().__init__(dataset, ratio, eqsize, min_purity, **kwargs)

    def _select_indices(self):
        selected_indices = {}
        
        num_samples_per_label = max(2, int(len(self.dataset_) * self.ratio_ / len(self.class_data_)))
        
        for (label, indices), (_, clusters) in zip(self.class_data_.items(), self.cluster_data_.items()):
            selected_indices[label] = []
                    
            combined = list(zip(indices, clusters))
            random.shuffle(combined)
            indices, clusters = zip(*combined)
            
            purities = dict(Counter(self.purity_data_[label]))
            purities = sorted(purities.items(), key=lambda item: item[1], reverse=True)
            
            print(label, purities)
            
            for select_from_cluster, _ in purities:
                for indice, cluster in zip(indices, clusters):
                    if cluster != select_from_cluster: continue
                    
                    selected_indices[label].append(indice)
                
                    if len(selected_indices[label]) == num_samples_per_label: break
                    if len(selected_indices[label]) == len(indices): break
                    
                if len(selected_indices[label]) == num_samples_per_label: break
                if len(selected_indices[label]) == len(indices): break
                
        return selected_indices
            
    def _get_cluster_data(self, class_data):
        cluster_data = defaultdict(list)
        
        print("Extracting features...")
        imgs = [img for img, _ in self.dataset_]
        data = self.feature_extractor(imgs)
        
        print("Clustering examples...")
        kmeans = KMeans(n_clusters=self.num_clusters_, n_init=1)
        clusters = kmeans.fit_predict(data)
        
        for label, indices in tqdm(class_data.items(), desc="Creating cluster data"):
            cluster_data[label] = [clusters[i] for i in indices]
            
        return cluster_data