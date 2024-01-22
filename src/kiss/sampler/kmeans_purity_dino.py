import torch
import faiss
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from .kmeans_purity import KMeansPuritySampler
from ..feature_extractor import DinoFeatureExtractor, Dinov2Models

from kiss.utils.configs import CONFIGS

class KMeansPurityDinoSampler(KMeansPuritySampler):
    def __init__(self, dataset, ratio=1.0, num_clusters: int = 10, eqsize=True, min_purity=0.5, dinov2_name: Dinov2Models = 'dinov2_vits14', device: torch.device = None, **kwargs):
        self.num_clusters_ = num_clusters
        self.dinov2_name = dinov2_name
        self.device = device or torch.device(CONFIGS.torch.device)
        self.feature_extractor = DinoFeatureExtractor(self.dinov2_name, self.device)
        super().__init__(dataset, ratio, num_clusters, eqsize, min_purity, **kwargs)

    def _get_cluster_data(self, class_data):
        cluster_data = defaultdict(list)

        print("Extracting features...")
        data = np.vstack([self._extract_features(img) for img, _ in self.dataset_]) 

        print("Clustering examples...")
        kmeans = faiss.Kmeans(d=self.feature_extractor.feature_size, k=self.num_clusters_, niter=20)
        kmeans.train(data)
        _, clusters = kmeans.index.search(data, 1)
        
        for label, indices in tqdm(class_data.items(), desc="Creating cluster data"):
            cluster_data[label] = [clusters[i] for i in indices]
            
        return cluster_data

    def _extract_features(self, image):
        features = self.feature_extractor(image.unsqueeze(0))
        features = np.float32(features)
        faiss.normalize_L2(features)

        return features
