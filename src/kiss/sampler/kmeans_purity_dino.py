import torch
import faiss
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from .kmeans_purity import KMeansPuritySampler
from ..feature_extractor import DinoFeatureExtractor, Dinov2Models

from kiss.utils.configs import CONFIGS

class KMeansPurityDinoSampler(KMeansPuritySampler):
    def __init__(self, dataset, ratio=1.0, num_clusters: int = 10, eqsize=True, min_purity=0.5, dinov2_name: Dinov2Models = 'dinov2_vits14', device: torch.device = None, dino_batch_size: int = 8, **kwargs):
        self.num_clusters_ = num_clusters
        self.dinov2_name = dinov2_name
        self.device = device or torch.device(CONFIGS.torch.device)
        self.feature_extractor = DinoFeatureExtractor(self.dinov2_name, self.device)
        self.dino_batch_size = dino_batch_size
        super().__init__(dataset, ratio, num_clusters, eqsize, min_purity, **kwargs)

    def _get_cluster_data(self, class_data):
        cluster_data = defaultdict(list)
        data = np.zeros((len(self.dataset_), self.feature_extractor.feature_size))
        data_loader = torch.utils.data.DataLoader(self.dataset_, batch_size=self.dino_batch_size, shuffle=False, num_workers=2)

        for i, (imgs, _) in tqdm(enumerate(data_loader), desc="Extracting features", total=len(data_loader)):
            start_idx = i*self.dino_batch_size
            data[start_idx: start_idx + self.dino_batch_size] = self._extract_features(imgs)
        
        print("Clustering examples...")
        kmeans = faiss.Kmeans(d=self.feature_extractor.feature_size, k=self.num_clusters_, niter=20)
        kmeans.train(data)
        _, clusters = kmeans.index.search(data, 1)
        
        clusters = clusters.flatten()

        for label, indices in tqdm(class_data.items(), desc="Creating cluster data"):
            cluster_data[label] = [clusters[i] for i in indices]
        return cluster_data

    def _extract_features(self, imgs):
        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)

        features = self.feature_extractor(imgs)
        features = np.float32(features)
        faiss.normalize_L2(features)

        return features
