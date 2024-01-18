import os
import numpy as np
import random
import cv2

from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from skimage import feature
from tqdm import tqdm
import scipy.stats
import pickle

from .base import Sampler

class KMeansSampler(Sampler):
    def __init__(self, dataset, ratio=1.0, num_clusters: int = 10, eqsize=True, **kwargs):
        super().__init__(dataset, ratio)
        self.num_clusters_ = num_clusters
        self.eqsize = eqsize
        
        saveto = kwargs.get("saveto", None)
        loadfrom = kwargs.get("loadfrom", None)
        
        if loadfrom is not None:
            with open(os.path.join(loadfrom, "class_data.picke"), 'w+') as file:
                self.class_data_ = pickle.load(file)
                
            with open(os.path.join(loadfrom, "cluster_data.picke"), 'w+') as file:
                self.cluster_data_ = pickle.load(file)
        else:
            self.class_data_ = self._get_class_data()
            self.cluster_data_ = self._get_cluster_data(self.class_data_, num_clusters)
        
        if saveto is not None:
            with open(os.path.join(saveto, "class_data.picke"), 'w+') as file:
                pickle.dump(self.class_data_, file)
                
            with open(os.path.join(saveto, "cluster_data.picke"), 'w+') as file:
                pickle.dump(self.cluster_data_, file)

        
            
    def calculate_indices(self, ratio: float = None):
        ratio = ratio or self.ratio_
        
        num_samples = int(len(self.dataset_) * ratio)
        
        # Equal number of samples from each cluster
        selected_indices = {}
        num_samples_per_cluster = max(2, int(len(self.dataset_) * ratio / len(self.class_data_) / self.num_clusters_))
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
        
        # Numerous clusters first/last
        # selected_indices = {}
        # for (label, indices), (_, clusters) in zip(self.class_data_.items(), self.cluster_data_.items()):
        #     selected_indices[label] = {}
            
        #     cluster_sizes = dict(Counter(clusters))
        #     cluster_sizes = dict(sorted(cluster_sizes.items(), key=lambda item: item[1], reverse=True))
            
        #     num_samples_per_label = int(len(indices) / len(self.dataset_) * num_samples)
        #     chosen = 0
            
        #     for current_cluster in cluster_sizes.keys():
        #         for indice, cluster in zip(indices, clusters):
        #             if cluster != current_cluster:
        #                 continue
                    
        #             if cluster not in selected_indices[label]:
        #                 selected_indices[label][cluster] = []
                        
        #             selected_indices[label][cluster].append(indice)
        #             chosen += 1
                    
        #             if chosen == num_samples_per_label: break
        #         if chosen == num_samples_per_label: break
        
        # Keep half of numerous clusters
        # selected_indices = {}
        # num_samples_per_cluster = max(2, int(len(self.dataset_) * ratio / len(self.class_data_) / self.num_clusters_ * 2))
        # for (label, indices), (_, clusters) in zip(self.class_data_.items(), self.cluster_data_.items()):
        #     selected_indices[label] = {}
            
        #     cluster_sizes = dict(Counter(clusters))
        #     cluster_sizes = dict(sorted(cluster_sizes.items(), key=lambda item: item[1], reverse=True))
        #     keep_clusters = list(cluster_sizes.keys())[:self.num_clusters_ // 2]
        
        #     combined = list(zip(indices, clusters))
        #     random.shuffle(combined)
        #     indices, clusters = zip(*combined)

        #     for indice, cluster in zip(indices, clusters):
        #         if cluster not in keep_clusters: continue
                
        #         if cluster not in selected_indices[label]:
        #             selected_indices[label][cluster] = []
                    
        #         if len(selected_indices[label][cluster]) == num_samples_per_cluster:
        #             continue
                
        #         selected_indices[label][cluster].append(indice)
                
        self.indices = []
        
        for label, clusters in selected_indices.items():
            for cluster, indices in clusters.items():
                for idx in indices:
                    self.indices.append(idx)
        
        if len(self.indices) < num_samples and self.eqsize:
            selected = set(self.indices)
            remained = set(range(len(self.dataset_)))
            available = remained - selected
            additional = random.sample(list(available), num_samples - len(self.indices))
            self.indices += list(additional)
            
                    
    def _get_class_data(self):
        class_data = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset_):
            class_data[label].append(idx)
            
        return class_data
            
            
    def _get_cluster_data(self, class_data, num_clusters):
        cluster_data = defaultdict(list)
        
        for label, indices in tqdm(class_data.items(), desc="Clustering"):
            data = [self.__extract_features(self.dataset_[i][0]) for i in indices]
            # data = [self.dataset_[i][0].numpy().flatten() for i in indices]
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            labels = kmeans.fit_predict(data)
            cluster_data[label] = labels
            
        return cluster_data
    
    def __extract_features(self, image):
        image = image.permute(1, 2, 0).numpy()
        image = np.round(image * 255).astype(np.uint8)
        
        # Color histogram
        hist_size = 16
        b_hist = cv2.calcHist([image], [0], None, [hist_size], (0, 256), accumulate=False)
        g_hist = cv2.calcHist([image], [1], None, [hist_size], (0, 256), accumulate=False)
        r_hist = cv2.calcHist([image], [2], None, [hist_size], (0, 256), accumulate=False)
        
        hist = np.concatenate([b_hist, g_hist, r_hist]).flatten()
        
        # Color moments
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray_image)
        central_moments = [moments['mu20'], moments['mu11'], moments['mu02']]
        
        # Texture features using Local Binary Pattern (LBP)
        lbp = feature.local_binary_pattern(gray_image, P=8, R=1, method='uniform')
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10))
        
        # Additional features
        mean_values = np.mean(image, axis=(0, 1))
        std_values = np.std(image, axis=(0, 1))
        skewness_values = [scipy.stats.skew(hist)]
        kurtosis_values = [scipy.stats.kurtosis(hist)]
        
        grid_size = 4
        grid_features = []
        for i in range(grid_size):
            for j in range(grid_size):
                grid = image[i * (image.shape[0] // grid_size):(i + 1) * (image.shape[0] // grid_size),
                             j * (image.shape[1] // grid_size):(j + 1) * (image.shape[1] // grid_size), :]
                grid_mean = np.mean(grid, axis=(0, 1))
                grid_std = np.std(grid, axis=(0, 1))
                grid_features.extend(grid_mean)
                grid_features.extend(grid_std)
        
        hog_features = feature.hog(gray_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
        
        edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)
        edge_density = [np.sum(edges) / (edges.shape[0] * edges.shape[1])]
        
        hu_moments = cv2.HuMoments(moments)
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        
        # Concatenate all features
        features = np.concatenate([
            hist / np.sum(hist), 
            central_moments, 
            hist_lbp / np.sum(hist_lbp),
            mean_values, 
            std_values, 
            skewness_values, 
            kurtosis_values,
            grid_features,
            hog_features,
            edge_density,
            hu_moments.flatten()
        ])
        
        return features