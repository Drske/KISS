import os
import cv2
import random
import pickle
import scipy.stats
import numpy as np

from skimage import feature
from abc import abstractmethod
from torchvision.datasets.vision import VisionDataset
from collections import defaultdict

from ._sampler import Sampler


class ClusterSampler(Sampler):
    def __init__(self, dataset: VisionDataset, ratio: float = 1.0, eqsize=True, load_clusters: str = None, save_clusters: str = None, **kwargs):
        super().__init__(dataset, ratio)
        self.eqsize = eqsize
        
        if load_clusters is not None:
            with open(os.path.join(load_clusters, "class_data.pickle"), 'rb') as file:
                self.class_data_ = pickle.load(file)
                
            with open(os.path.join(load_clusters, "cluster_data.pickle"), 'rb') as file:
                self.cluster_data_ = pickle.load(file)
        else:
            self.class_data_ = self._get_class_data()
            self.cluster_data_ = self._get_cluster_data(self.class_data_)
        
        if save_clusters is not None:
            if not os.path.exists(save_clusters):
                os.makedirs(save_clusters)
                
            with open(os.path.join(save_clusters, "class_data.pickle"), 'wb+') as file:
                pickle.dump(self.class_data_, file)
                
            with open(os.path.join(save_clusters, "cluster_data.pickle"), 'wb+') as file:
                pickle.dump(self.cluster_data_, file)

    def calculate_indices(self, ratio: float = None):
        self.ratio_ = ratio or self.ratio_
        
        selected_indices = self._select_indices()
                        
        self.indices = []
        
        for _, clusters in selected_indices.items():
            for _, indices in clusters.items():
                for idx in indices:
                    self.indices.append(idx)
                    
                    
        
        
        num_samples = int(len(self.dataset_) * ratio)
        print(num_samples, len(self.indices))
        if len(self.indices) < num_samples and self.eqsize:
            selected = set(self.indices)
            remained = set(range(len(self.dataset_)))
            available = remained - selected
            additional = random.sample(list(available), num_samples - len(self.indices))
            self.indices += list(additional)
    
    @abstractmethod
    def _select_indices(self):
        pass
    
    def _get_class_data(self):
        class_data = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset_):
            class_data[label].append(idx)
            
        return class_data
    
    @abstractmethod
    def _get_cluster_data(self, class_data):
        pass
    
    def _extract_features(self, image):
        image = image.permute(1, 2, 0).numpy()
        image = np.round(image * 255).astype(np.uint8)
        
        # Color histogram
        hist_size = 4
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
        
        grid_size = 2
        grid_features = []
        for i in range(grid_size):
            for j in range(grid_size):
                grid = image[i * (image.shape[0] // grid_size):(i + 1) * (image.shape[0] // grid_size),
                             j * (image.shape[1] // grid_size):(j + 1) * (image.shape[1] // grid_size), :]
                grid_mean = np.mean(grid, axis=(0, 1))
                grid_std = np.std(grid, axis=(0, 1))
                grid_features.extend(grid_mean)
                grid_features.extend(grid_std)
        
        hog_features = feature.hog(gray_image, orientations=3, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
        
        edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)
        edge_density = [np.sum(edges) / (edges.shape[0] * edges.shape[1])]
        
        hu_moments = cv2.HuMoments(moments)
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        
        # Concatenate all features
        features = np.concatenate([
            hist,
            # central_moments, 
            # hist_lbp,
            # mean_values, 
            # std_values, 
            # skewness_values, 
            # kurtosis_values,
            # grid_features,
            # hog_features,
            # edge_density,
            # hu_moments.flatten()
        ])
        
        return features