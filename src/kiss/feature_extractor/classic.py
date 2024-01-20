from torch import Tensor

import cv2
import scipy.stats
import numpy as np

from typing import List

from skimage import feature
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import hog

class ClassicFeatureExtractor:
    def __init__(self):
        self.scaler_ =  MinMaxScaler()
    
    def __call__(self, imgs: List[Tensor]):
        data = [self.__extract_features(img) for img in imgs]
        data = self.scaler_.fit_transform(data)
        
        return data
    
    def __extract_features(self, image):
        image = image.permute(1, 2, 0).numpy()
        image = np.round(image * 255).astype(np.uint8)
        
        # Colors
        mean_colors = np.mean(image, axis=(0,1))
        
        # Color histogram
        hist_size = 16
        b_hist = cv2.calcHist([image], [0], None, [hist_size], (0, 256), accumulate=False)
        g_hist = cv2.calcHist([image], [1], None, [hist_size], (0, 256), accumulate=False)
        r_hist = cv2.calcHist([image], [2], None, [hist_size], (0, 256), accumulate=False)
        
        hist = np.concatenate([b_hist, g_hist, r_hist]).flatten()
        
        # Hu Moments
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray_image)
        hu_moments = cv2.HuMoments(moments)
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        
        # Additional features
        mean_values = np.mean(image, axis=(0, 1))
        std_values = np.std(image, axis=(0, 1))
        skewness_values = [scipy.stats.skew(hist)]
        kurtosis_values = [scipy.stats.kurtosis(hist)]
        
        # Hog features
        hog_features = hog(gray_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
        
        # Edges
        edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)
        edge_density = [np.sum(edges) / (edges.shape[0] * edges.shape[1])]
    
        # Concatenate all features
        features = np.concatenate([
            mean_colors[:2],
            hist,
            mean_values, 
            std_values, 
            skewness_values, 
            kurtosis_values,
            hog_features,
            edge_density,
            hu_moments.flatten()
        ])
        
        return features