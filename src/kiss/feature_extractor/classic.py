from torch import Tensor

import cv2
import scipy.stats
import numpy as np

from typing import List

from skimage import feature
from sklearn.preprocessing import MinMaxScaler

class ClassicFeatureExtractor:
    def __init__(self):
        self.scaler_ =  MinMaxScaler()
    
    def __call__(self, imgs: List[Tensor]):
        data = [self.__extract_features(img) for img in imgs]
        # data = [img.numpy().flatten() for img in imgs]
        # data = [np.mean(img.permute(1, 2, 0).numpy(), axis=(0,1)) for img in imgs]
        # data = self.scaler_.fit_transform(data)
        
        return data
    
    def __extract_features(self, image):
        image = image.permute(1, 2, 0).numpy()
        mean_colors = np.mean(image, axis=(0,1))
            
        # Color moments
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray_image)
        central_moments = [moments['mu20'], moments['mu11'], moments['mu02']]
        
        hu_moments = cv2.HuMoments(moments)
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        
        # Concatenate all features
        features = np.concatenate([
            mean_colors,
            central_moments, 
            hu_moments.flatten()
        ])
        
        return features