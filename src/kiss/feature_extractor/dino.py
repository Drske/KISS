import torch
import numpy as np
from torchvision import transforms
from typing import Literal

from kiss.utils.configs import CONFIGS

Dinov2Models = Literal['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14']
FeatureSize = {
    'dinov2_vits14': 384, 
    'dinov2_vitb14': 768, 
    'dinov2_vitl14': 1024, 
    'dinov2_vitg14': 1536
}

class DinoFeatureExtractor:
    def __init__(
        self,
        dinov2_name: Dinov2Models = 'dinov2_vits14',
        device: torch.device = None,
    ):
        self.device = device or torch.device(CONFIGS.torch.device)
        self.model = torch.hub.load('facebookresearch/dinov2', dinov2_name).to(self.device)
        self.patch_size = 14
        self.patch_h = 520 // self.patch_size
        self.patch_w = 520 // self.patch_size
        self.feature_size = FeatureSize[dinov2_name]

    def __call__(self, imgs: torch.Tensor, mean=True) -> np.ndarray:
        preprocessed_imgs = self._preprocess(imgs)

        with torch.no_grad():
            features_dict = self.model.forward_features(preprocessed_imgs.to(self.device))
            features = features_dict['x_norm_patchtokens']

        features.reshape(imgs.shape[0], self.patch_h * self.patch_w, self.feature_size)        

        if mean:
            features = features.mean(axis=1)

        features.detach().cpu().numpy()
        
        return features.detach().cpu().numpy()

    def _preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize(520, antialias=True),
            transforms.CenterCrop(518),                         
            transforms.Normalize(mean=0.5, std=0.2)
        ])

        return transform(imgs)
