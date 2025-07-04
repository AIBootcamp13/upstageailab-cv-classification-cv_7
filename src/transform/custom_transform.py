import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from augraphy import AugraphyPipeline, Folding, InkBleed, Brightness, NoiseTexturize, PaperFactory

def get_augraphy_transform():
    return AugraphyPipeline(
            ink_phase=[InkBleed(
                intensity_range=(0.5, 0.9),
                kernel_size=(5, 5),
                severity=(0.2, 0.4)
            ),
            NoiseTexturize(
                sigma_range=(6, 9),
                turbulence_range=(2, 4),
                p=0.5
            )],
            paper_phase=[
            ],
            post_phase=[
                Brightness(
                    brightness_range=(0.7, 1.3),
                    min_brightness=0,
                ),
            ]
        )

def get_transform_rotation():
    return A.Compose([
        A.Rotate(limit=160, p=0.8),
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_transform_gaussNoise():
    return A.Compose([
        A.GaussNoise(std_range=(0.1, 0.2), p=0.8),
        A.Rotate(limit=160, p=0.8),
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_transform_blur():
  return A.Compose([
      A.MotionBlur(blur_limit=(8,13), p=0.8),
      A.Rotate(limit=160, p=0.8),
      A.Resize(height=224, width=224),
      A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ToTensorV2()
  ])

def get_transform_shadow():
  return A.Compose([
      A.RandomShadow(
        shadow_roi=(0, 0, 1, 1),
        num_shadows_limit=(1, 3), 
        shadow_dimension=6, 
        shadow_intensity_range=(0.2, 0.5),
        p=0.8
      ),
      A.Rotate(limit=160, p=0.8),
      A.Resize(height=224, width=224),
      A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ToTensorV2()
  ])