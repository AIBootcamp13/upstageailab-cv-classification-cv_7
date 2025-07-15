import numpy as np
import cv2
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
                sigma_range=(8, 11),
                turbulence_range=(2, 4),
                p=0.7
            )],
            paper_phase=[
            ],
            post_phase=[
                Brightness(
                    brightness_range=(1.04, 1.25),
                    min_brightness=0,
                ),
            ]
        )

def resize_padding(image_size=(224, 224), image_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}):
    return A.Compose([
        A.LongestMaxSize(max_size=image_size[0]),
        A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1], border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]),
        A.Normalize(mean=image_normalization["mean"], std=image_normalization["std"]),
        ToTensorV2()
    ])

def get_test_transform(image_size=(224, 224), image_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}):
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=image_normalization["mean"], std=image_normalization["std"]),
        ToTensorV2()
    ])

def get_transform_rotation(image_size=(224, 224), image_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}):
    return A.Compose([
        A.Rotate(limit=160, p=0.8),
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=image_normalization["mean"], std=image_normalization["std"]),
        ToTensorV2()
    ])

def get_transform_gaussNoise(image_size=(224, 224), image_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}):
    return A.Compose([
        A.GaussNoise(var_limit=(0.01, 0.04), p=0.8),
        A.Rotate(limit=160, p=0.8),
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=image_normalization["mean"], std=image_normalization["std"]),
        ToTensorV2()
    ])

def get_transform_blur(image_size=(224, 224), image_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}):
  return A.Compose([
      A.MotionBlur(blur_limit=(8,13), p=0.8),
      A.Rotate(limit=160, p=0.8),
      A.Resize(height=image_size[0], width=image_size[1]),
      A.Normalize(mean=image_normalization["mean"], std=image_normalization["std"]),
      ToTensorV2()
  ])

def get_transform_shadow(image_size=(224, 224), image_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}):
  return A.Compose([
      A.RandomShadow(
        shadow_roi=(0, 0, 1, 1),
        num_shadows_limit=(1, 3), 
        shadow_dimension=6, 
        shadow_intensity_range=(0.2, 0.5),
        p=0.8
      ),
      A.Rotate(limit=160, p=0.8),
      A.Resize(height=image_size[0], width=image_size[1]),
      A.Normalize(mean=image_normalization["mean"], std=image_normalization["std"]),
      ToTensorV2()
  ])

# 나중에 도입해보기
"""
def get_transform_custom(image_size=(224, 224), image_normalization={"mean": [...], "std": [...]}):
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=120, p=0.5),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.4),
        A.MotionBlur(blur_limit=(3, 7), p=0.4),
        A.RandomShadow(p=0.4),
        A.ImageCompression(quality_lower=30, quality_upper=70, p=0.3),
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=image_normalization["mean"], std=image_normalization["std"]),
        ToTensorV2()
    ])
"""