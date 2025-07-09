import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from augraphy.augmentations.jpeg import Jpeg
from augraphy.augmentations.geometric import Geometric  
from augraphy.augmentations.inkbleed import InkBleed
from augraphy.augmentations.brightness import Brightness  # 새로 추가
from augraphy import AugraphyPipeline

# ImageNet Normalization (공통)
IMAGENET_STATS = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
}

MODEL_INPUT_SIZES = {
    "resnet": (224, 224),
    "resnet18": (224, 224),
    "resnet34": (224, 224),
    "resnet50": (224, 224),
    "efficientnet_b0": (224, 224),
    "efficientnet_b1": (240, 240),
    "efficientnet_b2": (260, 260),
    "efficientnet_b3": (300, 300),
    "vit": (224, 224),
    "vit_large": (384, 384)
}

def get_common_transfom(img_size=(224, 224), normalization=IMAGENET_STATS):
    """공통 증강 (Motion Blur 포함)"""
    return A.Compose([
        A.PadIfNeeded(min_height=max(img_size), min_width=max(img_size), border_mode=cv2.BORDER_CONSTANT, value=255),
        A.Resize(img_size[0], img_size[1]),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5)
        ], p=0.6),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.5),
        A.Normalize(mean=normalization["mean"], std=normalization["std"]),
        ToTensorV2()
    ])

# 정확한 Augraphy 함수 사용 (brightness 추가)
def get_compression_pipeline():
    """JPEG 압축 파이프라인 (jpeg.py)"""
    return AugraphyPipeline([
        Jpeg(quality_range=(10, 50))
    ])

def get_skew_pipeline():
    """기하학적 변형 파이프라인 (geometric.py)"""
    return AugraphyPipeline([
        Geometric(rotate_range=(5, 10))
    ])

def get_inkbleed_pipeline():
    """잉크 번짐 파이프라인 (inkbleed.py)"""
    return AugraphyPipeline([
        InkBleed(intensity_range=(0.1, 0.3))
    ])

def get_brightness_pipeline():
    """밝기 조정 파이프라인 (brightness.py)"""
    return AugraphyPipeline([
        Brightness(brightness_range=(0.8, 1.3))  # 밝은 쪽으로 더 조정
    ])

def get_totensor_transform():
    return A.Compose([ToTensorV2()])

def get_test_transform(img_size=(224, 224), normalization=IMAGENET_STATS):
    """테스트용 변환 (증강 없음)"""
    return A.Compose([
        A.PadIfNeeded(min_height=max(img_size), min_width=max(img_size), border_mode=cv2.BORDER_CONSTANT, value=255),
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=normalization["mean"], std=normalization["std"]),
        ToTensorV2()
    ])

def get_transform_for_model(model_name="resnet", train=True):
    """backbone별 transform 자동 반환"""
    img_size = MODEL_INPUT_SIZES.get(model_name, (224, 224))
    if train:
        return get_common_transfom(img_size=img_size)
    else:
        return get_test_transform(img_size=img_size)