import cv2
import numpy as np
import torch
import albumentations as A

def get_rotation_angle_from_numpy(image):
    """
    이미지(NumPy 배열)의 회전 각도를 추출하는 함수

    Args:
        image (np.ndarray): 분석할 이미지 (OpenCV로 읽은 BGR 형식의 NumPy 배열)

    Returns:
        float: 계산된 회전 각도 (단위: 도)
    """

    # uint8 타입으로 변환 (필요시)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    # 1. 전처리 (이미지가 이미 NumPy 배열이므로 읽기 과정이 필요 없음)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러로 노이즈 제거
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 이진화 (Threshold)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. 윤곽선 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 3. 가장 큰 윤곽선 찾기
    if not contours:
        return 0
        
    largest_contour = max(contours, key=cv2.contourArea)

    # 4. 최소 영역 사각형으로 각도 계산
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]

    # 5. 각도 보정
    if rect[1][0] < rect[1][1]:
        angle = angle + 90
    
    if angle < -45:
        angle = 90 + angle

    return angle

def rotate_with_albumentations(image: np.ndarray) -> np.ndarray:
    """
    Albumentations를 이용해 이미지를 회전시키는 함수 (numpy array 입력)
    
    Args:
        image (np.ndarray): 입력 이미지, shape (H, W, C), dtype uint8
        angle (int): 회전 각도 (양수면 시계 반대방향)

    Returns:
        np.ndarray: 회전된 이미지, shape (H, W, C)
    """
    assert image.ndim == 3 and image.shape[2] in [1, 3], "입력은 (H, W, C) 형식의 컬러 또는 흑백 이미지여야 합니다."
    
    angle = int(get_rotation_angle_from_numpy(image))

    transform = A.Compose([
        A.Rotate(limit=(angle, angle), border_mode=cv2.BORDER_CONSTANT,
                 fill=(255, 255, 255), interpolation=cv2.INTER_LINEAR, p=1.0)
    ])

    augmented = transform(image=image)
    rotated_image = augmented["image"]

    return rotated_image

def denoise_and_contrast_enhance(
    image: np.ndarray,
    h: float = 2.0,
    h_color: float = 3.0,
    template_window_size: int = 7,
    search_window_size: int = 15,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    이미지 디노이즈 및 대비 향상 (numpy array 입력용)

    Args:
        image (np.ndarray): 입력 RGB 이미지, shape (H, W, C), dtype: float32 or uint8
        나머지는 OpenCV 디노이징 및 CLAHE 설정

    Returns:
        np.ndarray: 처리된 RGB 이미지 (np.uint8)
    """
    assert image.ndim == 3 and image.shape[2] in [1, 3], "입력은 (H, W, C) 형식의 이미지여야 합니다."

    img = image.copy()

    # float32 [0,1]인 경우 → uint8 [0,255]
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # 1. fast Non-Local Means Denoising
    img = cv2.fastNlMeansDenoisingColored(
        img,
        h=h,
        hColor=h_color,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size
    )

    # 2. CLAHE 대비 향상
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_enhanced = cv2.merge((l_clahe, a, b))
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    return enhanced_rgb    