# src/utils/ocr_module.py
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T


class OCRModule:
    def __init__(self, lang='korean'):
        self.ocr = PaddleOCR(
            # ✅ 모델 버전 및 언어 설정
            lang='korean',                       # 문서가 한글 중심이라면
            ocr_version='PP-OCRv5',              # 가장 최신 모델 (인식률 우수)

            # ✅ 디바이스 설정 (GPU가 있으면 사용)
            device='gpu',                      # 'cpu' 또는 'gpu:0'으로 지정

            # ✅ 성능 최적화 옵션
            enable_mkldnn=False,                 # GPU 사용 시 비활성화
            precision='fp16',                    # GPU + FP16 환경이면 성능 향상 가능

            # ✅ 감지/인식 관련 세부 설정
            text_det_limit_side_len= 456,        # 큰 문서 이미지 처리
            text_det_limit_type='max',           # 긴 변 기준으로 리사이즈
            text_det_thresh=0.3,
            text_det_box_thresh=0.5,
            text_det_unclip_ratio=2.0,

            text_rec_score_thresh=0.3         # 신뢰도 낮은 텍스트도 포함
        )

        # 이미지 전처리 (Tensor → PIL)
        self.tensor_to_pil = T.ToPILImage()

    def run_ocr(self, image_tensors: torch.Tensor):
        """
        단일 이미지(Tensor)에 대해 OCR을 수행하고 추출된 텍스트를 문자열로 반환.
        """
        pil_imgs = [self.tensor_to_pil(image_tensor) for image_tensor in image_tensors]
        np_imgs = [np.array(pil_img) for pil_img in pil_imgs]

        result = self.ocr.predict(np_imgs, use_doc_orientation_classify=True, use_doc_unwarping=True, use_textline_orientation=True)
        if result is None or len(result[0]) == 0:
            return ""

        return result

    def batch_ocr(self, batch_tensor: torch.Tensor) -> list:
        """
        배치 이미지(Tensor)에 대해 OCR 수행 → 텍스트 리스트 반환
        """
        results = []
        for i in range(batch_tensor.size(0)):
            text = self.run_ocr(batch_tensor[i])
            results.append(text)
        return results
