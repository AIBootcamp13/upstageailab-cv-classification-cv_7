# Document Type Classification | 문서 타입 분류
## Team 7조 | 남녀노소

| ![홍상호](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김정빈](https://avatars.githubusercontent.com/u/156163982?v=4) | ![소제목](https://avatars.githubusercontent.com/u/156163982?v=4) | ![박성진](https://avatars.githubusercontent.com/u/156163982?v=4) | ![고민주](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [홍상호](https://github.com/UpstageAILab)             |            [김정빈](https://github.com/UpstageAILab)             |            [소제목](https://github.com/UpstageAILab)             |            [박성진](https://github.com/psj2024p)              |            [고민주](https://github.com/UpstageAILab)             |
|                            팀장                             |                            팀원 </br>Wandb hydra 구축 </br>모델 학습 파이프라이닝 </br>모델실험                             |                            팀원                             |                            팀원                             |                            팀원                             |

## 0. Overview
### Environment
- 코드 버전 관리: GitHub를 활용한 코드 관리 및 협업
- 실험 관리: Weights & Biases(WandB) + Hydra를 통한 실험 트래킹
- 필수 패키지 설치: PyTorch Lightning, Albumentations, Augraphy 등 환경 세팅
- 재현성 확보: requirements.txt, requirements2.txt 관리


### Requirements
* Python 3.10+
* pytorch-lightning==2.0.2
* torchvision==0.15.2
* timm==0.9.2
* albumentations==1.3.0
* augraphy==8.0.0
* hydra-core==1.3.2
* wandb==0.15.12

## 1. Competiton Info

### Overview

- **Task**: 문서 이미지 분류 (Document Image Classification)
- **Objective**: 금융, 의료, 보험, 물류 등 산업 전반에 가장 많은 데이터인 아날로그 문서를 디지털화하기 위한 문서 타입 자동 분류 모델 개발
- **Dataset**: 570장의 학습 이미지를 통해 3140장의 평가 이미지를 예측
- **Classes**: 총 17개 문서 타입
- **Metric**: Macro F1 Score

### Timeline

- 프로젝트 전체 기간 (2주) : 6월 30일 (월) 10:00 ~ 7월 10일 (목) 19:00
- ### 📅 Week 1 (7/2~7/4)

#### **Day 1 (7/2, 수)** - 환경 구축

&nbsp;&nbsp; - **AM**: 팀 킥오프, 대회 규칙 이해, 데이터셋 파악 <br>
&nbsp;&nbsp; - **PM**: GitHub 저장소 생성, 디렉토리 구조 설계, WandB 설정<br>
&nbsp;&nbsp; - **Goal**: 개발환경 완료

#### **Day 2 (7/3, 목)** - 베이스라인 분석 + EDA

&nbsp;&nbsp; - **AM**: 베이스라인 모델 구현 (ResNet50)<br>
&nbsp;&nbsp; - **PM**: 기본 학습, 첫 제출<br>
&nbsp;&nbsp; - **Goal**: 베이스라인 완료

#### **Day 3 (7/4, 금)** - EDA + 데이터 증강

&nbsp;&nbsp; - **AM**: EDA (클래스 분포, 이미지 특성 분석)<br>
&nbsp;&nbsp; - **PM**: 데이터 전처리 파이프라인, 데이터로더 구현<br>
&nbsp;&nbsp; - **Goal**: 데이터 증강, 데이터 이해 완료<br>

- ### 📅 Week 2 (7/7~7/11)

#### **Day 4 (7/7, 월)** - EDA 합치기, Validation set 구축

&nbsp;&nbsp; - **AM**: 틀린 케이스 분석, 하이퍼파라미터 튜닝<br>
&nbsp;&nbsp; - **PM**: Albumentations 적용, 다른 모델 실험<br>
&nbsp;&nbsp; - **Goal**: 성능 개선

#### **Day 5 (7/8, 화)** - 2차 사이클 시작

&nbsp;&nbsp; - **AM**: Augraphy 적용, 고급 데이터 증강<br>
&nbsp;&nbsp; - **PM**: 개별 모델 여러 개 학습 (앙상블 준비)<br>
&nbsp;&nbsp; - **Goal**: 다양한 모델 확보

#### **Day 6 (7/9, 수)** - 3차 사이클 시작

&nbsp;&nbsp; - **AM**: 모델 앙상블 구현<br>
&nbsp;&nbsp; - **PM**: 앙상블 최적화, 2차 제출<br>
&nbsp;&nbsp; - **Goal**: 앙상블 완료

#### **Day 7 (7/10, 목)** - 마무리

&nbsp;&nbsp; - **AM**: 최종 제출<br>
&nbsp;&nbsp; - **PM**: 코드 정리, 프로젝트 회고<br>
&nbsp;&nbsp; - **Goal**: 프로젝트 완료

## 2. Components

### Directory
```

upstageailab-cv-classification-cv_7/
├──────── .github/
├──────── artifacts/
├──────── configs/
│    ├──────── data/
│    │    ├─────── document_kfold.yaml/
│    │    └──────── ...
│    ├──────── model/
│    │    ├──────── dinov2.yaml/
│    │    ├──────── efficientnetb5.yaml/
│    │    └──────── ...
│    ├──────── train/
│    │    └──────── default.yaml
│    ├──────── wandb/
│    │    └──────── wandb.yaml
│    └──────── config.yaml
├──────── data/
├──────── Notebook/
├──────── outputs/
├──────── src/
│    ├──────── dataset/
│    ├──────── ensemble/
│    ├──────── model/
│    ├──────── test/
│    ├──────── train/
│    ├──────── transform/
│    └──────── utils/
├──────── main.py/
├──────── requirements2.txt
└──────── csv outputs/

```

## 3. Data descrption

### Dataset overview

이 프로젝트는 아날로그 문서를 디지털로 자동 분류하기 위한 문서 이미지 분류 태스크입니다.
주어진 데이터셋은 학습 데이터(train)와 평가 데이터(test)로 나뉩니다.

#### ✅ 학습 데이터셋 (train)
- **이미지**: train/ 폴더에 1,570장의 이미지가 저장되어 있습니다.
- **라벨**: train.csv 파일에 각 이미지에 대한 정답 클래스 번호(target)가 제공됩니다.
    - ID: 학습 샘플의 파일명
    - target: 학습 샘플의 정답 클래스 번호
- **클래스 정보**: meta.csv 파일에 17개 클래스에 대한 매핑 정보가 포함되어 있습니다.
    - target: 클래스 번호
    - class_name: 클래스 이름 (ex: 주민등록증, 여권 등)

#### ✅ 평가 데이터셋 (test)
- **이미지**: test/ 폴더에 3,140장의 이미지가 저장되어 있습니다.
- **제출 파일 형식**: sample_submission.csv 파일은 총 3,140개의 샘플 ID가 포함되어 있으며, target 컬럼은 예측 결과가 들어갈 자리로 모든 값이 0으로 초기화되어 있습니다.
- 평가 데이터는 학습 데이터와 달리 랜덤으로 Rotation, Flip 등의 변형이 되어 있으며, 일부는 훼손된 이미지가 포함되어 있습니다.


### EDA

- 클래스 분포 시각화 및 품질 분석
- 이미지 크기 및 품질 편차 확인
- 틀린 케이스 패턴 분석


### Data Processing

- Augraphy, albumentations 활용하여 custom augmentation 적용
- 확률적 로테이션 적용하여 회전 강건 학습
- Cutmix 기법 활용

## 4. Modeling

### Model descrition

* **ResNet18 / ResNet50**: 가장 기본적인 CNN 기반 분류기로써 baseline 성능 측정용
* **EfficientNetB0 / B5**: 파라미터 수 대비 높은 정확도를 자랑하며 실험 효율성이 높음
* **Swin Transformer**: Transformer 기반으로 전역 정보 추출에 강점
* **DINOv2 (ViT 기반)**: Self-Supervised 학습으로 사전학습된 backbone 사용

### Modeling Process

* Train/Validation split: Stratified K-Fold (n=5)
* CE Loss + Triplet Loss 기반 학습
* Pytorch Lightning 기반으로 재사용 가능한 Module 구성
* 주요 실험에는 WandB + Hydra 활용하여 실험 기록

## 5. Result

### Leader Board

- <img width="1040" height="808" alt="image" src="https://github.com/user-attachments/assets/949e5898-c41e-47a0-91a5-d9f16be1a6d5" />

- <img width="1015" height="383" alt="image" src="https://github.com/user-attachments/assets/23f9619f-1533-4677-9994-fc84d23bca63" />


### Presentation

- https://docs.google.com/presentation/d/1dj_4yDZeUex89qat1wzjUi1n5G0Ajmrv/edit?slide=id.g36dabdc4904_0_152#slide=id.g36dabdc4904_0_152

## etc

### Reference

- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [Dataset & DataLoader 튜토리얼](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [Transfer Learning 튜토리얼](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [PyTorch Lightning (선택)](https://lightning.ai/docs/pytorch/stable/)
- [Hydra 공식 문서](https://hydra.cc/docs/intro/)
- [Weights & Biases](https://docs.wandb.ai/)
- [albumentations](https://albumentations.ai/docs/)
- [augraphy](https://augraphy.com/)
- [프로젝트 발표 자료](https://docs.google.com/presentation/d/1dj_4yDZeUex89qat1wzjUi1n5G0Ajmrv/edit?slide=id.g36dabdc4904_0_152#slide=id.g36dabdc4904_0_152)
- [7조 노션](https://www.notion.so/7-21140cb3731d80028eafddf88e9cdb17?source=copy_link)
