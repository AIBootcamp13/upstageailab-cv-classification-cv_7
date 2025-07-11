# Document Type Classification | 문서 타입 분류
## Team 7조 | 남녀노소

| ![홍상호](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김정빈](https://avatars.githubusercontent.com/u/156163982?v=4) | ![소제목](https://avatars.githubusercontent.com/u/156163982?v=4) | ![박성진](https://avatars.githubusercontent.com/u/156163982?v=4) | ![고민주](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [홍상호](https://github.com/UpstageAILab)             |            [김정빈](https://github.com/UpstageAILab)             |            [소제목](https://github.com/UpstageAILab)             |            [박성진](https://github.com/psj2024p)              |            [고민주](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            팀원                             |                            팀원                             |                            팀원                             |                            팀원                             |

## 0. Overview
### Environment
- 코드 버전 관리: GitHub를 활용한 코드 관리 및 협업
- 실험 관리: Weights & Biases(WandB) + Hydra를 통한 실험 트래킹
- 필수 패키지 설치: PyTorch Lightning, Albumentations, Augraphy 등 환경 세팅
- 재현성 확보: requirements.txt, requirements2.txt 관리


### Requirements
- _Write Requirements_

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

#### **Day 7 (7/10, 목)** - 3차 사이클 완료

&nbsp;&nbsp; - **AM**: TTA 적용, 파인튜닝<br>
&nbsp;&nbsp; - **PM**: 최종 모델 완성, 성능 검증<br>
&nbsp;&nbsp; - **Goal**: 최종 모델 완료

#### **Day 8 (7/11, 금)** - 마무리

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

## 클래스 불균형 분석 보고서

### 분석 목적
문서 데이터의 클래스 분포를 분석하여 클래스 간 불균형이 존재하는지 확인하고, 이에 따른 대응 방안을 도출한다.

---

### 분석 결과
- 전체 문서 중 `임신 진료비 지급 신청서` 및 `소견서` 문서가 약 **50개 정도**로, 다른 문서 클래스에 비해 **상대적으로 개수가 적음**을 확인함.
- 이로 인해 클래스 불균형이 존재하며, 학습 시 특정 클래스에 대한 모델의 인식률 저하 가능성이 있음.

---

### 의견 도출 및 대응 방안
- **데이터 증강(Data Augmentation)** 등을 활용하여 해당 클래스의 데이터를 충분히 확보한다면, 클래스 불균형으로 인한 모델 성능 저하를 **상당 부분 완화**할 수 있을 것으로 판단됨.
- 따라서, 현재는 데이터 증강을 통해 보완하는 방향으로 프로젝트를 **계속 진행**하기로 함.





### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- <img width="1040" height="808" alt="image" src="https://github.com/user-attachments/assets/949e5898-c41e-47a0-91a5-d9f16be1a6d5" />

- _Write rank and score_
- <img width="1015" height="383" alt="image" src="https://github.com/user-attachments/assets/23f9619f-1533-4677-9994-fc84d23bca63" />


### Presentation

- _Insert your presentaion file(pdf) link_
- https://docs.google.com/presentation/d/1dj_4yDZeUex89qat1wzjUi1n5G0Ajmrv/edit?slide=id.g36dabdc4904_0_152#slide=id.g36dabdc4904_0_152

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

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
