# Document Type Classification | 문서 타입 분류
## Team 7조 | 남녀노소

| ![홍상호](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김정빈](https://avatars.githubusercontent.com/u/156163982?v=4) | ![소제목](https://avatars.githubusercontent.com/u/156163982?v=4) | ![박성진](https://avatars.githubusercontent.com/u/156163982?v=4) | ![고민주](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [홍상호](https://github.com/UpstageAILab)             |            [김정빈](https://github.com/UpstageAILab)             |            [소제목](https://github.com/UpstageAILab)             |            [박성진](https://github.com/UpstageAILab)             |            [고민주](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview
### Environment
- _Write Development environment_

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

## 2. Components

### Directory
```
cv-document-classification/
├── config/
│   └── model_configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── utils/
│   └── ensemble/
├── scripts/
├── notebooks/
├── experiments/
├── outputs/
│   ├── models/
│   ├── predictions/
│   └── submissions/
└── docs/

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

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
