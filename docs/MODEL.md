# 암석 분류 모델 및 평가

- **개발 및 학습 환경** Ubuntu 24.04, NVIDIA GeForce RTX 4080 SUPER, PyTorch, CUDA
- **모델** 사전 학습된 ConvNeXt-Base를 사용한 7개 암석 클래스 분류
- [Ubuntu 환경 설정 및 데이터 배치](../README.md) | [실행 스크립트](../scripts/rock.sh) | [학습 코드](../model.py)

## 데이터셋

- 학습용 JPG 380,020장, 테스트용 JPG 95,006장
- 학습 이미지 경로는 `/home/kangjungmin1/open/train/`
- 클래스 번호(0~6)는 실제 학습 폴더명을 오름차순으로 정렬해 부여
- 체크포인트의 예측 번호를 클래스명으로 바꿀 때는 학습에 사용한 폴더명 기준으로 매핑

| 클래스 | 학습 이미지 수 |
| --- | ---: |
| Andesite | 43,802 |
| Basalt | 26,810 |
| Etc | 15,935 |
| Gneiss | 73,914 |
| Granite | 92,923 |
| Mud_Sandstone | 89,467 |
| Weathered_Rock | 37,169 |
| **합계** | **380,020** |

## Ubuntu 학습 및 제출

```bash
./scripts/rock.sh train
```

| 항목 | 설정 |
| --- | --- |
| 전처리 | LAB 밝기 채널에 CLAHE 적용, 224×224 크기 조정, ImageNet 정규화 |
| 검증 분할 | 층화 5폴드, 5개 검증 폴드에 모든 클래스를 포함하려면 클래스별 최소 5장 필요 |
| 클래스 불균형 보정 | 가중치 샘플링, 클래스 가중 손실 |
| 최적화 | AdamW, 코사인 학습률 스케줄, label smoothing, CUDA 혼합 정밀도 |
| 종료 조건 | 최대 50에폭, 검증 정확도가 8에폭 연속 개선되지 않으면 해당 폴드 종료 |
| 체크포인트 | 각 폴드에서 검증 정확도가 개선될 때 모델 파라미터와 버퍼 저장 |

- `test_modified.csv`의 `img_path`에 테스트 이미지 경로 지정
- 테스트 이미지의 상대 경로는 명령 실행 디렉터리 기준
- 제출용 추론은 기본 전처리, 좌우 반전, 밝기 및 대비 조정, 가우시안 블러로 얻은 네 가지 예측 확률을 모델별로 평균
- 폴드별 최고 검증 정확도에 비례해 각 모델의 예측 확률을 가중 평균
- 최종 클래스 번호를 `sample_submission.csv`의 `target` 열에 채워 `submit_convnext.csv`로 저장
- 폴드별 체크포인트는 `best_model_fold1.pth`부터 `best_model_fold5.pth`까지 데이터 디렉터리에 저장(각 약 334MiB)
- 같은 경로에서 다시 학습하면 기존 체크포인트와 제출 CSV를 덮어쓸 수 있음
- 최고 검증 정확도는 학습 중 메모리에 보관하므로 가중 앙상블을 다시 구성하려면 폴드별 검증 정확도가 필요

## 체크포인트 추론 및 평가

### 이미지 한 장 분류

```bash
./scripts/rock.sh predict /path/to/rock.jpg --checkpoint /path/to/best_model_fold1.pth --device cpu
```

- [predict.py](../predict.py)에서 학습된 가중치를 `weights_only=True`로 읽고 검증용 전처리 적용
- 한 번의 순전파로 가장 높은 softmax 확률과 해당 클래스 번호를 JSON으로 출력

### 폴드별 성능 평가

- 검증 CSV에 이미지 경로(`img_path`)와 정답 클래스 번호(`target`) 지정
- 검증 이미지의 상대 경로는 CSV가 있는 디렉터리 기준
- 기존 학습의 폴드별 성능을 확인하려면 해당 폴드의 검증 데이터와 체크포인트를 연결

```csv
img_path,target
images/sample_a.jpg,0
images/sample_b.jpg,3
```

```bash
./scripts/rock.sh evaluate /path/to/fold1.csv --checkpoint /path/to/best_model_fold1.pth \
  --batch-size 8 > evaluation.json
./scripts/rock.sh evaluate-folds /path/to/validation-folds /path/to/checkpoints --batch-size 8
```

- [evaluate.py](../evaluate.py)에서 이미지 배치별로 추론한 뒤 정확도, macro F1, 클래스별 precision, recall, F1, support와 혼동행렬을 JSON으로 출력
- 혼동행렬의 행은 정답, 열은 예측이며 클래스 번호 0~6 순서
- 분모가 0인 지표는 0으로 계산하고 검증 이미지가 없는 클래스도 7개 클래스 macro F1 계산에 포함

## 모델별 F1

| 모델 | F1 |
| --- | ---: |
| ViT-L/16 | 0.66588 |
| Swin V2 | 0.72668 |
| ConvNeXt-Base | 0.83635 |

- [모델 비교표(PDF 23쪽)](presentation.pdf#page=23)
