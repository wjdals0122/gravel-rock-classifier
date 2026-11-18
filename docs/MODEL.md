# 암석 분류 데이터와 코드 설명

## 원본 데이터

팀 발표 자료 「건설용 자갈 암석 종류 분류 AI」 5~6쪽에 따르면, 학습 데이터는 클래스별 폴더에 저장한 JPG 이미지 380,020장이고 테스트 데이터는 95,006장입니다. 발표 자료의 EDA 그래프에 기재된 학습 이미지 수는 다음과 같습니다.

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

`model.py`는 실제 학습 폴더 이름을 오름차순으로 정렬해 클래스 번호 0~6을 부여합니다. 위 표의 이름은 발표 자료 기준입니다. 기존 체크포인트의 숫자 라벨을 클래스 이름과 연결하려면 원래 폴더명으로 순서를 검증해야 합니다.

## 체크포인트로 단일 이미지 추론

```bash
python predict.py /path/to/rock.jpg --checkpoint best_model_fold1.pth --device cpu
```

`predict.py`는 7개 클래스를 출력하는 ConvNeXt-Base를 만들고, `torch.load(..., weights_only=True)`로 체크포인트를 읽어 모델에 적용합니다. `model.py`의 이미지 읽기 함수와 검증용 변환을 공유하므로, CLAHE, 224×224 크기 조정, ImageNet 정규화, 텐서 변환이 같은 순서로 적용됩니다. 한 번의 순전파에서 가장 높은 softmax 확률과 해당 클래스 번호를 JSON으로 출력합니다.

## 정답 데이터로 평가

검증 CSV에 `img_path`와 `target` 열을 준비합니다. `target`에는 원래 학습의 클래스 번호를 사용합니다. 상대 이미지 경로는 검증 CSV가 있는 디렉터리를 기준으로 해석됩니다.

```csv
img_path,target
images/sample_a.jpg,0
images/sample_b.jpg,3
```

```bash
python evaluate.py /path/to/validation.csv --checkpoint best_model_fold1.pth \
  --batch-size 8 > evaluation.json
```

`evaluate.py`는 `RockDataset`과 검증용 변환을 사용해 이미지들을 배치로 처리합니다. 결과 JSON에는 정확도, 7개 클래스 기준 macro F1, 클래스별 precision·recall·F1·support, 7×7 혼동행렬이 들어갑니다. 혼동행렬의 행은 정답, 열은 예측이며 양쪽 순서는 0~6입니다. 분모가 0인 클래스 지표는 0으로 기록되므로 검증 이미지가 없는 클래스도 macro F1에 영향을 줍니다. 기존 폴드의 성능을 확인하려면 체크포인트와 동일한 폴드의 원래 검증 분할을 사용합니다.

## Ubuntu에서 사용한 학습·제출 파이프라인

`python model.py`는 원래 Ubuntu 개발 환경에서 쓰던 아래 디렉터리 구조를 사용합니다.

```text
/home/kangjungmin1/open/
├── train/
│   ├── <클래스 폴더 1>/
│   │   └── <이미지>.jpg
│   └── … (총 7개 클래스 폴더)
├── test_modified.csv
└── sample_submission.csv
```

- 5개 검증 폴드 모두에 각 클래스가 들어가려면 클래스마다 학습 이미지가 최소 5장 필요합니다.
- `test_modified.csv`의 `img_path` 열에 테스트 이미지 경로를 둡니다. 이 CSV의 상대 경로는 실행 시 작업 디렉터리를 기준으로 해석됩니다.
- `sample_submission.csv`의 행 수와 순서를 테스트 CSV와 맞춥니다. 예측한 클래스 번호는 제출 양식의 `target` 열에 저장됩니다.
- 학습이 끝나면 `data_dir`에 `best_model_fold1.pth`부터 `best_model_fold5.pth`까지와 `submit_convnext.csv`가 생성됩니다. 원본 체크포인트를 보존하려면 학습 전에 별도 위치에 복사합니다.

학습에는 AdamW, 코사인 학습률 스케줄, label smoothing과 층화 5폴드 분할을 사용합니다. 최대 50에폭까지 학습하며, 검증 정확도가 8에폭 연속 개선되지 않으면 해당 폴드를 종료합니다. 제출용 추론은 기본 전처리, 좌우 반전, 밝기·대비 조정, 가우시안 블러의 네 가지 입력에서 얻은 확률을 모델별로 평균하고 폴드별 최고 검증 정확도의 비율로 합산합니다.

보유한 체크포인트 5개는 각각 약 334MiB입니다. 현재 체크포인트 저장 코드는 모델 파라미터와 버퍼를 저장하고, 가중 앙상블에 쓰는 최고 검증 정확도는 학습 중 메모리에 보관합니다. `predict.py`와 `evaluate.py`는 검증 점수 없이 실행 가능한 단일 모델 경로입니다. 원래 가중 앙상블을 별도로 복구하려면 각 폴드의 검증 점수가 추가로 필요합니다.

## 발표 자료의 성능 수치

발표 자료 23쪽의 F1은 ViT-L/16 0.66588, Swin V2 0.72668, ConvNeXt-Base 0.83635입니다. 발표 표에 F1 평균 방식과 평가 분할이 명시되어 있지 않으므로, 현재 평가 코드가 계산하는 7개 클래스 기준 macro F1과 동일한 조건인지 확인이 필요합니다.

## 복구 코드 실행 확인

macOS의 CPU 환경에서 Python 3.14.8, PyTorch 2.15.0, timm 1.0.31, Albumentations 2.0.8로 기존 체크포인트 5개를 로드해 순전파를 확인했습니다. 임시 JPG로 `predict.py`의 JSON 출력과 `evaluate.py`의 지표 및 혼동행렬 형식을 확인했습니다. 원본 데이터가 있는 Ubuntu 환경에서의 재학습과 성능 재평가는 별도 작업으로 남아 있습니다.
