# 건설용 자갈 암석 이미지 분류

**Ubuntu 24.04 Linux · NVIDIA RTX 4080 SUPER · Bash 자동화**

건설용 자갈 사진을 7개 암석 클래스로 분류하는 컴퓨터 비전 프로젝트입니다. Ubuntu에서 PyTorch와 CUDA로 ConvNeXt-Base를 학습하고, 환경 구성부터 5폴드 학습, 체크포인트 추론, 평가까지 Bash 명령으로 실행할 수 있게 구성했습니다.

[프로젝트 발표 자료 PDF](docs/presentation.pdf) · [데이터 및 모델 상세 설명](docs/MODEL.md)

## 개발 및 학습 환경

| 구성 | 사양 |
| --- | --- |
| 운영체제 | Ubuntu 24.04 Linux |
| GPU | NVIDIA GeForce RTX 4080 SUPER, VRAM 16GB |
| CPU | Intel Core i9-14900K |
| 메모리 | 64GB |
| 주요 도구 | Bash, Python, PyTorch, CUDA, timm, Albumentations, OpenCV |

학습에는 CUDA 혼합 정밀도를 사용했습니다. 단일 이미지 추론과 체크포인트 평가는 자동으로 CUDA 사용 가능 여부를 확인하며, `--device cpu`로 CPU를 지정할 수 있습니다.

## Bash로 실행하기

저장소 루트에서 시작합니다. `setup`은 `.venv`를 만들고 패키지를 설치합니다. 이후 명령은 해당 환경의 Python을 직접 실행합니다.

```bash
./scripts/rock.sh setup
./scripts/rock.sh --help
```

| 명령 | 입력과 결과 |
| --- | --- |
| `train` | 원본 학습·테스트 데이터로 5폴드를 학습하고, TTA 앙상블로 제출 CSV를 생성합니다. 실행 로그는 `runs/train-*/train.log`에 저장합니다. |
| `predict` | 자갈 이미지 한 장과 기존 체크포인트를 받아 클래스 번호와 softmax 확률을 JSON으로 출력합니다. |
| `evaluate` | 검증 CSV와 체크포인트 하나로 정확도, macro F1, 클래스별 지표, 혼동행렬을 계산합니다. |
| `evaluate-folds` | 5개 검증 CSV와 체크포인트를 폴드 번호로 연결해 평가한 뒤 `runs/evaluate-*/foldN.json`에 저장합니다. |

원본 학습 환경에서 `./scripts/rock.sh train`을 실행하려면 `/home/kangjungmin1/open/` 아래에 클래스별 학습 이미지가 담긴 `train/`, 테스트 이미지 경로를 담은 `test_modified.csv`, 제출 양식인 `sample_submission.csv`가 필요합니다. 테스트 CSV에 지정된 이미지도 해당 위치에서 읽을 수 있어야 합니다. 학습 결과는 같은 데이터 디렉터리에 저장되며 기존 체크포인트와 제출 CSV를 덮어쓸 수 있습니다.

보유한 체크포인트로 추론하거나 평가할 때는 파일 경로를 직접 지정합니다. 5폴드 평가는 검증 디렉터리의 `fold1.csv`부터 `fold5.csv`까지와 체크포인트 디렉터리의 `best_model_fold1.pth`부터 `best_model_fold5.pth`까지를 짝지어 실행합니다.

```bash
./scripts/rock.sh predict /path/to/rock.jpg --checkpoint /path/to/best_model_fold1.pth
./scripts/rock.sh evaluate /path/to/fold1.csv --checkpoint /path/to/best_model_fold1.pth
./scripts/rock.sh evaluate-folds /path/to/validation-folds /path/to/checkpoints --batch-size 8
```

원본 데이터와 모델 체크포인트는 별도로 준비해야 합니다. 숫자 라벨과 클래스명의 대응은 원래 학습 폴더 이름의 정렬 순서로 결정됩니다. [데이터 및 검증 CSV 형식](docs/MODEL.md)

## 모델과 실험 결과

1. LAB 색 공간의 밝기 채널에 CLAHE를 적용하고, 입력을 224×224로 변환해 ImageNet 기준으로 정규화했습니다.
2. 사전 학습된 ConvNeXt-Base를 층화 5폴드로 학습했습니다. 가중치 샘플링과 클래스 가중 손실로 데이터 불균형을 보정했습니다.
3. 제출용 추론에는 테스트 시점 증강(TTA)과 폴드별 검증 정확도를 반영한 가중 앙상블을 적용했습니다.

발표 자료에 기록된 데이터 규모는 학습 이미지 **380,020장**, 테스트 이미지 **95,006장**입니다.

| 모델 | 발표 자료의 F1 |
| --- | ---: |
| ViT-L/16 | 0.66588 |
| Swin V2 | 0.72668 |
| **ConvNeXt-Base** | **0.83635** |

발표 자료 23쪽의 비교 결과입니다. 해당 표에는 F1 평균 방식과 평가 분할이 명시되어 있지 않습니다. [발표 자료 PDF](docs/presentation.pdf)

## 저장소 구성

- `scripts/rock.sh` — Ubuntu에서 사용하는 실행 및 반복 평가 명령
- `model.py` — 전처리, 학습, TTA 앙상블 및 제출 CSV 생성
- `predict.py`, `evaluate.py` — 체크포인트를 활용한 단일 이미지 추론 및 검증 데이터 평가
- `requirements.txt` — Python 의존성 정의
- `docs/` — 모델 상세 설명과 발표 PDF
