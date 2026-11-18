import argparse
import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from model import NUM_CLASSES, RockDataset, build_eval_transform
from predict import load_model, select_device


# Data Load | 검증 CSV에서 이미지 경로와 정답 클래스 번호를 읽고 검증합니다.
def read_manifest(manifest):
    images = []
    labels = []
    with manifest.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        if not {'img_path', 'target'}.issubset(reader.fieldnames or []):
            raise ValueError('검증 CSV에 img_path와 target 열이 필요합니다.')

        for row in reader:
            image_path = (row['img_path'] or '').strip()
            if not image_path:
                raise ValueError(f'CSV {reader.line_num}행에 이미지 경로가 없습니다.')
            try:
                label = int(row['target'])
            except (TypeError, ValueError):
                raise ValueError(f'CSV {reader.line_num}행의 target은 정수여야 합니다.') from None
            if not 0 <= label < NUM_CLASSES:
                raise ValueError(f'CSV {reader.line_num}행의 target은 0~{NUM_CLASSES - 1}이어야 합니다.')
            images.append(manifest.parent / image_path)
            labels.append(label)

    if not images:
        raise ValueError('검증 CSV에 이미지가 없습니다.')
    return images, labels


def main():
    parser = argparse.ArgumentParser(description='라벨이 있는 이미지에서 체크포인트 하나를 평가합니다.')
    parser.add_argument('manifest', type=Path, help='img_path와 target 열이 있는 검증 CSV')
    parser.add_argument('--checkpoint', type=Path, required=True, help='폴드 체크포인트 경로')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size는 1 이상이어야 합니다.')

    try:
        device = select_device(args.device)
        images, labels = read_manifest(args.manifest)
        # Data Preprocessing | 검증용 전처리를 적용하고 CSV 순서대로 배치를 구성합니다.
        dataset = RockDataset(images, labels, transform=build_eval_transform())
        batches = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        model = load_model(args.checkpoint, device)

        # Inference | 체크포인트를 사용해 배치별 클래스 번호를 예측합니다.
        predictions = []
        with torch.inference_mode():
            for batch, _ in batches:
                predictions.extend(model(batch.to(device)).argmax(dim=1).tolist())
    except (OSError, ValueError, RuntimeError, csv.Error) as exc:
        parser.exit(1, f'평가 실행 오류: {exc}\n')

    # Evaluation | 정답과 예측을 비교해 정확도, macro F1, 클래스별 지표와 혼동행렬을 계산합니다.
    class_ids = list(range(NUM_CLASSES))
    report = classification_report(
        labels, predictions, labels=class_ids, output_dict=True, zero_division=0
    )
    result = {
        'num_images': len(labels),
        'accuracy': accuracy_score(labels, predictions),
        'macro_f1': report['macro avg']['f1-score'],
        'per_class': {class_id: report[str(class_id)] for class_id in class_ids},
        'confusion_matrix': confusion_matrix(labels, predictions, labels=class_ids).tolist(),
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
