import argparse
import json
from pathlib import Path

import timm
import torch

from model import MODEL_NAME, NUM_CLASSES, build_eval_transform, read_rgb_image


def select_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA를 사용할 수 없습니다. --device cpu를 지정하세요.')
    return torch.device(name)


# Inference | 학습된 가중치를 불러와 평가 모드로 전환합니다.
def load_model(checkpoint, device):
    state_dict = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description='체크포인트로 자갈 이미지 한 장을 분류합니다.')
    parser.add_argument('image', type=Path, help='입력 이미지 경로')
    parser.add_argument('--checkpoint', type=Path, required=True, help='폴드 체크포인트 경로')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()

    try:
        device = select_device(args.device)
        # Data Load / Data Preprocessing | 이미지 한 장을 읽어 검증과 같은 전처리를 적용합니다.
        image = build_eval_transform()(image=read_rgb_image(args.image))['image']
        model = load_model(args.checkpoint, device)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'추론 입력 또는 모델 로드 오류: {exc}\n')

    # Inference | 한 장을 배치 형태로 입력하고 가장 높은 확률의 클래스를 선택합니다.
    with torch.inference_mode():
        probabilities = model(image.unsqueeze(0).to(device)).softmax(dim=1)[0]
        probability, class_id = probabilities.max(dim=0)

    print(json.dumps({'class_id': class_id.item(), 'probability': probability.item()}))


if __name__ == '__main__':
    main()
