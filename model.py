import os
from glob import glob

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.amp import autocast, GradScaler


MODEL_NAME = 'convnext_base'
NUM_CLASSES = 7


# Data Preprocessing | LAB 밝기 채널의 국소 대비를 보정합니다.
def apply_clahe(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final_img


# Data Load | 이미지 파일을 읽고 CLAHE를 적용한 RGB 이미지를 반환합니다.
def read_rgb_image(path):
    img = cv2.imread(os.fspath(path))
    if img is None:
        raise ValueError(f'이미지를 읽을 수 없습니다: {path}')
    return apply_clahe(img)


# Data Preprocessing | 검증·추론에 공통으로 사용하는 224×224 변환과 ImageNet 정규화입니다.
def build_eval_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# Data Load | 학습·검증 이미지와 정답을 샘플 단위로 불러옵니다.
class RockDataset(Dataset):
    def __init__(self, file_list, labels=None, transform=None, mode='train'):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = read_rgb_image(img_path)
        if self.transform:
            img = self.transform(image=img)['image']
        if self.mode == 'test':
            return img
        else:
            return img, self.labels[idx]


# Data Load | 테스트 이미지를 읽고 지정된 TTA 변환을 적용합니다.
class TTADataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = read_rgb_image(img_path)
        img = self.transform(image=img)['image']
        return img


def main():
    # 설정 | Ubuntu 데이터 경로와 학습·추론 장치를 지정합니다.
    data_dir = '/home/kangjungmin1/open'
    train_dir = os.path.join(data_dir, 'train')
    test_csv_path = os.path.join(data_dir, 'test_modified.csv')
    sample_submission_path = os.path.join(data_dir, 'sample_submission.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda_amp = device.type == 'cuda'

    # Data Preprocessing | 학습용 무작위 증강 후 입력 크기와 정규화를 맞춥니다.
    transform_train = A.Compose([
        A.RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0), ratio=(0.8, 1.2), p=0.8),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.2, rotate_limit=20, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(p=1.0),
            A.CLAHE(p=1.0),
        ], p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(1, 32),
            hole_width_range=(1, 32),
            fill=0,
            p=0.5,
        ),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Data Preprocessing | 검증용 기본 변환과 테스트 시점 증강(TTA)을 구성합니다.
    transform_test = build_eval_transform()

    tta_transforms = [
        transform_test,
        A.Compose([A.HorizontalFlip(p=1.0)] + transform_test.transforms),
        A.Compose([A.RandomBrightnessContrast(p=1.0)] + transform_test.transforms),
        A.Compose([A.GaussianBlur(blur_limit=(3, 5), p=1.0)] + transform_test.transforms),
    ]

    # Data Load | 클래스 폴더를 이름순으로 정렬해 이미지 경로와 숫자 라벨을 수집합니다.
    rock_classes = sorted(os.listdir(train_dir))
    file_list = []
    labels = []
    for label_idx, rock_class in enumerate(rock_classes):
        files = glob(os.path.join(train_dir, rock_class, '*.jpg'))
        file_list.extend(files)
        labels.extend([label_idx] * len(files))
    labels = np.array(labels)

    # Train | 클래스 불균형을 보정할 손실 가중치와 층화 5폴드 분할을 준비합니다.
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(file_list, labels)):
        checkpoint_path = os.path.join(data_dir, f'best_model_fold{fold + 1}.pth')
        print(f"\n🌀 Fold {fold + 1}/5 시작")
        model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        scaler = GradScaler('cuda', enabled=use_cuda_amp)

        # Data Load | 현재 폴드의 학습·검증 배치를 구성하고 학습에 가중치 샘플링을 적용합니다.
        train_dataset = RockDataset([file_list[i] for i in train_idx], labels[train_idx], transform=transform_train)
        val_dataset = RockDataset([file_list[i] for i in val_idx], labels[val_idx], transform=transform_test)

        sample_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels[train_idx])
        sample_weights = sample_weights[labels[train_idx]]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        best_acc = -1.0
        counter = 0

        # Train | 배치별 손실을 계산하고 역전파로 모델을 업데이트합니다.
        for epoch in range(50):
            model.train()
            correct, total = 0, 0
            for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Fold {fold + 1}]", leave=False):
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                with autocast(device_type=device.type, enabled=use_cuda_amp):
                    preds = model(imgs)
                    loss = criterion(preds, lbls)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                correct += (preds.argmax(1) == lbls).sum().item()
                total += lbls.size(0)

            scheduler.step()

            # Train | 에폭별 검증 정확도를 계산합니다.
            model.eval()
            val_correct = 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    with autocast(device_type=device.type, enabled=use_cuda_amp):
                        outputs = model(imgs)
                    val_correct += (outputs.argmax(1) == lbls).sum().item()

            val_acc = val_correct / len(val_loader.dataset)
            print(f"Epoch {epoch + 1}: Train Acc={correct / total:.4f}, Val Acc={val_acc:.4f}")

            # Train | 정확도가 개선되면 체크포인트를 저장하고, 8에폭 동안 개선이 없으면 중단합니다.
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), checkpoint_path)
                counter = 0
            else:
                counter += 1
                if counter >= 8:
                    print("🚨 Early stopping")
                    break

        best_scores.append(best_acc)

    # Data Load | 제출 행 순서에 맞춰 테스트 이미지 경로를 불러옵니다.
    test_df = pd.read_csv(test_csv_path)
    test_files = test_df['img_path'].tolist()

    # Inference | 폴드별 체크포인트를 다시 로드해 테스트 이미지를 예측합니다.
    ensemble_probs = None
    total_score = sum(best_scores)
    for fold in range(5):
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES).to(device)
        checkpoint_path = os.path.join(data_dir, f'best_model_fold{fold + 1}.pth')
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model.eval()

        weight = best_scores[fold] / total_score if total_score else 1 / len(best_scores)
        print(f"[Fold {fold + 1}] 가중치: {weight:.4f}")

        total_preds = np.zeros((len(test_files), NUM_CLASSES))

        for tta_tf in tta_transforms:
            dataset = TTADataset(test_files, transform=tta_tf)
            loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
            fold_preds = []

            with torch.no_grad():
                for imgs in tqdm(loader, desc=f"Fold {fold + 1} TTA"):
                    imgs = imgs.to(device)
                    with autocast(device_type=device.type, enabled=use_cuda_amp):
                        outputs = model(imgs)
                    fold_preds.append(outputs.softmax(dim=1).cpu().numpy())

            total_preds += np.vstack(fold_preds)

        # Inference | TTA 확률을 평균한 뒤 폴드의 검증 정확도에 따른 가중치를 적용합니다.
        total_preds /= len(tta_transforms)
        total_preds *= weight

        if ensemble_probs is None:
            ensemble_probs = total_preds
        else:
            ensemble_probs += total_preds

    # Inference | 합산 확률이 가장 높은 클래스 번호를 제출 CSV에 저장합니다.
    final_preds = np.argmax(ensemble_probs, axis=1)
    submission = pd.read_csv(sample_submission_path)
    submission['target'] = final_preds
    submission.to_csv(os.path.join(data_dir, 'submit_convnext.csv'), index=False)

    print("✅ 전체 학습 + TTA 추론 완료 및 제출 파일 저장")


if __name__ == '__main__':
    main()
