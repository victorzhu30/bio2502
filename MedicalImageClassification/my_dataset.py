import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import warnings
import UNet_model_accessible

#   图像加载 → 预处理 → 模型前向传播 → 计算损失 → 反向传播 → 参数更新

#数据加载模块（已修复维度问题）
class MedicalSegmentationDataset(Dataset):
    def __init__(self, root_dir, mode='training', target_size=256):
        self.img_dir = os.path.normpath(os.path.join(root_dir, mode, "images"))
        self.mask_dir = os.path.normpath(os.path.join(root_dir, mode, "masks"))
        self.target_size = target_size
        # 获取所有.tif图像文件
        self.img_names = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.tif', '.tiff'))
        ])
        # 匹配图像和掩码
        self.valid_pairs = []
        for img_name in self.img_names:
            base_name = os.path.splitext(img_name)[0]
            possible_masks = [
                f for f in os.listdir(self.mask_dir)
                if f.startswith(base_name) and f.lower().endswith('.png')
            ]
            if possible_masks:
                self.valid_pairs.append((img_name, possible_masks[0]))
        print(f"成功加载 {len(self.valid_pairs)} 个有效图像-掩码对")
        # MRI专用标准化
        self.normalize = Normalize(mean=[0.5], std=[0.2])

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.valid_pairs[idx]

        try:
            # 加载图像和掩码
            image = Image.open(os.path.join(self.img_dir, img_name)).convert('L')
            mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert('L')
            # 调整尺寸
            image = image.resize((self.target_size, self.target_size))
            mask = mask.resize((self.target_size, self.target_size))
            # 转换为numpy数组
            image = np.array(image, dtype=np.float32)[None, ...] / 255.0  # [1, H, W]
            mask = np.array(mask, dtype=np.int64)
            mask = (mask > 0).astype(np.int64)[None, ...]  # 增加通道维度 [1, H, W]
            # 转换为Tensor并标准化
            image = torch.from_numpy(image)
            image = self.normalize(image)
            mask = torch.from_numpy(mask)

            return image, mask.squeeze(0)  # 返回 [C=1, H, W] -> [H, W]

        except Exception as e:
            print(f"加载 {img_name} 和 {mask_name} 时出错: {str(e)}")
            dummy_image = torch.rand(1, self.target_size, self.target_size)
            dummy_mask = torch.zeros(1, self.target_size, self.target_size, dtype=torch.int64)
            return dummy_image, dummy_mask.squeeze(0)

#损失函数
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred).squeeze(1)  # 压缩通道维度 [B,1,H,W] -> [B,H,W]
        intersection = (pred * target).sum()
        dice = (2. * intersection + 1e-5) / (pred.sum() + target.sum() + 1e-5)
        bce = F.binary_cross_entropy(pred, target.float())
        return 1 - dice + bce

#训练流程
def train_and_validate(model, root_dir, epochs=20, batch_size=4, target_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    # 数据加载
    train_set = MedicalSegmentationDataset(root_dir, 'training', target_size)
    test_set = MedicalSegmentationDataset(root_dir, 'test', target_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=2, num_workers=2, pin_memory=True)

    # 优化器和学习率调度
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceBCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    best_dice = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f'Train Epoch {epoch + 1}/{epochs}')
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 验证阶段
        val_dice = evaluate(model, test_loader, device)
        scheduler.step(val_dice)
        print(f'Epoch {epoch + 1} | Loss: {epoch_loss / len(train_loader):.4f} | Val Dice: {val_dice:.4f}')

        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'新的最佳模型保存，Dice: {best_dice:.4f}')

def evaluate(model, loader, device):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = torch.sigmoid(model(images)['out']).squeeze(1)
            preds = (outputs > 0.5).float()
            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum()
            dice = (2. * intersection + 1e-8) / (union + 1e-8)
            dice_scores.append(dice.item())
    return sum(dice_scores) / len(dice_scores)

if __name__ == '__main__':
    model = UNet_model_accessible.UNet(in_channels=1, num_classes=1, base_channels=32)
    print(f'使用设备: {"CUDA" if torch.cuda.is_available() else "CPU"}')
    data_root = "D:/code/deep_learning/medical_segmentation"
    train_and_validate(model, data_root, epochs=20, batch_size=4, target_size=256)