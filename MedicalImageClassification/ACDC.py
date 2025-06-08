import torch
import cv2
import os
import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from cardiac_UNet import CardiacUNet
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from medical_segmentation.my_dataset import DiceBCELoss


class ACDCDataset(Dataset):
    def __init__(self, data_dir, mode='train', target_size=(256, 256), dim=2, val_ratio=0.2, transform=None):
        self.data_dir = os.path.join(data_dir, f'ACDC_training_{"slices" if dim == 2 else "volumes"}')
        self.target_size = target_size
        self.transform = transform
        self.is_3d = (dim == 3)
        # 按患者ID划分
        all_files = self._load_valid_files()
        patient_ids = sorted(list(set(
            os.path.basename(f).split('_')[0] for f in all_files
        )))

        train_ids, val_ids = train_test_split(patient_ids, test_size=val_ratio, random_state=42)

        if mode == 'train':
            self.samples = [f for f in all_files if os.path.basename(f).split('_')[0] in train_ids]
        elif mode == 'val':
            self.samples = [f for f in all_files if os.path.basename(f).split('_')[0] in val_ids]
        else:
            self.samples = all_files

    def _load_valid_files(self):
        valid_files = []
        for fname in os.listdir(self.data_dir):
            if not fname.endswith(".h5"):
                continue
            with h5py.File(os.path.join(self.data_dir, fname), 'r') as f:
                if 'image' in f and 'label' in f:
                    valid_files.append(os.path.join(self.data_dir, fname))
        return valid_files

    def __getitem__(self,index):
        with h5py.File(self.samples[index], 'r') as f:
            #访问HDF5文件中名为data的数据集（dataset），类似于字典的键值访问，但值可能是多维数组
            #f['image']返回的是HDF5数据集对象（未加载数据），[()]表示立即将所有数据加载到内存（相当于numpy.array(f['image'])）
            image = f['image'][()]
            mask = f['label'][()]
            # 医学影像特性：
            # MRI图像的原始值（灰度）没有固定范围，不同扫描仪/协议产生的数值差异大
            # 标准化目的：
            # 使数据均值为0、标准差为1，提升模型训练稳定性
            # +1e-8防止除零错误（某些区域可能标准差为0）
            image = (image - image.mean()) / image.std()
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_LINEAR)
            sample =  {
                'image': torch.FloatTensor(image).unsqueeze(0),  # [1, H, W]  MRI 本质是单通道（灰度），但模型需要统一的 4D 输入格式 [B, C, H, W]，所以需要.unsqueeze(0)来增加channel维度
                'mask': torch.LongTensor(mask)  # [H, W]  分割任务的特性：标签每个像素值为整数类别ID，不需要通道维度，损失函数（如 CrossEntropyLoss）直接接受 2D 标签
            }
            if self.transform:
                sample = self.transform(sample)  # 应用数据增强
            return sample
            # 若为3D体积数据（如 [D, H, W]）
            # image_tensor = torch.FloatTensor(image).unsqueeze(0)  # [1, D, H, W]
            # mask_tensor = torch.LongTensor(mask)                  # [D, H, W]
            # 3D模型需要5D输入 [B, C, D, H, W]
            #
            # 假设是多模态数据（如 MRI+T1）
            # image = np.stack([mri, t1], axis=0)  # [2, H, W]
            # image_tensor = torch.FloatTensor(image)  # 自动有通道维度
    def __len__(self):
        return len(self.samples)
#---医学影像增强技巧---
class MedicalTransform:
    def __init__(self, p=0.5, is_3d=False):
        self.p = p
        self.is_3d = is_3d

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        #随机旋转90°倍数，避免插值伪影
        if torch.rand(1) < self.p:
            k = torch.randint(0, 4, (1,)).item()
            image = torch.rot90(image, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])
        #随机翻转
        if torch.rand(1) < self.p:
            dim = -1 if torch.rand(1) < 0.5 else -2  # 直接选择水平或垂直翻转
            image = torch.flip(image, dims=[dim])
            mask = torch.flip(mask, dims=[dim])
        #弹性形变（模拟心脏运动）
        if torch.rand(1) < self.p and not self.is_3d:
            image, mask = self._elastic_deform(image, mask)

        return {'image': image, 'mask': mask}

    def _elastic_deform(self, image, mask):
        """弹性形变（2D专用）"""
        alpha = (torch.rand(1) * 100 + 100).item()
        sigma = (torch.rand(1) * 5 + 5).item()

        # 生成随机位移场
        h, w = image.shape[-2], image.shape[-1]
        dx = torch.randn(h, w) * alpha
        dy = torch.randn(h, w) * alpha

        # 高斯模糊位移场
        dx = cv2.GaussianBlur(dx.numpy(), (0, 0), sigma)
        dy = cv2.GaussianBlur(dy.numpy(), (0, 0), sigma)

        # 生成映射网格
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x + dx).astype(np.float32)
        map_y = (map_y + dy).astype(np.float32)

        # 处理图像
        image_np = image.squeeze().numpy()  # 移除 batch 和 channel 维度 -> [H, W]
        image_np = image_np.astype(np.float32)  # 确保是 float32
        image_deformed = cv2.remap(
            image_np, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        # 处理标签（用最近邻插值）
        mask_np = mask.squeeze().numpy()  # [H, W]
        mask_np = mask_np.astype(np.float32)  # 转为 float32（OpenCV 要求）
        mask_deformed = cv2.remap(
            mask_np, map_x, map_y,
            interpolation=cv2.INTER_NEAREST,  # 分割标签必须用最近邻
            borderMode=cv2.BORDER_REFLECT
        )

        # 转回 Tensor
        return (
            torch.FloatTensor(image_deformed).unsqueeze(0),  # [1, H, W]
            torch.LongTensor(mask_deformed)                   # [H, W]
        )

#---训练---
def train_acdc(data_dir, batch_size=8, epochs=50, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #数据加载
    train_set = ACDCDataset(data_dir=r"D:\kaggle\ACDC_preprocessed", mode="train", transform=MedicalTransform(p=0.5))
    val_set = ACDCDataset(data_dir=r"D:\kaggle\ACDC_preprocessed", mode="val")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size // 2)

    model = CardiacUNet(dim=2 if not train_set.is_3d else 3).to(device)

    #损失函数(Dice + CrossEntropy)
    criterion = DiceCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            images, masks = batch['image'].to(device), batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        #验证
        val_dice = evaluate(model, val_loader, device)
        print(f"Val Dice: {val_dice:.4f}")

class DiceCELoss(nn.Module):
    def forward(self, pred, target):
        #Dice损失
        pred_probs = torch.softmax(pred, dim=1)
        target_oh = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)
        intersection = (pred_probs * target_oh).sum(dim=(2,3))
        union = pred_probs.sum(dim=(2,3)) + target_oh.sum(dim=(2,3))
        dice_loss = 1 - (2 * intersection / (union + 0.000001)).mean()

        #交叉熵
        ce_loss = F.cross_entropy(pred, target)
        return dice_loss + ce_loss

def evaluate(model, data_loader, device):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for batch in data_loader:
            images, masks = batch['image'].to(device), batch['mask'].to(device)
            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)
            dice = dice_score(preds, masks)
            dice_scores.append(dice)
    return np.mean(dice_scores)

def dice_score(pred, target):
    """Dice系数"""
    smooth = 0.000001
    intersection = (pred == target).sum()
    union = pred.numel() + target.numel()
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.item()

#---可视化分析---
def visualize_samples(dataset, n=3):
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    for i in range(n):
        sample = dataset[i]
        axes[i,0].imshow(sample['image'].squeeze(), cmap='gray')
        axes[i,0].set_title("MRI Image")
        axes[i,1].imshow(sample['mask'].squeeze(), cmap='jet')
        axes[i,1].set_title("Ground Truth")
        axes[i,2].hist(sample['mask'].flatten(), bins=50)
        axes[i,2].set_title("Intensity Distribution")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_set = ACDCDataset(data_dir="D:\kaggle\ACDC_preprocessed", mode="train")
    visualize_samples(train_set)

    train_acdc(data_dir=r"D:\kaggle\ACDC_preprocessed", epochs=20)


