## 用户手册

### 1. 安装指南

#### 系统要求

- Python 3.8+

- CUDA 11.3+ (如需GPU加速)

- 内存 ≥16GB (3D模型建议≥32GB)

  ##### 安装依赖

  ###### 1.创建 Conda 环境

  ```bash
  conda create -n cardiac python=3.8
  conda activate cardiac
  ```

  ###### 2.安装pytorch和相关依赖

  - CPU版本

  ```bash
  conda install pytorch torchvision torchaudio cpuonly -c pytorch
  ```

  - GPU版本

  ```bash
  conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
  ```

  ###### 3.安装其他依赖

  ```bash
  conda install h5py opencv-python scikit-learn tqdm matplotlib
  ```

  ###### 4.验证安装

  ```bash
  python -c "import torch; print(torch.__version__)"
  ```

​	

### 2. 使用说明

#### 数据准备

```plaintext
data/
└── ACDC_preprocessed/
    ├── ACDC_training_slices/  # 2D切片数据
    │   ├── patient001.h5
    │   └── ...
    └── ACDC_training_volumes/ # 3D体积数据
        ├── patient001.h5
        └── ...
```

#### 数据验证

```python
from utils import validate_dataset
validate_dataset("./data/ACDC_preprocessed")
```

#### 训练模型

```bash
# 基础训练（2D模型）
python train.py \
    --data_dir ./data/ACDC_preprocessed \
    --model 2d_unet \
    --epochs 50 \
    --batch_size 8

# 高级选项（3D模型）
python train.py \
    --data_dir ./data/ACDC_preprocessed \
    --model 3d_unet \
    --dim 3 \
    --patch_size 128 128 16 \
    --epochs 100
```



### 完整参数说明

| 参数           | 类型  | 必需 | 默认值 | 描述                   |
| -------------- | ----- | ---- | ------ | ---------------------- |
| `--data_dir`   | str   | 是   | 无     | 数据集根目录路径       |
| `--epochs`     | int   | 否   | 50     | 训练总轮数             |
| `--batch_size` | int   | 否   | 4      | 训练批大小             |
| `--lr`         | float | 否   | 1e-4   | 初始学习率             |
| `--dim`        | int   | 否   | 2      | 数据维度 (2或3)        |
| `--model`      | str   | 否   | "unet" | 模型类型 (unet/unet++) |
| `--resume`     | str   | 否   | 无     | 继续训练的检查点路径   |