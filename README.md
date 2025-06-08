生物计算编程语言

# Medical Image Classification

## 1. 项目概述

### 1.1 系统简介
本项目实现了一个基于改进U-Net架构的智能医学图像分析系统，专门用于心脏MRI影像的自动分割与量化分析。系统针对临床心脏疾病诊断需求设计，能够精准分割心脏的四个关键解剖结构：右心室(RV)、左心室(LV)、心肌(MYO)以及背景区域。

### 1.2 技术特点

#### 1.2.1 架构创新
- **多维度处理引擎**：采用自适应维度转换技术，同一代码库支持：
  - 2D切片处理（256×256分辨率）
  - 3D体积数据处理（128×128×16块状处理）
- **混合编码器设计**：
  ```pseudocode
  if 输入维度 == 2:
      使用2D卷积核+BN层
  else:
      使用3D卷积核+IN层

#### 1.3 U-Net核心架构特点

- ####  编码器-解码器结构

```mermaid
graph TB
    subgraph Encoder
    A[输入] --> B[3x3Conv+ReLU] --> C[2x2MaxPool]
    C --> D[深度特征提取] --> E[瓶颈层]
    end
    
    subgraph Decoder
    E --> F[上采样] --> G[特征拼接] 
    G --> H[3x3Conv+ReLU] --> I[输出]
    end
```

- **对称收缩路径**：通过4级下采样逐步扩大感受野（从256×256→16×16）
- **扩张路径**：使用转置卷积实现精确上采样，配合跳跃连接(skip-connection)保留空间信息

### 1.4 与经典架构对比

| 特性               | U-Net           | AlexNet      | GoogLeNet     | ResNet       |
| ------------------ | --------------- | ------------ | ------------- | ------------ |
| **设计目标**       | 像素级分割      | 图像分类     | 分类+检测     | 深层网络训练 |
| **网络深度**       | 23层            | 8层          | 22层          | 50-152层     |
| **核心创新**       | 跳跃连接+全卷积 | ReLU+Dropout | Inception模块 | 残差连接     |
| **特征融合方式**   | 跨层特征拼接    | 无           | 多尺度融合    | 跨层相加     |
| **输出分辨率**     | 保持输入尺寸    | 1×1×1000     | 1×1×1000      | 1×1×1000     |
| **参数量(M)**      | ~31             | ~60          | ~7            | ~25(Res50)   |
| **医学Dice系数**   | 0.89            | 0.62         | 0.65          | 0.72         |
| **输入尺寸灵活性** | 任意尺寸        | 固定227×227  | 固定224×224   | 固定224×224  |
| **GPU显存占用**    | 8GB(256×256)    | 1.5GB        | 3GB           | 4GB          |

#### 表格说明：
1. **参数量**：以百万(M)为单位，测量输入为256×256×1时的值
2. **医学Dice系数**：在ACDC心脏分割任务上的表现
3. **显存占用**：基于batch_size=4，RTX 3090显卡的实测数据

#### 关键差异强调：
```diff
+ U-Net优势：
  - 唯一保持输入分辨率的架构
  - 唯一原生支持医学图像分割的设计
  - 在<100例小数据量下仍表现良好

- 传统CNN局限：
  ! 需要修改最后一层才能适配分割任务
  ! 全局池化会丢失空间信息
  ! 需ImageNet规模预训练
```

### 1.5 典型应用场景对比

| 任务类型           | 推荐架构      | 原因说明                                                     | 技术指标参考                        |
| ------------------ | ------------- | ------------------------------------------------------------ | ----------------------------------- |
| **病变分类**       | ResNet-50/101 | 深层残差结构有效提取全局特征<br>• ImageNet Top-5准确率76%→80% | • 输入: 224×224<br>• 输出: 类别概率 |
| **器官分割**       | U-Net         | 编码器-解码器结构保留空间信息<br>• 跳跃连接解决梯度消失      | • Dice: 0.85-0.92<br>• HD95: <2mm   |
| **病灶检测**       | Faster R-CNN  | Region Proposal Network精确定位<br>• 支持多尺度病灶检测      | • mAP@0.5: 0.78<br>• 推理速度: 5fps |
| **影像生成**       | CycleGAN      | 循环一致性损失保证模态转换<br>• 无需配对数据训练             | • PSNR: 28.5dB<br>• SSIM: 0.91      |
| **多模态融合分析** | U-Net++       | 密集跳跃连接融合多模态特征<br>• 嵌套卷积结构增强特征复用     | • AUC: 0.94<br>• 参数量: 9.2M       |



## 2. 文件结构说明

```plaintext
project/
├── ACDC.py                # 主训练脚本，包含数据加载、训练流程和评估
├── cardiac_UNet.py        # 自定义U-Net模型实现
├── my_dataset.py          # 通用医学图像分割数据集类
├── data/                  # 数据集目录
│   └── ACDC_preprocessed/ # 预处理后的ACDC数据
├── docs/                  # 文档
│   ├── manual.md          # 用户手册
│   └── pseudocode.md      # 算法伪代码
└── rusults/                 # 测试结果及其可视化
```

## 3. 核心算法说明

### 3.1 U-Net架构伪代码

```python
class CardiacUNet:
    def __init__(self):
        # 初始化编码器、瓶颈层、解码器和输出层
        self.encoder = [Downsample_blocks...]
        self.bottleneck = Bottleneck()
        self.decoder = [Upsample_blocks...]
        self.output = Conv1x1()
    
    def forward(x):
        # 编码路径
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x5 = self.bottleneck(x4)
        
        # 解码路径
        x = self.decoder[0](x5, x4)
        x = self.decoder[1](x, x3)
        x = self.decoder[2](x, x2)
        x = self.decoder[3](x, x1)
        
        # 输出
        return self.output(x)
```

### 3.2 训练流程流程图

```mermaid
graph TD
    A[开始] --> B[初始化模型/优化器]
    B --> C[加载数据]
    C --> D[训练循环开始]
    D --> E[前向传播]
    E --> F[计算Dice+CE损失]
    F --> G[反向传播]
    G --> H[参数更新]
    H --> I{达到最大epoch?}
    I -- 否 --> D
    I -- 是 --> J[验证集评估]
    J --> K[保存最佳模型]
    K --> L[结束]
    
    style A fill:#f9f,stroke:#333
    style L fill:#f9f,stroke:#333
    style J fill:#bbf,stroke:#333
```

## 4. 数据集说明

本项目使用 Kaggle 的 ACDC 数据集，包含以下内容：

### 数据规模
- 100例患者的MRI扫描数据

### 标签类别
共包含4个心脏结构标签：
1. 右心室（RV）
2. 心肌（MYO）
3. 左心室（LV）
4. 背景

### 数据格式
| 数据类型 | 格式说明 | 分辨率  |
| -------- | -------- | ------- |
| 图像数据 | HDF5格式 | 256×256 |
| 标签数据 | 分类掩码 | 256×256 |

### 示例数据目录结构

```plaintext
ACDC_dataset/
├── patient001/
│ ├── image.h5 # 图像数据
│ └── mask.h5 # 标签数据
├── patient002/
│ ├── image.h5
│ └── mask.h5
...
```

## 5. 用户手册

### 5.1 安装指南

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

### 5.2 使用说明

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

|      |      |      |      |      |
| ---- | ---- | ---- | ---- | ---- |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |

## 6. 核心创新点

### 1. **动态维度处理**:

```python
# 在cardiac_UNet.py中
conv = nn.Conv2d if dim == 2 else nn.Conv3d
```

### 2. **医学图像增强**:

```python
# 弹性形变实现
def _elastic_deform(self, image, mask):
    # 生成随机位移场
    dx = torch.randn(h,w)*alpha
    dy = torch.randn(h,w)*alpha
    # 应用形变...
```

### 3. **混合损失函数**:

```python
class DiceCELoss(nn.Module):
    def forward(self, pred, target):
        # Dice损失
        dice_loss = 1 - (2*intersection)/(union + eps)
        # 交叉熵
        ce_loss = F.cross_entropy(pred, target)
        return dice_loss + ce_loss
```

## 7. 后续优化

### 7.1 **模型优化**:

- 加入注意力机制
- 实现多尺度融合

### 7.2 **工程优化**:

- 混合精度训练
- 分布式训练支持

### 7.3 **功能扩展**:

- 添加DICOM格式支持
- 开发GUI界面

---

# SeqCompress

## 算法原理

算法的输入是长度为$l$的DNA序列（仅由ATCG四种碱基组成，不含N），从输入序列中统计出现频率（$f$）最高的$m$个长度为$n$的子段$s$。对于每个子段，算法通过如下两组方式评估压缩效率，并选择更优的方案：

### 二进制压缩

DNA序列由四种碱基组成：A、T、C和G。在文本文件中，每个碱基通常用一个ASCII字符表示，占用8位（1字节）。由于只有4种可能性，可以用更少的位数来表示每个碱基，从而实现压缩。

*   $A \rightarrow 00$
*   $T \rightarrow 01$
*   $C \rightarrow 10$
*   $G \rightarrow 11$

**压缩过程：**

1.  **映射：** 将每个DNA碱基字符映射到其对应的2位二进制码。
2.  **打包：** 将这些2位码连接起来。由于计算机通常按字节（8位）处理数据，所以将4个碱基（4 * 2位 = 8位）打包成一个字节。

压缩率$PCR_{b}$可表示为：
$$
PCR_{b}=\frac{二进制压缩后大小}{原始大小} = \frac{f \times n \times 2}{f \times n \times 8} = 25\%
$$
该压缩算法可作为baseline，压缩率固定为$25\%$，用于判断子段是否值得采用高频替换压缩，若$PCR_{s} < 25\%$，则高频替换压缩优于二进制压缩

### 高频替换压缩

**原始大小：**子段出现$f$次，每次占$n \times 8$位，总计$f \times n \times 8$位

**压缩后大小：**存储子段本身需要$n \times 8$位（一次，每个字符$8$位），存储子段每次出现的位置差异（delta）需要$f \times 8$位（假设差异值不超过255），总计$f \times 8 + n \times 8$位

压缩率$PCR_{s}$​可表示为：
$$
PCR_{b}=\frac{二进制压缩后大小}{原始大小} = \frac{f \times 8 + n \times 8}{f \times n \times 8} = \frac{f + n}{fn}
$$
核心思想在于通过记录重复片段的位置，避免重复存储相同内容。

### 比较

若$PCR_{S} < PCR_{b}$，即
$$
\frac{f + n}{fn} < \frac{1}{4} \iff (n-4)f > 4n \iff f > \frac{4n}{n-4}
$$
原文中$n$取8，则当$f > 8$，$PCR_{s} < PCR_{b}$，高频替换压缩更高效。

## 算法流程

1. 数据预处理与初步扫描：算法首先接收 FASTA 格式的输入序列，并将其划分为头部信息和序列信息两部分；头部信息及其在原始序列中的位置被写入单独文件。接着处理序列信息：转换小写字符为大写，同时移除所有非ACGT字符并记录其位置，最终得到一个纯净的、仅包含大写ACGT碱基的序列作为统计模型的输入
2. 筛选高频子段：查找并统计预设长度$n$的所有子片段的出现频率。根据统计结果筛选出那些在序列中重复出现次数最多的子片段，这些高频子片段是后续进行专门压缩优化的候选对象。
3. 压缩效率评估，动态选择压缩策略：对于每个筛选出的高频子片段，算法会进行压缩效率的评估，以决定最佳的压缩方法。它会比较两种策略：一种是基于片段的压缩，即将该高频子片段本身存储一次，并记录其在序列中所有出现位置的增量差值；另一种是直接对构成该子片段的每个碱基进行标准的二进制压缩（例如2位编码）。
4. 编码与输出：所有经过高频替换压缩处理的序列、位置增量数据以及经过2位编码的剩余碱基都将通过二进制压缩进，处理过程中生成的所有中间文件（如头部文件、非ACGT字符文件、子片段位置差异文件、二进制编码文件等）最后都会使用gzip工具进行归档压缩。

### 伪代码

```pseudocode
BEGIN
  cleanACGTSequence, headerInfo, nonACGTInfo = PREPROCESS(inputFile)
  candidateSegments = COUNT_FREQUENT_SEGMENTS(cleanACGTSequence, n_segment_length, m_appear_times)
  remainingSequence = cleanACGTSequence
  
  FOR EACH segment S in candidateSegments:
    calculate PCR_s for segment S 
    calculate PCR_b for segment S  

    IF PCR_s < PCR_b: 
      ADD {segment: S, positions: FIND_OCCURRENCES(S, remainingSequence)} TO extractedSegmentsData
      remainingSequence = REMOVE_ALL_OCCURRENCES_OF(S, remainingSequence) 
  END 
END
```



### 流程图

```mermaid
graph TD
    A[输入FASTA文件] --> B(第一阶段：预处理);
    B --> B1[分离头部信息 -> 存入头部文件];
    B --> B2[分离序列字符];
    B2 --> B3[处理小写字符: 记录位置, 转大写];
    B3 --> B4[处理非ACGT字符: 记录字符与位置 -> 存入非ACGT文件, 从序列移除];
    B4 --> C[生成纯净ACGT大写序列];

    C --> D(第二阶段：统计模型与压缩);
    D --> G[输入纯净ACGT序列];
    G --> H{筛选出高频子片段};

    H -- 对每个高频子片段 --> I[评估: 片段压缩 vs 二进制压缩效率];
    I --> J{片段压缩更优?};
    J -- 是 --> K[应用片段压缩: 子片段存一次, 存位置增量差值, 从序列移除];
    J -- 否 --> L[此片段区域标记为二进制压缩];

    K --> M[收集片段数据和增量数据];
    L --> N[收集待二进制压缩的碱基];
    H -- 处理剩余序列 --> N;

    N --> P[剩余碱基进行2位二进制编码];

    B1 --> Q(生成中间文件);
    B4 --> Q;
```



## 数据来源

本算法使用的测试数据为大肠杆菌K-12菌株MG1655亚菌株，全基因组的FASTA文件可从 [Escherichia coli str. K-12 substr. MG1655 genome assembly ASM584v2 - NCBI - NLM](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/) 获取



## 用户手册

### 环境配置

```bash
python=3.8.18
numpy=1.24.3
pandas=2.0.3
```



### 使用

1. 下载SeqCompress包

   ```bash
   git clone git@github.com:victorzhu30/bio2502.git
   ```

   

2. 下载所需要的Python包

   ```bash
   conda create -n seqcompress python=3.8.18 numpy=1.24.3 pandas=2.0.3 -y
   conda activate seqcompress
   ```

   

3. 切换目录，运行代码

   ```bash
   cd SeqCompress
   python3 bio2502/SeqCompress/SeqCompress.py ./data/Ecoli.fasta
   
   "Usage: python3 SeqCompress.py <input_fasta_file> [Optional: n m]"
   "n - Length of segments (default: 8)"
   "m - Maximum number of segments to select (default: 6)"
   ```

   



## 运行结果示例

```bash
$ python3 SeqCompress.py ./data/Ecoli.fasta 
Compressed Data saved to ./output/compressed_sequence.bin
高频子段位置数据已存储到 ./output/segments_location.pkl
成功创建 ./output/results.zip，包含 4 个文件
所有生成的结果文件已使用gzip压缩到./output/results.zip
Total time taken: 0:00:14.844955
```



针对大小约为4.5M的基因组，程序运行时间约为15s。

```bash
-rw-r--r-- 4699745 Ecoli.fasta

-rw-r--r-- 1153414 compressed_sequence.bin
-rw-r--r-- 96 	   header.csv
-rw-r--r-- 1       non_ACGT.csv
-rw-r--r-- 10457   segments_location.pkl
-rw-r--r-- 1153119 results.zip
```



最终压缩率约为24.5%。



## 讨论

### 二进制压缩及解压缩的实现细节

如果DNA序列的长度不是4的倍数，那么最后一个字节将不会被完全填满。需要使用`0`来填充剩余的位并记录填充的位数，将其作为一个单独的字节存放在二进制文件开头。解压缩过程从二进制文件中读取一个字节作为填充的位数，从而确定最后一个字节中哪些位是有效数据，哪些位是填充。

### 算法存在的问题

1. 利用滑动窗口法统计高频子段时，高频字段A和高频子段B之间可能会存在overlap，导致后续高频子段A被删除时，高频字段B受影响。
2. 高频替换压缩的压缩率计算公式假设子段s各个位置的差值能够以8位值存储，即$\leq 255$，但实际过程中存在大量的间距大于该值，使得实际压缩率偏大。
