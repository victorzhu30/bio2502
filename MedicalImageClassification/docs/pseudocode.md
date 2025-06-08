# 算法伪代码文档

## 1. U-Net 核心算法

### 1.1 前向传播流程
```pseudocode
function FORWARD_PASS(input):
    // 编码器路径
    x1 = DoubleConv(input, base_channels)
    x2 = Downsample(x1, base_channels*2)
    x3 = Downsample(x2, base_channels*4)
    x4 = Downsample(x3, base_channels*8)
    x5 = Bottleneck(x4, base_channels*16)
    
    // 解码器路径
    x = Upsample(x5, x4, base_channels*8)
    x = Upsample(x, x3, base_channels*4)
    x = Upsample(x, x2, base_channels*2)
    x = Upsample(x, x1, base_channels)
    
    // 输出层
    output = Conv1x1(x, num_classes)
    return output
```

## 2.训练流程

### 2.1 主训练循环

```pseudocode
procedure TRAIN_MODEL(dataset, epochs):
    model = InitializeUNet()
    optimizer = Adam(lr=1e-4)
    loss_fn = DiceBCELoss()
    
    for epoch in 1..epochs:
        for batch in dataset:
            // 前向传播
            pred = model(batch.images)
            loss = loss_fn(pred, batch.masks)
            
            // 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        // 验证阶段
        val_score = EVALUATE(model, val_dataset)
        if val_score > best_score:
            SAVE_MODEL(model)
```

## 3.关键函数

### 3.1 Dice损失计算

$$
Dice系数公式：
Dice = \frac{2|X \cap Y|}{|X| + |Y|}
$$



```pseudocode
function DICE_LOSS(pred, target):
    smooth = 1e-5
    intersection = SUM(pred * target)
    union = SUM(pred) + SUM(target)
    return 1 - (2 * intersection + smooth) / (union + smooth)
```

### 3.2 数据增强

```pseudocode
procedure AUGMENT_SAMPLE(image, mask):
    // 随机旋转
    if RAND() < 0.5:
        angle = 90 * RAND_INT(0,3)
        image = ROTATE(image, angle)
        mask = ROTATE(mask, angle)
    
    // 弹性形变
    if RAND() < 0.3:
        displacement = GENERATE_DISPLACEMENT_FIELD()
        image = APPLY_DEFORMATION(image, displacement)
        mask = APPLY_DEFORMATION(mask, displacement, INTER_NEAREST)
```

