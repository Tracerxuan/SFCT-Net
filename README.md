# SFCT-Net

A Spatial–Frequency Collaborative Transformer for Robust Breast Ultrasound Tumor Segmentation
基于超像素和 SegFormer 的乳腺肿瘤图像分割项目。

## 项目结构

```text
SFCT-Net/
├── datasets/            # 数据集加载模块
│   └── breast.py        # BUSI、STU 数据集的 Dataset 实现
├── models/              # 模型定义模块
│   ├── sfct_net.py      # SFCT-Net (Stage_SSM) 核心模型
│   └── wtconv.py        # 小波卷积层 (WTConv2d)
├── utils/               # 工具函数模块
│   ├── dataloader_alone.py # 序列数据加载器[train.py](train.py)
│   ├── evaluation.py    # 评估指标计算 (Dice, IoU, Precision, Recall, HD95)
│   ├── loss.py          # 损失函数 (Dice Loss 等)
│   ├── metrics.py       # 评估指标核心计算
│   └── predict.py       # 预测与结果保存
└── train.py       # 5折交叉验证训练与评估脚本
```

## 环境要求

- Python 3.8+
- PyTorch >= 1.12
- torchvision
- opencv-python
- albumentations
- medpy (用于评估指标计算)
- scikit-learn
- tqdm
- GeodisTK (用于部分 Hausdorff 距离计算)
- mindspore (可选，用于 HausdorffDistance 评估指标)


### 训练模型

  ```
- **5折交叉验证：**
  ```bash
  python train_5fold.py
  ```
