# ViT 简单复现脚本

PyTorch 实现 "An Image is Worth 16x16 Words" (Dosovitskiy et al., ICLR 2021)，在 CIFAR-10 上训练。

## 训练

```bash
python train.py
```

100 epoch 完成后生成 `training.png` 和 `checkpoints/best.pth`。

## 文件

| 文件 | 内容 |
|------|------|
| `vit.py` | 模型：PatchEmbedding、Attention、Block、ViT |
| `data.py` | CIFAR-10 数据加载与增强 |
| `train.py` | 训练循环、评估、模型保存 |
| `utils.py` | AverageMeter、训练曲线 |

## 模型

默认 ViT-Tiny (5.36M)：patch=4, dim=192, depth=12, heads=3。也支持 `vit_small()` 和 `vit_base()`。
