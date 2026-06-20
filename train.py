"""
训练脚本
数据目录结构：
    data/augmented/
        anormaly/
            bent_wire/          ← 类别 0
            cable_swap/         ← 类别 1
            combined/           ← 类别 2
            cut_inner_insulation/ ← 类别 3
            cut_outer_insulation/ ← 类别 4
            missing_cable/      ← 类别 5
            missing_wire/       ← 类别 6
            poke_insulation/    ← 类别 7
        normal/                 ← 类别 8（正常样本）

类别映射：每种异常一个类别，最后一个类别为正常样本
损失函数：CrossEntropyLoss（不做归一化和预处理）

使用方式：
    直接运行：python train.py
    或在解释器中：from train import train; train()
"""

import os
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import ResNet50


# ======================== 日志 ========================

class Logger:
    """同时输出到终端和文件"""

    def __init__(self, log_dir='./logs'):
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = log_path / f'train_{timestamp}.log'

        self.f = open(self.log_file, 'w', encoding='utf-8')
        print(f"日志文件: {self.log_file}")

    def log(self, msg):
        print(msg)
        self.f.write(msg + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


# ======================== 热力图可视化 ========================

def save_cam_heatmap(image_tensor, cam_tensor, true_idx, pred_idx, class_names, save_path):
    """
    将 CAM 类激活图叠加到原图上，保存热力图

    Args:
        image_tensor: 原图 (C, H, W) float tensor，未归一化，RGB
        cam_tensor: CAM (num_classes, H, W) float tensor
        true_idx: 真实类别索引
        pred_idx: 预测类别索引
        class_names: 类别名列表
        save_path: 保存路径
    """
    # 转 numpy
    img = image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # (H, W, 3) RGB
    cam = cam_tensor[true_idx].cpu().numpy()  # (H, W)

    # 归一化 CAM 到 [0, 255]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = (cam * 255).astype(np.uint8)

    # 转热力图
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # CAM 尺寸可能与原图不同（resolution_factor < 1 时），resize 到原图尺寸
    if heatmap.shape[:2] != img.shape[:2]:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 叠加原图和热力图
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    # 拼接：原图 | 热力图 | 叠加图
    h, w = img.shape[:2]
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, :w] = img
    canvas[:, w:2*w] = heatmap
    canvas[:, 2*w:] = overlay

    # 添加文字标注：真实类别 + 预测类别
    true_name = class_names[true_idx]
    pred_name = class_names[pred_idx]
    match = "OK" if true_idx == pred_idx else "WRONG"
    label = f"True: {true_name} | Pred: {pred_name} [{match}]"
    cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    # 保存（BGR）
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), canvas_bgr)


def save_cam_all_classes(image_tensor, cam_tensor, true_idx, pred_idx, class_names, save_path):
    """
    为每个类别生成热力图并拼接成一张大图

    Args:
        image_tensor: 原图 (C, H, W) float tensor
        cam_tensor: CAM (num_classes, H, W) float tensor
        true_idx: 真实类别索引
        pred_idx: 预测类别索引
        class_names: 类别名列表
        save_path: 保存路径
    """
    num_classes = len(class_names)
    img = image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    h, w = img.shape[:2]

    # 每行放原图+各类别热力图，按列排
    # 第一列：原图，后续列：各类别 CAM 叠加
    cols = num_classes + 1
    canvas = np.zeros((h + 40, w * cols, 3), dtype=np.uint8)

    # 原图
    canvas[40:40+h, :w] = img
    cv2.putText(canvas, f"True:{class_names[true_idx]}", (5, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 各类别热力图
    for cls_idx in range(num_classes):
        cam = cam_tensor[cls_idx].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = (cam * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

        x_offset = (cls_idx + 1) * w
        canvas[40:40+h, x_offset:x_offset+w] = overlay

        # 标注类别名，预测类别标红
        color = (0, 200, 0) if cls_idx == true_idx else (200, 200, 200)
        if cls_idx == pred_idx:
            color = (200, 50, 50)
        cv2.putText(canvas, class_names[cls_idx], (x_offset + 5, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), canvas_bgr)


# ======================== 数据集 ========================

class AnomalyDataset(Dataset):
    """
    异常检测数据集
    自动扫描 data_dir 下的目录结构，建立类别映射
    """

    def __init__(self, data_dir):
        """
        Args:
            data_dir: 数据根目录，如 './data/augmented'
        """
        self.data_dir = Path(data_dir)
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        # 扫描目录，建立类别映射
        self.class_names = []   # 类别名列表
        self.class_to_idx = {}  # 类别名 -> 索引
        self.samples = []       # [(图像路径, 类别索引), ...]

        self._scan_directory()

    def _scan_directory(self):
        """扫描目录结构，建立类别映射"""
        # 1. 先收集 anormaly/ 下的子目录（异常类别）
        anormaly_dir = self.data_dir / 'anormaly'
        if anormaly_dir.exists():
            for subdir in sorted(anormaly_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                images = self._collect_images(subdir)
                if images:
                    self.class_names.append(subdir.name)

        # 2. 再收集 normal/（正常类别）
        normal_dir = self.data_dir / 'normal'
        if normal_dir.exists():
            images = self._collect_images(normal_dir)
            if images:
                self.class_names.append('normal')

        # 建立类别索引映射
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # 3. 收集所有样本
        for class_name in self.class_names:
            if class_name == 'normal':
                class_dir = self.data_dir / 'normal'
            else:
                class_dir = self.data_dir / 'anormaly' / class_name

            images = self._collect_images(class_dir)
            class_idx = self.class_to_idx[class_name]
            for img_path in images:
                self.samples.append((str(img_path), class_idx))

    def _collect_images(self, directory):
        """收集目录下的所有图像路径"""
        return sorted([
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in self.image_extensions
        ])

    def __len__(self):
        return len(self.samples)

    def __init__(self, data_dir, img_size=224):
        """
        Args:
            data_dir: 数据根目录，如 './data/augmented'
            img_size: 统一输入尺寸（resize 到正方形），默认 224
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        # 扫描目录，建立类别映射
        self.class_names = []   # 类别名列表
        self.class_to_idx = {}  # 类别名 -> 索引
        self.samples = []       # [(图像路径, 类别索引), ...]

        self._scan_directory()

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 读取图像（BGR -> RGB -> CHW Tensor，不做归一化）
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resize 到统一尺寸（避免显存溢出）
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (H,W,C) -> (C,H,W)

        return image, label


# ======================== 训练函数 ========================

def train(
    data_dir='./data/augmented',
    num_epochs=100,
    batch_size=16,
    lr=1e-3,
    img_size=224,
    use_dw=False,
    width_factor=1.0,
    resolution_factor=1.0,
    patience=15,
    device='cuda',
    save_path='best_model.pth',
    cam_dir='./cam_outputs',
    log_dir='./logs',
):
    """
    训练函数

    Args:
        data_dir: 数据目录
        num_epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        img_size: 统一输入尺寸（resize 到正方形），默认 224
        use_dw: 是否使用深度可分离残差块，默认 False
        width_factor: 宽度因子 α，缩放中间通道数，默认 1.0
        resolution_factor: 分辨率因子 ρ，缩放输入图像尺寸，默认 1.0
        device: 设备（'cuda' 或 'cpu'）
        save_path: 模型保存路径
        cam_dir: CAM 热力图输出目录
        log_dir: 日志目录
    """
    # ---------- 日志 ----------
    logger = Logger(log_dir)

    # ---------- 设备 ----------
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.log(f"[{datetime.now().strftime('%H:%M:%S')}] 使用设备: {device}")

    # ---------- 数据集 ----------
    dataset = AnomalyDataset(data_dir, img_size=img_size)
    num_classes = len(dataset.class_names)

    logger.log(f"\n类别数: {num_classes}")
    logger.log(f"类别映射:")
    for name, idx in dataset.class_to_idx.items():
        count = sum(1 for _, label in dataset.samples if label == idx)
        logger.log(f"  {idx}: {name} ({count} 张)")

    logger.log(f"\n总样本数: {len(dataset)}, 输入尺寸: {img_size}×{img_size}")

    # 划分训练集和验证集（8:2）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    logger.log(f"训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")

    # ---------- 样本分布图 ----------
    class_counts = [0] * num_classes
    for _, label in dataset:
        class_counts[label] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(dataset.class_names, class_counts, color='steelblue')
    for bar, count in zip(bars, class_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(count), ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Sample Distribution')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    dist_path = Path(log_dir) / 'sample_distribution.png'
    fig.savefig(str(dist_path), dpi=150)
    plt.close(fig)
    logger.log(f"样本分布图: {dist_path}")

    # ---------- 模型 ----------
    model = ResNet50(num_classes=num_classes, in_channels=3,
                     use_dw=use_dw, width_factor=width_factor,
                     resolution_factor=resolution_factor)
    model = model.to(device)
    block_type = 'DWBottleneck' if use_dw else 'Bottleneck'
    logger.log(f"模型: ResNet50({block_type}, α={width_factor}, ρ={resolution_factor})")
    logger.log(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ---------- CAM 输出目录 ----------
    cam_path = Path(cam_dir)
    cam_path.mkdir(parents=True, exist_ok=True)

    # ---------- 损失函数和优化器 ----------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ---------- 训练循环 ----------
    best_val_acc = 0.0
    patience_counter = 0

    # ---------- CSV 日志 ----------
    csv_path = Path(log_dir) / f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    csv_header = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr',
                  'epoch_time'] + [f'acc_{name}' for name in dataset.class_names]
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_header)
    logger.log(f"CSV 日志: {csv_path}")

    for epoch in range(num_epochs):
        epoch_start = datetime.now()

        # ---- 训练阶段 ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            # 反�传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # 打印 batch 日志
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                batch_acc = predicted.eq(labels).sum().item() / labels.size(0)
                logger.log(f"  [Epoch {epoch+1}] Batch [{batch_idx+1}/{len(train_loader)}]  "
                           f"Loss: {loss.item():.4f}  Acc: {batch_acc:.4f}")

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ---- 验证阶段 ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # 每类统计：正确数、总数
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        # 用于热力图的样本（每个类别取一张）
        cam_samples = {}  # {class_idx: (image_tensor, cam_tensor)}

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs, cam = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                # 每类统计
                for i in range(images.size(0)):
                    cls_idx = labels[i].item()
                    class_total[cls_idx] += 1
                    if predicted[i].item() == cls_idx:
                        class_correct[cls_idx] += 1

                # 记录每个类别的第一张样本用于 CAM
                for i in range(images.size(0)):
                    cls_idx = labels[i].item()
                    if cls_idx not in cam_samples:
                        pred_idx = predicted[i].item()
                        cam_samples[cls_idx] = (images[i].cpu(), cam[i].cpu(), pred_idx)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # 每类准确率
        class_accs = []
        for i in range(num_classes):
            if class_total[i] > 0:
                class_accs.append(class_correct[i] / class_total[i])
            else:
                class_accs.append(0.0)

        # 更新学习率
        scheduler.step()

        # 打印 epoch 日志
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        logger.log(f"\n{'='*70}")
        logger.log(f"Epoch [{epoch+1}/{num_epochs}]  耗时: {epoch_time:.1f}s")
        logger.log(f"  Train  Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        logger.log(f"  Val    Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
        logger.log(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # 打印每类准确率
        logger.log(f"  各类别准确率:")
        for i, name in enumerate(dataset.class_names):
            logger.log(f"    {name}: {class_accs[i]:.4f} ({class_correct[i]}/{class_total[i]})")

        # 写入 CSV
        row = [epoch + 1, f'{train_loss:.4f}', f'{train_acc:.4f}',
               f'{val_loss:.4f}', f'{val_acc:.4f}',
               f'{scheduler.get_last_lr()[0]:.6f}', f'{epoch_time:.1f}']
        row += [f'{acc:.4f}' for acc in class_accs]
        csv_writer.writerow(row)
        csv_file.flush()

        # ---- 保存热力图 ----
        epoch_cam_dir = cam_path / f'epoch_{epoch+1:03d}'
        epoch_cam_dir.mkdir(parents=True, exist_ok=True)

        for cls_idx, (img_tensor, cam_tensor, pred_idx) in cam_samples.items():
            cls_name = dataset.class_names[cls_idx]
            # 单类别热力图（真实类别激活）
            save_cam_heatmap(
                img_tensor, cam_tensor, cls_idx, pred_idx, dataset.class_names,
                save_path=epoch_cam_dir / f'{cls_name}.png'
            )
            # 所有类别热力图拼接
            save_cam_all_classes(
                img_tensor, cam_tensor, cls_idx, pred_idx, dataset.class_names,
                save_path=epoch_cam_dir / f'{cls_name}_all.png'
            )

        logger.log(f"  CAM 热力图已保存至: {epoch_cam_dir}")

        # ---- 保存最佳模型 ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': dataset.class_names,
                'class_to_idx': dataset.class_to_idx,
                'use_dw': use_dw,
                'width_factor': width_factor,
                'resolution_factor': resolution_factor,
                'img_size': img_size,
            }, save_path)
            logger.log(f"  -> 保存最佳模型 (Val Acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.log(f"  -> 验证准确率未提升 ({patience_counter}/{patience})")

        if patience_counter >= patience:
            logger.log(f"\n早停触发！连续 {patience} 轮验证准确率未提升，停止训练。")
            break

    logger.log(f"\n{'='*70}")
    logger.log(f"训练完成！最佳验证准确率: {best_val_acc:.4f}")
    logger.log(f"模型保存至: {save_path}")
    logger.log(f"日志保存至: {logger.log_file}")
    logger.log(f"CSV 日志: {csv_path}")

    csv_file.close()
    logger.close()
    return model, dataset


# ======================== 入口 ========================
if __name__ == '__main__':
    train(
        data_dir='./data/augmented',
        num_epochs=100,
        batch_size=8,
        lr=1e-3,
        img_size=224,
        use_dw=True,              # True 用深度可分离，False 用标准 Bottleneck
        width_factor=0.5,          # 宽度因子 α（0.5=轻量，1.0=标准，2.0=更宽）
        resolution_factor=1.0,     # 分辨率因子 ρ（0.5=低分辨率加速，1.0=标准）
        device='cuda',
        save_path='best_model.pth',
        cam_dir='./cam_outputs',
        log_dir='./logs',
    )
