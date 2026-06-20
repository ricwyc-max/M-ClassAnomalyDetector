"""
训练结果可视化脚本
从 CSV 日志和模型文件生成图表

功能：
    - 损失曲线（训练/验证）
    - 整体准确率曲线（训练/验证）
    - 各类别准确率曲线
    - 混淆矩阵热力图
    - 最终测试 CAM 热力图

使用方式：
    python visualize.py
    或在解释器中：
        from visualize import plot_all
        plot_all(csv_path='./logs/xxx.csv', model_path='best_model.pth', data_dir='./data/augmented')
"""

import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from model import ResNet50
from train import AnomalyDataset

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def read_csv(csv_path):
    """
    读取 CSV 日志文件

    Returns:
        dict: {
            'epochs': [1, 2, ...],
            'train_loss': [...],
            'train_acc': [...],
            'val_loss': [...],
            'val_acc': [...],
            'lr': [...],
            'epoch_time': [...],
            'class_names': ['bent_wire', ...],
            'class_accs': {'bent_wire': [...], ...}
        }
    """
    data = {
        'epochs': [], 'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'lr': [], 'epoch_time': [],
        'class_names': [], 'class_accs': {},
    }

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        # 提取类别名（acc_xxx 列）
        for h in headers:
            if h.startswith('acc_'):
                cls_name = h[4:]  # 去掉 'acc_' 前缀
                data['class_names'].append(cls_name)
                data['class_accs'][cls_name] = []

        for row in reader:
            data['epochs'].append(int(row['epoch']))
            data['train_loss'].append(float(row['train_loss']))
            data['train_acc'].append(float(row['train_acc']))
            data['val_loss'].append(float(row['val_loss']))
            data['val_acc'].append(float(row['val_acc']))
            data['lr'].append(float(row['lr']))
            data['epoch_time'].append(float(row['epoch_time']))

            for cls_name in data['class_names']:
                data['class_accs'][cls_name].append(float(row[f'acc_{cls_name}']))

    return data


def plot_loss_curve(data, save_path):
    """绘制损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['epochs'], data['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(data['epochs'], data['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  损失曲线: {save_path}")


def plot_acc_curve(data, save_path):
    """绘制整体准确率曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['epochs'], data['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax.plot(data['epochs'], data['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training & Validation Accuracy', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  准确率曲线: {save_path}")


def plot_class_acc_curves(data, save_path):
    """绘制各类别准确率曲线"""
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(data['class_names'])))
    for i, cls_name in enumerate(data['class_names']):
        ax.plot(data['epochs'], data['class_accs'][cls_name],
                color=colors[i], label=cls_name, linewidth=1.5)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Validation Accuracy', fontsize=14)
    ax.legend(fontsize=10, ncol=2, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  各类别准确率曲线: {save_path}")


def plot_lr_curve(data, save_path):
    """绘制学习率曲线"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data['epochs'], data['lr'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  学习率曲线: {save_path}")


def plot_confusion_matrix(model, dataset, device, save_path):
    """绘制混淆矩阵热力图"""
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    class_names = dataset.class_names
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs, _ = model(images)
            _, predicted = outputs.max(1)

            for t, p in zip(labels.numpy(), predicted.cpu().numpy()):
                confusion[t][p] += 1

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix', fontsize=14)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)

    # 在格子中标注数字
    thresh = confusion.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(confusion[i, j]),
                    ha='center', va='center',
                    color='white' if confusion[i, j] > thresh else 'black',
                    fontsize=8)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  混淆矩阵: {save_path}")


def plot_cam_samples(model, dataset, device, save_path):
    """绘制最终测试的 CAM 热力图（每个类别一张）"""
    from train import save_cam_heatmap

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    class_names = dataset.class_names
    num_classes = len(class_names)

    # 每个类别收集一张
    cam_samples = {}

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs, cam = model(images)

            for i in range(images.size(0)):
                cls_idx = labels[i].item()
                if cls_idx not in cam_samples:
                    cam_samples[cls_idx] = (images[i].cpu(), cam[i].cpu())

            if len(cam_samples) == num_classes:
                break

    # 每个类别生成一张热力图
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    for cls_idx, (img_tensor, cam_tensor) in cam_samples.items():
        cls_name = class_names[cls_idx]
        pred_idx = cam_tensor.mean(dim=(1, 2)).argmax().item()
        save_cam_heatmap(
            img_tensor, cam_tensor, pred_idx, class_names,
            save_path=save_dir / f'{cls_name}.png'
        )

    # 拼接所有类别为一张大图
    images_list = []
    for cls_idx in range(num_classes):
        if cls_idx in cam_samples:
            img_path = save_dir / f'{class_names[cls_idx]}.png'
            img = cv2.imread(str(img_path))
            if img is not None:
                images_list.append((class_names[cls_idx], img))

    if images_list:
        # 计算拼接布局
        n = len(images_list)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        # 统一尺寸
        h, w = images_list[0][1].shape[:2]
        canvas = np.zeros((rows * (h + 30), cols * w, 3), dtype=np.uint8)

        for idx, (name, img) in enumerate(images_list):
            r = idx // cols
            c = idx % cols
            y1 = r * (h + 30) + 30
            x1 = c * w
            canvas[y1:y1+h, x1:x1+w] = img
            # 类别名标注
            cv2.putText(canvas, name, (x1 + 10, r * (h + 30) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        combined_path = save_dir / '_all_classes.png'
        cv2.imwrite(str(combined_path), canvas)
        print(f"  CAM 热力图: {save_dir}")
        print(f"  拼接总图: {combined_path}")


def plot_all(csv_path, model_path=None, data_dir=None, output_dir='./visualizations'):
    """
    一键生成所有可视化图表

    Args:
        csv_path: CSV 日志文件路径
        model_path: 模型文件路径（可选，用于混淆矩阵和CAM）
        data_dir: 数据目录（可选，需要 model_path 时使用）
        output_dir: 图表输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 读取 CSV
    print(f"读取 CSV: {csv_path}")
    data = read_csv(csv_path)
    print(f"  共 {len(data['epochs'])} 轮, {len(data['class_names'])} 个类别")

    # 1. 损失曲线
    plot_loss_curve(data, output_path / 'loss_curve.png')

    # 2. 整体准确率曲线
    plot_acc_curve(data, output_path / 'acc_curve.png')

    # 3. 各类别准确率曲线
    plot_class_acc_curves(data, output_path / 'class_acc_curves.png')

    # 4. 学习率曲线
    plot_lr_curve(data, output_path / 'lr_curve.png')

    # 5. 混淆矩阵和 CAM（需要模型和数据）
    if model_path and data_dir:
        print(f"\n加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        class_names = checkpoint['class_names']
        num_classes = len(class_names)

        model = ResNet50(num_classes=num_classes, in_channels=3)
        model.load_state_dict(checkpoint['model_state_dict'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # 加载数据集
        img_size = 224
        dataset = AnomalyDataset(data_dir, img_size=img_size)

        # 5. 混淆矩阵
        plot_confusion_matrix(model, dataset, device, output_path / 'confusion_matrix.png')

        # 6. CAM 热力图
        plot_cam_samples(model, dataset, device, output_path / 'cam_heatmaps')

    print(f"\n所有图表已保存至: {output_dir}")


# ======================== 入口 ========================
if __name__ == '__main__':
    # ---------- 在这里修改参数 ----------

    # CSV 日志文件（在 ./logs/ 目录下找最新的）
    log_dir = Path('./logs')
    csv_files = sorted(log_dir.glob('train_*.csv'))
    if not csv_files:
        print("未找到 CSV 日志文件！请先运行训练。")
        exit()
    CSV_PATH = str(csv_files[-1])  # 最新的 CSV

    # 模型文件（可选，不填则只生成曲线图）
    MODEL_PATH = 'best_model.pth'

    # 数据目录（需要 MODEL_PATH 时使用）
    DATA_DIR = './data/augmented'

    # 输出目录
    OUTPUT_DIR = './visualizations'

    # ---------- 执行 ----------
    plot_all(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
    )
