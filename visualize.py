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
from colors import (
    R100, R50, R20, R10, R5, R1,
    get_class_colors, get_cmap_rmb, get_loss_colors, get_acc_colors,
    get_bar_colors_4, get_grid_bg, get_canvas_bg
)

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 人民币配色
GRID_COLOR = get_grid_bg()
CANVAS_BG = get_canvas_bg()


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
    train_color, val_color = get_loss_colors()

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(CANVAS_BG)
    ax.set_facecolor(CANVAS_BG)

    ax.plot(data['epochs'], data['train_loss'], color=train_color, label='Train Loss', linewidth=2)
    ax.plot(data['epochs'], data['val_loss'], color=val_color, label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, color=GRID_COLOR)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  损失曲线: {save_path}")


def plot_acc_curve(data, save_path):
    """绘制整体准确率曲线"""
    train_color, val_color = get_acc_colors()

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(CANVAS_BG)
    ax.set_facecolor(CANVAS_BG)

    ax.plot(data['epochs'], data['train_acc'], color=train_color, label='Train Acc', linewidth=2)
    ax.plot(data['epochs'], data['val_acc'], color=val_color, label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, color=GRID_COLOR)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  准确率曲线: {save_path}")


def plot_class_acc_curves(data, save_path):
    """绘制各类别准确率曲线"""
    n_classes = len(data['class_names'])
    class_colors = get_class_colors(n_classes)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(CANVAS_BG)
    ax.set_facecolor(CANVAS_BG)

    for i, cls_name in enumerate(data['class_names']):
        ax.plot(data['epochs'], data['class_accs'][cls_name],
                color=class_colors[i], label=cls_name, linewidth=1.5)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, ncol=2, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, color=GRID_COLOR)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  各类别准确率曲线: {save_path}")


def plot_lr_curve(data, save_path):
    """绘制学习率曲线"""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(CANVAS_BG)
    ax.set_facecolor(CANVAS_BG)

    # 使用10元蓝色系
    ax.fill_between(data['epochs'], data['lr'], alpha=0.3, color=R10[1])
    ax.plot(data['epochs'], data['lr'], color=R10[3], linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, color=GRID_COLOR)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, facecolor=fig.get_facecolor())
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

    # 使用10元蓝色系作为混淆矩阵配色
    cmap_rmb = get_cmap_rmb('blue')

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(CANVAS_BG)
    ax.set_facecolor(CANVAS_BG)

    im = ax.imshow(confusion, interpolation='nearest', cmap=cmap_rmb)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax)

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
    fig.savefig(str(save_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  混淆矩阵: {save_path}")


def plot_cam_samples(model, dataset, device, save_path):
    """绘制最终测试的 CAM 热力图（每个类别一张）"""
    from train import save_cam_heatmap, save_cam_all_classes

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
            _, predicted = outputs.max(1)

            for i in range(images.size(0)):
                cls_idx = labels[i].item()
                if cls_idx not in cam_samples:
                    pred_idx = predicted[i].item()
                    cam_samples[cls_idx] = (images[i].cpu(), cam[i].cpu(), pred_idx)

            if len(cam_samples) == num_classes:
                break

    # 每个类别生成热力图
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    for cls_idx, (img_tensor, cam_tensor, pred_idx) in cam_samples.items():
        cls_name = class_names[cls_idx]
        save_cam_heatmap(
            img_tensor, cam_tensor, cls_idx, pred_idx, class_names,
            save_path=save_dir / f'{cls_name}.png'
        )
        save_cam_all_classes(
            img_tensor, cam_tensor, cls_idx, pred_idx, class_names,
            save_path=save_dir / f'{cls_name}_all.png'
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


def plot_metrics_summary(csv_path, model_path, data_dir, save_path):
    """绘制指标汇总图（Precision、Recall、F1、AUC）"""
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    checkpoint = torch.load(model_path, map_location='cpu')
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    use_dw = checkpoint.get('use_dw', False)
    width_factor = checkpoint.get('width_factor', 1.0)
    resolution_factor = checkpoint.get('resolution_factor', 1.0)
    img_size = checkpoint.get('img_size', 224)

    model = ResNet50(num_classes=num_classes, in_channels=3,
                     use_dw=use_dw, width_factor=width_factor,
                     resolution_factor=resolution_factor)
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    dataset = AnomalyDataset(data_dir, img_size=img_size)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs, _ = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 计算每类指标
    precisions = []
    recalls = []
    f1s = []
    aucs = []

    for i, name in enumerate(class_names):
        p = precision_score(all_labels == i, all_preds == i, zero_division=0)
        r = recall_score(all_labels == i, all_preds == i, zero_division=0)
        f = f1_score(all_labels == i, all_preds == i, zero_division=0)

        binary_labels = (all_labels == i).astype(int)
        try:
            auc = roc_auc_score(binary_labels, all_probs[:, i])
        except ValueError:
            auc = 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        aucs.append(auc)

    # 绘图 - 使用人民币配色四宫格
    bar_colors = get_bar_colors_4()  # 红/褐/绿/紫

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(CANVAS_BG)

    x = np.arange(len(class_names))
    width = 0.6

    # Precision (100元红)
    ax = axes[0, 0]
    ax.set_facecolor(CANVAS_BG)
    bars = ax.bar(x, precisions, width, color=bar_colors[0], edgecolor='white', linewidth=0.5)
    ax.set_title('Precision per Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y', color=GRID_COLOR)
    for bar, v in zip(bars, precisions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.2f}', ha='center', fontsize=7)

    # Recall (20元褐)
    ax = axes[0, 1]
    ax.set_facecolor(CANVAS_BG)
    bars = ax.bar(x, recalls, width, color=bar_colors[1], edgecolor='white', linewidth=0.5)
    ax.set_title('Recall per Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('Recall', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y', color=GRID_COLOR)
    for bar, v in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.2f}', ha='center', fontsize=7)

    # F1-Score (50元绿)
    ax = axes[1, 0]
    ax.set_facecolor(CANVAS_BG)
    bars = ax.bar(x, f1s, width, color=bar_colors[2], edgecolor='white', linewidth=0.5)
    ax.set_title('F1-Score per Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y', color=GRID_COLOR)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.2f}', ha='center', fontsize=7)

    # AUC (5元紫)
    ax = axes[1, 1]
    ax.set_facecolor(CANVAS_BG)
    bars = ax.bar(x, aucs, width, color=bar_colors[3], edgecolor='white', linewidth=0.5)
    ax.set_title('AUC per Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y', color=GRID_COLOR)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.2f}', ha='center', fontsize=7)

    # 添加整体指标
    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f = np.mean(f1s)
    macro_auc = np.mean(aucs)
    fig.suptitle(f'Macro Avg — P: {macro_p:.3f}  R: {macro_r:.3f}  F1: {macro_f:.3f}  AUC: {macro_auc:.3f}',
                 fontsize=12, y=1.02, fontweight='bold')

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  指标汇总: {save_path}")


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

    # 5. 混淆矩阵、CAM、指标汇总（需要模型和数据）
    if model_path and data_dir:
        print(f"\n加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        class_names = checkpoint['class_names']
        num_classes = len(class_names)

        # 从 checkpoint 恢复模型配置
        use_dw = checkpoint.get('use_dw', False)
        width_factor = checkpoint.get('width_factor', 1.0)
        resolution_factor = checkpoint.get('resolution_factor', 1.0)
        img_size = checkpoint.get('img_size', 224)

        model = ResNet50(num_classes=num_classes, in_channels=3,
                         use_dw=use_dw, width_factor=width_factor,
                         resolution_factor=resolution_factor)
        model.load_state_dict(checkpoint['model_state_dict'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # 加载数据集
        dataset = AnomalyDataset(data_dir, img_size=img_size)

        # 5. 混淆矩阵
        plot_confusion_matrix(model, dataset, device, output_path / 'confusion_matrix.png')

        # 6. CAM 热力图
        plot_cam_samples(model, dataset, device, output_path / 'cam_heatmaps')

        # 7. 指标汇总（Precision、Recall、F1、AUC）
        plot_metrics_summary(csv_path, model_path, data_dir, output_path / 'metrics_summary.png')

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
