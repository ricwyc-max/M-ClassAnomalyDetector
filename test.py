"""
模型测试脚本
计算分类指标：Precision、Recall、F1-score、AUC
生成 ROC 曲线、PR 曲线、混淆矩阵、测试图片结果

使用方式：
    python test.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, average_precision_score
)
from model import ResNet50
from train import AnomalyDataset, save_cam_heatmap, save_cam_all_classes
import cv2

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_model(model_path, device='cpu'):
    """加载模型"""
    checkpoint = torch.load(model_path, map_location=device)

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
    model = model.to(device)
    model.eval()

    return model, class_names, img_size


def evaluate_model(model, dataloader, device, num_classes):
    """
    在数据集上评估模型

    Returns:
        all_labels: 真实标签
        all_preds: 预测标签
        all_probs: 预测概率
        all_images: 原始图像
        all_cams: CAM 热力图
    """
    all_labels = []
    all_preds = []
    all_probs = []
    all_images = []
    all_cams = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs, cam = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_images.extend(images.cpu())
            all_cams.extend(cam.cpu())

    return (np.array(all_labels), np.array(all_preds),
            np.array(all_probs), all_images, all_cams)


def plot_roc_curves(all_labels, all_probs, class_names, save_path):
    """绘制 ROC 曲线（每类 + 宏平均）"""
    n_classes = len(class_names)

    # 计算每类的 ROC
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        binary_labels = (all_labels == i).astype(int)
        if binary_labels.sum() == 0:
            continue
        fpr[i], tpr[i], _ = roc_curve(binary_labels, all_probs[:, i])
        roc_auc[i] = roc_auc_score(binary_labels, all_probs[:, i])

    # 宏平均 ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in fpr]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in fpr:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(fpr)
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = np.mean([roc_auc[i] for i in roc_auc])

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i in range(n_classes):
        if i in fpr:
            ax.plot(fpr[i], tpr[i], color=colors[i], lw=1.5,
                    label=f'{class_names[i]} (AUC={roc_auc[i]:.3f})')

    ax.plot(fpr['macro'], tpr['macro'], 'k--', lw=2,
            label=f'Macro Average (AUC={roc_auc["macro"]:.3f})')

    ax.plot([0, 1], [0, 1], 'gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  ROC 曲线: {save_path}")

    return roc_auc


def plot_pr_curves(all_labels, all_probs, class_names, save_path):
    """绘制 PR 曲线（每类 + 宏平均）"""
    n_classes = len(class_names)

    precision = {}
    recall = {}
    pr_auc = {}

    for i in range(n_classes):
        binary_labels = (all_labels == i).astype(int)
        if binary_labels.sum() == 0:
            continue
        precision[i], recall[i], _ = precision_recall_curve(binary_labels, all_probs[:, i])
        pr_auc[i] = average_precision_score(binary_labels, all_probs[:, i])

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i in range(n_classes):
        if i in precision:
            ax.plot(recall[i], precision[i], color=colors[i], lw=1.5,
                    label=f'{class_names[i]} (AP={pr_auc[i]:.3f})')

    macro_ap = np.mean([pr_auc[i] for i in pr_auc])
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curves (Macro AP={macro_ap:.3f})', fontsize=14)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  PR 曲线: {save_path}")

    return pr_auc


def plot_confusion_matrix(all_labels, all_preds, class_names, save_path):
    """绘制混淆矩阵（原始 + 归一化）"""
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 原始混淆矩阵
    ax = axes[0]
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix (Counts)', fontsize=13)
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=8)

    # 归一化混淆矩阵
    ax = axes[1]
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_title('Confusion Matrix (Normalized)', fontsize=13)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{cm_norm[i, j]:.2f}', ha='center', va='center',
                    color='white' if cm_norm[i, j] > 0.5 else 'black', fontsize=8)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  混淆矩阵: {save_path}")


def save_test_results(all_images, all_labels, all_preds, all_probs, all_cams,
                      class_names, save_dir, max_per_class=5):
    """
    保存测试图片结果：原图 + CAM 热力图 + 预测结果
    每个类别保存 max_per_class 张
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    saved_count = {i: 0 for i in range(n_classes)}

    # 统计正确/错误
    correct_dir = save_dir / 'correct'
    wrong_dir = save_dir / 'wrong'
    correct_dir.mkdir(exist_ok=True)
    wrong_dir.mkdir(exist_ok=True)

    for idx in range(len(all_labels)):
        cls_idx = all_labels[idx]
        pred_idx = all_preds[idx]

        if saved_count[cls_idx] >= max_per_class:
            continue

        saved_count[cls_idx] += 1

        img_tensor = all_images[idx]
        cam_tensor = all_cams[idx]
        prob = all_probs[idx][pred_idx]
        true_name = class_names[cls_idx]
        pred_name = class_names[pred_idx]

        # 生成 CAM 热力图
        is_correct = (cls_idx == pred_idx)
        out_dir = correct_dir if is_correct else wrong_dir

        # 使用 train.py 中的函数生成热力图
        filename = f'{true_name}_{saved_count[cls_idx]}_pred{pred_name}_p{prob:.2f}.png'
        save_cam_heatmap(
            img_tensor, cam_tensor, cls_idx, pred_idx, class_names,
            save_path=out_dir / filename
        )

    # 生成汇总图
    _save_summary_grid(all_images, all_labels, all_preds, all_probs, all_cams,
                       class_names, save_dir / 'summary.png')

    print(f"  测试结果图片: {save_dir}")
    print(f"    正确样本: {correct_dir}")
    print(f"    错误样本: {wrong_dir}")


def _save_summary_grid(all_images, all_labels, all_preds, all_probs, all_cams,
                       class_names, save_path, max_images=20):
    """生成汇总网格图"""
    n_classes = len(class_names)
    selected = []

    # 每类选一张，最多 max_images 张
    seen = set()
    for idx in range(len(all_labels)):
        cls_idx = all_labels[idx]
        if cls_idx not in seen and len(selected) < max_images:
            selected.append(idx)
            seen.add(cls_idx)

    if not selected:
        return

    # 计算布局
    n = len(selected)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    img_h, img_w = 224, 224
    cell_h = img_h + 60  # 图像 + 文字区域
    cell_w = img_w

    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for i, idx in enumerate(selected):
        r = i // cols
        c = i % cols
        y1 = r * cell_h
        x1 = c * cell_w

        # 获取 CAM 叠加图
        img = all_images[idx].permute(1, 2, 0).numpy().astype(np.uint8)
        cam = all_cams[idx][all_labels[idx]].numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = (cam * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        if heatmap.shape[:2] != img.shape[:2]:
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

        # 缩放到单元格大小
        overlay = cv2.resize(overlay, (cell_w, img_h))
        canvas[y1:y1+img_h, x1:x1+cell_w] = overlay

        # 标注
        true_name = class_names[all_labels[idx]]
        pred_name = class_names[all_preds[idx]]
        prob = all_probs[idx][all_preds[idx]]
        is_correct = all_labels[idx] == all_preds[idx]

        color = (0, 200, 0) if is_correct else (200, 50, 50)
        label = f"T:{true_name} P:{pred_name} {prob:.2f}"
        cv2.putText(canvas, label, (x1 + 5, y1 + img_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"  汇总图: {save_path}")


def test(
    model_path='best_model.pth',
    data_dir='./data/augmented',
    batch_size=16,
    device='cuda',
    output_dir='./test_results',
):
    """
    主测试函数

    Args:
        model_path: 模型文件路径
        data_dir: 数据目录
        batch_size: 批大小
        device: 设备
        output_dir: 结果输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print(f"\n加载模型: {model_path}")
    model, class_names, img_size = load_model(model_path, device)
    num_classes = len(class_names)
    print(f"  类别数: {num_classes}")
    print(f"  类别: {class_names}")

    # 加载数据集
    print(f"\n加载数据集: {data_dir}")
    dataset = AnomalyDataset(data_dir, img_size=img_size)

    # 使用与训练相同的随机种子划分
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  测试集大小: {len(val_dataset)}")

    # 评估
    print("\n评估中...")
    all_labels, all_preds, all_probs, all_images, all_cams = \
        evaluate_model(model, val_loader, device, num_classes)

    # 计算指标
    print("\n" + "=" * 60)
    print("分类指标")
    print("=" * 60)

    # 整体指标
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    print(f"\n整体指标 (Macro Average):")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall:    {recall_macro:.4f}")
    print(f"  F1-Score:  {f1_macro:.4f}")

    # 每类指标
    print(f"\n各类别指标:")
    print(f"{'类别':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)

    for i, name in enumerate(class_names):
        mask = (all_labels == i)
        if mask.sum() == 0:
            continue
        p = precision_score(all_labels == i, all_preds == i)
        r = recall_score(all_labels == i, all_preds == i)
        f = f1_score(all_labels == i, all_preds == i)
        s = mask.sum()
        print(f"{name:<25} {p:<12.4f} {r:<12.4f} {f:<12.4f} {s:<10}")

    # AUC（每类 + 宏平均）
    print(f"\nAUC:")
    auc_scores = {}
    for i, name in enumerate(class_names):
        binary_labels = (all_labels == i).astype(int)
        if binary_labels.sum() == 0:
            continue
        try:
            auc = roc_auc_score(binary_labels, all_probs[:, i])
            auc_scores[i] = auc
            print(f"  {name}: {auc:.4f}")
        except ValueError:
            print(f"  {name}: N/A (样本不足)")

    macro_auc = np.mean(list(auc_scores.values()))
    print(f"  Macro Average: {macro_auc:.4f}")

    # 分类报告（保存到文件）
    report = classification_report(all_labels, all_preds, target_names=class_names)
    report_path = output_path / 'classification_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write(f"\n\nMacro Average AUC: {macro_auc:.4f}\n")
    print(f"\n分类报告: {report_path}")

    # 生成图表
    print("\n生成图表...")
    plot_roc_curves(all_labels, all_probs, class_names, output_path / 'roc_curves.png')
    plot_pr_curves(all_labels, all_probs, class_names, output_path / 'pr_curves.png')
    plot_confusion_matrix(all_labels, all_preds, class_names, output_path / 'confusion_matrix.png')

    # 保存测试图片结果
    print("\n保存测试图片结果...")
    save_test_results(all_images, all_labels, all_preds, all_probs, all_cams,
                      class_names, output_path / 'test_images', max_per_class=5)

    # 汇总
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print(f"  结果目录: {output_dir}")
    print(f"  分类报告: {report_path}")
    print(f"  ROC 曲线: {output_path / 'roc_curves.png'}")
    print(f"  PR 曲线:  {output_path / 'pr_curves.png'}")
    print(f"  混淆矩阵: {output_path / 'confusion_matrix.png'}")
    print(f"  测试图片: {output_path / 'test_images'}")

    return {
        'precision': precision_macro,
        'recall': recall_macro,
        'f1': f1_macro,
        'auc': macro_auc,
        'auc_per_class': auc_scores,
    }


if __name__ == '__main__':
    test(
        model_path='best_model.pth',
        data_dir='./data/augmented',
        batch_size=16,
        device='cuda',
        output_dir='./test_results',
    )
