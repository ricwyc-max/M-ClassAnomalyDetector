"""
人民币配色方案
参考第五套人民币设计色彩，用于数据可视化

使用方式：
    from colors import R100, R50, R20, R10, R5, R1
    from colors import get_class_colors, get_cmap_rmb
"""

import numpy as np
import matplotlib.colors as mcolors

# ── 人民币配色方案 ──
# 100元 — 红色系（多头/上涨/正面）
R100 = ["#E6A9B4", "#BBB1D8", "#C25C6C", "#D5513C", "#AB2930"]
# 50元 — 深绿系（空头/下跌/负面）
R50  = ["#C7C1D3", "#7D809F", "#98B7AB", "#597971", "#234437"]
# 20元 — 黄褐系（中性/分类/暖色）
R20  = ["#E5C484", "#929D84", "#897A6B", "#674F32", "#442E26"]
# 10元 — 蓝色系（技术/数据/冷色）
R10  = ["#D7E4F1", "#A9D7EB", "#C6C1D7", "#6881A9", "#536694"]
# 5元 — 紫色系（期权/特殊）
R5   = ["#D0C1AE", "#92A2A5", "#9888A9", "#746187", "#423559"]
# 1元 — 浅绿系（背景/辅助）
R1   = ["#CFD8C5", "#D6C690", "#BDC5A5", "#758364", "#47522F"]


def get_class_colors(n_classes):
    """
    为 n_classes 个类别生成配色
    优先使用100元红+10元蓝，不够时循环补充

    Returns:
        list of hex color strings
    """
    # 基础色板：从各面值中选取最具代表性的颜色
    base_colors = [
        R100[4], R10[4], R50[4], R20[3], R5[4],   # 深色系
        R100[2], R10[3], R50[3], R20[2], R5[3],   # 中色系
        R100[0], R10[0], R50[0], R20[0], R5[0],   # 浅色系
    ]
    return [base_colors[i % len(base_colors)] for i in range(n_classes)]


def get_cmap_rmb(name='blue'):
    """
    获取人民币风格的 LinearSegmentedColormap

    Args:
        name: 'blue'(10元), 'red'(100元), 'green'(50元), 'brown'(20元)

    Returns:
        matplotlib LinearSegmentedColormap
    """
    cmap_dict = {
        'blue': R10[::-1],   # 浅→深
        'red': R100[::-1],
        'green': R50[::-1],
        'brown': R20[::-1],
    }
    colors = cmap_dict.get(name, R10[::-1])
    return mcolors.LinearSegmentedColormap.from_list(f'rmb_{name}', colors)


def get_loss_colors():
    """损失曲线配色：训练(红) + 验证(绿)"""
    return R100[3], R50[3]  # 红 + 深绿


def get_acc_colors():
    """准确率曲线配色：训练(蓝) + 验证(绿)"""
    return R10[3], R50[3]  # 蓝 + 深绿


def get_bar_colors_4():
    """四宫格柱状图配色：Precision/Recall/F1/AUC"""
    return R100[3], R20[3], R50[3], R5[3]  # 红/褐/绿/紫


def get_grid_bg():
    """网格/辅助线颜色"""
    return R1[0]


def get_canvas_bg():
    """画布背景色"""
    return '#FAFAF8'
