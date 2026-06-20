"""
平衡 normal 样本数量：随机删除部分增强图片，使 normal 与异常类总量级一致

规则：
    - 每个原图的增强图片按相同比例保留/删除
    - 保留数量 = target_count / 原图数（向上取整）
    - 随机选择保留哪些增强，其余删除
"""

import os
import random
from pathlib import Path
from collections import defaultdict

# ======================== 配置 ========================
NORMAL_DIR = Path('./data/augmented/normal')
# 目标数量（与异常类平均值对齐，或直接指定）
TARGET_COUNT = None  # 设为 None 则自动计算异常类平均值
# 如果要手动指定，设为具体数字，如 1200
# TARGET_COUNT = 1200

RANDOM_SEED = 42  # 随机种子，保证可复现


def group_by_original(normal_dir):
    """按原图编号分组"""
    groups = defaultdict(list)
    for f in normal_dir.iterdir():
        if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
            # 文件名格式: XXX_augYYY.png
            parts = f.stem.split('_aug')
            if len(parts) == 2:
                orig_id = parts[0]
                groups[orig_id].append(f)
    return groups


def calc_target_count(normal_dir):
    """计算目标数量：异常类的平均值"""
    augmented_dir = normal_dir.parent
    anormaly_dir = augmented_dir / 'anormaly'

    total = 0
    count = 0
    for cls_dir in anormaly_dir.iterdir():
        if cls_dir.is_dir():
            n = len([f for f in cls_dir.iterdir() if f.is_file()])
            total += n
            count += 1

    return total // count if count > 0 else 1200


def balance(normal_dir, target_count, seed):
    """执行平衡操作"""
    groups = group_by_original(normal_dir)
    n_originals = len(groups)

    if n_originals == 0:
        print("未找到图片！")
        return

    # 每个原图保留的增强数量
    keep_per_orig = max(1, target_count // n_originals)
    actual_target = keep_per_orig * n_originals

    print(f"原图数量: {n_originals}")
    print(f"当前总数: {sum(len(v) for v in groups.values())}")
    print(f"目标总数: {target_count}")
    print(f"每个原图保留: {keep_per_orig} 张")
    print(f"实际保留: {actual_target} 张")

    random.seed(seed)
    deleted = 0
    kept = 0

    for orig_id, files in sorted(groups.items()):
        files_sorted = sorted(files)
        keep = random.sample(files_sorted, min(keep_per_orig, len(files_sorted)))
        keep_set = set(keep)

        for f in files_sorted:
            if f in keep_set:
                kept += 1
            else:
                f.unlink()
                deleted += 1

    print(f"\n完成！保留 {kept} 张，删除 {deleted} 张")


if __name__ == '__main__':
    target = TARGET_COUNT if TARGET_COUNT else calc_target_count(NORMAL_DIR)
    balance(NORMAL_DIR, target, RANDOM_SEED)
