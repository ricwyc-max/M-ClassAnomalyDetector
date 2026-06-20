"""
图像增强模块
支持从指定目录读取所有子目录中的图像，进行多种增强处理后输出到指定目录

增强方式：
    - 旋转（rotate）：随机旋转图像
    - 裁剪（crop）：随机裁剪后 resize 回原尺寸
    - 马赛克模糊（mosaic_blur）：将图像分块取平均色
    - 马赛克拼接（mosaic_shuffle）：将图像分块后随机打乱重排
    - 平移（translate）：随机平移图像
    - 亮度（brightness）：随机调整亮度
    - 缩放（scale）：随机缩放图像

在解释器中使用：
    from augment import ImageAugmentor, collect_images, batch_augment

    # 单张图像增强（指定方法和参数）
    aug = ImageAugmentor(seed=42)
    img = cv2.imread('test.jpg')

    # 方法1：用默认参数
    results = aug.augment(img, num_augments=3, methods=['rotate', 'brightness'])

    # 方法2：用自定义参数（传 kwargs 字典）
    results = aug.augment(img, num_augments=3, methods=['rotate', 'brightness'],
                          params={'rotate': {'angle_range': (-45, 45)},
                                  'brightness': {'factor_range': (0.8, 1.2)}})
"""

import cv2
import numpy as np
import random
from pathlib import Path


class ImageAugmentor:
    """图像增强器"""

    def __init__(self, seed=42):
        """
        Args:
            seed: 随机种子，保证可复现
        """
        random.seed(seed)
        np.random.seed(seed)

    def rotate(self, image, angle_range=(-30, 30)):
        """
        随机旋转图像

        Args:
            image: 输入图像 (H, W, C)
            angle_range: 旋转角度范围（度），默认 (-30, 30)

        Returns:
            旋转后的图像
        """
        h, w = image.shape[:2]
        angle = random.uniform(*angle_range)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated

    def crop(self, image, crop_ratio_range=(0.7, 1.0)):
        """
        随机裁剪图像（裁剪后 resize 回原尺寸）

        Args:
            image: 输入图像 (H, W, C)
            crop_ratio_range: 裁剪比例范围（保留原图的比例），默认 (0.7, 1.0)

        Returns:
            裁剪后的图像
        """
        h, w = image.shape[:2]
        ratio = random.uniform(*crop_ratio_range)
        new_h, new_w = int(h * ratio), int(w * ratio)

        # 随机选择裁剪起点
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        cropped = image[top:top+new_h, left:left+new_w]
        # resize 回原尺寸
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized

    def mosaic_blur(self, image, grid_size_range=(4, 16)):
        """
        马赛克模糊：将图像分块，每块用平均色填充

        Args:
            image: 输入图像 (H, W, C)
            grid_size_range: 马赛克块大小范围（像素），默认 (4, 16)

        Returns:
            马赛克处理后的图像
        """
        h, w = image.shape[:2]
        result = image.copy()
        grid_size = random.randint(*grid_size_range)

        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                y_end = min(y + grid_size, h)
                x_end = min(x + grid_size, w)
                block = image[y:y_end, x:x_end]
                avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
                result[y:y_end, x:x_end] = avg_color

        return result

    def mosaic_shuffle(self, image, grid_size_range=(32, 96)):
        """
        马赛克拼接：将图像分成多个块，随机打乱后重新拼接

        Args:
            image: 输入图像 (H, W, C)
            grid_size_range: 块大小范围（像素），默认 (32, 96)

        Returns:
            打乱拼接后的图像
        """
        h, w = image.shape[:2]
        grid_size = random.randint(*grid_size_range)

        # 计算行列数
        rows = h // grid_size
        cols = w // grid_size

        if rows < 2 or cols < 2:
            return image.copy()

        # 切割图像块
        blocks = []
        for r in range(rows):
            for c in range(cols):
                y1 = r * grid_size
                x1 = c * grid_size
                block = image[y1:y1+grid_size, x1:x1+grid_size]
                blocks.append(block)

        # 随机打乱
        random.shuffle(blocks)

        # 重新拼接
        result = image.copy()
        idx = 0
        for r in range(rows):
            for c in range(cols):
                y1 = r * grid_size
                x1 = c * grid_size
                result[y1:y1+grid_size, x1:x1+grid_size] = blocks[idx]
                idx += 1

        return result

    def translate(self, image, shift_range=(-0.2, 0.2)):
        """
        随机平移图像

        Args:
            image: 输入图像 (H, W, C)
            shift_range: 平移比例范围（相对于图像尺寸），默认 (-0.2, 0.2)

        Returns:
            平移后的图像
        """
        h, w = image.shape[:2]
        tx = int(w * random.uniform(*shift_range))
        ty = int(h * random.uniform(*shift_range))

        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return translated

    def brightness(self, image, factor_range=(0.5, 1.5)):
        """
        随机调整亮度

        Args:
            image: 输入图像 (H, W, C)
            factor_range: 亮度因子范围（<1变暗，>1变亮），默认 (0.5, 1.5)

        Returns:
            亮度调整后的图像
        """
        factor = random.uniform(*factor_range)
        brightened = image.astype(np.float32) * factor
        brightened = np.clip(brightened, 0, 255).astype(np.uint8)
        return brightened

    def scale(self, image, scale_range=(0.8, 1.2)):
        """
        随机缩放图像

        Args:
            image: 输入图像 (H, W, C)
            scale_range: 缩放比例范围，默认 (0.8, 1.2)

        Returns:
            缩放后的图像
        """
        h, w = image.shape[:2]
        factor = random.uniform(*scale_range)
        new_h, new_w = int(h * factor), int(w * factor)

        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 裁剪或填充回原尺寸
        result = np.zeros_like(image)
        src_y1 = max(0, (new_h - h) // 2)
        src_x1 = max(0, (new_w - w) // 2)
        dst_y1 = max(0, (h - new_h) // 2)
        dst_x1 = max(0, (w - new_w) // 2)

        copy_h = min(h, new_h)
        copy_w = min(w, new_w)

        result[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = \
            scaled[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w]

        return result

    def augment(self, image, num_augments=1, methods=None, params=None):
        """
        对单张图像进行增强

        Args:
            image: 输入图像 (H, W, C)
            num_augments: 生成的增强图像数量
            methods: 增强方法组合（按顺序执行），可选值：
                     ['rotate', 'crop', 'mosaic_blur', 'mosaic_shuffle',
                      'translate', 'brightness', 'scale']
                     默认全部方法依次执行
            params: 各方法的参数字典，格式：
                    {'方法名': {参数名: 参数值}}
                    未指定的方法使用默认参数

                    示例：
                    {
                        'rotate': {'angle_range': (-45, 45)},
                        'brightness': {'factor_range': (0.8, 1.2)},
                        'crop': {'crop_ratio_range': (0.5, 0.8)},
                        'mosaic_blur': {'grid_size_range': (8, 32)},
                        'mosaic_shuffle': {'grid_size_range': (48, 128)},
                        'translate': {'shift_range': (-0.3, 0.3)},
                        'scale': {'scale_range': (0.7, 1.3)},
                    }

        Returns:
            增强后的图像列表
        """
        if methods is None:
            methods = ['rotate', 'crop', 'mosaic_blur', 'mosaic_shuffle',
                       'translate', 'brightness', 'scale']

        if params is None:
            params = {}

        # 增强方法映射
        method_map = {
            'rotate': self.rotate,
            'crop': self.crop,
            'mosaic_blur': self.mosaic_blur,
            'mosaic_shuffle': self.mosaic_shuffle,
            'translate': self.translate,
            'brightness': self.brightness,
            'scale': self.scale,
        }

        results = []
        for _ in range(num_augments):
            aug_img = image.copy()

            # 按用户指定的顺序依次执行增强
            for method_name in methods:
                method_func = method_map[method_name]
                method_params = params.get(method_name, {})
                aug_img = method_func(aug_img, **method_params)

            results.append(aug_img)

        return results


def collect_images(input_dir):
    """
    递归收集输入目录下所有包含图像的叶子目录

    目录结构示例：
        input_dir/
            anormaly/
                bent_wire/     ← 包含图片，记录为 'anormaly/bent_wire'
                    000.png
                cable_swap/    ← 包含图片，记录为 'anormaly/cable_swap'
                    000.png
            normal/
                good/          ← 包含图片，记录为 'normal/good'
                    000.png

    Args:
        input_dir: 输入根目录

    Returns:
        dict: {相对路径: [图像路径列表]}
              如 {'anormaly/bent_wire': ['...000.png', '...001.png']}
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_dict = {}

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # 递归查找所有包含图片的目录
    for dirpath in sorted(input_path.rglob('*')):
        if not dirpath.is_dir():
            continue

        images = [str(f) for f in sorted(dirpath.iterdir())
                  if f.is_file() and f.suffix.lower() in image_extensions]

        if images:
            # 计算相对于 input_dir 的路径
            rel_path = str(dirpath.relative_to(input_path))
            image_dict[rel_path] = images

    return image_dict


def batch_augment(image_dict, output_dir, num=5, methods=None, params=None, seed=42):
    """
    批量增强整个目录的图像

    Args:
        image_dict: collect_images 返回的字典 {子目录名: [图像路径列表]}
        output_dir: 输出目录
        num: 每张原图生成的增强图像数量
        methods: 增强方法列表，默认全部
        params: 各方法的参数字典
        seed: 随机种子

    Returns:
        生成的图像总数
    """
    augmentor = ImageAugmentor(seed=seed)
    output_path = Path(output_dir)

    count = 0
    for subdir_name, image_paths in image_dict.items():
        # '.' 表示图片直接在 input_dir 下，输出也直接到 output_dir
        out_subdir = output_path / subdir_name if subdir_name != '.' else output_path
        out_subdir.mkdir(parents=True, exist_ok=True)

        print(f"处理: {subdir_name} ({len(image_paths)} 张)")

        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is None:
                print(f"  跳过: {img_path}")
                continue

            stem = Path(img_path).stem
            ext = Path(img_path).suffix

            aug_images = augmentor.augment(image, num_augments=num,
                                           methods=methods, params=params)

            for i, aug_img in enumerate(aug_images):
                out_name = f"{stem}_aug{i+1:03d}{ext}"
                cv2.imwrite(str(out_subdir / out_name), aug_img)
                count += 1

    print(f"完成！共生成 {count} 张图像 -> {output_dir}")
    return count


# ======================== 测试 / 调试入口 ========================
if __name__ == '__main__':
    # ---------- 在这里修改参数进行调试 ----------

    # 输入输出目录
    INPUT_DIR = r'D:\2025College Student Innovation and Entrepreneurship Project\project\M-ClassAnomalyDetector\data\data_root\test'
    OUTPUT_DIR = r'D:\2025College Student Innovation and Entrepreneurship Project\project\M-ClassAnomalyDetector\data\augmented'

    # 每张原图生成几张增强图像
    NUM_AUGMENTS = 100

    # 使用的增强方法（按顺序执行）
    METHODS = ['rotate', 'crop', 'brightness', 'scale']

    # 各方法的参数（可选，不填用默认值）
    PARAMS = {
        'rotate':    {'angle_range': (-180, 180)},
        'crop':      {'crop_ratio_range': (0.7, 1.0)},
        'brightness': {'factor_range': (0.6, 1.4)},
        'scale':     {'scale_range': (0.8, 1.2)},
        # 'mosaic_blur':    {'grid_size_range': (4, 16)},
        # 'mosaic_shuffle': {'grid_size_range': (32, 96)},
        # 'translate':      {'shift_range': (-0.2, 0.2)},
    }

    SEED = 42

    # ---------- 执行 ----------

    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"每张图生成: {NUM_AUGMENTS} 张")
    print(f"增强方法: {METHODS}")
    print(f"自定义参数: {PARAMS}")
    print("=" * 50)

    image_dict = collect_images(INPUT_DIR)
    if not image_dict:
        print("未找到图像！请检查输入目录。")
    else:
        total = sum(len(v) for v in image_dict.values())
        print(f"找到 {len(image_dict)} 个子目录，共 {total} 张原图\n")

        batch_augment(image_dict, OUTPUT_DIR, num=NUM_AUGMENTS,
                      methods=METHODS, params=PARAMS, seed=SEED)
