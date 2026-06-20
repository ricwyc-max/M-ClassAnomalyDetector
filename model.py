"""
ResNet-50 模型定义（多尺度特征融合版本）
基于 addBlock.py 中的 ResidualBottleneckBlock（前激活 / ResNet-v2 风格）

ResNet-50 多尺度特征融合结构：
    Input (B, 3, H, W)  — H, W 为任意尺寸，需 >= 224
    │
    ├── Stem ─────────────────────────────────────────────────────
    │   Conv2d(3→64, 7×7, stride=2, padding=3)   → 64×(H/2)×(W/2)
    │   MaxPool2d(3×3, stride=2, padding=1)       → 64×(H/4)×(W/4)
    │                                                ↓ 上采样到 H×W
    │
    ├── Stage 1 (layer1) ────────────────────────────────────────
    │   3 × Bottleneck(mid=64,  out=256)           → 256×(H/4)×(W/4)
    │                                                ↓ 上采样到 H×W
    │
    ├── Stage 2 (layer2) ────────────────────────────────────────
    │   1 × Bottleneck(mid=128, out=512, stride=2) → 512×(H/8)×(W/8)
    │   3 × Bottleneck(mid=128, out=512)           → 512×(H/8)×(W/8)
    │                                                ↓ 上采样到 H×W
    │
    ├── Stage 3 (layer3) ────────────────────────────────────────
    │   1 × Bottleneck(mid=256, out=1024, stride=2)→ 1024×(H/16)×(W/16)
    │   5 × Bottleneck(mid=256, out=1024)          → 1024×(H/16)×(W/16)
    │                                                ↓ 上采样到 H×W
    │
    ├── Stage 4 (layer4) ────────────────────────────────────────
    │   1 × Bottleneck(mid=512, out=2048, stride=2)→ 2048×(H/32)×(W/32)
    │   2 × Bottleneck(mid=512, out=2048)          → 2048×(H/32)×(W/32)
    │                                                ↓ 上采样到 H×W
    │
    ├── Concat ───────────────────────────────────────────────────
    │   拼接 5 个尺度的特征图                    → (64+256+512+1024+2048)×H×W
    │                                               = 3904×H×W
    │
    └── Head ────────────────────────────────────────────────────
        Conv1x1(3904→num_classes)                 → num_classes×H×W
        AdaptiveAvgPool2d(1)                      → num_classes×1×1
        Flatten                                   → num_classes
        Softmax(dim=1)                            → num_classes

    最小输入尺寸：224×224（确保 layer4 输出 >= 1×1）
    无最大尺寸限制（GAP 适配任意尺寸）

设计说明：
    - 多尺度特征融合：提取每个 stage 的输出，上采样到原始尺寸后拼接
    - 浅层特征（layer1）包含细节信息，深层特征（layer4）包含语义信息
    - 融合后的特征同时具备细节和语义，适合细粒度分类和异常检测
    - 最终 1×1 卷积 + GAP 实现全卷积分类，无全连接层

前激活（Pre-activation / ResNet-v2）设计说明：
    - 标准 ResNet-v1 顺序：Conv → BN → ReLU → ... → Add → ReLU
    - 前激活 ResNet-v2 顺序：BN → ReLU → Conv → ... → Add（末端无 ReLU）
    - 优势：残差分支末端没有 ReLU，梯度可以无阻碍地流过跳跃连接，
      在 100+ 层的深层网络中训练更稳定、收敛更好

参考论文：
    [1] He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
    [2] He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016 (ResNet-v2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from addBlock import ResidualBottleneckBlock


class ResNet50(nn.Module):
    """
    基于前激活 Bottleneck Block 的 ResNet-50（多尺度特征融合版本）

    Args:
        num_classes (int): 分类类别数，默认 1000（ImageNet）
        in_channels (int): 输入图像通道数，默认 3（RGB）

    最小输入尺寸：224×224（确保 layer4 输出 >= 1×1）
    """

    def __init__(self, num_classes=1000, in_channels=3):
        super().__init__()

        # 各阶段 Bottleneck 中间通道数（窄通道，即 1×1 降维后的通道数）
        #   Stage 1: mid=64  → out=64×4=256
        #   Stage 2: mid=128 → out=128×4=512
        #   Stage 3: mid=256 → out=256×4=1024
        #   Stage 4: mid=512 → out=512×4=2048
        mid_channels = [64, 128, 256, 512]
        out_channels = [256, 512, 1024, 2048]  # 各阶段输出通道数

        # 各阶段 Bottleneck 块数（ResNet-50 标准配置：3+4+6+3=16 个 Bottleneck）
        num_blocks = [3, 4, 6, 3]

        # ======================== Stem（主干网络入口）========================
        # 包含一个 7×7 大卷积核（提取底层特征 + 下采样）和一个最大池化层
        # 注意：前激活模式下，Conv1 后面不接 BN 和 ReLU，
        #       因为第一个 Bottleneck 的开头会执行 BN→ReLU，等效于对 stem 输出做激活
        self.stem = nn.Sequential(
            # 7×7 卷积，stride=2 将空间尺寸减半：224→112
            # bias=False：后面紧跟 BN，偏置是冗余的
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            # 3×3 最大池化，stride=2 再次减半：112→56
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ======================== 4 个残差阶段 ========================
        # 每个阶段的第一个 Bottleneck 可能需要下采样（stride≠1 或通道数变化）
        #   layer1: 64→256,   stride=1, 空间保持 56×56
        #   layer2: 256→512,  stride=2, 空间缩小 56→28
        #   layer3: 512→1024, stride=2, 空间缩小 28→14
        #   layer4: 1024→2048,stride=2, 空间缩小 14→7
        self.layer1 = self._make_layer(64,   mid_channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(256,  mid_channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(512,  mid_channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(1024, mid_channels[3], num_blocks[3], stride=2)

        # ======================== Head（分类头）========================
        # 拼接后的通道数：64 + 256 + 512 + 1024 + 2048 = 3904
        concat_channels = 64 + sum(out_channels)

        # 1×1 卷积：将拼接后的特征映射到类别数
        # 每个通道即为该类别的类激活图（CAM），可用于可视化
        self.cls_conv = nn.Conv2d(concat_channels, num_classes, kernel_size=1, bias=False)

        # GAP：压缩空间信息
        # Softmax：输出概率分布
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, in_channels, mid_channels, num_blocks, stride):
        """
        构建一个残差阶段（Stage）

        Args:
            in_channels  (int): 该阶段第一个块的输入通道数
            mid_channels (int): Bottleneck 中间通道数（瓶颈窄通道）
            num_blocks   (int): 该阶段包含的 Bottleneck 块数
            stride       (int): 第一个块的步长（>1 时空间尺寸减半）

        Returns:
            nn.Sequential: 由 num_blocks 个 Bottleneck 堆叠组成的阶段

        数据流示意（以 layer2 为例，in=256, mid=128, out=512, stride=2）：
            输入: (B, 256, 56, 56)
            │
            ├─ Block 0（stride=2, 带 downsample）── 主分支和 shortcut 都做空间减半
            │   主分支: BN→ReLU→Conv1x1(256→128)→BN→ReLU→Conv3x3(128→128, s=2)→BN→ReLU→Conv1x1(128→512)
            │   shortcut: BN→ReLU→Conv1x1(256→512, s=2)
            │   输出: (B, 512, 28, 28)
            │
            ├─ Block 1（stride=1, 无 downsample）
            │   输出: (B, 512, 28, 28)
            │
            └─ Block 2（stride=1, 无 downsample）
                输出: (B, 512, 28, 28)
        """
        out_channels = mid_channels * 4  # expansion=4，Bottleneck 输出通道 = 中间通道 × 4

        # ------ 下采样分支（shortcut 路径）------
        # 当 stride≠1（空间尺寸变化）或 in≠out（通道数不匹配）时需要下采样
        # 前激活风格：BN → ReLU → Conv1x1（而不是 Conv1x1 → BN）
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.BatchNorm2d(in_channels),     # 先对输入做归一化
                nn.ReLU(inplace=True),           # 激活
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),  # 1×1 卷积调整通道和尺寸
            )

        # ------ 堆叠 Bottleneck 块 ------
        layers = []

        # 第一个块：可能需要下采样（stride>1 或通道变化）
        layers.append(ResidualBottleneckBlock(
            in_channels, mid_channels, out_channels,
            stride=stride, downsample=downsample, preactivated=True,
        ))

        # 后续块：通道数不变，stride=1，无需下采样
        for _ in range(1, num_blocks):
            layers.append(ResidualBottleneckBlock(
                out_channels, mid_channels, out_channels,
                stride=1, downsample=None, preactivated=True,
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播（多尺度特征融合）

        Args:
            x (Tensor): 输入图像张量，形状 (B, in_channels, H, W)
                        通常为 (B, 3, 224, 224)

        Returns:
            Tensor: 各类别概率分布，形状 (B, num_classes)

        数据流（以输入 224×224 为例）：
            (B, 3, H, W)  — H, W 为任意尺寸，需 >= 224
              → stem   → (B, 64, H/4, W/4)    → 上采样 → (B, 64, H, W)
              → layer1 → (B, 256, H/4, W/4)   → 上采样 → (B, 256, H, W)
              → layer2 → (B, 512, H/8, W/8)   → 上采样 → (B, 512, H, W)
              → layer3 → (B, 1024, H/16, W/16) → 上采样 → (B, 1024, H, W)
              → layer4 → (B, 2048, H/32, W/32) → 上采样 → (B, 2048, H, W)
              → concat → (B, 3904, H, W)
              → head   → (B, num_classes)  概率分布
        """
        # 记录输入尺寸，用于上采样目标
        input_size = x.shape[2:]  # (H, W)

        x = self.stem(x)       # 主干：7×7 Conv + MaxPool，输出 64×(H/4)×(W/4)

        # 提取每个阶段的输出
        feat0 = x                  # (B, 64, H/4, W/4)    stem 输出
        feat1 = self.layer1(x)     # (B, 256, H/4, W/4)
        feat2 = self.layer2(feat1) # (B, 512, H/8, W/8)
        feat3 = self.layer3(feat2) # (B, 1024, H/16, W/16)
        feat4 = self.layer4(feat3) # (B, 2048, H/32, W/32)

        # 上采样到原始输入尺寸（动态获取，无需固定 target_size）
        feat0 = F.interpolate(feat0, size=input_size, mode='bilinear', align_corners=False)  # (B, 64, H, W)
        feat1 = F.interpolate(feat1, size=input_size, mode='bilinear', align_corners=False)  # (B, 256, H, W)
        feat2 = F.interpolate(feat2, size=input_size, mode='bilinear', align_corners=False)  # (B, 512, H, W)
        feat3 = F.interpolate(feat3, size=input_size, mode='bilinear', align_corners=False)  # (B, 1024, H, W)
        feat4 = F.interpolate(feat4, size=input_size, mode='bilinear', align_corners=False)  # (B, 2048, H, W)

        # 在通道维度拼接多尺度特征
        x = torch.cat([feat0, feat1, feat2, feat3, feat4], dim=1)  # (B, 3904, 224, 224)

        # 分类头：1×1 Conv + GAP + Softmax
        cam = self.cls_conv(x)      # (B, num_classes, H, W) 类激活图
        x = self.gap(cam)           # (B, num_classes, 1, 1)
        x = self.flatten(x)         # (B, num_classes)
        x = self.softmax(x)         # (B, num_classes) 概率分布
        return x, cam


# ======================== 测试代码 ========================
if __name__ == '__main__':
    # 创建 ResNet-50，10 分类任务
    model = ResNet50(num_classes=10)

    # 测试 1：标准 224×224 输入
    print("=" * 50)
    print("测试 1：标准 224×224 输入")
    x = torch.randn(1, 3, 224, 224)
    out, cam = model(x)
    print(f"输入形状: {x.shape}")           # (1, 3, 224, 224)
    print(f"输出形状: {out.shape}")         # (1, 10)
    print(f"CAM 形状: {cam.shape}")         # (1, 10, 224, 224)
    print(f"概率和: {out.sum(dim=1).item():.4f}")  # ≈ 1.0

    # 测试 2：更大的输入尺寸（如 320×320）
    print("\n" + "=" * 50)
    print("测试 2：更大输入 320×320")
    x = torch.randn(1, 3, 320, 320)
    out, cam = model(x)
    print(f"输入形状: {x.shape}")           # (1, 3, 320, 320)
    print(f"输出形状: {out.shape}")         # (1, 10)
    print(f"CAM 形状: {cam.shape}")         # (1, 10, 320, 320)
    print(f"概率和: {out.sum(dim=1).item():.4f}")  # ≈ 1.0

    # 测试 3：非正方形输入（如 256×384）
    print("\n" + "=" * 50)
    print("测试 3：非正方形输入 256×384")
    x = torch.randn(1, 3, 256, 384)
    out, cam = model(x)
    print(f"输入形状: {x.shape}")           # (1, 3, 256, 384)
    print(f"输出形状: {out.shape}")         # (1, 10)
    print(f"CAM 形状: {cam.shape}")         # (1, 10, 256, 384)
    print(f"概率和: {out.sum(dim=1).item():.4f}")  # ≈ 1.0
