"""
ResNet-50 模型定义（多尺度特征融合版本）
基于 addBlock.py 中的 ResidualBottleneckBlock 和 DWBottleneckBlock

支持两种残差块：
    - 标准 Bottleneck（前激活 ResNet-v2）
    - 深度可分离 Bottleneck（前激活 + Depthwise Separable Conv）

支持因子调整：
    - width_factor (α): 宽度因子，缩放中间通道数
    - resolution_factor (ρ): 分辨率因子，缩放输入图像尺寸

使用方式：
    # 标准 ResNet-50
    model = ResNet50(num_classes=9)

    # 深度可分离版（轻量）
    model = ResNet50(num_classes=9, use_dw=True, width_factor=0.5, resolution_factor=1.0)

参考论文：
    [1] He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
    [2] He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016 (ResNet-v2)
    [3] Howard et al., "MobileNets: Efficient CNN for Mobile Vision Applications", 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from addBlock import ResidualBottleneckBlock, DWBottleneckBlock


class ResNet50(nn.Module):
    """
    基于前激活 Bottleneck Block 的 ResNet-50（多尺度特征融合版本）

    Args:
        num_classes (int): 分类类别数，默认 1000（ImageNet）
        in_channels (int): 输入图像通道数，默认 3（RGB）
        use_dw (bool): 是否使用深度可分离卷积残差块，默认 False
        width_factor (float): 宽度因子 α，缩放中间通道数，默认 1.0
        resolution_factor (float): 分辨率因子 ρ，缩放空间下采样步长，默认 1.0

    最小输入尺寸：224×224（ρ < 1.0 时，模型内部会缩放到更小尺寸）
    """

    def __init__(self, num_classes=1000, in_channels=3, use_dw=False,
                 width_factor=1.0, resolution_factor=1.0):
        super().__init__()

        self.use_dw = use_dw
        self.width_factor = width_factor
        self.resolution_factor = resolution_factor

        # 选择残差块类型
        Block = DWBottleneckBlock if use_dw else ResidualBottleneckBlock

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
        #
        # DW 模式下，各阶段实际输出通道受 width_factor 缩放，
        # 因此每个 layer 的 in_channels 必须等于前一个 layer 的实际输出通道数
        if use_dw:
            actual_out = [max(1, int(c * width_factor)) for c in out_channels]
        else:
            actual_out = out_channels

        self.layer1 = self._make_layer(Block, 64,             mid_channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(Block, actual_out[0],  mid_channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(Block, actual_out[1],  mid_channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(Block, actual_out[2],  mid_channels[3], num_blocks[3], stride=2)

        # ======================== Head（分类头）========================
        # 拼接后的通道数（DW 模式下通道受 width_factor 缩放）
        if use_dw:
            # stem(64) + 各阶段实际输出通道
            actual_out_channels = [max(1, int(c * width_factor)) for c in out_channels]
            concat_channels = 64 + sum(actual_out_channels)
        else:
            concat_channels = 64 + sum(out_channels)

        # 1×1 卷积：将拼接后的特征映射到类别数
        # 每个通道即为该类别的类激活图（CAM），可用于可视化
        self.cls_conv = nn.Conv2d(concat_channels, num_classes, kernel_size=1, bias=False)

        # GAP：压缩空间信息
        # 注意：不在此处加 Softmax，CrossEntropyLoss 内部已包含 log_softmax
        # 训练时输出 raw logits，推理时可手动加 softmax 得到概率分布
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def _make_layer(self, Block, in_channels, mid_channels, num_blocks, stride):
        """
        构建一个残差阶段（Stage）

        Args:
            Block          (class): 残差块类（ResidualBottleneckBlock 或 DWBottleneckBlock）
            in_channels  (int): 该阶段第一个块的输入通道数
            mid_channels (int): Bottleneck 中间通道数（瓶颈窄通道）
            num_blocks   (int): 该阶段包含的 Bottleneck 块数
            stride       (int): 第一个块的步长（>1 时空间尺寸减半）

        Returns:
            nn.Sequential: 由 num_blocks 个 Block 堆叠组成的阶段
        """
        # 扩展后通道数
        out_channels = mid_channels * 4  # expansion=4

        # DW 模式下，实际输出通道受 width_factor 缩放
        if Block is DWBottleneckBlock:
            actual_out = max(1, int(out_channels * self.width_factor))
        else:
            actual_out = out_channels

        # ------ 下采样分支（shortcut 路径）------
        # downsample 输出通道必须和块的实际输出通道一致
        downsample = None
        if stride != 1 or in_channels != actual_out:
            downsample = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, actual_out, kernel_size=1,
                          stride=stride, bias=False),
            )

        # ------ 堆叠残差块 ------
        layers = []

        # 第一个块：可能需要下采样
        if Block is DWBottleneckBlock:
            layers.append(Block(
                in_channels, mid_channels, out_channels,
                stride=stride, downsample=downsample,
                width_factor=self.width_factor,
            ))
            # 后续块：输入通道 = actual_out
            for _ in range(1, num_blocks):
                layers.append(Block(
                    actual_out, mid_channels, out_channels,
                    stride=1, downsample=None,
                    width_factor=self.width_factor,
                ))
        else:
            # ResidualBottleneckBlock（标准模式）
            layers.append(Block(
                in_channels, mid_channels, out_channels,
                stride=stride, downsample=downsample, preactivated=True,
            ))
            for _ in range(1, num_blocks):
                layers.append(Block(
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

        数据流（以输入 224×224，ρ=1.0 为例）：
            (B, 3, H, W)  — H, W 为任意尺寸，需 >= 224
              → stem   → (B, 64, H/4, W/4)    → 上采样 → (B, 64, H, W)
              → layer1 → (B, 256, H/4, W/4)   → 上采样 → (B, 256, H, W)
              → layer2 → (B, 512, H/8, W/8)   → 上采样 → (B, 512, H, W)
              → layer3 → (B, 1024, H/16, W/16) → 上采样 → (B, 1024, H, W)
              → layer4 → (B, 2048, H/32, W/32) → 上采样 → (B, 2048, H, W)
              → concat → (B, 3904, H, W)
              → head   → (B, num_classes)  概率分布

        分辨率因子 ρ 的作用（以 ρ=0.5 为例）：
            输入 224×224 → 缩放到 112×112 → 网络处理 → 特征图 112×112 → CAM 112×112
            减少空间维度的计算量和显存占用
        """
        # 分辨率因子：缩放输入图像尺寸，后续特征图和 CAM 均为缩放后的分辨率
        if self.resolution_factor != 1.0:
            h, w = x.shape[2:]
            scaled_h = max(32, int(h * self.resolution_factor))
            scaled_w = max(32, int(w * self.resolution_factor))
            x = F.interpolate(x, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)

        # 记录处理尺寸，用于上采样目标
        input_size = x.shape[2:]  # (H', W')

        x = self.stem(x)       # 主干：7×7 Conv + MaxPool，输出 64×(H'/4)×(W'/4)

        # 提取每个阶段的输出（通道数受 width_factor 影响）
        feat0 = x                  # (B, 64, H/4, W/4)    stem 输出
        feat1 = self.layer1(x)     # (B, C1, H/4, W/4)
        feat2 = self.layer2(feat1) # (B, C2, H/8, W/8)
        feat3 = self.layer3(feat2) # (B, C3, H/16, W/16)
        feat4 = self.layer4(feat3) # (B, C4, H/32, W/32)

        # 上采样到原始输入尺寸（动态获取，无需固定 target_size）
        feat0 = F.interpolate(feat0, size=input_size, mode='bilinear', align_corners=False)  # (B, 64, H, W)
        feat1 = F.interpolate(feat1, size=input_size, mode='bilinear', align_corners=False)  # (B, 256, H, W)
        feat2 = F.interpolate(feat2, size=input_size, mode='bilinear', align_corners=False)  # (B, 512, H, W)
        feat3 = F.interpolate(feat3, size=input_size, mode='bilinear', align_corners=False)  # (B, 1024, H, W)
        feat4 = F.interpolate(feat4, size=input_size, mode='bilinear', align_corners=False)  # (B, 2048, H, W)

        # 在通道维度拼接多尺度特征
        x = torch.cat([feat0, feat1, feat2, feat3, feat4], dim=1)  # (B, 3904, 224, 224)

        # 分类头：1×1 Conv + GAP（输出 raw logits，不加 softmax）
        cam = self.cls_conv(x)      # (B, num_classes, H, W) 类激活图
        x = self.gap(cam)           # (B, num_classes, 1, 1)
        x = self.flatten(x)         # (B, num_classes) raw logits
        return x, cam


# ======================== 测试代码 ========================
if __name__ == '__main__':
    print("=" * 60)
    print("测试 1：标准 ResNet-50（Bottleneck）")
    print("=" * 60)
    model = ResNet50(num_classes=10, use_dw=False)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(1, 3, 224, 224)
    out, cam = model(x)
    print(f"输入: {x.shape}  输出: {out.shape}  CAM: {cam.shape}")
    print(f"logits 范围: [{out.min().item():.2f}, {out.max().item():.2f}]")

    print("\n" + "=" * 60)
    print("测试 2：深度可分离 ResNet-50（DWBottleneck, α=1.0, ρ=1.0）")
    print("=" * 60)
    model_dw = ResNet50(num_classes=10, use_dw=True, width_factor=1.0, resolution_factor=1.0)
    print(f"参数量: {sum(p.numel() for p in model_dw.parameters()):,}")

    x = torch.randn(1, 3, 224, 224)
    out, cam = model_dw(x)
    print(f"输入: {x.shape}  输出: {out.shape}  CAM: {cam.shape}")
    print(f"logits 范围: [{out.min().item():.2f}, {out.max().item():.2f}]")

    print("\n" + "=" * 60)
    print("测试 3：深度可分离 + 宽度因子 α=0.5（轻量版）")
    print("=" * 60)
    model_dw_half = ResNet50(num_classes=10, use_dw=True, width_factor=0.5, resolution_factor=1.0)
    print(f"参数量: {sum(p.numel() for p in model_dw_half.parameters()):,}")

    x = torch.randn(1, 3, 224, 224)
    out, cam = model_dw_half(x)
    print(f"输入: {x.shape}  输出: {out.shape}  CAM: {cam.shape}")
    print(f"logits 范围: [{out.min().item():.2f}, {out.max().item():.2f}]")

    print("\n" + "=" * 60)
    print("测试 4：深度可分离 + 分辨率因子 ρ=0.5（加速版）")
    print("=" * 60)
    model_dw_lowres = ResNet50(num_classes=10, use_dw=True, width_factor=1.0, resolution_factor=0.5)
    print(f"参数量: {sum(p.numel() for p in model_dw_lowres.parameters()):,}")

    # ρ=0.5 会将输入缩放到 112×112，减少计算量
    x = torch.randn(1, 3, 224, 224)
    out, cam = model_dw_lowres(x)
    print(f"输入: {x.shape}  输出: {out.shape}  CAM: {cam.shape}")
    print(f"logits 范围: [{out.min().item():.2f}, {out.max().item():.2f}]")
