"""Channel-preserving CBAM used by the SkinSeg-YOLO26 P2 model."""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pooled = torch.cat((torch.mean(x, dim=1, keepdim=True), torch.amax(x, dim=1, keepdim=True)), dim=1)
        return self.sigmoid(self.conv(pooled)) * x


class CBAM(nn.Module):
    """Apply channel and spatial attention without changing tensor shape."""
    def __init__(self, c1, c2=None, reduction=16, kernel_size=7):
        super().__init__()
        channels = c1 if c2 is None else c2
        if channels != c1:
            raise ValueError(f"CBAM must preserve channels (received {c1} -> {channels})")
        if channels < reduction:
            raise ValueError(f"CBAM channels ({channels}) must be at least reduction ({reduction})")
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


def register_cbam():
    """Register exactly this class in both namespaces used by the YAML parser."""
    import ultralytics.nn.modules as block_modules
    import ultralytics.nn.tasks as task_modules

    setattr(block_modules, "CBAM", CBAM)
    setattr(task_modules, "CBAM", CBAM)
    assert block_modules.CBAM is CBAM and task_modules.CBAM is CBAM, "CBAM registration identity mismatch"
