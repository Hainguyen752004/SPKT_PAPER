"""Channel-preserving CBAM used by the SkinSeg-YOLO26 P2 model."""
import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.block import Proto26
from ultralytics.nn.modules.head import Segment26 as UltralyticsSegment26


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


class GatedFusion(nn.Module):
    """Learn channel and spatial gates for an already-concatenated fusion tensor."""

    def __init__(self, c1, c2=None, reduction=16, kernel_size=7):
        super().__init__()
        channels = c1 if c2 is None else c2
        if channels != c1:
            raise ValueError(f"GatedFusion must preserve channels (received {c1} -> {channels})")
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        hidden = max(channels // reduction, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.channel_gate(x) * self.spatial_gate(x)


class P2CompatibleSegment26(UltralyticsSegment26):
    """Segment26 variant that keeps P2 predictions but uses stride-4 mask prototypes."""

    def __init__(self, nc=80, nm=32, npr=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        proto_channels = ch[1:] if len(ch) >= 4 else ch
        self.proto = Proto26(proto_channels, self.npr, self.nm, nc)

    def forward(self, x):
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto_input = x[1:] if len(x) >= 4 else x
        proto = self.proto(proto_input)
        if isinstance(preds, dict):
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


def register_cbam():
    """Register custom modules in both namespaces used by the YAML parser."""
    import ultralytics.nn.modules as block_modules
    import ultralytics.nn.tasks as task_modules

    setattr(block_modules, "CBAM", CBAM)
    setattr(task_modules, "CBAM", CBAM)
    setattr(block_modules, "GatedFusion", GatedFusion)
    setattr(task_modules, "GatedFusion", GatedFusion)
    setattr(block_modules, "Segment26", P2CompatibleSegment26)
    setattr(task_modules, "Segment26", P2CompatibleSegment26)
    assert block_modules.CBAM is CBAM and task_modules.CBAM is CBAM, "CBAM registration identity mismatch"
    assert (
        block_modules.GatedFusion is GatedFusion
        and task_modules.GatedFusion is GatedFusion
    ), "GatedFusion registration identity mismatch"
    assert (
        block_modules.Segment26 is P2CompatibleSegment26
        and task_modules.Segment26 is P2CompatibleSegment26
    ), "Segment26 registration identity mismatch"
