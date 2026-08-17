"""Gated-fusion extension for the SkinSeg-YOLO26 P2-CBAM v2B model."""
import torch.nn as nn

from cbam import CBAM, P2CompatibleSegment26


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


def register_v2b():
    """Register baseline modules plus the v2B GatedFusion block for YAML parsing."""
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
