from pathlib import Path
import sys

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_gated_fusion_is_channel_preserving_and_registered_by_identity():
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks
    from cbam_v2b import GatedFusion, register_v2b

    register_v2b()
    assert modules.GatedFusion is GatedFusion
    assert tasks.GatedFusion is GatedFusion

    layer = GatedFusion(128, 128)
    x = torch.randn(2, 128, 16, 16)
    output = layer(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()

    with pytest.raises(ValueError, match="preserve channels"):
        GatedFusion(128, 64)


def test_v2b_yaml_adds_two_gated_fusion_layers_without_modifying_v1():
    baseline = yaml.safe_load((ROOT / "models" / "yolo26n-seg-p2-cbam.yaml").read_text(encoding="utf-8"))
    v2b = yaml.safe_load(
        (ROOT / "models" / "yolo26n-seg-p2-cbam-v2b-gatedfusion.yaml").read_text(encoding="utf-8")
    )

    baseline_modules = [row[2] for row in baseline["backbone"] + baseline["head"]]
    v2b_modules = [row[2] for row in v2b["backbone"] + v2b["head"]]

    assert baseline_modules.count("GatedFusion") == 0
    assert baseline["head"][-1][0] == [21, 24, 27, 30]

    assert v2b_modules.count("GatedFusion") == 2
    assert v2b["head"][5][2] == "GatedFusion"
    assert v2b["head"][9][2] == "GatedFusion"
    assert v2b["head"][-1][0] == [23, 26, 29, 32]


def test_v2b_model_build_and_eval_forward_has_four_finite_scales():
    from cbam import P2CompatibleSegment26
    from cbam_v2b import register_v2b
    from ultralytics import YOLO

    register_v2b()
    model = YOLO(str(ROOT / "models" / "yolo26n-seg-p2-cbam-v2b-gatedfusion.yaml")).model
    head = model.model[-1]

    assert isinstance(head, P2CompatibleSegment26)
    assert head.nl == 4
    assert head.reg_max == 1
    assert head.end2end is True
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]

    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 256, 256))

    predictions, auxiliary = output
    detections, prediction_prototypes = predictions
    assert torch.isfinite(detections).all()
    assert torch.isfinite(prediction_prototypes).all()
    assert prediction_prototypes.shape[-2:] == (64, 64)
    for branch in ("one2many", "one2one"):
        assert len(auxiliary[branch]["feats"]) == 4
        assert torch.isfinite(auxiliary[branch]["mask_coefficient"]).all()
        assert torch.isfinite(auxiliary[branch]["proto"]).all()
