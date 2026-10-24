"""The box branch must not send gradients into the shared features when detached."""

import torch

from skylens_model.models import SkyLensForDisasterPerception
from skylens_model.models.skylensnet.configuration_skylensnet import SkyLensConfig


def _box_grad_reaches(detach: bool) -> tuple[bool, bool]:
    cfg = SkyLensConfig(wh_detach=detach, person_head_stride=2, use_offset_head=True)
    model = SkyLensForDisasterPerception(cfg)
    model.train()
    out = model(pixel_values=torch.randn(1, 4, 128, 128),
                modality_mask=torch.tensor([[True, False]]))
    named = list(model.named_parameters())
    shared = [p for n, p in named if "wh_" not in n and p.requires_grad]
    box = [p for n, p in named if "wh_" in n and p.requires_grad]

    def any_grad(params):
        g = torch.autograd.grad(out.person_wh.sum(), params, retain_graph=True, allow_unused=True)
        return any(x is not None and bool(x.abs().sum() > 0) for x in g)

    return any_grad(shared), any_grad(box)


def test_box_branch_feeds_the_backbone_by_default():
    shared, box = _box_grad_reaches(detach=False)
    assert shared, "without --wh-detach the box loss should still shape the shared features"
    assert box


def test_wh_detach_isolates_the_box_branch():
    shared, box = _box_grad_reaches(detach=True)
    assert not shared, "with --wh-detach no box gradient may reach the shared features"
    assert box, "the box stem and head must still learn"
