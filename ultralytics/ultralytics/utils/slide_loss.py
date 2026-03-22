"""Slide Loss: IoU-adaptive BCE weighting for classification branch.

Paper: APMLF-YOLO Section 3.3, Eq. 2
Applied ONLY to classification BCE loss (not box/DFL loss).
μ = auto_iou (mean CIoU of positive-assigned samples from TAL).

Weighting function (paper Eq. 2 — using e^(1-μ) NOT e^μ):
  f(x) = 1.0        if x <= μ - 0.1       (easy negatives: full weight)
  f(x) = e^(1-μ)    if μ-0.1 < x < μ     (transition: constant per batch)
  f(x) = e^(1-x)    if x >= μ             (positives: decreasing with IoU)

Key difference from YOLO-FaceV2 (which uses e^μ): the paper specifies 1-μ.
This makes the transition zone weight > 1 when μ < 1, emphasizing hard samples.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlideLoss(nn.Module):
    """IoU-adaptive classification loss weighting.

    Wraps BCE with IoU-derived per-sample weights.

    Args:
        reduction: 'mean' or 'sum' (default: 'mean')
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        auto_iou: float = 0.5,
    ) -> torch.Tensor:
        # Clamp auto_iou to minimum 0.2 (avoid degenerate weights at start of training)
        auto_iou = max(float(auto_iou), 0.2)
        # Clamp targets to valid IoU range; prevents exp overflow if caller passes raw logits
        true = true.clamp(0.0, 1.0)

        # Base BCE loss (unreduced)
        loss = F.binary_cross_entropy_with_logits(pred, true, reduction='none')

        # Three zones based on paper Eq. 2
        b1 = true <= (auto_iou - 0.1)                          # easy negatives
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)    # transition zone
        b3 = true >= auto_iou                                   # positive zone

        # Weights: PAPER formula (e^(1-μ) for middle, NOT e^μ)
        w = (
            b1.float()
            + math.exp(1.0 - auto_iou) * b2.float()
            + torch.exp(1.0 - true) * b3.float()
        )
        loss = loss * w

        return loss.mean() if self.reduction == 'mean' else loss.sum()


from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors


class v8DetectionLossWithSlide(v8DetectionLoss):
    """v8DetectionLoss with IoU-adaptive SlideLoss for classification branch.

    Replaces BCEWithLogitsLoss with SlideLoss. Overrides __call__ to:
    1. Run TAL assignment (same as parent)
    2. Compute mean CIoU of TAL-positive predictions as auto_iou
    3. Pass auto_iou to SlideLoss for classification weighting

    auto_iou = mean CIoU of TAL-positive predictions (paper: "mu" in Eq.2).
    Falls back to 0.5 when no positives assigned (e.g. first iterations).
    """

    def __init__(self, model):
        # Ensure model.args is present (may be absent when constructing model directly
        # outside of training; training always attaches args via the checkpoint loader).
        if not hasattr(model, 'args'):
            from ultralytics.utils import DEFAULT_CFG
            model.args = DEFAULT_CFG
        super().__init__(model)
        self.bce = SlideLoss()  # Replace BCEWithLogitsLoss with SlideLoss

    # NOTE: This method is a copy of v8DetectionLoss.__call__ from Ultralytics v8.2.103.
    # If upgrading ultralytics, re-sync this method body with the parent class.
    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Compute auto_iou: mean CIoU of TAL-positive assignments
        # This mirrors the iou computation inside BboxLoss.forward (same formula).
        # We compute it here (before bbox_loss call) to pass to SlideLoss.
        if fg_mask.sum() > 0:
            # target_bboxes are still in stride-scaled space here (before /= stride_tensor below)
            _pred_pos = (pred_bboxes.detach() * stride_tensor)[fg_mask]
            _tgt_pos = target_bboxes[fg_mask]  # already stride-scaled (from assigner output)
            _iou = bbox_iou(_pred_pos, _tgt_pos, xywh=False, CIoU=True)
            auto_iou = float(_iou.mean().clamp(0.2, 1.0))
        else:
            auto_iou = 0.5

        # Cls loss — use SlideLoss with dynamic auto_iou
        loss[1] = self.bce(pred_scores, target_scores.to(dtype), auto_iou=auto_iou).sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
