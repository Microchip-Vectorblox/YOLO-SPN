from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss

from yolo.config.config import Config, LossConfig
from yolo.utils.bounding_box_utils import BoxMatcher, Vec2Box, calculate_iou
from yolo.utils.logger import logger


class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Refactor the device, should be assign by config
        # TODO: origin v9 assing pos_weight == 1?
        self.bce = BCEWithLogitsLoss(reduction="none")

    def forward(self, predicts_cls: Tensor, targets_cls: Tensor, cls_norm: Tensor) -> Any:
        return self.bce(predicts_cls, targets_cls).sum() / cls_norm


class BoxLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, predicts_bbox: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        picked_predict = predicts_bbox[valid_bbox].view(-1, 4)
        picked_targets = targets_bbox[valid_bbox].view(-1, 4)

        iou = calculate_iou(picked_predict, picked_targets, "ciou").diag()
        loss_iou = 1.0 - iou
        loss_iou = (loss_iou * box_norm).sum() / cls_norm
        return loss_iou


class DFLoss(nn.Module):
    def __init__(self, vec2box: Vec2Box, reg_max: int) -> None:
        super().__init__()
        self.anchors_norm = (vec2box.anchor_grid / vec2box.scaler[:, None])[None]
        self.reg_max = reg_max

    def forward(
        self, predicts_anc: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        bbox_lt, bbox_rb = targets_bbox.chunk(2, -1)
        targets_dist = torch.cat(((self.anchors_norm - bbox_lt), (bbox_rb - self.anchors_norm)), -1).clamp(
            0, self.reg_max - 1.01
        )
        picked_targets = targets_dist[valid_bbox].view(-1)
        picked_predict = predicts_anc[valid_bbox].view(-1, self.reg_max)

        label_left, label_right = picked_targets.floor(), picked_targets.floor() + 1
        weight_left, weight_right = label_right - picked_targets, picked_targets - label_left

        loss_left = F.cross_entropy(picked_predict, label_left.to(torch.long), reduction="none")
        loss_right = F.cross_entropy(picked_predict, label_right.to(torch.long), reduction="none")
        loss_dfl = loss_left * weight_left + loss_right * weight_right
        loss_dfl = loss_dfl.view(-1, 4).mean(-1)
        loss_dfl = (loss_dfl * box_norm).sum() / cls_norm
        return loss_dfl


class YOLOLoss:
    def __init__(self, loss_cfg: LossConfig, vec2box: Vec2Box, class_num: int = 80, reg_max: int = 16) -> None:
        self.class_num = class_num
        self.vec2box = vec2box

        self.cls = BCELoss()
        self.dfl = DFLoss(vec2box, reg_max)
        self.iou = BoxLoss()

        self.matcher = BoxMatcher(loss_cfg.matcher, self.class_num, vec2box, reg_max)

    def separate_anchor(self, anchors):
        """
        separate anchor and bbouding box
        """
        anchors_cls, anchors_box = torch.split(anchors, (self.class_num, 4), dim=-1)
        anchors_box = anchors_box / self.vec2box.scaler[None, :, None]
        return anchors_cls, anchors_box

    def __call__(self, predicts: List[Tensor], targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        predicts_cls, predicts_anc, predicts_box = predicts
        # For each predicted targets, assign a best suitable ground truth box.
        align_targets, valid_masks = self.matcher(targets, (predicts_cls.detach(), predicts_box.detach()))

        targets_cls, targets_bbox = self.separate_anchor(align_targets)
        predicts_box = predicts_box / self.vec2box.scaler[None, :, None]

        cls_norm = max(targets_cls.sum(), 1)
        box_norm = targets_cls.sum(-1)[valid_masks]

        ## -- CLS -- ##
        loss_cls = self.cls(predicts_cls, targets_cls, cls_norm)
        ## -- IOU -- ##
        loss_iou = self.iou(predicts_box, targets_bbox, valid_masks, box_norm, cls_norm)
        ## -- DFL -- ##
        loss_dfl = self.dfl(predicts_anc, targets_bbox, valid_masks, box_norm, cls_norm)

        return loss_iou, loss_dfl, loss_cls


class YOLOSPLoss:
    def __init__(self, loss_cfg: LossConfig, vec2box: Vec2Box, class_num: int = 80, reg_max: int = 16) -> None:
        self.class_num = class_num
        self.vec2box = vec2box

        self.cls = BCELoss()
        self.dfl = DFLoss(vec2box, reg_max)
        self.iou = BoxLoss()
        self.sp = SPLoss()

        self.matcher = BoxMatcher(loss_cfg.matcher, self.class_num, vec2box, reg_max)

    def separate_anchor(self, anchors):
        """
        separate anchor and bbouding box
        """
        anchors_cls, anchors_box = torch.split(anchors, (self.class_num, 4), dim=-1)
        anchors_box = anchors_box / self.vec2box.scaler[None, :, None]
        return anchors_cls, anchors_box

    def __call__(self, predicts: List[Tensor], targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        predicts_cls, predicts_anc, predicts_box, predicts_trans, predicts_rot = predicts
        # For each predicted targets, assign a best suitable ground truth box.
        align_targets, valid_masks = self.matcher(targets['bboxes'], (predicts_cls.detach(), predicts_box.detach()))

        targets_cls, targets_bbox = self.separate_anchor(align_targets)
        predicts_box = predicts_box / self.vec2box.scaler[None, :, None]

        cls_norm = max(targets_cls.sum(), 1)
        box_norm = targets_cls.sum(-1)[valid_masks]

        ## -- CLS -- ##
        loss_cls = self.cls(predicts_cls, targets_cls, cls_norm)
        ## -- IOU -- ##
        loss_iou = self.iou(predicts_box, targets_bbox, valid_masks, box_norm, cls_norm)
        ## -- DFL -- ##
        loss_dfl = self.dfl(predicts_anc, targets_bbox, valid_masks, box_norm, cls_norm)
        ## -- SP  -- ##
        #anchor_states = compute_anchor_state(self.anchors, targets['bboxes'])
        N = predicts_rot.shape[1]
        targets_rot = targets['rotationmatrix'].unsqueeze(1).expand(-1,N,-1,-1)
        targets_trans = targets['translation'].unsqueeze(1).expand(-1,N,-1)
        loss_pos,loss_rot = self.sp(predicts_rot, predicts_trans, targets['rotationmatrix'], targets['translation'], valid_masks)
        #loss_pose = self.pose(rotation_raw_6d,
        #                      translation,
        #                      targets['rotationmatrix'],
        #                      targets['translation'],
        #                      anchor_states)

        return loss_iou, loss_dfl, loss_cls, loss_pos, loss_rot



class SPLoss(nn.Module):
    def __init__(self):
        super(SPLoss, self).__init__()

    def forward(self, r_raw_pr, t_pr, R_gt, t_gt, anchor_states):
        B, N = anchor_states.shape

        # loss_total = []
        # for b in range(B):
        # Apply transformation loss to anchors with IoU > 0.5
        positive_indices = torch.eq(anchor_states, 1)

        # if num_positive_indices > 0:
        # 6D rotation -> DCM [sum(Ni), 3, 3]
        r6d  = r_raw_pr[positive_indices, :]
        R_pr = rot_6d_to_matrix(r6d)

        # Translation [sum(Ni), 3]
        t_pr = t_pr[positive_indices, :]

        # Ground-truth pose into same shape, then take positive indices
        R_gt = R_gt.view(B,1,3,3).repeat(1,N,1,1)
        R_gt = R_gt[positive_indices, :, :]
        t_gt = t_gt.view(B,1,3).repeat(1,N,1)
        t_gt = t_gt[positive_indices, :]

        # Rotation error
        Rdot  = torch.bmm(R_pr, R_gt.transpose(1,2)) # [sum(Ni), 3, 3]
        trace = Rdot.diagonal(offset=0, dim1=1, dim2=2).sum(-1) # Batch trace
        trace = (trace - 1.0)/2.0

        # Gradient diverges as argument to torch.acos approaches +/- 1 ...
        # Solution: clamp the value to never reach exactly +/- 1
        rot_err = torch.acos(trace.clamp(-1+1e-6, 1-1e-6)) # [rad]

        # Translation error (normalized)
        pos_err = torch.linalg.vector_norm(t_pr - t_gt, ord=2, dim=1)
        pos_err = pos_err.div(torch.linalg.vector_norm(t_gt, ord=2, dim=1))

        # Total loss
        loss_rot = rot_err.mean()
        loss_pos = pos_err.mean()

        return loss_pos, loss_rot

@torch.jit.script
def rot_6d_to_matrix(r):
    # r : (.., 6)
    r1_raw = r[..., :3]
    r2_raw = r[..., 3:]

    # First column
    r1 = F.normalize(r1_raw, p=2.0, dim=-1) # [..., 3]

    # Second column
    dot = torch.sum(r1 * r2_raw, dim=-1, keepdim=True)
    r2 = r2_raw - dot * r1
    r2 = F.normalize(r2, p=2.0, dim=-1)

    # Third column
    r3 = torch.cross(r1, r2, dim=-1)

    # Into matrix
    dcm = torch.stack((r1, r2, r3), dim=-1)

    return dcm


def compute_anchor_state(anchors, bbox_gt):
    """ Assign anchor states based on IoU w.r.t. ground-truth bounding box.
        +1: Positive, 0: Negative, -1: Ignore.

        Bounding boxes must be in [xmin, ymin, xmax, ymax] format.

    Args:
        anchors (torch.Tensor): [sum(AHW), 4]
        bbox_gt (torch.Tensor): [B, 4] ground-truth bounding box
    """
    # if bbox_gt.ndim == 1:
    #     assert len(bbox_gt) == 4
    #     bbox_gt = bbox_gt[None, :]

    B = bbox_gt.shape[0]
    N = anchors.shape[0]

    # Anchors and bboxes into same shape
    anchors = anchors.unsqueeze(0).repeat(B,1,1)
    bbox_gt = bbox_gt.unsqueeze(1).repeat(1,N,1)

    anchor_state = torch.ones((B, N), dtype=bbox_gt.dtype,
                                 device=bbox_gt.device).mul(-1)

    iou = ciou_loss(anchors, bbox_gt, return_iou=True)

    # IoU > 0.5 -> positive
    anchor_state[iou > 0.5] = 1

    # IoU < 0.4 -> negative
    anchor_state[iou < 0.4] = 0

    return anchor_state


class DualLoss:
    def __init__(self, cfg: Config, vec2box) -> None:
        loss_cfg = cfg.task.loss
        self.loss = YOLOLoss(loss_cfg, vec2box, class_num=cfg.dataset.class_num, reg_max=cfg.model.anchor.reg_max)

        self.aux_rate = loss_cfg.aux

        self.iou_rate = loss_cfg.objective["BoxLoss"]
        self.dfl_rate = loss_cfg.objective["DFLoss"]
        self.cls_rate = loss_cfg.objective["BCELoss"]

    def __call__(
        self, aux_predicts: List[Tensor], main_predicts: List[Tensor], targets: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        # TODO: Need Refactor this region, make it flexible!
        aux_iou, aux_dfl, aux_cls = self.loss(aux_predicts, targets)
        main_iou, main_dfl, main_cls = self.loss(main_predicts, targets)

        total_loss = [
            self.iou_rate * (aux_iou * self.aux_rate + main_iou),
            self.dfl_rate * (aux_dfl * self.aux_rate + main_dfl),
            self.cls_rate * (aux_cls * self.aux_rate + main_cls),
        ]
        loss_dict = {
            f"Loss/{name}Loss": value.detach().item() for name, value in zip(["Box", "DFL", "BCE"], total_loss)
        }
        return sum(total_loss), loss_dict


class DualLossSpacecraftPose:
    def __init__(self, cfg: Config, vec2box) -> None:
        loss_cfg = cfg.task.loss
        self.loss = YOLOSPLoss(loss_cfg, vec2box, class_num=cfg.dataset.class_num, reg_max=cfg.model.anchor.reg_max)

        self.aux_rate = loss_cfg.aux

        self.iou_rate = loss_cfg.objective["BoxLoss"]
        self.dfl_rate = loss_cfg.objective["DFLoss"]
        self.cls_rate = loss_cfg.objective["BCELoss"]
        self.pos_rate = loss_cfg.objective["SPPosLoss"]
        self.rot_rate = loss_cfg.objective["SPRotLoss"]

    def __call__(
        self, aux_predicts: List[Tensor], main_predicts: List[Tensor], targets: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        # TODO: Need Refactor this region, make it flexible!
        aux_iou, aux_dfl, aux_cls, aux_pos, aux_rot = self.loss(aux_predicts, targets)
        main_iou, main_dfl, main_cls, main_pos, main_rot = self.loss(main_predicts, targets)

        total_loss = [
            self.iou_rate * (aux_iou * self.aux_rate + main_iou),
            self.dfl_rate * (aux_dfl * self.aux_rate + main_dfl),
            self.cls_rate * (aux_cls * self.aux_rate + main_cls),
            self.pos_rate  * (aux_pos  * self.aux_rate + main_pos),
            self.rot_rate  * (aux_rot  * self.aux_rate + main_rot),
        ]
        loss_dict = {
            f"Loss/{name}Loss": value.detach().item() for name, value in zip(["Box", "DFL", "BCE", "SPPos", "SPRot"], total_loss)
        }
        return sum(total_loss), loss_dict




def create_loss_function(cfg: Config, vec2box) -> DualLoss:
    # TODO: make it flexible, if cfg doesn't contain aux, only use SingleLoss
    if 'MultiheadSpacecraftPose' in cfg.model.model.detection[0]:
        loss_function = DualLossSpacecraftPose(cfg, vec2box)
    else:
        loss_function = DualLoss(cfg, vec2box)
    logger.info(":white_check_mark: Success load loss function")
    return loss_function
