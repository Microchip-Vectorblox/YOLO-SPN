import os
from copy import deepcopy
from math import exp
from pathlib import Path
from typing import List, Optional, Type, Union

import torch
import torch.distributed as dist
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from omegaconf import ListConfig
from torch import Tensor, no_grad
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, _LRScheduler

import numpy as np

from yolo.config.config import (
    IDX_TO_ID,
    DataConfig,
    NMSConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from yolo.model.yolo import YOLO
from yolo.utils.bounding_box_utils import Anc2Box, Vec2Box, bbox_nms, transform_bbox, bbox_sp_nms, rot_6d_to_matrix
from yolo.utils.logger import logger


def lerp(start: float, end: float, step: Union[int, float], total: int = 1):
    """
    Linearly interpolates between start and end values.

    start * (1 - step) + end * step

    Parameters:
        start (float): The starting value.
        end (float): The ending value.
        step (int): The current step in the interpolation process.
        total (int): The total number of steps.

    Returns:
        float: The interpolated value.
    """
    return start + (end - start) * step / total


class EMA(Callback):
    def __init__(self, decay: float = 0.9999, tau: float = 2000):
        super().__init__()
        logger.info(":chart_with_upwards_trend: Enable Model EMA")
        self.decay = decay
        self.tau = tau
        self.step = 0
        self.batch_step_counter = 0
        self.ema_state_dict = None

    def setup(self, trainer, pl_module, stage):
        pl_module.ema = deepcopy(pl_module.model)
        self.tau /= trainer.world_size
        for param in pl_module.ema.parameters():
            param.requires_grad = False

    def on_validation_start(self, trainer: "Trainer", pl_module: "LightningModule"):
        self.batch_step_counter = 0
        if self.ema_state_dict is None:
            self.ema_state_dict = deepcopy(pl_module.model.state_dict())
        pl_module.ema.load_state_dict(self.ema_state_dict)

    @no_grad()
    def on_train_batch_end(self, trainer: "Trainer", pl_module: "LightningModule", *args, **kwargs) -> None:
        self.batch_step_counter += 1
        if self.batch_step_counter % trainer.accumulate_grad_batches:
            return
        self.step += 1
        decay_factor = self.decay * (1 - exp(-self.step / self.tau))
        for key, param in pl_module.model.state_dict().items():
            self.ema_state_dict[key] = lerp(param.detach(), self.ema_state_dict[key], decay_factor)


class GradientAccumulation(Callback):
    def __init__(self, data_cfg: DataConfig, scheduler_cfg: SchedulerConfig):
        super().__init__()
        self.equivalent_batch_size = data_cfg.equivalent_batch_size
        self.actual_batch_size = data_cfg.batch_size
        self.warmup_epochs = getattr(scheduler_cfg.warmup, "epochs", 0)
        self.current_batch = 0
        self.max_accumulation = 1
        self.warmup_batches = 0
        logger.info(":arrows_counterclockwise: Enable Gradient Accumulation")

    def setup(self, trainer: "Trainer", pl_module: "LightningModule", stage: str) -> None:
        effective_batch_size = self.actual_batch_size * trainer.world_size
        self.max_accumulation = max(1, round(self.equivalent_batch_size / effective_batch_size))
        batches_per_epoch = int(len(pl_module.train_loader) / trainer.world_size)
        self.warmup_batches = int(self.warmup_epochs * batches_per_epoch)

    def on_train_epoch_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self.current_batch = trainer.global_step

    def on_train_batch_start(self, trainer: "Trainer", pl_module: "LightningModule", *args, **kwargs) -> None:
        if self.current_batch < self.warmup_batches:
            current_accumulation = round(lerp(1, self.max_accumulation, self.current_batch, self.warmup_batches))
        else:
            current_accumulation = self.max_accumulation
        trainer.accumulate_grad_batches = current_accumulation

    def on_train_batch_end(self, trainer: "Trainer", pl_module: "LightningModule", *args, **kwargs) -> None:
        self.current_batch += 1


def create_optimizer(model: YOLO, optim_cfg: OptimizerConfig) -> Optimizer:
    """Create an optimizer for the given model parameters based on the configuration.

    Returns:
        An instance of the optimizer configured according to the provided settings.
    """
    optimizer_class: Type[Optimizer] = getattr(torch.optim, optim_cfg.type)

    bias_params = [p for name, p in model.named_parameters() if "bias" in name]
    norm_params = [p for name, p in model.named_parameters() if "weight" in name and "bn" in name]
    conv_params = [p for name, p in model.named_parameters() if "weight" in name and "bn" not in name]

    model_parameters = [
        {"params": bias_params, "momentum": 0.937, "weight_decay": 0},
        {"params": conv_params, "momentum": 0.937},
        {"params": norm_params, "momentum": 0.937, "weight_decay": 0},
    ]

    def next_epoch(self, batch_num, epoch_idx):
        self.min_lr = self.max_lr
        self.max_lr = [param["lr"] for param in self.param_groups]
        # TODO: load momentum from config instead a fix number
        #       0.937: Start Momentum
        #       0.8  : Normal Momemtum
        #       3    : The warm up epoch num
        self.min_mom = lerp(0.8, 0.937, min(epoch_idx, 3), 3)
        self.max_mom = lerp(0.8, 0.937, min(epoch_idx + 1, 3), 3)
        self.batch_num = batch_num
        self.batch_idx = 0

    def next_batch(self):
        self.batch_idx += 1
        lr_dict = dict()
        for lr_idx, param_group in enumerate(self.param_groups):
            min_lr, max_lr = self.min_lr[lr_idx], self.max_lr[lr_idx]
            param_group["lr"] = lerp(min_lr, max_lr, self.batch_idx, self.batch_num)
            param_group["momentum"] = lerp(self.min_mom, self.max_mom, self.batch_idx, self.batch_num)
            lr_dict[f"LR/{lr_idx}"] = param_group["lr"]
            lr_dict[f"momentum/{lr_idx}"] = param_group["momentum"]
        return lr_dict

    optimizer_class.next_batch = next_batch
    optimizer_class.next_epoch = next_epoch

    optimizer = optimizer_class(model_parameters, **optim_cfg.args)
    optimizer.max_lr = [0.1, 0, 0]
    return optimizer


def create_scheduler(optimizer: Optimizer, schedule_cfg: SchedulerConfig) -> _LRScheduler:
    """Create a learning rate scheduler for the given optimizer based on the configuration.

    Returns:
        An instance of the scheduler configured according to the provided settings.
    """
    scheduler_class: Type[_LRScheduler] = getattr(torch.optim.lr_scheduler, schedule_cfg.type)
    schedule = scheduler_class(optimizer, **schedule_cfg.args)
    if hasattr(schedule_cfg, "warmup"):
        wepoch = schedule_cfg.warmup.epochs
        lambda1 = lambda epoch: (epoch + 1) / wepoch if epoch < wepoch else 1
        lambda2 = lambda epoch: 10 - 9 * ((epoch + 1) / wepoch) if epoch < wepoch else 1
        warmup_schedule = LambdaLR(optimizer, lr_lambda=[lambda2, lambda1, lambda1])
        schedule = SequentialLR(optimizer, schedulers=[warmup_schedule, schedule], milestones=[wepoch - 1])
    return schedule


def initialize_distributed() -> None:
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    logger.info(f"🔢 Initialized process group; rank: {rank}, size: {world_size}")
    return local_rank


def get_device(device_spec: Union[str, int, List[int]]) -> torch.device:
    ddp_flag = False
    if isinstance(device_spec, (list, ListConfig)):
        ddp_flag = True
        device_spec = initialize_distributed()
    if torch.cuda.is_available() and "cuda" in str(device_spec):
        return torch.device(device_spec), ddp_flag
    if not torch.cuda.is_available():
        if device_spec != "cpu":
            logger.warning(f"❎ Device spec: {device_spec} not support, Choosing CPU instead")
        return torch.device("cpu"), False

    device = torch.device(device_spec)
    return device, ddp_flag


class PostProcess:
    """
    TODO: function document
    scale back the prediction and do nms for pred_bbox
    """

    def __init__(self, converter: Union[Vec2Box, Anc2Box], nms_cfg: NMSConfig) -> None:
        self.converter = converter
        self.nms = nms_cfg

    def __call__(
        self, predict, rev_tensor: Optional[Tensor] = None, image_size: Optional[List[int]] = None
    ) -> List[Tensor]:
        if image_size is not None:
            self.converter.update(image_size)
        prediction = self.converter(predict["Main"])
        pred_class, _, pred_bbox = prediction[:3]
        pred_conf = prediction[3] if len(prediction) == 4 else None
        if rev_tensor is not None:
            pred_bbox = (pred_bbox - rev_tensor[:, None, 1:]) / rev_tensor[:, 0:1, None]
        pred_bbox = bbox_nms(pred_class, pred_bbox, self.nms, pred_conf)
        return pred_bbox


class PostProcessSpacecraftPose:
    """
    TODO: function document
    scale back the prediction and do nms for pred_bbox
    """

    def __init__(self, converter: Union[Vec2Box, Anc2Box], nms_cfg: NMSConfig) -> None:
        self.converter = converter
        self.nms = nms_cfg

    def __call__(
        self, predict, rev_tensor: Optional[Tensor] = None, image_size: Optional[List[int]] = None
    ) -> List[Tensor]:
        if image_size is not None:
            self.converter.update(image_size)
        prediction = self.converter(predict["Main"])
        pred_class, _, pred_bbox, pred_trans, pred_rot = prediction
        pred_conf = None
        if rev_tensor is not None:
            assert(0) # MCHP what is rev_tensor?
            pred_bbox = (pred_bbox - rev_tensor[:, None, 1:]) / rev_tensor[:, 0:1, None]
        pred_bbox, pred_trans, pred_rot = bbox_sp_nms(pred_class, pred_bbox, pred_trans, pred_rot, self.nms, pred_conf)
        return pred_bbox, pred_trans, pred_rot


class SpacecraftPoseMetric:
    def __init__(self, domain, spacecraft_cfg):
        metric_names = ['effi_eR', 'effi_eT', 'effi_eR_HIL', 'effi_eT_norm_HIL', 'effi_pose']
        self.metrics = {}
        self.domain = domain
        self.SPEED_THRESHOLD_Q = spacecraft_cfg.SPEED_THRESHOLD_Q
        self.SPEED_THRESHOLD_T = spacecraft_cfg.SPEED_THRESHOLD_T
        for name in metric_names:
            self.metrics[name] = [] 

    def update(self, predicts, predicts_trans, predicts_rot, targets):
        R_reg = []
        t_reg = []
        B = len(predicts)
        #if len(predicts_trans)!=B or len(predicts_rot)!=B or len(targets)!=B:
        #    print('SpacecraftPoseMetric warning: predicts batch sizes are different')
        for b in range(len(predicts)):
            if predicts[b].numel()==0:
                continue
            ind = torch.argmax(predicts[b][:,5])
            R_pr    = rot_6d_to_matrix(predicts_rot[b][ind,:].to(torch.float32)).cpu().numpy()
            t_pr    = predicts_trans[b][ind,:].to(torch.float32).cpu().numpy()
            R_gt    = targets['rotationmatrix'][b,:,:].to(torch.float32).cpu().numpy()
            t_gt    = targets['translation'][b,:].to(torch.float32).cpu().numpy()

            # Pose metric(s)
            err_q = error_orientation(R_pr, R_gt, 'rotationmatrix') # [deg]
            err_t = error_translation(t_pr, t_gt)
            speed_t, speed_q, speed = speed_score(t_pr, R_pr, t_gt, R_gt,
                representation='rotationmatrix',
                applyThreshold=self.domain in ['lightbox', 'sunlamp'],
                theta_q=self.SPEED_THRESHOLD_Q,
                theta_t=self.SPEED_THRESHOLD_T)

            # Save & record
            R_reg.append(R_pr)
            t_reg.append(t_pr)
            result = {'_eR': err_q, '_eT': err_t, '_eR_HIL': speed_q, '_eT_norm_HIL': speed_t, '_pose': speed}
            for m, v in result.items():
                self.metrics['effi'+m].append(v)

    def compute(self) -> dict:
        results = {'pose':np.mean(self.metrics['effi_pose']),
                   'pose_t':np.mean(self.metrics['effi_eT_norm_HIL']),
                   'pose_r':np.mean(self.metrics['effi_eR_HIL']),}
        return results

    def reset(self):
       for k in self.metrics.keys():
            self.metrics[k] = [] 

def error_translation(t_pr, t_gt):
    t_pr = np.reshape(t_pr, (3,))
    t_gt = np.reshape(t_gt, (3,))

    return np.sqrt(np.sum(np.square(t_gt - t_pr)))

def _error_orientation_quaternion(q_pr, q_gt):
    # q must be [qvec, qcos]
    q_pr = np.reshape(q_pr, (4,))
    q_gt = np.reshape(q_gt, (4,))

    qdot = np.abs(np.dot(q_pr, q_gt))
    qdot = np.minimum(qdot, 1.0)
    return np.rad2deg(2*np.arccos(qdot)) # [deg]

def _error_orientation_rotationmatrix(R_pr, R_gt):
    assert R_pr.shape == R_gt.shape and R_pr.ndim == 2
    assert np.abs(np.linalg.det(R_pr) - 1) < 1e-6, \
        f'Determinant of R_pr is {np.linalg.det(R_pr)}'

    Rdot = np.dot(R_pr, np.transpose(R_gt))
    trace = (np.trace(Rdot) - 1.0)/2.0
    trace = np.clip(trace, -1.0, 1.0)
    return np.rad2deg(np.arccos(trace))

def error_orientation(ori_pr, ori_gt, representation='quaternion'):
    assert representation in ['quaternion', 'rotationmatrix'], \
        'Orientation representation must be either quaternion or rotationmatrix'

    if representation == 'quaternion':
        return _error_orientation_quaternion(ori_pr, ori_gt)
    else:
        return _error_orientation_rotationmatrix(ori_pr, ori_gt)

def speed_score(t_pr, ori_pr, t_gt, ori_gt, representation='quaternion',
                    applyThreshold=True, theta_q=0.5, theta_t=0.005):
    # theta_q: rotation threshold [deg]
    # theta_t: normalized translation threshold [m/m]
    err_t = error_translation(t_pr, t_gt)
    err_q = error_orientation(ori_pr, ori_gt, representation) # [deg]

    t_gt = np.reshape(t_gt, (3,))
    speed_t = err_t / np.sqrt(np.sum(np.square(t_gt)))
    speed_q = np.deg2rad(err_q)

    # Check if within threshold
    if applyThreshold and err_q < theta_q:
        speed_q = 0.0

    if applyThreshold and speed_t < theta_t:
        speed_t = 0.0

    speed = speed_t + speed_q

    return speed_t, speed_q, speed


def collect_prediction(predict_json: List, local_rank: int) -> List:
    """
    Collects predictions from all distributed processes and gathers them on the main process (rank 0).

    Args:
        predict_json (List): The prediction data (can be of any type) generated by the current process.
        local_rank (int): The rank of the current process. Typically, rank 0 is the main process.

    Returns:
        List: The combined list of predictions from all processes if on rank 0, otherwise predict_json.
    """
    if dist.is_initialized() and local_rank == 0:
        all_predictions = [None for _ in range(dist.get_world_size())]
        dist.gather_object(predict_json, all_predictions, dst=0)
        predict_json = [item for sublist in all_predictions for item in sublist]
    elif dist.is_initialized():
        dist.gather_object(predict_json, None, dst=0)
    return predict_json


def predicts_to_json(img_paths, predicts, rev_tensor):
    """
    TODO: function document
    turn a batch of imagepath and predicts(n x 6 for each image) to a List of diction(Detection output)
    """
    batch_json = []
    for img_path, bboxes, box_reverse in zip(img_paths, predicts, rev_tensor):
        scale, shift = box_reverse.split([1, 4])
        bboxes = bboxes.clone()
        bboxes[:, 1:5] = (bboxes[:, 1:5] - shift[None]) / scale[None]
        bboxes[:, 1:5] = transform_bbox(bboxes[:, 1:5], "xyxy -> xywh")
        for cls, *pos, conf in bboxes:
            bbox = {
                "image_id": int(Path(img_path).stem),
                "category_id": IDX_TO_ID[int(cls)],
                "bbox": [float(p) for p in pos],
                "score": float(conf),
            }
            batch_json.append(bbox)
    return batch_json
