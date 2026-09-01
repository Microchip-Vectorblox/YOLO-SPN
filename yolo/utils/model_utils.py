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
import cv2
from scipy.spatial.transform import Rotation as R

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
        self.heatmap_threshold = [0.0,0.9]    #cfg.TEST.HEATMAP_THRESHOLD

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
        pred_bbox, pose_trans, pose_rot = bbox_sp_nms(pred_class, pred_bbox, pred_trans, pred_rot, self.nms, pred_conf)

        # heatmap stuff
        heat_kps = []
        heat_trans = []
        heat_rot = []
        heat_reject = []
        for b in range(predict["MainHeat"].shape[0]):
            keypts_pr, q_pr, t_pr, reject = solve_pose_from_heatmaps(
                predict["MainHeat"][b].cpu(), #outputs[i].squeeze(0).cpu(),
                (self.converter.camera['Nu'],self.converter.camera['Nv']),
                self.heatmap_threshold,
                self.converter.camera, self.converter.keypts_true_3D
            )
            heat_kps.append(keypts_pr)
            heat_trans.append(t_pr)
            heat_rot.append(q_pr)
            heat_reject.append(reject)

        predictOut = {'bbox':pred_bbox,
                      'pose_trans':pose_trans,
                      'pose_rot':pose_rot,
                      'heat_trans':heat_trans,
                      'heat_rot':heat_rot,
                      'heat_reject':heat_reject,
                      'heat_kps':heat_kps}
        return predictOut


def solve_pose_from_heatmaps_old(heatmaps, image_size, threshold, camera, keypts_true_3D):
    # heatmaps: [Nk x H x W]
    nK, nH, nW = heatmaps.shape

    # Take max
    maxVal, maxInd = heatmaps.view(nK, -1).max(-1)

    # Flag to reject measurements
    reject = False

    # Apply heatmaps until we have enough keypoints detected
    visibleIdx = []
    while len(visibleIdx) < 5 and threshold >= 0:
        visibleIdx = torch.nonzero(maxVal > threshold)
        threshold -= 0.1

    if len(visibleIdx) < 5:
        reject = True
        return None, None, None, reject
    else:
        # Extract keypoint locations in [pix]
        keypts = torch.zeros(nK, 2)
        for k in range(nK):
            if k in visibleIdx:
                y = torch.floor(maxInd[k] / nW)
                x = maxInd[k] - nW * y
                keypts[k,0] = x / nW * image_size[0]
                keypts[k,1] = y / nH * image_size[1]

        # PnP
        q, t = pnp(keypts_true_3D[visibleIdx], keypts[visibleIdx].numpy(),
                        camera['cameraMatrix'], camera['distCoeffs'])

        #! Added on 2023/07/02
        # Correction based on camera physics
        z_max = 50 # [m]

        if t[2] < 0 or t[2] > z_max:
            # --- 1. Check z limit
            reject = True
        else:
            # --- 2. Check xy limits based on estimated z distance
            x_max = t[2] * np.tan(camera["horizontalFOV"]/2)
            y_max = x_max / camera["Nu"] * camera["Nv"]

            if t[0] < -x_max or t[0] > x_max or t[1] < -y_max or t[1] > y_max:
                reject = True

        return keypts, q, t, reject


def solve_pose_from_heatmaps(heatmaps, image_size, threshold_bound, camera, keypts_true_3D):
    from scipy.ndimage import map_coordinates

    # heatmaps: [Nk x H x W]
    nK, nH, nW = heatmaps.shape
        
    peakVal = np.zeros(nK)
    peakY = np.zeros(nK)
    peakX = np.zeros(nK)
    peakYC = np.zeros(nK)
    peakXC = np.zeros(nK)
    for k in range(nK):
        maxInd = heatmaps[k].argmax().item()
        y = maxInd // nW
        x = maxInd - nW * y
        ymin = max(0,y-1)
        ymax = min(nH-1,y+1)
        xmin = max(0,x-1)
        xmax = min(nW-1,x+1)
        yrange = np.arange(ymin,ymax,0.125)
        xrange = np.arange(xmin,xmax,0.125)
        coord = np.stack([yrange.repeat(len(xrange)),np.tile(xrange,len(yrange))])
        heatInterp = map_coordinates(heatmaps[k].numpy().astype('float32'),coord)
        maxInterpInd = heatInterp.argmax()
        peakVal[k] = heatInterp[maxInterpInd]
        peakY[k] = coord[0,maxInterpInd]
        peakX[k] = coord[1,maxInterpInd]
        peakXC[k] = x
        peakYC[k] = y

    minNumPeaks = 5
    threshold = np.sort(peakVal)[-minNumPeaks]
    threshold = max(min(threshold, threshold_bound[1]),threshold_bound[0])
    visibleIdx = peakVal >= threshold
    peaks = np.stack([peakX*image_size[0]/nW, peakY*image_size[1]/nH, peakVal, visibleIdx]).transpose() # X (pix), Y (pix), Val, Visible

    if np.sum(visibleIdx)<minNumPeaks:
        reject = True
        return peaks, None, None, reject
    else:
        reject = False
    
    # PnP
    q, t = pnp(keypts_true_3D[visibleIdx], peaks[visibleIdx,:2], camera['cameraMatrix'], camera['distCoeffs'])

    #! Added on 2023/07/02
    # Correction based on camera physics
    z_max = 50 # [m]

    if t[2] < 0 or t[2] > z_max:
        # --- 1. Check z limit
        reject = True
    else:
        # --- 2. Check xy limits based on estimated z distance
        x_max = t[2] * np.tan(camera["horizontalFOV"]/2)
        y_max = x_max / camera["Nu"] * camera["Nv"]

        if t[0] < -x_max or t[0] > x_max or t[1] < -y_max or t[1] > y_max:
            reject = True

    return peaks, q, t, reject


def pnp(points_3D, points_2D, cameraMatrix, distCoeffs=None, rvec=None, tvec=None, useExtrinsicGuess=False):
    if distCoeffs is None:
        distCoeffs = np.zeros((5, 1), dtype=np.float32)

    assert points_3D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    points_3D = np.ascontiguousarray(points_3D).reshape((-1,1,3))
    points_2D = np.ascontiguousarray(points_2D).reshape((-1,1,2))

    _, R_exp, t = cv2.solvePnP(points_3D,
                               points_2D,
                               cameraMatrix,
                               distCoeffs, rvec, tvec, useExtrinsicGuess,
                               flags=cv2.SOLVEPNP_EPNP)

    R_pr, _ = cv2.Rodrigues(R_exp)

    # To (q, t)
    q = R.from_matrix(R_pr).as_quat() # [qvec, qcos]
    t = np.squeeze(t)

    return q[[3,0,1,2]], t


class SpacecraftPoseMetric:
    def __init__(self, domain, spacecraft_cfg):
        metric_names = ['effi_eR', 'effi_eT', 'effi_eR_HIL', 'effi_eT_norm_HIL', 'effi_pose',
                        'heat_eR', 'heat_eT', 'heat_eR_HIL', 'heat_eT_norm_HIL', 'heat_pose',
                        'final_pose', 'heat_reject']
        self.metrics = {}
        self.domain = domain
        self.SPEED_THRESHOLD_Q = spacecraft_cfg.SPEED_THRESHOLD_Q
        self.SPEED_THRESHOLD_T = spacecraft_cfg.SPEED_THRESHOLD_T
        for name in metric_names:
            self.metrics[name] = []
    
    def update(self, predicts, targets):
        batch_results = []
        for b in range(len(predicts['bbox'])):
            result = {}

            # ground truth
            R_gt    = targets['rotationmatrix'][b].to(torch.float32).cpu().numpy()
            q_gt    = targets['quaternion'][b].to(torch.float32).cpu().numpy()
            t_gt    = targets['translation'][b].to(torch.float32).cpu().numpy()

            # heatmap
            result = result | {'heat_reject':predicts['heat_reject'][b]}
            if not predicts['heat_reject'][b]:
                q_pr    = predicts['heat_rot'][b]
                t_pr    = predicts['heat_trans'][b]
                err_q = error_orientation(q_pr, q_gt, 'quaternion') # [deg]
                err_t = error_translation(t_pr, t_gt)
                speed_t, speed_q, speed = speed_score(t_pr, q_pr, t_gt, q_gt,
                        representation='quaternion',
                        applyThreshold=self.domain in ['lightbox', 'sunlamp'],
                        theta_q=self.SPEED_THRESHOLD_Q,
                        theta_t=self.SPEED_THRESHOLD_T)
                result = result | {'heat_eR': err_q, 'heat_eT': err_t, 'heat_eR_HIL': speed_q, 'heat_eT_norm_HIL': speed_t, 'heat_pose': speed}

            # pose
            assert predicts['bbox'][b].numel()>0
            R_pr    = rot_6d_to_matrix(predicts['pose_rot'][b].to(torch.float32)).cpu().numpy()
            t_pr    = predicts['pose_trans'][b].to(torch.float32).cpu().numpy()
            err_q = error_orientation(R_pr, R_gt, 'rotationmatrix') # [deg]
            err_t = error_translation(t_pr, t_gt)
            speed_t, speed_q, speed = speed_score(t_pr, R_pr, t_gt, R_gt,
                representation='rotationmatrix',
                applyThreshold=self.domain in ['lightbox', 'sunlamp'],
                theta_q=self.SPEED_THRESHOLD_Q,
                theta_t=self.SPEED_THRESHOLD_T)
            result = result | {'effi_eR': err_q, 'effi_eT': err_t, 'effi_eR_HIL': speed_q, 'effi_eT_norm_HIL': speed_t, 'effi_pose': speed}

            # final pose score
            if not predicts['heat_reject'][b]:
                result['final_pose'] = result['heat_eR_HIL'] + result['effi_eT_norm_HIL']

            # Save & record
            for m, v in result.items():
                self.metrics[m].append(v)
            batch_results.append(result)

        return batch_results

    def compute(self) -> dict:
        results = {}
        for k,v in self.metrics.items():
            results[k] = np.mean(v)
        results['heat_reject'] = np.sum(self.metrics['heat_reject'])
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
