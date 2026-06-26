from math import ceil
from pathlib import Path

from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format
from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler, PostProcessSpacecraftPose, SpacecraftPoseMetric


class BaseModel(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight, dataset_cfg=cfg.dataset)
        self.spacecraft_pose = 'MultiheadSpacecraftPose' in cfg.model.model.detection[0]

    def forward(self, x):
        return self.model(x)


class ValidateModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.debug = False
        if self.cfg.task.task == "validation":
            self.validation_cfg = self.cfg.task
        else:
            self.validation_cfg = self.cfg.task.validation
        self.val_loader = create_dataloader(self.validation_cfg.data, self.cfg.dataset, self.validation_cfg.task)
        if isinstance(self.val_loader, list):
            self.metric = []
            self.val_label = []
            for folder in self.cfg.dataset.val_folder:
                self.metric.append(MeanAveragePrecision(iou_type="bbox", box_format="xyxy", backend="faster_coco_eval"))
                self.metric[-1].warn_on_many_detections = False
                self.val_label.append(folder)
            if self.spacecraft_pose:
                self.sp_metric = []
                for folder in self.cfg.dataset.val_folder:
                    self.sp_metric.append(SpacecraftPoseMetric(folder, self.validation_cfg.spacecraft))
        else:
            self.val_label = ['val']
            self.metric = [MeanAveragePrecision(iou_type="bbox", box_format="xyxy", backend="faster_coco_eval")]
            self.metric[-1].warn_on_many_detections = False
            if self.spacecraft_pose:
                self.sp_metric = [SpacecraftPoseMetric('val', self.validation_cfg.spacecraft)]
        self.ema = self.model

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        if self.spacecraft_pose:
            self.post_process = PostProcessSpacecraftPose(self.vec2box, self.validation_cfg.nms)
        else:
            self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)

    def val_dataloader(self):
        return self.val_loader

    def debug_image(self, image, bp, tp, rp, bt, tt, Rt, image_path):
        import torch
        import numpy as np
        from yolo.utils.bounding_box_utils import project_keypoints, quat2dcm, rot_6d_to_matrix
        from yolo.utils.bounding_box_utils import quat2dcm
        predict = tp is not None
        bt = bt[0].to(torch.float32).numpy()
        tt = tt.to(torch.float32).numpy()
        Rt = Rt.to(torch.float32).numpy()
        keypts3d_origin = np.concatenate((np.zeros((3,1), dtype=np.float32), np.transpose(self.model.keypts3d)), axis=1) # [3, 12]
        kt = project_keypoints(Rt, tt, self.model.camera['cameraMatrix'], self.model.camera['distCoeffs'], keypts3d_origin) # (2, 12)
        if predict:
            bp = bp[0].to(torch.float32).numpy()
            tp = tp[0].to(torch.float32).numpy()
            Rp = rot_6d_to_matrix(rp[0].to(torch.float32)).numpy()
            kp = project_keypoints(Rp, tp, self.model.camera['cameraMatrix'], self.model.camera['distCoeffs'], keypts3d_origin) # (2, 12)
            kRp = project_keypoints(Rp, tt, self.model.camera['cameraMatrix'], self.model.camera['distCoeffs'], keypts3d_origin)
            ktp = project_keypoints(Rt, tp, self.model.camera['cameraMatrix'], self.model.camera['distCoeffs'], keypts3d_origin)

        from torchvision.transforms.functional import to_pil_image
        from PIL import Image, ImageDraw, ImageFont
        img = to_pil_image(image).copy()
        kt[0,:] *= img.size[0]/self.model.camera['Nu']
        kt[1,:] *= img.size[1]/self.model.camera['Nv']
        if predict:
            kp[0,:] *= img.size[0]/self.model.camera['Nu']
            kp[1,:] *= img.size[1]/self.model.camera['Nv']
            ktp[0,:] *= img.size[0]/self.model.camera['Nu']
            ktp[1,:] *= img.size[1]/self.model.camera['Nv']
            kRp[0,:] *= img.size[0]/self.model.camera['Nu']
            kRp[1,:] *= img.size[1]/self.model.camera['Nv']
            kp_rms = np.sqrt(np.mean(np.square(kp-kt)))
            ktp_rms = np.sqrt(np.mean(np.square(ktp-kt)))
            kRp_rms = np.sqrt(np.mean(np.square(kRp-kt)))

        label_size = img.size[1] / 30
        draw = ImageDraw.Draw(img, "RGBA")
        font = ImageFont.load_default(int(label_size))
        connections = [[1,2],[2,3],[3,4],[4,1],[5,6],[6,7],[7,8],[8,5],[1,5],[2,6],[3,7],[4,8],[3,10],[2,9],[4,11]]
        for n in range(kt.shape[1]):
            draw.circle(kt[:,n],3,fill="forestgreen")
        for nc in connections:
            line = (kt[0,nc[0]],kt[1,nc[0]],kt[0,nc[1]],kt[1,nc[1]])
            draw.line(line, fill="forestgreen", width=1)
        for n in range(kt.shape[1]):
            draw.text(kt[:,n], str(n), fill="lightgreen", font=font)
        if predict:
            for n in range(kp.shape[1]):
                draw.circle(kp[:,n],3,fill="red")
            for nc in connections:
                line = (kp[0,nc[0]],kp[1,nc[0]],kp[0,nc[1]],kp[1,nc[1]])
                draw.line(line, fill="red", width=1)
            for n in range(kp.shape[1]):
                draw.text(kt[:,n], str(n), fill="lightgreen", font=font)
            draw.text((0,0),f"keypoint err: {kp_rms:.2f} (pix rms)", fill="white", font=font)
            draw.text((0,30),f"  translation: {ktp_rms:.2f}", fill="white", font=font)
            draw.text((0,60),f"  rotation: {kRp_rms:.2f}", fill="white", font=font)
    
        save_name = image_path.with_suffix(".png").name
        img.save(save_name)
        print('saved debug image: '+save_name)


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch_size, images, targets, rev_tensor, img_paths = batch
        H, W = images.shape[2:]
        if self.spacecraft_pose:
            predicts, predicts_trans, predicts_rot = self.post_process(self.ema(images), image_size=[W, H])
            self.sp_metric[dataloader_idx].update(predicts, predicts_trans, predicts_rot, targets)
            if self.debug: # testing
                self.debug_image(images[0],predicts[0],predicts_trans[0],predicts_rot[0],targets['bboxes'][0],targets['translation'][0],targets['rotationmatrix'][0],img_paths[0])
        else:
            predicts = self.post_process(self.ema(images), image_size=[W, H])
        mAP = self.metric[dataloader_idx](
            [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets['bboxes']]
        )
        return predicts, mAP

    def on_validation_epoch_end(self):
        for n,label in enumerate(self.val_label):
            epoch_metrics = self.metric[n].compute()
            if self.spacecraft_pose:
                epoch_metrics = epoch_metrics | self.sp_metric[n].compute()
            del epoch_metrics["classes"]
            if n==0:
                self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
            self.log_dict(
                {"val/"+label+"@.5:.95": epoch_metrics["map"], "val/"+label+"@.5": epoch_metrics["map_50"]},
                sync_dist=True,
                rank_zero_only=True,
            )
            self.metric[n].reset()
            if self.spacecraft_pose:
                self.log_dict({"val/"+label+" pose": epoch_metrics["pose"],
                               "val/"+label+" pose_t": epoch_metrics["pose_t"],
                               "val/"+label+" pose_r": epoch_metrics["pose_r"]}, sync_dist=True, rank_zero_only=True)
                self.sp_metric[n].reset()


class TrainModel(ValidateModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.train_loader = create_dataloader(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task)
        #img, targets, rev_tensor, img_path = self.train_loader.dataset.__getitem__(0)
        #print('hi')

    def setup(self, stage):
        super().setup(stage)
        self.loss_fn = create_loss_function(self.cfg, self.vec2box)

    def train_dataloader(self):
        return self.train_loader

    def on_train_epoch_start(self):
        self.trainer.optimizers[0].next_epoch(
            ceil(len(self.train_loader) / self.trainer.world_size), self.current_epoch
        )
        self.vec2box.update(self.cfg.image_size)

    def training_step(self, batch, batch_idx):
        lr_dict = self.trainer.optimizers[0].next_batch()
        batch_size, images, targets, *_ = batch
        predicts = self(images)
        aux_predicts = self.vec2box(predicts["AUX"])
        main_predicts = self.vec2box(predicts["Main"])
        loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)
        self.log_dict(
            loss_item,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)
        if self.debug:
            self.debug_image(images[0],None,None,None,targets['bboxes'][0],targets['translation'][0],targets['rotationmatrix'][0],batch[-1][0])
        return loss * batch_size

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
        scheduler = create_scheduler(optimizer, self.cfg.task.scheduler)
        return [optimizer], [scheduler]
    

class InferenceModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        # TODO: Add FastModel
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        if self.spacecraft_pose:
            self.post_process = PostProcessSpacecraftPose(self.vec2box, self.cfg.task.nms)
        else:
            self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx):
        images, rev_tensor, origin_frame = batch
        if self.spacecraft_pose:
            predicts, predicts_trans, predicts_rot = self.post_process(self(images))
        else:
            predicts = self.post_process(self(images), rev_tensor=rev_tensor)
        img = draw_bboxes(origin_frame, predicts, idx2label=self.cfg.dataset.class_list)
        # TODO: draw spacecraft pose
        if getattr(self.predict_loader, "is_stream", None):
            fps = self._display_stream(img)
        else:
            fps = None
        if getattr(self.cfg.task, "save_predict", None):
            self._save_image(img, batch_idx)
        return img, fps

    def _save_image(self, img, batch_idx):
        save_image_path = Path(self.trainer.default_root_dir) / f"frame{batch_idx:03d}.png"
        img.save(save_image_path)
        print(f"💾 Saved visualize image at {save_image_path}")
