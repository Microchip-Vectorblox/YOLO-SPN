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

    def debug_heatmap(self, predict, target):
        import numpy as np
        from yolo.utils.bounding_box_utils import project_keypoints
        keypts3d_origin = np.concatenate((np.zeros((3,1), dtype=np.float32), np.transpose(self.model.keypts3d)), axis=1) # [3, 12]
        kt = project_keypoints(target['quaternion'].cpu().numpy(), target['translation'].cpu().numpy(), self.model.camera['cameraMatrix'], self.model.camera['distCoeffs'], keypts3d_origin) # (2, 12)
        
        heat_err = np.sqrt(np.sum(np.square(kt[:,1:].transpose() - predict['heat_kps'][:,:2]),axis=1))
        heat_val = predict['heat_kps'][:,2]

        return heat_err, heat_val

    def debug_image(self, image, predict, target, metric, image_path, output):
        import torch
        import numpy as np
        from yolo.utils.bounding_box_utils import project_keypoints, quat2dcm, rot_6d_to_matrix
        from yolo.utils.bounding_box_utils import quat2dcm
        from torchvision.transforms.functional import to_pil_image
        from PIL import Image, ImageDraw, ImageFont

        img = to_pil_image(image).copy()
        
        keypts3d_origin = np.concatenate((np.zeros((3,1), dtype=np.float32), np.transpose(self.model.keypts3d)), axis=1) # [3, 12]
        kt = project_keypoints(target['quaternion'].cpu().numpy(), target['translation'].cpu().numpy(), self.model.camera['cameraMatrix'], self.model.camera['distCoeffs'], keypts3d_origin) # (2, 12)
        kt[0,:] *= img.size[0]/self.model.camera['Nu']
        kt[1,:] *= img.size[1]/self.model.camera['Nv']
        if not predict==None:
            kp = project_keypoints(predict['pose_rot'].cpu().numpy(), predict['pose_trans'].cpu().numpy(), self.model.camera['cameraMatrix'], self.model.camera['distCoeffs'], keypts3d_origin) # (2, 12)
            kp[0,:] *= img.size[0]/self.model.camera['Nu']
            kp[1,:] *= img.size[1]/self.model.camera['Nv']
            if not predict['heat_reject']:
                kh = project_keypoints(predict['heat_rot'], predict['heat_trans'], self.model.camera['cameraMatrix'], self.model.camera['distCoeffs'], keypts3d_origin) # (2, 12)
                kf = project_keypoints(predict['heat_rot'], predict['pose_trans'].cpu().numpy(), self.model.camera['cameraMatrix'], self.model.camera['distCoeffs'], keypts3d_origin) # (2, 12)
                #kp_rms = np.sqrt(np.mean(np.square(kp-kt)))
                kh[0,:] *= img.size[0]/self.model.camera['Nu']
                kh[1,:] *= img.size[1]/self.model.camera['Nv']
                #kh_rms = np.sqrt(np.mean(np.square(kh-kt)))
                kf[0,:] *= img.size[0]/self.model.camera['Nu']
                kf[1,:] *= img.size[1]/self.model.camera['Nv']
            heat_peaks = predict['heat_kps'].copy()
            heat_peaks[:,0] *= img.size[0]/self.model.camera['Nu']
            heat_peaks[:,1] *= img.size[1]/self.model.camera['Nv']
            heat_peaks_err = np.sqrt(np.sum(np.square(kt[:,1:].transpose() - heat_peaks[:,:2]),axis=1))

        # heatmap debugging
        #from yolo.utils.model_utils import solve_pose_from_heatmaps
        #heatmain = torch.zeros((11,64,96))
        #for n in range(11):
        #    heatmain[n,int(kt[1,n+1]/8),int(kt[0,n+1]/8)] = 1.0
        #keypts_pr, q_pr, t_pr, reject = solve_pose_from_heatmaps(output['MainHeat'][0].cpu(), (1920,1200), 0.5, self.model.camera, self.model.keypts3d)
        #keypts_pri, q_pri, t_pri, rejecti = solve_pose_from_heatmaps(heatmain, (1920,1200), 0.5, self.model.camera, self.model.keypts3d)
        #scale = np.tile(np.array(img.size) / [self.model.camera['Nu'],self.model.camera['Nv']],(11,1))
        #keypts_pr = np.transpose(keypts_pr * scale)
        #kh2 = project_keypoints(q_pr, t_pr, self.model.camera['cameraMatrix'], self.model.camera['distCoeffs'], keypts3d_origin) # (2, 12)
        #kh2[0,:] *= img.size[0]/self.model.camera['Nu']
        #kh2[1,:] *= img.size[1]/self.model.camera['Nv']
        #khi = project_keypoints(q_pri, t_pri, self.model.camera['cameraMatrix'], self.model.camera['distCoeffs'], keypts3d_origin) # (2, 12)
        #khi[0,:] *= img.size[0]/self.model.camera['Nu']
        #khi[1,:] *= img.size[1]/self.model.camera['Nv']
        #predict['heat_rot'] == q_pr
        #predict['heat_trans'] == t_pr

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
        if not predict==None:
            for n in range(kp.shape[1]):
                draw.circle(kp[:,n],3,fill="red")
            for nc in connections:
                line = (kp[0,nc[0]],kp[1,nc[0]],kp[0,nc[1]],kp[1,nc[1]])
                draw.line(line, fill="red", width=1)
            if not predict['heat_reject']:
                for n in range(kh.shape[1]):
                    draw.circle(kh[:,n],3,fill="yellow")
                for nc in connections:
                    line = (kh[0,nc[0]],kh[1,nc[0]],kh[0,nc[1]],kh[1,nc[1]])
                    draw.line(line, fill="yellow", width=1)
            draw.text((0,0),f"pose: {metric['effi_eR']:.4f}, {metric['effi_eT']:.4f} (deg,m?)", fill="red", font=font)
            draw.text((0,30),f"pose: {metric['effi_eR_HIL']:.4f}, {metric['effi_eT_norm_HIL']:.4f}, {metric['effi_pose']:.4f} (rot,trans,final)", fill="red", font=font)
            if not predict['heat_reject']:
                draw.text((0,60),f"heat: {metric['heat_eR']:.4f}, {metric['heat_eT']:.4f} (deg,m?)", fill="yellow", font=font)
                draw.text((0,90),f"heat: {metric['heat_eR_HIL']:.4f}, {metric['heat_eT_norm_HIL']:.4f}, {metric['heat_pose']:.4f} (rot,trans,final)", fill="yellow", font=font)
            # heatmap peaks
            for n in range(heat_peaks.shape[0]):
                if heat_peaks[n,3]:
                    draw.circle(heat_peaks[n,:],3,fill="blue")
                    draw.text(heat_peaks[n,:], str(n+1), fill="blue", font=font)
            #for n in range(khi.shape[1]):
            #    draw.circle(khi[:,n],3,fill="yellow")
            #for nc in connections:
            #    line = (khi[0,nc[0]],khi[1,nc[0]],khi[0,nc[1]],khi[1,nc[1]])
            #    draw.line(line, fill="yellow", width=1)
            #for n in range(keypts_pr.shape[0]):
            #    draw.circle(keypts_pr[n,:],3,fill="blue")
            #    draw.text(keypts_pr[n,:], str(n+1), fill="blue", font=font)

        save_name = image_path.with_suffix(".png").name
        img.save(save_name)
        print('saved debug image: '+save_name)

        if 0:
            heatmapTarget = torch.zeros_like(target['heatmap'])
            x = torch.arange(96).reshape((1,96))
            y = torch.arange(64).reshape((64,1))
            sigma = 2.0
            for h in range(11):
                x0 = kt[0,h+1]/8
                y0 = kt[1,h+1]/8
                g = torch.exp(-(torch.tile((x-x0)**2,(64,1)) + torch.tile((y-y0)**2,(1,96)))/(2*sigma**2))
                heatmapTarget[h] = g

            from yolo.utils.model_utils import solve_pose_from_heatmaps
            keypts_ht, q_ht, t_ht, reject_ht = solve_pose_from_heatmaps(heatmapTarget.cpu(), (1920,1200), 0.5, self.model.camera, self.model.keypts3d)
            keypts_ht[:,0] *= 768/self.model.camera['Nu']
            keypts_ht[:,1] *= 512/self.model.camera['Nv']
            kg = keypts_ht[:,:2].transpose()
            kht = project_keypoints(q_ht, t_ht, self.model.camera['cameraMatrix'], self.model.camera['distCoeffs'], keypts3d_origin) # (2, 12)
            kht[0,:] *= 768/self.model.camera['Nu']
            kht[1,:] *= 512/self.model.camera['Nv']

            # h = 0
            img = to_pil_image(target['heatmap'][0:3]).copy()
            img.save('img000014_ht012.png')
            
            img = to_pil_image(heatmapTarget[0:3]).copy()
            img.save('img000014_hg012.png')

            img = to_pil_image(output['MainHeat'][0,h:3].clip(0,1)).copy()
            img.save('img000014_hp012.png')


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch_size, images, targets, rev_tensor, img_paths = batch
        H, W = images.shape[2:]
        if self.spacecraft_pose:
            predicts = self.post_process(self.ema(images), image_size=[W, H])
            batch_metrics = self.sp_metric[dataloader_idx].update(predicts, targets)
            if self.debug:
                import torch
                import numpy as np
                peak_err = np.array([])
                peak_val = np.array([])
                for n in range(batch_size):
                    #metric = {}
                    #for k,v in self.sp_metric[dataloader_idx].metrics.items():
                        #metric[k] = v[n-batch_size]
                    predict = {}
                    for k,v in predicts.items():
                        predict[k] = v[n]
                    target = {}
                    for k,v in targets.items():
                        target[k] = v[n]
                    output = self.ema(images[n:n+1])
                    err,val = self.debug_heatmap(predict, target)
                    peak_err = np.concat((peak_err,err))
                    peak_val = np.concat((peak_val,val))
                    self.debug_image(images[n],predict,target,batch_metrics[n],img_paths[n],output)
        else:
            predicts = self.post_process(self.ema(images), image_size=[W, H])
        mAP = self.metric[dataloader_idx](
            [to_metrics_format(bbox.unsqueeze(0)) for bbox in predicts['bbox']], [to_metrics_format(bboxes) for bboxes in targets['bboxes']]
        )
        return predicts, mAP

    def on_validation_epoch_end(self):
        for n,label in enumerate(self.val_label):
            epoch_metrics = self.metric[n].compute()
            del epoch_metrics["classes"]
            if n==0:
                self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
            self.log_dict(
                {"val_"+label+"/mAP@.5:.95": epoch_metrics["map"], "val_"+label+"/mAP@.5": epoch_metrics["map_50"]},
                sync_dist=True,
                rank_zero_only=True,
            )
            self.metric[n].reset()
            if self.spacecraft_pose:
                epoch_sp_metrics = self.sp_metric[n].compute()
                epoch_log = {}
                for k,v in epoch_sp_metrics.items():
                    epoch_log["val_"+label+"/"+k] = v
                self.log_dict(epoch_log, sync_dist=True, rank_zero_only=True)
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
        aux_predicts = self.vec2box(predicts["AUX"]) + (predicts["AUXHeat"],)
        main_predicts = self.vec2box(predicts["Main"]) + (predicts["MainHeat"],)
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
            for n in range(batch_size):
                target = {}
                for k,v in targets.items():
                    target[k] = v[n]
                self.debug_image(images[n],None,target,None,batch[-1][n],None)
            #self.debug_image(images[0],None,None,None,targets['bboxes'][0],targets['translation'][0],targets['rotationmatrix'][0],None,None,batch[-1][0])
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
