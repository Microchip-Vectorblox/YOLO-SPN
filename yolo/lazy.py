import sys
from pathlib import Path

import torch
import hydra
from lightning import Trainer
from omegaconf import open_dict

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.solver import InferenceModel, TrainModel, ValidateModel, BaseModel
from yolo.utils.logging_utils import setup


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    callbacks, loggers, save_path = setup(cfg)

    trainer = Trainer(
        accelerator=getattr(cfg, "accelerator", "auto"),
        devices=cfg.device,
        max_epochs=getattr(cfg.task, "epoch", None),
        precision="16-mixed",
        callbacks=callbacks,
        sync_batchnorm=True,
        logger=loggers,
        log_every_n_steps=1,
        gradient_clip_val=10,
        gradient_clip_algorithm="norm",
        deterministic=True,
        enable_progress_bar=not getattr(cfg, "quiet", False),
        default_root_dir=save_path,
    )

    if cfg.task.task == "train":
        model = TrainModel(cfg)
        trainer.fit(model)
    if cfg.task.task == "validation":
        model = ValidateModel(cfg)
        trainer.validate(model)
    if cfg.task.task == "inference":
        model = InferenceModel(cfg)
        trainer.predict(model)
    if cfg.task.task == "export":
        with open_dict(cfg):
            del cfg.model.model['auxiliary']
        model = BaseModel(cfg)
        model.eval().float().to('cpu')
        input_sample = torch.randn((1,3,cfg.image_size[1],cfg.image_size[0]))
        #model.to_onnx('yolov9s-sp_73.onnx', input_sample, dynamo=True, external_data=False)
        export_path = 'yolov9_export.onnx'
        with torch.inference_mode():
            torch.onnx.export(
                model,
                input_sample,
                export_path,
                opset_version=13,
                external_data=False,
                dynamo=True,
            )


if __name__ == "__main__":
    main()
