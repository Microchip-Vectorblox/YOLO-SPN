python yolo/lazy.py task=train weight=weights/v9-s.pt model=v9-s-sp use_wandb=False dataset='spin' use_tensorboard=True image_size=[768,512] name=v9-s-sp task.data.batch_size=32 task.data.equivalent_batch_size=32 task.epoch=20 task.optimizer.args.lr=0.01 task.data.style=0.0 task.data.data_augment.CoarseDropout=0.0
#accelerator='cpu' device=1 +quiet=True cfg.task.data.cpu_num=0
