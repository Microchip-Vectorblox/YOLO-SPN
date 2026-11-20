
#python yolo/lazy.py task=train weight=weights/v9-t.pt model=v9-t-sp use_wandb=False dataset='spinrgb' use_tensorboard=True image_size=[768,512] name=v9-t-sp task.data.batch_size=64 task.data.equivalent_batch_size=64
#python yolo/lazy.py task=train weight=weights/v9-s.pt model=v9-s-sp use_wandb=False dataset='spinrgb' use_tensorboard=True image_size=[768,512] name=v9-s-sp task.data.batch_size=32 task.data.equivalent_batch_size=32
#python yolo/lazy.py task=train weight=weights/v9-m.pt model=v9-m-sp use_wandb=False dataset='spinrgb' use_tensorboard=True image_size=[768,512] name=v9-m-sp task.data.batch_size=16 task.data.equivalent_batch_size=16 task.optimizer.args.lr=0.005


#python yolo/lazy.py task=train weight=weights/v9-t.pt model=v9-t-sp use_wandb=False dataset='speedplus' use_tensorboard=True image_size=[768,512] name=v9-t-sp task.data.batch_size=64 task.data.equivalent_batch_size=64 task.data.style=0.5 task.data.data_augment.CoarseDropout=0.5 task.data.data_augment.RandomSunFlare=0.5
#python yolo/lazy.py task=train weight=weights/v9-s.pt model=v9-s-sp use_wandb=False dataset='speedplus' use_tensorboard=True image_size=[768,512] name=v9-s-sp task.data.batch_size=32 task.data.equivalent_batch_size=32 task.data.style=0.5 task.data.data_augment.CoarseDropout=0.5 task.data.data_augment.RandomSunFlare=0.5
#python yolo/lazy.py task=train weight=weights/v9-m.pt model=v9-m-sp use_wandb=False dataset='speedplus' use_tensorboard=True image_size=[768,512] name=v9-m-sp task.data.batch_size=16 task.data.equivalent_batch_size=16 task.optimizer.args.lr=0.002 task.loss.objective.HeatLoss=0.6 task.optimizer.args.weight_decay=0.0001 task.loss.objective.BCELoss=0.25 task.data.style=0.5 task.data.data_augment.CoarseDropout=0.5 task.data.data_augment.RandomSunFlare=0.5

#task.data.style=0.5 task.data.data_augment.CoarseDropout=0.5
#accelerator='cpu' device=1 +quiet=True cfg.task.data.cpu_num=0
