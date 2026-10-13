import os,sys,json,time
from pathlib import Path
ROOT=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse');sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import torch
from omegaconf import OmegaConf
from diffusion_policy.workspace.train_diffusion_unet_sim_hand_workspace import TrainDiffusionUnetSimHandWorkspace
name=sys.argv[1];seed=int(sys.argv[2]) if len(sys.argv)>2 else 42;out=ROOT/'runs'/f'domain10k_20260920_{name}_seed{seed}'
torch.set_num_threads(4)
cfg=OmegaConf.load(ROOT/'runs/sim_hand_10k_seed42/.hydra/config.yaml');cfg.task.dataset_path=str(ROOT/'data'/f'domain10k_20260920_{name}');cfg.training.seed=seed;cfg.training.resume=False;cfg.training.checkpoint_every=200;cfg.training.num_epochs=200;cfg.dataloader.num_workers=0;cfg.val_dataloader.num_workers=0;cfg.logging.mode='disabled';cfg.logging.name=f'domain10k_{name}_seed{seed}';cfg.training.tqdm_interval_sec=30.;out.mkdir(parents=True,exist_ok=False)
(out/'training_config.yaml').write_text(OmegaConf.to_yaml(cfg));start=time.monotonic();w=TrainDiffusionUnetSimHandWorkspace(cfg,output_dir=str(out));w.run();(out/'training_complete.json').write_text(json.dumps({'complete':True,'epoch':w.epoch,'global_step':w.global_step,'dataset':cfg.task.dataset_path,'seed':seed,'wall_seconds':time.monotonic()-start},indent=2));print('TRAIN COMPLETE',name,flush=True)
