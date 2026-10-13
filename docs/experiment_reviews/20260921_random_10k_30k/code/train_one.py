import os, sys, json, time, hashlib
from pathlib import Path
ROOT=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')
sys.path.insert(0,str(ROOT)); os.chdir(ROOT)
import torch
from omegaconf import OmegaConf
from diffusion_policy.workspace.train_diffusion_unet_sim_hand_workspace import TrainDiffusionUnetSimHandWorkspace
name=sys.argv[1]
out=ROOT/'runs'/f'random_20260921_{name}_train42'
torch.set_num_threads(4)
cfg=OmegaConf.load(ROOT/'runs/sim_hand_10k_seed42/.hydra/config.yaml')
cfg.task.dataset_path=str(ROOT/'data'/f'random_20260921_{name}')
cfg.training.seed=42; cfg.training.resume=False
cfg.training.checkpoint_every=200; cfg.training.num_epochs=200
cfg.dataloader.num_workers=0; cfg.val_dataloader.num_workers=0
cfg.logging.mode='disabled'; cfg.logging.name=f'random_20260921_{name}'
cfg.training.tqdm_interval_sec=30.
out.mkdir(parents=True,exist_ok=False)
(out/'training_config.yaml').write_text(OmegaConf.to_yaml(cfg))
start=time.monotonic()
w=TrainDiffusionUnetSimHandWorkspace(cfg,output_dir=str(out)); w.run()
# BaseWorkspace writes its checkpoint asynchronously; join before hashing or eval.
thread=getattr(w,'_saving_thread',None)
if thread is not None: thread.join()
checkpoint=out/'checkpoints/latest.ckpt'
(out/'training_complete.json').write_text(json.dumps({'complete':True,'epoch':w.epoch,'global_step':w.global_step,'dataset':cfg.task.dataset_path,'training_seed':42,'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'wall_seconds':time.monotonic()-start},indent=2))
print('TRAIN COMPLETE',name,flush=True)
