"""Opt-in exact NOKOV preprocessing cache; recreate physics normally each trial.

Cache serialized outputs plus consumed RNG transitions, never task states.
On a hit, reject differing inputs or pre-RNG states instead of using stale data.
"""
import hashlib,json,pickle,random
from pathlib import Path
import numpy as np
import torch

def rng_states():
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
def digest(x):
    if isinstance(x,torch.Tensor):return hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
    return hashlib.sha256(pickle.dumps(x,protocol=4)).hexdigest()
def restore(key,state):
    if key=='cpu':torch.set_rng_state(state)
    elif key=='cuda':torch.cuda.set_rng_state(state)
    elif key=='numpy':np.random.set_state(state)
    else:random.setstate(state)
def install_demo_cache(root):
    from main.dataset.nokov3_dataset_dexhand_rh import Nokov3DatasetDexHandRH
    cls=Nokov3DatasetDexHandRH
    if getattr(cls,'_astra_cache_installed',False):return
    original=cls.__getitem__;source=Path(__import__(cls.__module__,fromlist=['x']).__file__)
    root.mkdir(parents=True,exist_ok=True)
    code_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    def cached(self,index):
        obj,idx=self.parse_index(index)
        data_path=Path(self.data_paths[obj][idx])
        retarget=Path(self.retarget_dir)/('mano2'+str(self.dexhand))/obj/(str(idx)+'.pkl')
        paths=[data_path,retarget,Path(self.obj_mesh_paths[obj]),Path(self.obj_urdf_paths[obj])]
        token=dict(code=code_hash,index=str(index),device=str(self.device),hand=str(self.dexhand),skip=self.skip,
            transform=self.mujoco2gym_transf.cpu().numpy().tolist() if hasattr(self,'mujoco2gym_transf') else None,
            files=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths])
        file=root/(hashlib.sha256(json.dumps(token,sort_keys=True).encode()).hexdigest()+'.pt')
        before=rng_states()
        if file.exists():
            saved=torch.load(str(file))
            assert saved['token']==token
            for key in saved['changed']:
                if digest(before[key])!=saved['before'][key]:raise RuntimeError('demo cache RNG mismatch: '+str(index)+' '+key)
                restore(key,saved['after'][key])
            return saved['data']
        result=original(self,index);after=rng_states()
        changed=[k for k in before if digest(before[k])!=digest(after[k])]
        saved=dict(token=token,data=result,changed=changed,before={k:digest(before[k]) for k in changed},after={k:after[k] for k in changed})
        temp=file.with_suffix('.tmp');torch.save(saved,str(temp));temp.replace(file)
        return result
    cls.__getitem__=cached;cls._astra_cache_installed=True
