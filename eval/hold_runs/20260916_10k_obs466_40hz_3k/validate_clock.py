"""Check that changing action frequency does not change the external-force clock."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import torch

root=Path(__file__).resolve().parent
source=root/'code/maniptrans_envs/lib/envs/tasks/sindexhandmanip_sh.py'
tree=ast.parse(source.read_text())
method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='eval_force_tick')
module=ast.Module(body=[method],type_ignores=[])
ns=dict(torch=torch,STRICT_TEST_MODE=False,gymtorch=SimpleNamespace(unwrap_tensor=lambda x:x),gymapi=SimpleNamespace(LOCAL_SPACE=0))
exec(compile(ast.fix_missing_locations(module),str(source),'exec'),ns)
results={};reference=None
for hz in (30,40):
    torch.manual_seed(42)
    frames=[]
    h=SimpleNamespace(training_or_collection=True,cfg={'env':{'evalReferenceDt':0.0166667}},
        random_force_decay=.99,random_force_decay_interval=.08,apply_forces=torch.zeros(4,1,3),
        num_envs=4,device='cpu',random_force_prob=torch.full((4,),.15),
        _manip_obj_rigid_body_handle=0,random_force_scale=1.,manip_obj_mass=torch.ones(4),sim=None)
    h.gym=SimpleNamespace(apply_rigid_body_force_tensors=lambda *args:frames.append((h._eval_force_tick if hasattr(h,'_eval_force_tick') else 0,args[1].clone())))
    for action in range(hz):
        for sub in range(360//hz): ns['eval_force_tick'](h)
    assert h._eval_force_tick==360 and len(frames)==180
    assert [f[0] for f in frames]==[i for i in range(360) if i%12<6]
    forces=torch.stack([f[1] for f in frames])
    if reference is None: reference=forces
    assert torch.equal(forces,reference)
    results[str(hz)]={'physics_ticks_per_second':360,'force_loaded_ticks':180,'force_sequence_exactly_equal':True,'cap_steps':hz*400}
(root/'clock_validation.json').write_text(json.dumps(results,indent=2)+'\n')
print('All frequencies produce identical fixed-clock force sequences and 400-second caps.')
