import ast,sys,importlib.util,json,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
sys.path.insert(0,str(ROOT/'code/eval'))
from step_clamp import clamp_targets
spec=importlib.util.spec_from_file_location('hardware',REPO/'eval/real/hardware.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
source=REPO/'eval/real/inference_real.py'
tree=ast.parse(source.read_text());method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='safe_target')
module=ast.Module(body=[method],type_ignores=[])
ns=dict(np=np,argparse=argparse,HAND_DIM=22,POLICY_LOWER_LIMITS=h.POLICY_LOWER_LIMITS,POLICY_UPPER_LIMITS=h.POLICY_UPPER_LIMITS)
exec(compile(ast.fix_missing_locations(module),str(source),'exec'),ns)
args=argparse.Namespace(joint_limit_margin=0.,disable_step_clamp=False,max_hand_step=.03)
rng=np.random.default_rng(42);previous=rng.uniform(h.POLICY_LOWER_LIMITS,h.POLICY_UPPER_LIMITS,(1000,22));raw=previous+rng.normal(0,.3,(1000,22))
sent,step,limit=clamp_targets(raw,previous,h.POLICY_LOWER_LIMITS,h.POLICY_UPPER_LIMITS)
for i in range(len(raw)):
 expected,lc,sc=ns['safe_target'](raw[i],previous[i],args)
 np.testing.assert_array_equal(sent[i],expected)
 assert lc==limit[i].sum() and sc==step[i].sum()
assert np.max(abs(sent-previous))<.0300000001
# A blocked measured joint does not change the previous-command reference.
previous=np.zeros((1,22));raw=np.ones((1,22))*.09
path=[]
for _ in range(3):
 previous,_,_=clamp_targets(raw,previous,h.POLICY_LOWER_LIMITS,h.POLICY_UPPER_LIMITS);path.append(previous[0,0])
np.testing.assert_allclose(path,[.03,.06,.09],atol=1e-12)
report=dict(passed=True,random_cases=1000,exact_match_to_real_safe_target=True,reference='previous sent target, not measured qpos',three_step_target_path=path)
(ROOT/'clamp_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
