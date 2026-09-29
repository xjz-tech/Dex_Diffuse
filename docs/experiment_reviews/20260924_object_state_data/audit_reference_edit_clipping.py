from pathlib import Path
import json,numpy as np,torch
from reference_action_editor import ReferenceActionEditor
P=Path(__file__).resolve().parent;R=P/'reference_turn_baseline_20260926';O=R/'episode54_reference_edit_20260928'
torch.set_num_threads(2);ed=ReferenceActionEditor('/home/carus/data_usb/10B_obs_4-66.ckpt',.1);a=np.load(R/'qualified_comparison/episode_54/reference_full.npz')['hand_target_rad'][0];t=torch.as_tensor(a,device='cuda:0');norm=ed.policy.normalizer['action'].normalize(t);back=ed.policy.normalizer['action'].unnormalize(norm.clamp(-1,1));d=(back-t).detach().cpu().numpy();rows=[]
for name,n in [('first2',2),('first9',9),('first50',50),('whole',len(a))]:rows.append(dict(window=name,rmse_rad=float(np.sqrt((d[:n]**2).mean())),max_abs_rad=float(np.abs(d[:n]).max()),clipped_fraction=float((norm[:n].abs()>1).float().mean())))
result=dict(reference_only_normalization_clip_error=rows);(O/'reference_clipping_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
