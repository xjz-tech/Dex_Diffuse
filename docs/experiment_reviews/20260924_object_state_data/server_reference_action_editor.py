"""IPC server: noise a recorded reference window, edit with the existing prior."""
import argparse,json,socket,time
from pathlib import Path
import numpy as np,torch
from reference_action_editor import ReferenceActionEditor
from reference_resampling import interpolate_large_jumps
from ipc import recv_message,send_message

def main():
 p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--log',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--noise-ratio',type=float,required=True);p.add_argument('--ddim-steps',type=int,default=4);p.add_argument('--execution-steps',type=int,default=2);p.add_argument('--checkpoint',default='/home/carus/data_usb/10B_obs_4-66.ckpt');p.add_argument('--reference-interpolation-threshold',type=float);args=p.parse_args()
 torch.set_num_threads(2);editor=ReferenceActionEditor(args.checkpoint,args.noise_ratio,steps=args.ddim_steps,execution_steps=args.execution_steps);ref=np.load(args.reference)['hand_target_rad'];path=Path(args.socket);assert not path.exists();records=[];server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
 if args.reference_interpolation_threshold is not None:
  ref,_=interpolate_large_jumps(ref,args.reference_interpolation_threshold)
 try:
  server.bind(str(path));server.listen(1);print('READY',json.dumps(editor.metadata),flush=True);client,_=server.accept()
  with client:
   while True:
    msg,history=recv_message(client)
    if msg['type']=='shutdown':break
    if msg['type']=='hello':
     send_message(client,dict(ok=True,prior=args.checkpoint,spec=editor.spec,ddim=args.ddim_steps,execution_steps=args.execution_steps,guidance_steps=editor.future_steps,guidance_scale=0.,reference=str(args.reference),reference_repeat=1,reference_interpolation=0,reference_interpolation_threshold=args.reference_interpolation_threshold,reference_interpolation_equal_jump=None,reference_mode='expanded',editor=editor.metadata));continue
    assert msg['type']=='predict' and msg['guidance_scale']==0
    j=int(msg['reference_index']);indices=np.minimum(np.arange(j,j+editor.future_steps),ref.shape[1]-1);future=np.stack([ref[int(ri),indices] for ri in msg['reference_ids']]);start=time.monotonic();action,stats=editor.predict(history,future,msg['seeds']);row=dict(reference_index=j,guidance_scale=0.,noise_ratio=args.noise_ratio,seeds=msg['seeds'],inference_seconds=time.monotonic()-start,**stats);records.append(row);send_message(client,dict(ok=True,**row),action.astype(np.float32))
 finally:
  server.close();path.unlink(missing_ok=True);args.log.write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
