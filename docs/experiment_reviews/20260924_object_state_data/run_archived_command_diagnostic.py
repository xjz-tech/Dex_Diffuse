"""Hold every emitted command fixed, intervene on physics; no new policy test."""
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
from run_mass_scale_diagnostic import P,OUT,CASE,REF,SIMPY,DPPY,SIMENV,DPENV

def run(source,mass,mu,label):
    folder=OUT/label;folder.mkdir(exist_ok=True)
    sock=Path(f'/tmp/command_replay_diag_{os.getpid()}.sock')
    servercmd=[DPPY,str(P/'replay_archived_command_server.py'),'--socket',str(sock),'--source',str(source)]
    simcmd=[SIMPY,str(P/'compare_corrected_rollouts.py'),'--mode','guided','--out',str(folder),'--reference',str(REF),'--reference-id','0','--source-episode','54','--front-only','--audit-recording','--grasp-evidence','--no-video','--object-mass-kg',str(mass),'--friction',str(mu),'--socket',str(sock),'--guidance-steps','2','--execution-steps','1','--guidance-scale','50','--reference-interpolation-threshold','0.12']
    (folder/'commands.json').write_text(json.dumps(dict(sim=simcmd,server=servercmd,meaning='Replay archived scale50 commands without online prediction; physical intervention from import onward'),indent=2)+'\n')
    if not (folder/'summary.json').exists():
        server=None
        try:
            with (folder/'server.log').open('w') as log:server=subprocess.Popen(servercmd,env=DPENV,stdout=log,stderr=subprocess.STDOUT)
            start=time.monotonic()
            while not sock.exists():
                if server.poll() is not None:raise RuntimeError(label)
                if time.monotonic()-start>30:raise TimeoutError(label)
                time.sleep(.1)
            with (folder/'sim.log').open('w') as log:subprocess.run(simcmd,env=SIMENV,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert server.wait(timeout=30)==0
        finally:
            if server is not None and server.poll() is None:server.terminate();server.wait()
            sock.unlink(missing_ok=True)
    old=json.loads((source/'trace.json').read_text());new=json.loads((folder/'trace.json').read_text())
    assert len(new)==len(old)
    assert all(a['command']==b['command'] for a,b in zip(old,new))
    if label=='replay_m044_commands_in_m044':
        assert all(all(a[k]==b[k] for k in ('q','object_pose','executed_target')) for a,b in zip(old,new))
    a=[x for x in new if x['phase']=='action'];result=dict(name=label,commands_exact=True,source=str(source),mass=mass,mu=mu,angle_at_steps_30_60_90_120=[a[i-1]['vertical_error_deg'] for i in (30,60,90,120)])
    (folder/'replay_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    light=CASE/'guide1_adaptive012_ddim4_scale50_m044_mu11';heavy=CASE/'guide1_adaptive012'
    run(light,.044,1.1,'replay_m044_commands_in_m044')
    run(light,.17,2.2,'replay_m044_commands_in_m170')
    run(heavy,.044,1.1,'replay_m170_commands_in_m044')
