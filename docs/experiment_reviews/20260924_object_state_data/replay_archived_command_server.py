"""Counterfactual command replay, explicitly bypassing online DP prediction."""
import argparse
import json
import socket
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'eval'))
from ipc import recv_message,send_message

def main():
    p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    trace=json.loads((a.source/'trace.json').read_text());s=json.loads((a.source/'summary.json').read_text())
    commands=np.array([x['command'] for x in trace if x['phase']=='action'],dtype=np.float32)
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(a.socket);server.listen(1)
    try:
        client,_=server.accept()
        with client:
            while True:
                msg,_=recv_message(client)
                if msg['type']=='shutdown':break
                if msg['type']=='hello':
                    info=dict(s['prior'],diagnostic_command_replay=True,source=str(a.source))
                    # Older archives predate this optional selector; no equality interpolation was used.
                    info.setdefault('reference_interpolation_equal_jump',None)
                    send_message(client,info);continue
                assert msg['type']=='predict'
                j=msg['reference_index'];send_message(client,dict(ok=True,guidance_scale=s['guidance_scale'],diagnostic_command_replay=True),commands[j][None,None])
    finally:
        server.close();Path(a.socket).unlink(missing_ok=True)

if __name__=='__main__':main()
