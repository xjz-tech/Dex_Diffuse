"""Astra-authored small left-pad pushes and exact 60-step static references.

Closed-loop numeric helper; no model API, demonstrations, or state writes.
"""
import argparse,json,os,sys,subprocess,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'eval'))
from astra_bridge import atomic_json,validate_response
from astra_kinematic_step import propose

class Sequence:
 def __init__(self,root,degrees=3.,inward=.00025,max_delta=.08,pause_scale=None,approach_scale=None,approach_gain=.15):
  self.root=root;self.degrees=degrees;self.inward=inward;self.max_delta=max_delta
  self.stage=1;self.phase='turn';self.pause_end=None;self.hold=None;self.events=[]
  self.pause_scale=pause_scale;self.intervals=[]
  self.approach_scale=approach_scale;self.approach_gain=approach_gain
 def event(self,kind,step,**data):
  event=dict(event=kind,step=step,stage=self.stage,**data);self.events.append(event)
  atomic_json(self.root/'sequence_events.json',self.events);print(event,flush=True)
 def turn_plan(self,request,start=None):
  if start is not None:
   request=dict(request,state=dict(request['state'],target_before=start.tolist()))
  remaining=180*self.stage-request['measured_twist_degrees']
  degrees=max(.4,min(self.degrees,self.approach_gain*remaining))
  return propose(request,degrees,16,self.inward,max_delta=self.max_delta),degrees
 def plan(self,request):
  step=request['step'];angle=request['measured_twist_degrees'];diag=[];degrees=0.
  if step==0:
   actions=np.repeat(np.asarray(request['state']['target_before'])[None,:],16,axis=0)
   labels=['initialize']*16
  else:
   if self.phase=='pause' and step>=self.pause_end and True:
    self.event('resume_turn',self.pause_end,observed_at_step=step,observed_angle_deg=angle)
    self.stage+=1;self.phase='turn'
   if self.phase=='turn' and angle>=180*self.stage-5:
    self.phase='pause';self.hold=np.asarray(request['state']['target_before']);self.pause_end=step+60
    self.event('start_pause',step,target_deg=180*self.stage,actual_angle_deg=angle,
               pause_end_step=self.pause_end,hold_target=self.hold.tolist())
    if self.pause_scale is not None:
     self.intervals.append(dict(start_step=step,end_step=self.pause_end,scale=self.pause_scale))
     atomic_json(self.root/'guidance_schedule.json',dict(intervals=self.intervals))
   if self.phase=='turn':
    if self.approach_scale is not None and 180*self.stage-angle<=40:
     self.intervals.append(dict(start_step=step,end_step=step+8,scale=self.approach_scale))
     atomic_json(self.root/'guidance_schedule.json',dict(intervals=self.intervals))
    (actions,diag),degrees=self.turn_plan(request);labels=['turn%d'%self.stage]*16
   else:
    actions=np.repeat(self.hold[None,:],16,axis=0);labels=['pause%d'%self.stage]*16
    n_hold=max(0,min(16,self.pause_end-step))
    if n_hold<16 and True:
     # The next segment is explicitly authored in the lookahead. No last-target padding.
     previous=self.stage;self.stage+=1
     (moving,diag),degrees=self.turn_plan(request,self.hold);self.stage=previous
     actions[n_hold:]=moving[:16-n_hold];labels[n_hold:]=['turn%d'%(self.stage+1)]*(16-n_hold)
  return dict(session_id=request['session_id'],request_id=request['request_id'],
    joint_names=request['joint_names'],units='absolute_joint_radians',actions=actions.tolist(),
    rationale='Astra: small positive pad rotation about bulb axis; at each cumulative half-turn keep the exact same reference for60 control steps before resuming.',
    helper='URDF Jacobian from current simulator feedback; no LLM API or demonstration replay',
    action_phases=labels,sequence_stage=self.stage,phase=self.phase,target_angle_deg=180*self.stage,
    pause_end_step=self.pause_end,parameters=dict(degrees=degrees,inward=self.inward,max_delta=self.max_delta),diagnostics=diag)

