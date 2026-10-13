"""Differentiable SharpA fingertip FK in metres, in the palm link frame.

Joint order matches Isaac Gym and the 22-D policy action. URDF joint origins
are composed before joint motion. No angle clamping or loss rescaling is used.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import torch

DOF_NAMES = (
    'right_index_MCP_FE', 'right_index_MCP_AA', 'right_index_PIP', 'right_index_DIP',
    'right_middle_MCP_FE', 'right_middle_MCP_AA', 'right_middle_PIP', 'right_middle_DIP',
    'right_pinky_CMC', 'right_pinky_MCP_FE', 'right_pinky_MCP_AA', 'right_pinky_PIP', 'right_pinky_DIP',
    'right_ring_MCP_FE', 'right_ring_MCP_AA', 'right_ring_PIP', 'right_ring_DIP',
    'right_thumb_CMC_FE', 'right_thumb_CMC_AA', 'right_thumb_MCP_FE', 'right_thumb_MCP_AA', 'right_thumb_IP',
)
TIP_NAMES = tuple(f'right_{finger}_fingertip' for finger in ('thumb', 'index', 'middle', 'ring', 'pinky'))
DEFAULT_URDF = Path(__file__).resolve().parents[2] / 'maniptrans_envs/assets/sharpa_hand/v3right_sharpa_wave-forhammer5.urdf'


class SharpAFingertipFK(torch.nn.Module):
    def __init__(self, urdf=DEFAULT_URDF):
        super().__init__()
        joints = {j.find('child').get('link'): j for j in ET.parse(urdf).getroot().findall('joint')}
        chains = []
        for tip in TIP_NAMES:
            chain = []
            while tip != 'right_hand_C_MC':
                j = joints[tip]
                chain.insert(0, j)
                tip = j.find('parent').get('link')
            chains.append(chain)
        depth = max(map(len, chains))
        origins = np.tile(np.eye(4), (depth, 5, 1, 1))
        axes = np.zeros((depth, 5, 3))
        indices = np.full((depth, 5), 22, dtype=np.int64)
        for f, chain in enumerate(chains):
            for d, j in enumerate(chain):
                o = j.find('origin')
                if o is not None:
                    origins[d,f,:3,3] = np.fromstring(o.get('xyz', '0 0 0'), sep=' ')
                    origins[d,f,:3,:3] = Rotation.from_euler('xyz', np.fromstring(o.get('rpy', '0 0 0'), sep=' ')).as_matrix()
                if j.get('type') != 'fixed':
                    if j.get('type') != 'revolute' or j.find('mimic') is not None:
                        raise ValueError('Only independent revolute joints supported')
                    indices[d,f] = DOF_NAMES.index(j.get('name'))
                    a = np.fromstring(j.find('axis').get('xyz'), sep=' ')
                    axes[d,f] = a / np.linalg.norm(a)
        self.register_buffer('origins', torch.tensor(origins, dtype=torch.float32))
        self.register_buffer('axes', torch.tensor(axes, dtype=torch.float32))
        self.register_buffer('indices', torch.tensor(indices))
        # Rodrigues rotation basis, precomputed for every joint axis.
        x,y,z = self.axes.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero), -1).reshape(depth,5,3,3)
        self.register_buffer('skew', skew)
        self.register_buffer('skew2', skew @ skew)

    def forward(self, q):
        if q.shape[-1] != 22:
            raise ValueError('Expected 22 joints in Isaac Gym order')
        shape = q.shape[:-1]
        q = q.reshape(-1,22)
        q = torch.cat((q, torch.zeros_like(q[:,:1])), -1)
        rot = torch.eye(3, dtype=q.dtype, device=q.device).expand(len(q),5,3,3)
        pos = q.new_zeros((len(q),5,3))
        for d in range(len(self.indices)):
            origin = self.origins[d]
            pos = pos + (rot @ origin[:,:3,3,None]).squeeze(-1)
            rot = rot @ origin[:,:3,:3]
            angle = q[:,self.indices[d],None,None]
            motion = torch.eye(3, dtype=q.dtype, device=q.device) + angle.sin()*self.skew[d] + (1-angle.cos())*self.skew2[d]
            rot = rot @ motion
        return pos.reshape(*shape,5,3)


class FingertipGuidanceLoss:
    def __init__(self, action_normalizer, device, urdf=DEFAULT_URDF):
        self.normalizer = action_normalizer
        self.fk = SharpAFingertipFK(urdf).to(device)

    def __call__(self, pred, reference, guidance_slice):
        actual = self.fk(self.normalizer.unnormalize(pred[:,guidance_slice]))
        with torch.no_grad():
            target = self.fk(self.normalizer.unnormalize(reference))
        return (actual-target).square().mean(dim=(1,2,3))
