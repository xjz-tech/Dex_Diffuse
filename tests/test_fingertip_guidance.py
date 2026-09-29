from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import xml.etree.ElementTree as ET
from diffusion_policy.guidance.fingertip_fk import SharpAFingertipFK, FingertipGuidanceLoss, DOF_NAMES, TIP_NAMES, DEFAULT_URDF
from diffusion_policy.guidance.guided_ddim import guided_ddim_step
from diffusers import DDIMScheduler


def test_fk_against_independent_scalar_urdf_composition():
    rng = np.random.default_rng(23)
    q = rng.uniform(-1, 1, (3,22))
    children = {j.find('child').get('link'):j for j in ET.parse(DEFAULT_URDF).getroot().findall('joint')}
    expected = []
    for angles in q:
        def transform(link):
            if link == 'right_hand_C_MC':
                return np.eye(4)
            j = children[link];o=j.find('origin');t=np.eye(4);m=np.eye(4)
            t[:3,:3] = Rotation.from_euler('xyz',np.fromstring(o.get('rpy'),sep=' ')).as_matrix()
            t[:3,3] = np.fromstring(o.get('xyz'),sep=' ')
            if j.get('type') != 'fixed':
                axis=np.fromstring(j.find('axis').get('xyz'),sep=' ')
                m[:3,:3]=Rotation.from_rotvec(axis*angles[DOF_NAMES.index(j.get('name'))]).as_matrix()
            return transform(j.find('parent').get('link')) @ t @ m
        expected.append([transform(t)[:3,3] for t in TIP_NAMES])
    actual=SharpAFingertipFK()(torch.tensor(q,dtype=torch.float32)).detach().numpy()
    np.testing.assert_allclose(actual,expected,atol=8e-8)


def test_fk_autograd_and_loss_units_and_guidance_window():
    fk=SharpAFingertipFK().double()
    q=torch.randn(1,22,dtype=torch.float64,requires_grad=True)*0.2
    assert torch.autograd.gradcheck(fk,(q,),eps=1e-6,atol=1e-5)
    loss=FingertipGuidanceLoss(SimpleNamespace(unnormalize=lambda x:x*2+0.3), 'cpu')
    p=torch.randn(2,12,22,requires_grad=True);ref=torch.randn(2,2,22)
    energy=loss(p,ref,slice(3,5))
    expected=(loss.fk(p[:,3:5]*2+.3)-loss.fk(ref*2+.3)).square().mean((1,2,3))
    torch.testing.assert_close(energy,expected)
    grad=torch.autograd.grad(energy.sum(),p)[0]
    assert grad[:,3:5].abs().sum()>0
    assert grad[:,:3].count_nonzero()==0 and grad[:,5:].count_nonzero()==0
    torch.testing.assert_close(loss(p,p[:,3:5].detach(),slice(3,5)),torch.zeros(2))


def test_fingertip_loss_reaches_ddim_score_update():
    sched=DDIMScheduler(num_train_timesteps=100,prediction_type='epsilon',clip_sample=True)
    sched.set_timesteps(4)
    loss=FingertipGuidanceLoss(SimpleNamespace(unnormalize=lambda x:x), 'cpu')
    torch.manual_seed(2)
    x=torch.randn(2,12,22,requires_grad=True);eps=torch.randn_like(x);ref=torch.randn(2,2,22)
    t=int(sched.timesteps[0]);alpha=sched.alphas_cumprod[t]
    raw=(x-(1-alpha).sqrt()*eps.detach())/alpha.sqrt()
    grad=torch.autograd.grad(loss(raw,ref,slice(3,5)).sum(),x)[0]
    output=guided_ddim_step(sched,eps,t,x,ref,25.,slice(3,5),guidance_loss_fn=loss)
    expected=raw.clamp(-1,1)-(1-alpha)/alpha.sqrt()*25*grad
    torch.testing.assert_close(output.pred_original_sample,expected)
    assert torch.isfinite(output.prev_sample).all()
