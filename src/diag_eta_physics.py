"""Does the weak-constraint eta recover the known missing physics M_true(x)-M_F1(x) (F1: rho 27.5)? (2026-11-17)"""
import sys; sys.path.insert(0,'src')
import torch, xarray as xr, numpy as np
from dameta.dynamics.l63 import L63Params, l63_tendency, rk4_step
torch.set_default_dtype(torch.float64)
d=xr.open_dataset('tmp/data/l63v2d_test_seed42.nc'); xt=torch.tensor(d.x_true.values)[:, :111]
def M(x,rho): p=L63Params(rho=rho); return rk4_step(lambda v: l63_tendency(v,p), x, 0.01)
for meth in ['lbfgs','adagrad']:
  r=xr.open_dataset(f'tmp/results/dense/l63v2d_f1_weak_q0.0316228_{meth}_test.nc'); c=torch.tensor(r.control_a.values)
  x0=c[:,:3]; eta=c[:,3:].reshape(200,110,3)
  # analysis trajectory with F1 + eta
  xs=[x0]; x=x0
  for t in range(110): x=M(x,27.5)+eta[:,t]; xs.append(x)
  xa=torch.stack(xs,1)
  mp_a=M(xa[:,:-1],28.0)-M(xa[:,:-1],27.5)   # true missing physics at analysis states
  mp_t=M(xt[:,:-1],28.0)-M(xt[:,:-1],27.5)   # at true states
  e=eta.numpy(); 
  for name,mp in [('at analysis',mp_a.numpy()),('at truth',mp_t.numpy())]:
    cor=[np.corrcoef(e[...,k].ravel(),mp[...,k].ravel())[0,1] for k in range(3)]
    reg=[(e[...,k]*mp[...,k]).sum()/(mp[...,k]**2).sum() for k in range(3)]
    print(meth,name,'corr xyz',np.round(cor,3),'regression slope',np.round(reg,3))
  print(meth,'rms eta xyz',np.round(np.sqrt((e**2).mean((0,1))),4),'rms missing phys xyz',np.round(np.sqrt((mp_t.numpy()**2).mean((0,1))),4))
  # residual (noise) part: eta minus its projection
  res=e-mp_a.numpy(); print(meth,'rms(eta - missing phys)',np.round(np.sqrt((res**2).mean((0,1))),4))
