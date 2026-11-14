"""Unseen independent virtual species / reporting-effort challenge.

Habitat suitability, recorder effort and environmental availability
are separate generative components. True habitat intensity and effort
are NEVER response labels for training the species estimator.

PO observation process: lambda_observed(s) proportional to
lambda_ecological(s) * q_effort(s).
Background: uniformly sampled accessible area, NOT biological absence.
A third 'imperfect effort proxy' comes from a prespecified independent
observer process, not labels for the focal species.

Spatial outer test x-stripes and inner stage-selection stripes are
strictly disjoint from model fitting. No true habitat labels or held-
out test metrics are used in selecting any model hyperparameter.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

PREDICTORS=("temperature","evi","humidity","host",
            "saturation_deficit","land_cover")
LANDSCAPES=("ecotone","interaction")
EFFORTS=("uniform","biased")
SEEDS=(11711,11807,11909)
OUTER_FOLDS=(1,2,3)
BLOCKS=5

@dataclass
class VirtualEffortCase:
    landscape:str
    effort_condition:str
    seed:int
    outer_fold:int
    inner_fold:int
    p_train:pd.DataFrame
    b_train:pd.DataFrame
    p_val:pd.DataFrame
    b_val:pd.DataFrame
    p_test:pd.DataFrame
    b_test:pd.DataFrame
    q_p_train:np.ndarray
    q_b_train:np.ndarray
    q_p_val:np.ndarray
    q_b_val:np.ndarray
    proxy_b_train:np.ndarray
    proxy_b_val:np.ndarray
    true_b_test:np.ndarray
    effort_b_test:np.ndarray
    training_cells:set[int]
    validation_cells:set[int]
    outer_cells:set[int]

def surface(mode:str,nside=55):
    if mode not in LANDSCAPES:
        raise ValueError("Unknown preregistered ecological process")
    axis=np.linspace(-.98,.98,nside)
    xx,yy=np.meshgrid(axis,axis,indexing="xy")
    x=xx.ravel();y=yy.ravel()
    temp=x+.4*y+.3*np.sin(3*y)
    evi=np.sin(2.8*np.pi*x)*np.cos(2*np.pi*y)
    humidity=.7*np.cos(np.pi*y)+.35*x
    host=1.2*np.exp(-((x-.45)**2+(y+.25)**2)/.2)+(
       .8*np.exp(-((x+.5)**2+(y-.42)**2)/.16))
    deficit=.7*x*y+.3*x*x-.4*y*y
    cover=np.digitize(x+.15*y,[-.35,.35]).astype(float)
    frame=pd.DataFrame(dict(
       temperature=temp,evi=evi,humidity=humidity,
       host=host,saturation_deficit=deficit,land_cover=cover))
    if mode=="ecotone":
       eta=2.1*np.exp(-((temp-.35)**2)/.18)+(
           .7*host-.4*deficit+.35*evi)
    else:
       eta=1.65*np.tanh(3.1*temp*evi)+.75*host+(
           .55*humidity*deficit-.32*cover)
    habitat=np.exp(np.clip(eta-np.max(eta),-30,0))
    # "Recorder" activity independent of focal species labels,
    # but can share environmental and geographic causes with habitat.
    raw_effort=np.exp(1.2*x+.6*humidity+.4*np.sin(3*y+1.4*x))
    effort=raw_effort/raw_effort.mean()
    # Independent imperfect target-group effort proxy: biased exponent
    # plus smooth observation-network noise; target labels not used.
    proxy=np.exp(.65*np.log(effort)+.27*np.sin(4*y-2*x))
    proxy=proxy/proxy.mean()
    block=np.clip(np.floor((x+1.)*BLOCKS/2).astype(int),
                  0,BLOCKS-1)
    if set(block)!=set(range(BLOCKS)):
        raise AssertionError("Spatial block partition invalid")
    return frame,habitat,effort,proxy,block

def sample_unique(rng,weights,n):
    w=np.asarray(weights,float)
    if n>=len(w) or not np.isfinite(w).all() or np.any(w<=0):
        raise ValueError("Invalid virtual species quadrature weights")
    return rng.choice(len(w),size=n,replace=False,p=w/w.sum())

def cases(landscape,effort_condition,seed):
    X,h,q,proxy,block=surface(landscape)
    if effort_condition=="uniform":
       observation=np.ones_like(q)
       proxy=np.ones_like(q)
    elif effort_condition=="biased":
       observation=q
    else:raise ValueError("Unknown reporting process")
    rng=np.random.default_rng(seed+331)
    presence=sample_unique(rng,h*observation,550)
    background=sample_unique(rng,np.ones_like(h),1500)
    for outer in OUTER_FOLDS:
      held=block==outer
      p_outer=presence[~held[presence]]
      b_outer=background[~held[background]]
      p_test=presence[held[presence]]
      b_test=np.flatnonzero(held)
      candidates=[]
      for inner in sorted(set(block[p_outer])):
         mask_p=block[p_outer]==inner
         mask_b=block[b_outer]==inner
         if mask_p.sum()>=25 and mask_b.sum()>=30 and (
           (~mask_p).sum()>=100) and ((~mask_b).sum()>=100):
            candidates.append((abs(mask_p.mean()-.2),inner,
                               mask_p,mask_b))
      if not candidates:
         raise RuntimeError("Prespecified spatial validation infeasible")
      _,inner,pi,bi=min(candidates,key=lambda z:(z[0],z[1]))
      tp=p_outer[~pi];vp=p_outer[pi]
      tb=b_outer[~bi];vb=b_outer[bi]
      training=set(tp).union(tb)
      validation=set(vp).union(vb)
      test=set(b_test)
      if (training & validation or training & test or
          validation & test):
          raise AssertionError("Spatial blocked leakage")
      output=VirtualEffortCase(
        landscape=landscape,effort_condition=effort_condition,
        seed=seed,outer_fold=outer,inner_fold=inner,
        p_train=X.iloc[tp].reset_index(drop=True),
        b_train=X.iloc[tb].reset_index(drop=True),
        p_val=X.iloc[vp].reset_index(drop=True),
        b_val=X.iloc[vb].reset_index(drop=True),
        p_test=X.iloc[p_test].reset_index(drop=True),
        b_test=X.iloc[b_test].reset_index(drop=True),
        q_p_train=observation[tp],q_b_train=observation[tb],
        q_p_val=observation[vp],q_b_val=observation[vb],
        proxy_b_train=proxy[tb],proxy_b_val=proxy[vb],
        true_b_test=h[b_test],effort_b_test=observation[b_test],
        training_cells=training,validation_cells=validation,
        outer_cells=test)
      yield output

def validate_case(c:VirtualEffortCase):
    assert not (c.training_cells & c.validation_cells)
    assert not (c.training_cells & c.outer_cells)
    assert not (c.validation_cells & c.outer_cells)
    for d in (c.p_train,c.b_train,c.p_val,c.b_val,
              c.p_test,c.b_test):
        assert tuple(d.columns)==PREDICTORS
        assert np.isfinite(d.to_numpy()).all()
    assert len(c.p_train)>=100
    assert len(c.p_val)>=25
    assert len(c.b_train)>=100
    assert len(c.b_val)>=30
    assert len(c.p_test)>=20
    assert len(c.b_test)>100
    assert np.min(c.true_b_test)>0
    assert len(c.q_b_train)==len(c.b_train)
    assert len(c.q_b_val)==len(c.b_val)
    return True
