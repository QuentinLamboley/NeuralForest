"""Virtual-species spatial presence-only design.

TRUE ecological suitability is known to the simulator but NEVER
included in any estimator training CSV. A sampled presence is an
observation at a known species location, NOT evidence that any other
location is unoccupied. Background always describes available space.

The design explicitly separates an effort-weighted detection process
from biological relative intensity and includes two background policies:
uniform-accessibility availability and effort-matched target-group
availability. No independent true negative labels are fabricated.

All spatial folds are determined by x-coordinate BEFORE any fits.
Each withheld block has no overlapping training spatial cells.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

PREDICTORS=("land_cover","temperature","humidity","vegetation",
            "host_availability","saturation_deficit")
SCENARIOS=("linear","nonlinear_interaction","multimodal")
BACKGROUND_POLICIES=("uniform","effort_matched")
EFFORT_CONDITIONS=("unbiased","effort_biased")
BLOCKS=5
SEEDS=(8101,8231)

@dataclass(frozen=True)
class SpatialCase:
    scenario:str
    effort_condition:str
    background_policy:str
    seed:int
    fold:int
    presence_train:pd.DataFrame
    background_train:pd.DataFrame
    presence_test:pd.DataFrame
    background_test:pd.DataFrame
    oracle_true_intensity:np.ndarray
    oracle_effort:np.ndarray
    all_train_cells:set[int]
    all_test_cells:set[int]
    background_overlap_with_presence:int
    present_n:int

def landscape(scenario:str,nside:int=70):
    axis=np.linspace(-.985,.985,nside)
    xx,yy=np.meshgrid(axis,axis,indexing="xy")
    x=xx.ravel();y=yy.ravel()
    temperature=1.1*x+.5*y+.25*np.sin(3*y)
    vegetation=np.sin(2*np.pi*x)*np.cos(np.pi*y)
    humidity=.8*np.cos(np.pi*y)+.4*x
    hosts=np.exp(-((x-.25)**2+(y+.3)**2)/.13)+.65*np.exp(
          -((x+.55)**2+(y-.4)**2)/.24)
    saturation=x*y+.55*x*x-y*y
    landcover=(np.digitize(x+.2*y,[-.3,.3])).astype(float)
    features=pd.DataFrame({
      "land_cover":landcover,
      "temperature":temperature,
      "humidity":humidity,
      "vegetation":vegetation,
      "host_availability":hosts,
      "saturation_deficit":saturation})
    if scenario=="linear":
        eta=.85*temperature+.72*hosts-.48*saturation+.2*humidity
    elif scenario=="nonlinear_interaction":
        eta=2.3*np.tanh(3.0*temperature*vegetation)+1.0*hosts-.35*humidity
    elif scenario=="multimodal":
        eta=(3.0*np.exp(-((x-.50)**2+(y+.43)**2)/.12)
             +2.7*np.exp(-((x+.53)**2+(y-.42)**2)/.10)
             -.5*temperature-.35*saturation)
    else:raise ValueError(scenario)
    true=np.exp(np.clip(eta-eta.max(),-30,0))
    effort=np.exp(.92*x+.40*y)*(1+.75*np.exp(-((y-x+.12)/.13)**2))
    block=np.minimum(BLOCKS-1,
      np.floor((x+1.)*BLOCKS/2).astype(int))
    if set(block)!=set(range(BLOCKS)):
        raise RuntimeError("Spatial fold assignment incomplete")
    return features,true,effort,block

def draw_indices(rng:np.random.Generator,weights,n:int):
    w=np.asarray(weights,dtype=float)
    if (w<0).any() or not np.isfinite(w).all() or w.sum()<=0:
        raise ValueError("Sampling weights invalid")
    return rng.choice(len(w),size=n,replace=False,p=w/w.sum())

def spatial_cases(scenario:str,effort_condition:str,
                  background_policy:str,seed:int):
    features,truth,effort,folds=landscape(scenario)
    npres=700;nbg=1500
    rng=np.random.default_rng(seed+37)
    if effort_condition=="unbiased":
        detection=np.ones(len(features))
    elif effort_condition=="effort_biased":
        detection=effort
    else:raise ValueError(effort_condition)
    present=draw_indices(rng,truth*detection,npres)
    if background_policy=="uniform":
        design=np.ones(len(features))
    elif background_policy=="effort_matched":
        design=detection
    else:raise ValueError(background_policy)
    bg=draw_indices(rng,design,nbg)
    # BG is allowed to overlap observed presences by design; no
    # true-absence status is read, nor is overlap artificially removed.
    overlap=len(set(present).intersection(bg))
    for fold in range(BLOCKS):
        held=(folds==fold)
        ptr=present[~held[present]]
        ptest=present[held[present]]
        btr=bg[~held[bg]]
        btest=np.flatnonzero(held) # exhaustive, uniform availability in block
        if len(ptr)<30 or len(ptest)<10 or len(btr)<40:
            raise ValueError("Too little spatial-block support")
        traincells=set(ptr).union(btr)
        testcells=set(ptest).union(btest)
        if not traincells.isdisjoint(testcells):
            raise AssertionError("Spatial leakage detected")
        yield SpatialCase(
          scenario=scenario,effort_condition=effort_condition,
          background_policy=background_policy,seed=seed,fold=fold,
          presence_train=features.iloc[ptr].reset_index(drop=True),
          background_train=features.iloc[btr].reset_index(drop=True),
          presence_test=features.iloc[ptest].reset_index(drop=True),
          background_test=features.iloc[btest].reset_index(drop=True),
          oracle_true_intensity=np.asarray(truth[btest],float),
          oracle_effort=np.asarray(effort[btest],float),
          all_train_cells=traincells,all_test_cells=testcells,
          background_overlap_with_presence=overlap,present_n=npres)

def validate_case(case:SpatialCase):
    assert case.all_train_cells.isdisjoint(case.all_test_cells)
    assert list(case.presence_train)==list(PREDICTORS)
    assert list(case.background_train)==list(PREDICTORS)
    assert list(case.presence_test)==list(PREDICTORS)
    assert list(case.background_test)==list(PREDICTORS)
    assert case.oracle_true_intensity.shape==(
      len(case.background_test),)
    assert len(case.presence_train)>=30
    assert len(case.presence_test)>=10
    assert (case.oracle_true_intensity>0).all()
    return True
