"""Experimental design using td-machinelearning's unmodified vampire generator."""
from __future__ import annotations
import dataclasses
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "td-source"
if not SOURCE.is_dir():
    raise RuntimeError("Missing td-source checkout. See benchmark/README.md.")
sys.path.insert(0, str(SOURCE))

from vampire_lab.config import initial_parameters, KEYS
from vampire_lab.simulation import parameters_with_proximity, simulate_population

# Pinned external engine: QuentinLamboley/td-machinelearning@48c2b28ca7874cbef1f7dde15de962c6e56c5027
@dataclasses.dataclass(frozen=True)
class Scenario:
    key: str
    n: int
    proximity: int
    prevalence: float
    target_metric: str
    extension: str = "none"
    test_proximity: int | None = None

SCENARIOS = {
    "core": [
        Scenario("distinct_0",600,0,0.5,"auc"),
        Scenario("intermediate_45",600,45,0.5,"auc"),
        Scenario("overlap_80",600,80,0.5,"auc"),
        Scenario("similar_97",600,97,0.5,"brier"),
        Scenario("identical_100",600,100,0.5,"logloss")
    ],
    "imbalance": [
        Scenario("rare_10pct",900,55,0.10,"pr_auc"),
        Scenario("rare_3pct",1800,55,0.03,"pr_auc"),
        Scenario("small_n",240,70,0.5,"balanced_accuracy"),
        Scenario("large_n",2400,70,0.5,"auc")
    ],
    "structure": [
        Scenario("noise_50",800,65,0.5,"auc","noise"),
        Scenario("correlated_30",800,65,0.5,"auc","correlated"),
        Scenario("nonlinear_xor",950,100,0.5,"f1","xor"),
        Scenario("distribution_shift",800,45,0.5,"auc","none",test_proximity=78)
    ],
    "presence": [
        Scenario("presence_bg_1to1",1400,60,0.28,"boyce","bg1"),
        Scenario("presence_bg_1to5",2800,60,0.28,"boyce","bg5"),
        Scenario("pseudoabsence_hard",2400,70,0.28,"boyce","hard")
    ],
    "stress": [
        Scenario("rare_1pct",5000,55,0.01,"pr_auc"),
        Scenario("label_noise_20pct",1100,55,0.5,"auc","labelnoise"),
        Scenario("missing_15pct",1200,65,0.5,"auc","missing")
    ]
}
ALL_SCENARIOS = {s.key:s for v in SCENARIOS.values() for s in v}

def augment(df: pd.DataFrame, kind: str, seed: int) -> pd.DataFrame:
    data=df.copy()
    rng=np.random.default_rng(seed)
    if kind=="noise":
        for j in range(50):
            data[f"noise_{j:02d}"]=rng.normal(0,1,len(data))
    elif kind=="correlated":
        for j in range(30):
            source=KEYS[j%len(KEYS)]
            data[f"derived_{j:02d}"]=data[source]+rng.normal(
                0,0.01*max(1,df[source].std()),len(data))
    elif kind=="xor":
        data["Vampirisme"]=(
            (data["Regeneration"].to_numpy()>5.5) ^
            (data["Fer"].to_numpy()>95.0)).astype(int)
    elif kind=="labelnoise":
        flips=rng.random(len(data))<0.20
        data.loc[flips,"Vampirisme"]=1-data.loc[flips,"Vampirisme"]
    elif kind=="missing":
        for key in KEYS:
            mask=rng.random(len(data))<0.15
            data.loc[mask,key]=np.nan
    return data

def create_data(spec: Scenario, seed: int):
    parameters=parameters_with_proximity(initial_parameters(),spec.proximity)
    test_parameters=parameters_with_proximity(
        initial_parameters(),
        spec.proximity if spec.test_proximity is None else spec.test_proximity)
    # Full independence between generated individuals in train and test.
    source=simulate_population(parameters,spec.n,spec.prevalence,seed)
    test=simulate_population(test_parameters,
                             max(1000,min(2400,spec.n*2)),
                             spec.prevalence,seed+100_003)
    source=augment(source,spec.extension,seed+3)
    test=augment(test,spec.extension,seed+100_005)
    target_status="verified labels"
    if spec.extension.startswith("bg") or spec.extension=="hard":
        rng=np.random.default_rng(seed+53)
        positive=np.where(source["Vampirisme"].to_numpy()==1)[0]
        observed=rng.choice(positive,size=min(220,len(positive)),replace=False)
        remaining=np.setdiff1d(np.arange(len(source)),observed)
        ratio=5 if spec.extension=="bg5" else 1
        size=min(len(remaining),len(observed)*ratio)
        if spec.extension=="hard":
            # Background candidates are selected using features only.
            # Their true labels are never consulted, including the labels
            # of any unobserved vampires in the candidate pool.
            scaler=StandardScaler().fit(source.iloc[observed][KEYS])
            center=scaler.transform(source.iloc[observed][KEYS]).mean(axis=0)
            distance=((scaler.transform(source.iloc[remaining][KEYS])-center)**2).sum(axis=1)
            background=remaining[np.argsort(distance)[:size]]
        else:
            background=rng.choice(remaining,size=size,replace=False)
        observed_df=source.iloc[observed].copy()
        observed_df["Vampirisme"]=1
        background_df=source.iloc[background].copy()
        background_df["Vampirisme"]=0
        source=pd.concat([observed_df,background_df],ignore_index=True)
        source=source.iloc[rng.permutation(len(source))].reset_index(drop=True)
        target_status="labeled presences vs unlabeled background (contaminated negatives)"
    if source["Vampirisme"].nunique()!=2 or test["Vampirisme"].nunique()!=2:
        raise RuntimeError(f"Insufficient classes in {spec.key}")
    train_X=source.drop(columns=["Vampirisme"])
    test_X=test.drop(columns=["Vampirisme"])
    if list(train_X)!=list(test_X):
        raise AssertionError("Training and test columns must be identical")
    return source,test,dict(
       source_repository="QuentinLamboley/td-machinelearning",
       source_commit="48c2b28ca7874cbef1f7dde15de962c6e56c5027",
       scenario=dataclasses.asdict(spec),
       seed=seed,train_n=len(source),test_n=len(test),
       train_positive=int(source["Vampirisme"].sum()),
       test_positive=int(test["Vampirisme"].sum()),
       label_semantics=target_status)
