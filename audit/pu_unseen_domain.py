"""Predeclared fourth-source PU transport challenge (retrospective).

Diabetes progression target is dichotomized at a fixed threshold of
180, chosen as a source-independent task definition, not a clinical
diagnosis. This dataset did not occur in earlier NeuralForest suites.
No test outcomes enter any learner or hyperparameter selection.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"benchmark"))
from run import model_candidates,evaluate_sklearn,evaluate_R,metric_table
from audit.pu_attribution import write_report

OUT=ROOT/"audit"/"pu_unseen_results"
SEEDS=(1601,1709,1801)
REGIMES=("SCAR","SAR")
THRESHOLD=180.0
DATASET="Diabetes_Progression180"
BASELINES=("RandomForest","RF_BalancedUndersampled","XGBoost",
           "Voting_Heterogeneous","Stacking_RF_XGB")
TOTAL=10

def run_case(seed,regime):
    raw=load_diabetes()
    X=pd.DataFrame(raw.data,columns=[
        f"x{j:02d}" for j in range(raw.data.shape[1])])
    truth=(np.asarray(raw.target)>=THRESHOLD).astype(int)
    train_ids,hold_ids=train_test_split(
        np.arange(len(truth)),train_size=.6,stratify=truth,
        random_state=seed)
    test_ids,ext_ids=train_test_split(
        hold_ids,test_size=.5,stratify=truth[hold_ids],
        random_state=seed+19)
    xtr=X.iloc[train_ids].reset_index(drop=True)
    xte=X.iloc[test_ids].reset_index(drop=True)
    xext=X.iloc[ext_ids].reset_index(drop=True)
    truth_train=truth[train_ids]
    ytest=truth[test_ids]
    yext=truth[ext_ids]
    pos=np.flatnonzero(truth_train==1)
    n_confirmed=max(20,int(round(.70*len(pos))))
    if n_confirmed>=len(pos):raise RuntimeError("Too few positives")
    rng=np.random.default_rng(seed+1001)
    p=None;selected_feature=-1
    if regime=="SAR":
        xpositive=xtr.iloc[pos].to_numpy(float)
        selected_feature=int(np.argmax(np.var(xpositive,axis=0)))
        z=xpositive[:,selected_feature]
        z=(z-np.median(z))/max(np.std(z),1e-8)
        w=np.exp(np.clip(2.0*z,-6,6))
        p=w/w.sum()
    elif regime!="SCAR":
        raise ValueError("Unknown label confirmation mechanism")
    verified=rng.choice(pos,size=n_confirmed,replace=False,p=p)
    pu=np.zeros(len(xtr),dtype=int)
    pu[verified]=1
    training=xtr.assign(Vampirisme=pu)
    test=xte.assign(Vampirisme=ytest)
    external=xext.assign(Vampirisme=yext)
    rows=[]
    def add(name,pred,pexternal,tr,predtime,selection,refit,config):
        m=metric_table(ytest,np.asarray(pred,dtype=float))
        ex=metric_table(yext,np.asarray(pexternal,dtype=float))
        if not np.isfinite(m["roc_auc"]) or not np.isfinite(ex["roc_auc"]):
            raise RuntimeError(f"Missing AUC for {name}")
        rows.append(dict(dataset=DATASET,regime=regime,seed=seed,
            source="sklearn.datasets.load_diabetes",
            label_definition=f"progression>=180 (not clinical diagnosis)",
            n_train=len(xtr),n_test=len(xte),n_external=len(xext),
            true_train_positive=int(truth_train.sum()),
            labeled_train_positive=int(pu.sum()),
            sar_feature=selected_feature,
            method=name,total_seconds=float(tr+predtime),
            train_seconds=float(tr),predict_seconds=float(predtime),
            selection_seconds=float(selection),
            final_fit_seconds=float(refit),
            configuration=config,**m,
            **{"external_"+k:v for k,v in ex.items()}))
    candidates=model_candidates(seed)
    selected={k:candidates[k] for k in BASELINES}
    for name,pred,tr,pt,inner,cfg,sel,refit,pexternal in evaluate_sklearn(
       selected,xtr,pu,xte,ytest,xext,"auc",seed):
        add(name,pred,pexternal,tr,pt,sel,refit,cfg)
    for model in evaluate_R(training,test,external,"auc",seed,ROOT,
                            script="fit_pu_r.R"):
        add(model["name"],model["probability"],
            model["external_probability"],
            model["train_seconds"],model["prediction_seconds"],
            model["selection_seconds"],model["final_fit_seconds"],
            model["configuration"])
    if len(rows)!=TOTAL or len({r["method"] for r in rows})!=TOTAL:
        raise RuntimeError(f"Expected {TOTAL} prediction methods")
    print(f"{DATASET}/{regime}/{seed}: {len(rows)} methods",flush=True)
    return rows

def main():
    import csv
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for regime in REGIMES:
        for seed in SEEDS:
            rows.extend(run_case(seed,regime))
    if len(rows)!=TOTAL*len(SEEDS)*len(REGIMES):
        raise RuntimeError("Incomplete new-source validation")
    with (OUT/"all_runs.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    manifest={
        "source":"sklearn.datasets.load_diabetes",
        "predeclared_positive_definition":"progression>=180",
        "clinical_diagnosis":False,
        "seeds":list(SEEDS),"regimes":list(REGIMES),
        "baseline_models":list(BASELINES),
        "native_pu_predictions":5,
        "records":len(rows),"independent_fits":len(rows)-len(SEEDS)*len(REGIMES),
        "trusted_negative_test_labels_used_only_for_evaluation":True,
        "independent_external_site":False,
        "dataset_appeared_in_earlier_NeuralForest_benchmarks":False}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",
                                      encoding="utf8")
    write_report("real",OUT/"all_runs.csv",OUT)
    with (OUT/"README.md").open("w",encoding="utf8") as f:
        f.write("# Unseen source: diabetes progression threshold challenge\n\n"
                "This study uses a new public dataset with a prespecified "
                "binary high-progression threshold of 180. It is neither "
                "a clinical diagnostic threshold nor external hospital "
                "validation. Three seeds, SCAR and feature-dependent SAR, "
                "and two disjoint test partitions are evaluated. The original "
                "true outcomes are masked during all PU training and used "
                "only for untouched test scoring. The five prior-established "
                "baseline algorithms are NOT all possible competitors; "
                "the 31-method full synthetic suite remains separate.\n"
                "The matched neural teacher and both test cohorts support "
                "tree-attribution and conservative rank-safety checks.\n")
    print("Unseen-source audit complete:",len(rows),"prediction records")

if __name__=="__main__":
    main()
