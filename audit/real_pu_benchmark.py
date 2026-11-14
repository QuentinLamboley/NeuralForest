"""Out-of-simulator PU validation on real, publicly bundled datasets.

Verified original labels are used ONLY to construct stratified splits and
to simulate which *positive* training cases receive confirmation.
The learner sees confirmed-positive (1) versus unlabeled (0) cases.
External test labels are never used for selection/tuning.
These are independently held-out splits from public datasets, NOT
multi-institutional external validation of Lyme/clinical deployment.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer,load_wine,load_digits
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"benchmark"))
from run import model_candidates,evaluate_sklearn,evaluate_R,metric_table

SEEDS=(911,947,983)
REGIMES=("SCAR","SAR")
OUT=ROOT/"audit"/"real_pu_results"
METHODS=36
SOURCE_INFO={
    "Wisconsin_Breast_Cancer":"sklearn.datasets.load_breast_cancer (Wisconsin Diagnostic Breast Cancer, UCI)",
    "Wine_Class0":"sklearn.datasets.load_wine (Wine recognition, UCI)",
    "Digits_Eight":"sklearn.datasets.load_digits (optical recognition, UCI)",
}
def load_public(name):
    if name=="Wisconsin_Breast_Cancer":
        raw=load_breast_cancer()
        y=(raw.target==0).astype(int)
    elif name=="Wine_Class0":
        raw=load_wine()
        y=(raw.target==0).astype(int)
    else:
        raw=load_digits()
        y=(raw.target==8).astype(int)
    X=pd.DataFrame(raw.data,columns=[f"x{j:02d}" for j in range(raw.data.shape[1])])
    return X,np.asarray(y,dtype=int)

def evaluate_case(name,seed,regime):
    X,y=load_public(name)
    train_ids,hold_ids=train_test_split(
        np.arange(len(y)),train_size=.6,stratify=y,random_state=seed)
    test_ids,external_ids=train_test_split(
        hold_ids,test_size=.5,stratify=y[hold_ids],random_state=seed+17)
    training=X.iloc[train_ids].reset_index(drop=True)
    test=X.iloc[test_ids].reset_index(drop=True)
    external=X.iloc[external_ids].reset_index(drop=True)
    truth_train=y[train_ids]
    ytest=y[test_ids];yext=y[external_ids]
    rng=np.random.default_rng(seed+191)
    known=np.where(truth_train==1)[0]
    n_known=max(20,int(round(.7*len(known))))
    if n_known>=len(known):
        n_known=len(known)-2
    if n_known<20: raise RuntimeError("Need enough observed positives")
    # Feature-dependent positive confirmation (SAR) is constructed using
    # TRAINING positives only. The learner never sees the confirmation
    # propensity or the hidden positive identities.
    feature_index=-1
    selection_prob=None
    if regime=="SAR":
        positive_features=training.iloc[known].to_numpy(dtype=float)
        # Fixed, training-only high-variance feature to create an explicit
        # selection mechanism (no access to held-out test outcomes).
        feature_index=int(np.argmax(positive_features.var(axis=0)))
        v=positive_features[:,feature_index]
        z=(v-np.median(v))/max(1e-8,np.std(v))
        weight=np.exp(np.clip(2*z,-6,6))
        selection_prob=weight/weight.sum()
    elif regime!="SCAR":
        raise ValueError("Unknown positive labeling mechanism")
    visible=rng.choice(known,size=n_known,replace=False,p=selection_prob)
    pseudo=np.zeros(len(training),dtype=int)
    pseudo[visible]=1
    if len(np.unique(pseudo))<2:raise RuntimeError("No unlabeled cases")
    train=training.assign(Vampirisme=pseudo)
    test_df=test.assign(Vampirisme=ytest)
    external_df=external.assign(Vampirisme=yext)
    target_metric="pr_auc" if name=="Digits_Eight" else "auc"
    rows=[]
    def record(model,p,extp,trainsec,predsec,sel,fit,config):
        m=metric_table(ytest,np.asarray(p,dtype=float))
        e=metric_table(yext,np.asarray(extp,dtype=float))
        if not np.isfinite(m["roc_auc"]) or not np.isfinite(e["roc_auc"]):
            raise RuntimeError(f"Invalid AUC from {model}")
        rows.append(dict(dataset=name,seed=seed,regime=regime,
           labeling_feature=feature_index,method=model,
           primary_metric=target_metric,
           source=SOURCE_INFO[name],
           train_n=len(training),train_true_positive=int(truth_train.sum()),
           train_confirmed_positive=int(pseudo.sum()),
           unlabeled_contamination=int((truth_train==1).sum()-pseudo.sum()),
           test_n=len(test_ids),external_n=len(external_ids),
           train_seconds=float(trainsec),predict_seconds=float(predsec),
           selection_seconds=float(sel),final_fit_seconds=float(fit),
           total_seconds=float(trainsec+predsec),
           config=config,**m,
           **{"external_"+k:v for k,v in e.items()}))
    for mod,p,tr,pt,inner,config,sel,fit,extp in evaluate_sklearn(
            model_candidates(seed),training,pseudo,test,ytest,external,
            target_metric,seed):
        record(mod,p,extp,tr,pt,sel,fit,config)
    for script in ("fit_r.R","fit_pu_r.R"):
        for row in evaluate_R(train,test_df,external_df,
                              target_metric,seed,ROOT,script=script):
            record(row["name"],row["probability"],
                   row["external_probability"],
                   row["train_seconds"],row["prediction_seconds"],
                   row["selection_seconds"],row["final_fit_seconds"],
                   row.get("configuration",""))
    if len(rows)!=METHODS or len({r["method"] for r in rows})!=METHODS:
        raise RuntimeError(f"Expected {METHODS} unique models in {regime}, got {len(rows)}")
    for r in rows:
        print(f"{name}/{regime}/{seed} {r['method']:32s} auc={r['roc_auc']:.3f} "
              f"AP={r['pr_auc']:.3f} ext_AUC={r['external_roc_auc']:.3f}",
              flush=True)
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    for regime in REGIMES:
      for name in SOURCE_INFO:
        for seed in SEEDS:
            all_rows.extend(evaluate_case(name,seed,regime))
    if len(all_rows)!=len(SOURCE_INFO)*len(REGIMES)*len(SEEDS)*METHODS:
        raise RuntimeError("Missing real-dataset outcomes")
    fields=list(all_rows[0])
    with (OUT/"all_runs.csv").open("w",newline="",encoding="utf8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(all_rows)
    df=pd.DataFrame(all_rows)
    summary=df.groupby(["dataset","regime","method"],as_index=False).agg(
        roc_auc_mean=("roc_auc","mean"),
        pr_auc_mean=("pr_auc","mean"),
        brier_mean=("brier","mean"),
        logloss_mean=("logloss","mean"),
        ext_auc_mean=("external_roc_auc","mean"),
        ext_pr_auc_mean=("external_pr_auc","mean"),
        ext_brier_mean=("external_brier","mean"),
        seconds_mean=("total_seconds","mean"),
        selection_mean=("selection_seconds","mean"),
        fit_mean=("final_fit_seconds","mean"),
        predict_both_mean=("predict_seconds","mean"))
    summary.to_csv(OUT/"summary.csv",index=False)
    lines=[
      "# PU validation beyond the vampire simulator",
      "",
      "Three real publicly bundled datasets are used: Wisconsin Diagnostic",
      "Breast Cancer, Wine recognition, and handwritten Digits (digit eight",
      "vs other digits). Each dataset is evaluated under SCAR (uniform",
      "positive confirmation) and SAR (feature-dependent positive",
      "confirmation) training-label mechanisms. All negative cases and",
      "30% of positive training cases have unknown labels. Ground truth",
      "from two reserved independent evaluation partitions is never",
      "used by model selection. SAR confirmation propensities depend",
      "only on a high-variance feature identified among training positives.",
      "",
      "This is out-of-simulator validation on real records, but NOT an",
      "independent hospital, veterinary, geographical or field cohort.",
      "PU priors remain fixed at 0.12, 0.25 and 0.40; no test-label",
      "estimate of the class prior is permitted. Three seeds, 36 predictions,",
      "two confirmation regimes and 16 metrics yield 648 evaluations.",
      "The matched two-restart teacher is a component prediction sharing",
      "all of PUStable's training: only 630 distinct model fits occur.",
      "",
      "| Dataset | Regime | Best AUC method | AUC | NF safe | NF stable | NF PU neural | NF PU informed |",
      "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for (name,regime),g in summary.groupby(["dataset","regime"]):
        top=g.sort_values("roc_auc_mean",ascending=False).iloc[0]
        safe=g[g.method=="NeuralForest_PUSafe"].iloc[0]
        stable=g[g.method=="NeuralForest_PUStable"].iloc[0]
        teacher=g[g.method=="NeuralForest_PUNeural"].iloc[0]
        informed=g[g.method=="NeuralForest_PUInformed"].iloc[0]
        lines.append(
            f"| {name} | {regime} | {top.method} | "
            f"{top.roc_auc_mean:.4f} | {safe.roc_auc_mean:.4f} | "
            f"{stable.roc_auc_mean:.4f} | "
            f"{teacher.roc_auc_mean:.4f} | {informed.roc_auc_mean:.4f} |")
    lines += ["","## Stable PU vs predecessor: all metrics and computation","",
       "PUStableTeacher is the exact fitted six-network teacher component",
       "of PUStable and isolates the tree contribution from extra restarts.",
       "No test-based selection is performed. The stable method's",
       "cross-fitting, candidate forests, teacher fitting and holdout",
       "selection are counted under selection_seconds. Its refit and",
       "inference costs are recorded separately. All model scores and",
       "times appear in summary.csv and all_runs.csv.",
       "",
       "| Dataset | Regime | Safe AP | Stable AP | Neural AP | Safe total (s) | Stable total (s) | Neural total (s) |",
       "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for (name,regime),g in summary.groupby(["dataset","regime"]):
        sa=g[g.method=="NeuralForest_PUSafe"].iloc[0]
        st=g[g.method=="NeuralForest_PUStable"].iloc[0]
        nn=g[g.method=="NeuralForest_PUNeural"].iloc[0]
        lines.append(f"| {name} | {regime} | {sa.pr_auc_mean:.4f} | "
                     f"{st.pr_auc_mean:.4f} | {nn.pr_auc_mean:.4f} | "
                     f"{sa.seconds_mean:.3f} | {st.seconds_mean:.3f} | "
                     f"{nn.seconds_mean:.3f} |")
    lines+=["","## Limits","",
     "The pseudo-unlabeled regime is simulated from known real labels.",
     "Outcome labels from the independent test partitions do not train",
     "any model. However, the number of original datasets and seeds is",
     "small, and no independent clinical site is represented.",
     "No universal superiority is established by these comparisons.",
     "The method's label-selection assumptions are more plausible here",
     "than in environmentally selected hard backgrounds but still need",
     "domain-specific evaluation.",
     "",
     "Full results: all_runs.csv, summary.csv and manifest.json."]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    manifest=dict(datasets=list(SOURCE_INFO),seeds=list(SEEDS),
                  model_types=METHODS,regimes=list(REGIMES),rows=len(all_rows),
                  real_source_public=True,hidden_labels_simulated=True,
                  truly_independent_external_sites=False,
                  comparator_count=31,native_pu_methods=5,
                  metrics=16)
    (OUT/"manifest.json").write_text(
        json.dumps(manifest,indent=2),encoding="utf8")
    print("Real-data benchmark complete:",len(all_rows),flush=True)
if __name__=="__main__":
    main()
