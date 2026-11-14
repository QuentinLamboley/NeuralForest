"""Leakage-free, paired, test-only attribution audit for PU guidance.

The 2-restart PUStableTeacher shares precisely the same fitted native
networks/encoder as PUStable and is NEVER an additional fitted estimator.
This script does not select models or use evaluation outcomes for training.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TEACHER="NeuralForest_PUStableTeacher"
STABLE="NeuralForest_PUStable"
LEGACY="NeuralForest_PUNeural"
SAFE="NeuralForest_PUSafe"
PAIRS=(
    (STABLE,TEACHER,"incremental_tree_effect"),
    (TEACHER,LEGACY,"extra_teacher_restarts"),
    (STABLE,LEGACY,"combined_effect"),
    (SAFE,LEGACY,"rank_guard_vs_legacy"))
METRICS=("roc_auc","pr_auc","boyce","boyce_rank","brier","logloss")
MINIMIZE={"brier","logloss"}
STRICT_RANK_METRICS=("roc_auc","pr_auc")
DISPLAY=("RandomForest","RF_BalancedUndersampled","RF_Subsample50",
         "XGBoost","Voting_Heterogeneous","Stacking_Boosting",
         LEGACY,TEACHER,STABLE,SAFE)

def paired_values(frame,keys,model_col,student,teacher,metric):
    a=frame[frame[model_col]==student].set_index(keys)
    b=frame[frame[model_col]==teacher].set_index(keys)
    if a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError("Duplicated comparison keys")
    if set(a.index)!=set(b.index):
        raise ValueError(f"Unmatched rows for {student} and {teacher}")
    a=a.sort_index();b=b.reindex(a.index)
    sign=-1 if metric.removeprefix("external_") in MINIMIZE else 1
    delta=sign*(a[metric].to_numpy(float)-b[metric].to_numpy(float))
    if metric.removeprefix("external_") in STRICT_RANK_METRICS and (
        not np.all(np.isfinite(delta))):
        raise ValueError("Missing or non-finite discrimination metrics")
    return a,b,delta

def audit_frame(frame,domain):
    real=domain=="real"
    keys=(["dataset","regime","seed"] if real else ["scenario","seed"])
    group=keys[:-1]
    model_col="method" if real else "model"
    for key in keys+[model_col]:
        if key not in frame: raise ValueError(f"Missing {key}")
    if frame[keys+[model_col]].duplicated().any():
        raise ValueError("Duplicated per-model case keys")
    required=set(sum(([p[0],p[1]] for p in PAIRS),[]))
    if not required.issubset(set(frame[model_col])):
        raise ValueError("Matched teacher or required PU models are missing")
    for m in METRICS:
        for prefix in ("","external_"):
            if prefix+m not in frame:
                raise ValueError(f"Missing metric: {prefix+m}")
    pair_rows=[]
    for labels,sub in frame.groupby(group,sort=True):
        if not isinstance(labels,tuple): labels=(labels,)
        for student,teacher,comparison in PAIRS:
            for prefix,cohort in (("","test"),("external_","external")):
                for metric in METRICS:
                    a,b,d=paired_values(sub,keys,model_col,student,
                                      teacher,prefix+metric)
                    finite=d[np.isfinite(d)]
                    sd=float(finite.std(ddof=1)) if len(finite)>1 else float("nan")
                    pair_rows.append(dict(
                      **dict(zip(group,labels)),comparison=comparison,
                      student=student,teacher=teacher,cohort=cohort,
                      metric=metric,n=len(finite),n_attempted=len(d),
                      mean_gain=float(finite.mean()) if len(finite) else float("nan"),
                      sd_gain=sd,
                      min_gain=float(finite.min()) if len(finite) else float("nan"),
                      max_gain=float(finite.max()) if len(finite) else float("nan"),
                      gains=int((finite>1e-9).sum()),
                      losses=int((finite< -1e-9).sum()),
                      ties=int((abs(finite)<=1e-9).sum()),
                      mean_student=float(a[prefix+metric].mean()),
                      mean_teacher=float(b[prefix+metric].mean()),
                      mean_student_seconds=float(a.total_seconds.mean()),
                      mean_teacher_seconds=float(b.total_seconds.mean())))
    comparisons=pd.DataFrame(pair_rows)
    # Strict rank-check is a *test-only release invariant*. This is not
    # model selection; a violation fails the experiment instead of being
    # silently hidden in a mean.
    for prefix in ("","external_"):
        for metric in STRICT_RANK_METRICS:
            _,_,diff=paired_values(frame,keys,model_col,SAFE,LEGACY,prefix+metric)
            if np.any(abs(diff)>1e-7):
                raise AssertionError(
                  f"Safe rank preservation violated for {prefix+metric}: "
                  f"max delta {np.abs(diff).max():.9g}")
    controls=[]
    for labels,sub in frame.groupby(group,sort=True):
        if not isinstance(labels,tuple):labels=(labels,)
        for model in DISPLAY:
            x=sub[sub[model_col]==model]
            if x.empty:continue
            controls.append(dict(**dict(zip(group,labels)),
                model=model,n=len(x),
                auc=float(x.roc_auc.mean()),
                ap=float(x.pr_auc.mean()),
                boyce=float(x.boyce.mean()),
                shifted_auc=float(x.external_roc_auc.mean()),
                shifted_ap=float(x.external_pr_auc.mean()),
                seconds=float(x.total_seconds.mean())))
    controls=pd.DataFrame(controls)
    return comparisons,controls

def write_report(domain,raw,output):
    frame=pd.read_csv(raw)
    comparisons,controls=audit_frame(frame,domain)
    output.mkdir(parents=True,exist_ok=True)
    comparisons.to_csv(output/"pu_paired_attribution.csv",index=False)
    controls.to_csv(output/"pu_baseline_controls.csv",index=False)
    lines=[
       "# PU guided forest: paired teacher attribution and transport audit","",
       f"Evaluated commit: {os.environ.get('GITHUB_SHA','local/unspecified')}.",
       f"Domain: {domain}. Both evaluation cohorts are test-only.",
       "No test outcomes were used to fit, select or calibrate any model.",
       "",
       "**Crucial control:** PUStableTeacher is the exact fitted two-restart",
       "neural component of PUStable. It shares training and search time,",
       "so its total time is NOT an independent neural-only compute cost.",
       "Incremental-tree comparisons use identical networks and encoders.",
       "The cheaper PUNeural reference has only one neural restart.",
       "",
       "## Matching teacher, tree contribution, and transport",
       "",
       "| Group | Test metric | Comparison | Mean gain | Losses / seeds | Shifted gain |",
       "| --- | --- | --- | ---: | ---: | ---: |"]
    group=["dataset","regime"] if domain=="real" else ["scenario"]
    select=comparisons[comparisons.metric.isin(["roc_auc","pr_auc","boyce"])]
    for g,sub in select.groupby(group,sort=True):
        key="/".join(map(str,g if isinstance(g,tuple) else (g,)))
        for (comparison,metric),cohort in sub.groupby(["comparison","metric"]):
            base=cohort[cohort.cohort=="test"].iloc[0]
            shifted=cohort[cohort.cohort=="external"].iloc[0]
            lines.append(f"| {key} | {metric} | {comparison} | "
             f"{base.mean_gain:+.4f} | {int(base.losses)}/{int(base.n)} | "
             f"{shifted.mean_gain:+.4f} |")
    lines += ["","## Core comparators (not selected by these results)","",
       "| Group | Method | AUC | AP | Shifted AUC | Seconds, both cohorts |",
       "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in controls.itertuples(index=False):
        key="/".join(str(getattr(r,k)) for k in group)
        lines.append(f"| {key} | {r.model} | {r.auc:.4f} | "
                     f"{r.ap:.4f} | {r.shifted_auc:.4f} | {r.seconds:.3f} |")
    lines += ["","## Important limits","",
      "Matched-forest gains can be positive or negative, especially under",
      "unrepresentative background selection and feature-dependent",
      "positive confirmation. APUUC/SCAR assumptions must not be inferred",
      "from these results. The synthetic shifted cohort is generated by",
      "the same simulator, and the real-data partitions come from the",
      "same original datasets; neither is an external clinical site.",
      "Only 3 seeds per real regime and 10 synthetic seeds were used.",
      "Intervals across reused simulation designs are exploratory.",
      "Outcome-based gate tuning on these test results is prohibited.",
      "A separate reserved verified-label audit must be used to make",
      "prospective selection decisions.",
      "",
      "For precise scores and phase-level timings see the paired CSV",
      "and the full benchmark all_runs.csv, as well as",
      "pu_baseline_controls.csv."]
    (output/"PU_ATTRIBUTION.md").write_text(
       "\n".join(lines)+"\n",encoding="utf-8")
    print(f"PU attribution ({domain}): {len(comparisons)} pair rows; "
          f"{len(controls)} comparator rows; rank invariant holds",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--domain",choices=("synthetic","real"),required=True)
    args=ap.parse_args()
    if args.domain=="synthetic":
        write_report(args.domain,ROOT/"benchmark/results/all_runs.csv",
                     ROOT/"benchmark/results")
    else:
        write_report(args.domain,ROOT/"audit/real_pu_results/all_runs.csv",
                     ROOT/"audit/real_pu_results")
