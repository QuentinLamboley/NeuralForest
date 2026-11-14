"""Paired PU residual add-on to the pinned 0.9.2 synthetic benchmark.

Uses the EXACT same 3 presence/background scenarios, 10 seeds, simulator
revision, test outcomes, and independent shifted cohorts; runs ONLY the
new C++ gradient-based experimental learner and its exact neural teacher.
Prior benchmark results are test-only controls, not used for selection.

There is a shared training cost: the matched teacher is not a second fit.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"benchmark"))
from design import SCENARIOS,create_data,create_external_population
from run import evaluate_R,metric_table,SELECTION_SEEDS

OUT=ROOT/"audit"/"pu_residual_legacy_results"
OLD=ROOT/"benchmark"/"results"/"all_runs.csv"
PU=("NeuralForest_PUResidual","NeuralForest_PUResidualTeacher")
BASELINES=("NeuralForest_PUNeural","NeuralForest_PUStableTeacher",
           "NeuralForest_PUSafe","NeuralForest_PUStable",
           "RandomForest","RF_Subsample50","RF_BalancedUndersampled",
           "XGBoost","CatBoost","Voting_Heterogeneous",
           "Stacking_Boosting")

def one(spec,seed):
    train,test,_=create_data(spec,seed)
    ext,_=create_external_population(spec,seed)
    results=evaluate_R(train,test,ext,spec.target_metric,seed,
                       ROOT,script="pu_residual_legacy_r.R")
    if len(results)!=2 or {x["name"] for x in results}!=set(PU):
        raise RuntimeError("Missing residual or shared teacher prediction")
    y=test["Vampirisme"].to_numpy(int)
    ye=ext["Vampirisme"].to_numpy(int)
    rows=[]
    for result in results:
        p=np.asarray(result["probability"],float)
        q=np.asarray(result["external_probability"],float)
        if len(p)!=len(y) or len(q)!=len(ye):
            raise RuntimeError("Model predictions do not align with held-out labels")
        m=metric_table(y,p)
        external=metric_table(ye,q)
        rows.append(dict(scenario=spec.key,seed=seed,model=result["name"],
          train_n=len(train),test_n=len(test),external_n=len(ext),
          selection_seconds=float(result["selection_seconds"]),
          fit_seconds=float(result["final_fit_seconds"]),
          train_seconds=float(result["train_seconds"]),
          predict_seconds=float(result["prediction_seconds"]),
          total_seconds=float(result["train_seconds"]+
                              result["prediction_seconds"]),
          selected_candidate=result["configuration"].split("selected=")[1].split(";")[0],
          shared_teacher_fit=result["name"]==PU[1],
          **m,**{"external_"+k:v for k,v in external.items()}))
    print(f"{spec.key}/{seed}: "+
          ", ".join(f"{r['model']} AUC {r['roc_auc']:.4f}"
                    for r in rows),flush=True)
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    new=[]
    for spec in SCENARIOS["presence"]:
        for seed in SELECTION_SEEDS:
            new.extend(one(spec,seed))
    result=pd.DataFrame(new)
    expected=len(SCENARIOS["presence"])*len(SELECTION_SEEDS)*len(PU)
    if len(result)!=expected or result[["scenario","seed","model"]].duplicated().any():
        raise RuntimeError("Incomplete or duplicated residual-vs-teacher trial")
    result.to_csv(OUT/"all_runs.csv",index=False)
    base=pd.read_csv(OLD)
    base=base[base.scenario.isin(result.scenario.unique())&
              base.model.isin(BASELINES)].copy()
    if base.empty: raise RuntimeError("No pinned reference baselines")
    if set(zip(base.scenario,base.seed))!=set(zip(result.scenario,result.seed)):
        raise RuntimeError("Comparator seed/design mismatch")
    if base[["scenario","seed","model"]].duplicated().any():
        raise RuntimeError("Duplicate comparator baseline records")
    comparison=pd.concat([result,base],ignore_index=True,sort=False)
    comparison.to_csv(OUT/"comparison.csv",index=False)
    pairs=[]
    for (scenario,seed),group in result.groupby(["scenario","seed"]):
        a=group[group.model==PU[0]].iloc[0]
        b=group[group.model==PU[1]].iloc[0]
        for col in ("roc_auc","pr_auc","brier","logloss","boyce",
                    "boyce_rank","external_roc_auc","external_pr_auc",
                    "external_brier","external_logloss","external_boyce",
                    "external_boyce_rank"):
            direction=-1 if col.endswith(("brier","logloss")) else 1
            pairs.append(dict(scenario=scenario,seed=seed,metric=col,
                gain=direction*(a[col]-b[col]),
                selected=a.selected_candidate))
    paired=pd.DataFrame(pairs)
    paired.to_csv(OUT/"paired_effect.csv",index=False)
    lines=["# PU residual gradient: matched comparison on the pinned synthetic benchmark","",
      "This predeclared add-on runs the new C++17 learner and its exact",
      "matched neural teacher on the same 3 presence/background scenarios,",
      "10 seeds, true-label test samples and shifted cohort as the v0.9.2",
      "reference benchmark. Comparators reuse recorded frozen v0.9.2",
      "outcomes, not retrained estimates. Test labels never select models.",
      "The simulator has been inspected repeatedly and is NOT independent",
      "confirmatory evidence. All 60 prediction records reflect 30 new",
      "actual model fits, since the teacher is a shared-fit ablation.",
      "",
      "| Scenario | Original nnPU AUC | Stable matched teacher AUC | New teacher AUC | Residual AUC | Incremental AUC | Boyce gain | Shifted AUC gain | Selected rounds/10 |",
      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for scenario,g in comparison.groupby("scenario"):
        pick=lambda name:g[g.model==name]
        newteacher=pick(PU[1])
        candidate=pick(PU[0])
        oldteacher=pick("NeuralForest_PUStableTeacher")
        oldnn=pick("NeuralForest_PUNeural")
        delta=(candidate.roc_auc.mean()-newteacher.roc_auc.mean())
        boyce=(candidate.boyce.mean()-newteacher.boyce.mean())
        shift=(candidate.external_roc_auc.mean()-newteacher.external_roc_auc.mean())
        selected=int((candidate.selected_candidate!="neural_fallback").sum())
        lines.append(f"| {scenario} | {oldnn.roc_auc.mean():.4f} | "
          f"{oldteacher.roc_auc.mean():.4f} | {newteacher.roc_auc.mean():.4f} | "
          f"{candidate.roc_auc.mean():.4f} | {delta:+.4f} | "
          f"{boyce:+.4f} | {shift:+.4f} | {selected}/{len(SELECTION_SEEDS)} |")
    lines+=["","## Controls and compute time","",
      "| Scenario | Method | AUC | AP | Boyce | Shifted AUC | Total seconds |",
      "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for (scenario,model),g in comparison.groupby(["scenario","model"]):
        lines.append(f"| {scenario} | {model} | {g.roc_auc.mean():.4f} | "
          f"{g.pr_auc.mean():.4f} | {g.boyce.mean():.4f} | "
          f"{g.external_roc_auc.mean():.4f} | {g.total_seconds.mean():.3f} |")
    lines+=["","## Methodological limitations","",
      "The PU-risk gate is only a heuristic under SCAR and predeclared",
      "class priors; SAR, SNAR or hard-selected backgrounds can produce",
      "strongly misleading decisions. The new method does not claim",
      "universally superior discrimination. The same fixed simulator",
      "seeds have been used in earlier iterations; the experiment is",
      "an engineering audit, not an untouched external validation.",
      "Runtime measurements are not directly comparable across jobs",
      "or different GitHub CPU runner vintages. Shared-fit teacher",
      "training time must not be added to the residual learner total.",
      "",
      "Full rows: all_runs.csv, comparison.csv, paired_effect.csv.",
      ""]
    (OUT/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    manifest=dict(run_sha=os.environ.get("GITHUB_SHA","local"),
       generator_revision="48c2b28ca7874cbef1f7dde15de962c6e56c5027",
       scenarios=[s.key for s in SCENARIOS["presence"]],
       seeds=list(SELECTION_SEEDS),
       records=len(result),independent_training_fits=len(result)//2,
       same_seed_as_prior_audits=True,independent_real_world_validation=False)
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
