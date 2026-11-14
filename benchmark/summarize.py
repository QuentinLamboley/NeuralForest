"""Aggregate benchmark performance, selection overhead and time to predict."""
from __future__ import annotations
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
from multimetric import build_reports

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
RAW=HERE/"out"
RESULTS=HERE/"results"
BASELINE=ROOT/"audit"/"baseline-v0.2-all_runs.csv"
SEEDS=(17,43,71,97,131,197,251,347,401,503)
NEW_SEEDS=(401,503)
SCENARIOS=19
MODELS=31
PU_MODELS=("NeuralForest_PUNeural","NeuralForest_PUInformed",
           "NeuralForest_PUStable","NeuralForest_PUStableTeacher",
           "NeuralForest_PUSafe")
PU_SCENARIOS=("presence_bg_1to1","presence_bg_1to5","pseudoabsence_hard")
METRICS=("roc_auc","pr_auc","accuracy","balanced_accuracy","precision",
         "recall","specificity","f1","mcc","brier","logloss","boyce",
         "boyce_rank","ece","precision_at_10pct","recall_at_10pct")
MINIMIZE={"brier","logloss"}
def fmt(x,n=3):
    return "NA" if not np.isfinite(x) else f"{x:.{n}f}"
def scores_for(g,metric):
    column="roc_auc" if metric=="auc" else metric
    return g[column]
def summarize():
    paths=sorted(RAW.glob("*.csv"))
    if not paths: raise RuntimeError("No trial CSV files")
    data=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    # Each (scenario, seed, model) is unique. Missing fits remain explicit.
    expected=SCENARIOS*len(SEEDS)*MODELS+len(PU_SCENARIOS)*len(SEEDS)*len(PU_MODELS)
    independent_fits=expected-len(PU_SCENARIOS)*len(SEEDS)
    if len(data)!=expected: raise RuntimeError(
        f"Expected {expected} score rows but found {len(data)}")
    if data[["scenario","seed","model"]].duplicated().any():
        raise RuntimeError("Duplicated model-scenario-seed result")
    failed=data[data.status!="ok"]
    if len(failed):
        detail=failed[["scenario","seed","model","error"]].head(12)
        raise RuntimeError(
            f"{len(failed)} of {expected} models failed; refusing to "
            f"publish an incomplete benchmark:\n{detail.to_string(index=False)}")
    # Specialized PU methods apply only to presence/background.
    for scenario,group in data.groupby("scenario"):
        expected_models=MODELS+(len(PU_MODELS) if scenario in PU_SCENARIOS else 0)
        if group.model.nunique()!=expected_models or len(group)!=len(SEEDS)*expected_models:
            raise RuntimeError(f"Incomplete model coverage for {scenario}")
        if scenario in PU_SCENARIOS and not set(PU_MODELS).issubset(group.model):
            raise RuntimeError(f"Missing a native PU variant for {scenario}")
    external_cols=["external_"+metric for metric in METRICS]
    if any(col not in data for col in external_cols):
        raise RuntimeError("Independent shifted-cohort metrics are missing")
    if data.external_roc_auc.isna().any() or data.external_pr_auc.isna().any():
        raise RuntimeError("Independent cohort is missing AUC or average precision")
    pair=data[["scenario","seed"]].drop_duplicates()
    if len(pair)!=SCENARIOS*len(SEEDS) or set(data.seed)!=set(SEEDS):
        raise RuntimeError("Missing or duplicate scenario seeds")
    RESULTS.mkdir(parents=True,exist_ok=True)
    data.to_csv(RESULTS/"all_runs.csv",index=False)
    summary=[]
    for (scenario,model),g in data.groupby(["scenario","model"],sort=True):
        metric=g.selected_metric.iloc[0]
        v=scores_for(g,metric)
        valid=v[np.isfinite(v)]
        fresh=scores_for(g[g.seed.isin(NEW_SEEDS)],metric)
        fresh=fresh[np.isfinite(fresh)]
        extkey="external_"+("roc_auc" if metric=="auc" else metric)
        external_primary=g[extkey]
        row=dict(scenario=scenario,model=model,primary_metric=metric,
           external_primary_mean=external_primary.mean(),
           external_primary_sd=external_primary.std(ddof=1),
           primary_mean=valid.mean() if len(valid) else np.nan,
           primary_sd=valid.std(ddof=1) if len(valid)>1 else np.nan,
           new_seed_primary_mean=fresh.mean() if len(fresh) else np.nan,
           new_seed_primary_sd=fresh.std(ddof=1) if len(fresh)>1 else np.nan,
           n_seeds=len(g),n_success=int((g.status=="ok").sum()),
           mean_selection_seconds=g.selection_seconds.mean(),
           mean_final_fit_seconds=g.final_fit_seconds.mean(),
           mean_train_seconds=g.train_seconds.mean(),
           mean_predict_seconds=g.predict_seconds.mean(),
           mean_total_seconds=g.total_seconds.mean())
        for m in METRICS:
            row[m+"_mean"]=g[m].mean()
            row[m+"_sd"]=g[m].std()
            row["external_"+m+"_mean"]=g["external_"+m].mean()
            row["external_"+m+"_sd"]=g["external_"+m].std()
        summary.append(row)
    sm=pd.DataFrame(summary)
    sm["rank"]=np.nan
    sm["new_seed_rank"]=np.nan
    sm["external_rank"]=np.nan
    for scenario,g in sm.groupby("scenario"):
        increasing=g.primary_metric.iloc[0] in MINIMIZE
        sm.loc[g.index,"rank"]=g.primary_mean.rank(
            method="min",ascending=increasing,na_option="bottom")
        sm.loc[g.index,"new_seed_rank"]=g.new_seed_primary_mean.rank(
            method="min",ascending=increasing,na_option="bottom")
        sm.loc[g.index,"external_rank"]=g.external_primary_mean.rank(
            method="min",ascending=increasing,na_option="bottom")
    sm=sm.sort_values(["scenario","rank","model"])
    sm.to_csv(RESULTS/"summary.csv",index=False)
    # External shift cohort uses independent seeds and different proximity;
    # it never informs the hyperparameter or ensemble-weight selection.
    extfields=["scenario","model","primary_metric","primary_mean",
       "primary_sd","external_primary_mean","external_primary_sd",
       "external_rank","mean_selection_seconds","mean_final_fit_seconds",
       "mean_predict_seconds","mean_total_seconds"]
    sm[extfields].sort_values(["scenario","external_rank","model"]).to_csv(
       RESULTS/"external_validation.csv",index=False)
    multi_manifest=build_reports(data,sm,RESULTS)
    # Keep classical/undersampled forest baselines side by side.
    forest_models=[
        "RandomForest","RF_Bootstrap50","RF_Subsample50","RF_Subsample80",
        "RF_BalancedUndersampled","RF_BalancedUndersampledPrior",
        "ranger","NeuralForest_CART",
        "NeuralForest_SubSample50_Informed",
        "NeuralForest_UndersampledPlain",
        "NeuralForest_UndersampledInformed",
        "NeuralForest_Adaptive","NeuralForest_Calibrated"]
    comparisons=sm[sm.model.isin(forest_models)].copy()
    comparisons["is_undersampled"]=comparisons.model.str.contains(
        "Subsample|SubSample|Undersampled|Bootstrap50",case=False,regex=True)
    comparisons["relative_time_vs_classic_rf"]=np.nan
    comparisons["primary_gap_vs_classic_rf"]=np.nan
    comparisons["primary_advantage_vs_classic_rf"]=np.nan
    for scenario,g in comparisons.groupby("scenario"):
        rf=g[g.model=="RandomForest"]
        if len(rf)!=1: raise RuntimeError(f"Missing classic RF for {scenario}")
        root=rf.iloc[0]
        same=comparisons.scenario==scenario
        base_time=float(root.mean_total_seconds)
        comparisons.loc[same,"relative_time_vs_classic_rf"]=(
            comparisons.loc[same,"mean_total_seconds"]/base_time)
        gap=comparisons.loc[same,"primary_mean"]-float(root.primary_mean)
        comparisons.loc[same,"primary_gap_vs_classic_rf"]=gap
        comparisons.loc[same,"primary_advantage_vs_classic_rf"]=(
            -gap if root.primary_metric in MINIMIZE else gap)
    comparisons.to_csv(RESULTS/"sampling_comparison.csv",index=False)
    # Generated from the same score table, never a manually copied timing report.
    t=sm.groupby("model",as_index=False).agg(
        scenarios=("scenario","nunique"),
        selection=("mean_selection_seconds","mean"),
        final_fit=("mean_final_fit_seconds","mean"),
        prediction=("mean_predict_seconds","mean"),
        total=("mean_total_seconds","mean"))
    t=t.sort_values(["total","model"])
    timing=[
        "# Computation time: NeuralForest and forest baselines",
        "",
        "Observed wall-clock seconds averaged across 19 scenarios and ten seeds.",
        "Numerical libraries were restricted to one CPU thread. Different runners",
        "may have distinct performance. Times do not include data generation",
        "or dependency installation. Internal search budgets differ by estimator.",
        "",
        "| Algorithm | Scenarios | Selection (s) | Final fit (s) | Prediction both cohorts (s) | Total (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in t.itertuples():
        timing.append(f"| {item.model} | {item.scenarios} | {fmt(item.selection)} | "
          f"{fmt(item.final_fit)} | {fmt(item.prediction,4)} | {fmt(item.total)} |")
    timing.extend(["",
      "## Comparisons by scenario",
      "",
      "A full per-scenario table of the standard RF, 50% and 80% row",
      "subsampling, class-balanced under-sampling, ranger and informed",
      "NeuralForest variants is in sampling_comparison.csv. Both the",
      "primary objective gap and the total-time ratio versus RandomForest",
      "are stored. Positive primary_advantage_vs_classic_rf is favorable.",
      "",
      f"All {expected:,} per-seed score and timing records appear in all_runs.csv;",
      "the cross-seed means and standard deviations are in summary.csv.",
      "Prediction time covers BOTH independent true-label test cohorts.",
      "PU methods are evaluated on three presence/background scenarios",
      "only. Their mean times are not directly comparable with the",
      "19-scenario averages of other methods.",
      "The training budget and the initial predictive objective are",
      "not perfectly matched across algorithm families."])
    (RESULTS/"COMPUTE_TIME.md").write_text("\n".join(timing)+"\n",encoding="utf8")
    # Historical comparisons are based only on seeds 17 and 43.
    compared=None
    if BASELINE.is_file():
        old=pd.read_csv(BASELINE)
        old=old[old.seed.isin((17,43))]
        new=data[data.seed.isin((17,43))]
        columns=["scenario","seed","model","primary_metric","baseline_value",
                 "updated_value","delta_improvement","baseline_train_seconds",
                 "updated_train_seconds","baseline_predict_seconds",
                 "updated_predict_seconds"]
        records=[]
        oldidx={(r.scenario,int(r.seed),r.model):r for r in old.itertuples()}
        for row in new.itertuples():
            prior=oldidx.get((row.scenario,int(row.seed),row.model))
            if prior is None:continue
            target=row.selected_metric
            oldcol="roc_auc" if target=="auc" else target
            oldscore=float(getattr(prior,oldcol))
            newscore=float(getattr(row,oldcol))
            sign=-1 if target in MINIMIZE else 1
            records.append(dict(zip(columns,[
              row.scenario,int(row.seed),row.model,target,
              oldscore,newscore,sign*(newscore-oldscore),
              float(prior.train_seconds),float(row.train_seconds),
              float(prior.predict_seconds),float(row.predict_seconds)])))
        compared=pd.DataFrame(records,columns=columns)
        compared.to_csv(RESULTS/"paired_v02_comparison.csv",index=False)
    manifest=dict(
        generator="QuentinLamboley/td-machinelearning",
        generator_revision="48c2b28ca7874cbef1f7dde15de962c6e56c5027",
        seeds=list(SEEDS),new_confirmation_seeds=list(NEW_SEEDS),
        scenarios=sorted(data.scenario.unique().tolist()),
        models=sorted(data.model.unique().tolist()),
        rows=len(data),independent_model_fits=independent_fits,
        successes=int((data.status=="ok").sum()),
        failures=data.loc[data.status!="ok",[
            "scenario","seed","model","error"]].fillna("").to_dict("records"),
        paired_old_rows=0 if compared is None else len(compared),
        multi_metric_report=multi_manifest,
        external_evaluation=dict(type="synthetic shifted generator",
            source="independent same simulator, not real external data",
            external_seed_offset=300003,
            selected_using_external_labels=False,
            metrics=external_cols),
        python_version=sys.version,platform=platform.platform())
    (RESULTS/"manifest.json").write_text(json.dumps(
        manifest,indent=2,ensure_ascii=False),encoding="utf8")
    lines=[
        "# Vampire benchmark: audited NeuralForest",
        "",
        "All times are measured wall-clock values on GitHub-hosted CPU",
        "runners with mathematical libraries constrained to one thread.",
        "Selection time includes internal hyperparameter and expert search.",
        "Final fit time measures the separately trained final estimator.",
        "Prediction time includes evaluation inference on the held-out set.",
        "The three stages are also available independently in summary.csv.",
        "",
        "The experiment covers 19 scenarios, ten seeds and 31 standard methods",
        "plus five PU-specific predictions in three presence/background scenarios.",
        "The matched two-restart teacher shares the fit and selection cost",
        "of PUStable: it is not a separate independently trained algorithm.",
        f"Successful prediction records: {manifest['successes']} / {len(data)}.",
        f"Independent model fits: {independent_fits}; the additional",
        "PUStableTeacher prediction shares its fitted model with PUStable.",
        "Seeds 401 and 503 remain confirmation seeds from the v0.6 audit.",
        "A second shifted simulator population is evaluated independently",
        "without using any of its labels during fitting or model selection.",
        "This is synthetic transfer validation, not real-world external validation.",
        "",
        "## Overall ranking for each scenario",
        "",
        "| Scenario | Metric | Top three (ten-seed mean) | NeuralForest adaptatif | Rank | Selection (s) | Final fit (s) | Prediction (s) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for name,g in sm.groupby("scenario"):
        top=g.loc[np.isfinite(g.primary_mean)].sort_values("rank").head(3)
        winners=", ".join(f"{r.model} {fmt(r.primary_mean)}"
                          for r in top.itertuples())
        nf=g[g.model=="NeuralForest_Adaptive"]
        if len(nf):
            row=nf.iloc[0]
            values=(fmt(row.primary_mean),str(int(row["rank"])),
                    fmt(row.mean_selection_seconds,2),
                    fmt(row.mean_final_fit_seconds,2),
                    fmt(row.mean_predict_seconds,3))
        else:
            values=("NA","NA","NA","NA","NA")
        lines.append(f"| {name} | {g.primary_metric.iloc[0]} | {winners} | "
                     f"{values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The validation set selects algorithms, model complexity and mixtures",
        "without consulting independent test outcomes. Non-differentiable",
        "metrics use a differentiable proxy in soft trees. Average precision",
        "uses a smooth listwise ranking surrogate, while AUC and Boyce use",
        "a pairwise logistic surrogate. Test performance always uses the",
        "original metric, not its surrogate.",
        "",
        "NeuralForest_Adaptive and NeuralForest_Calibrated include ridge-regularized linear and constant",
        "experts. Its performance must be interpreted as a hybrid ensemble,",
        "not attributed exclusively to learned tree divisions.",
        "",
        "Historical seed-matched v0.2 comparisons are in",
        "paired_v02_comparison.csv. Means over seeds 401 and 503 appear in",
        "new_seed_primary_mean and new_seed_rank in summary.csv.",
        "",
        "Presence-background training negatives are not confirmed human",
        "labels. The independent test population has true synthetic labels,",
        "which changes the interpretation of reported metrics.",
        "",
        "The identical-distribution negative control should give chance",
        "discrimination; a constant predictor often minimizes log loss.",
        "These synthetic datasets do not establish universal superiority.",
        "For 15 metric-specific comparisons and voting/stacking ensembles,",
        "consult METRICS_REPORT.md. The extra metrics are evaluated only",
        "on held-out test sets and were not used to select candidates.",
        "",
        "## Result files",
        "",
        "- all_runs.csv: all metrics, configurations and timed stages",
        "- balanced under-sampling uses equal minority/majority per tree",
        "  without replacement. Each tree uses a training-only class count.",
        "- summary.csv: mean, standard deviation, ranking and phase timings",
        "- paired_v02_comparison.csv: paired original-seed differences",
        "- paired_v03_comparison.csv: 4-seed paired v0.3 versus v0.4 comparison",
        "- paired_v041_comparison.csv: six matched seeds against NeuralForest v0.4.1",
        "- paired_v05_comparison.csv: eight matched seeds against v0.5",
        "- METRICS_REPORT.md: rankings across fifteen independent test metrics",
        "- external_validation.csv: out-of-distribution shifted cohort scores",
        "- EXTERNAL_VALIDATION.md: shifted-cohort method comparisons",
        "- PU_STABILITY.md: cross-fitted uncertainty-gating decisions",
        "- pu_stability_choices.csv: every selected sampling strategy",
        "- pu_safe_choices.csv: fidelity-gated candidate decisions",
        "- paired_v07_comparison.csv: same-seed reference against v0.7",
        "- all_metric_ranks.csv, metric_winners.csv: all metric rankings",
        "- ensemble_comparison.csv: voted and stacked ensemble comparison",
        "- paired_multi_metrics.csv: paired differences and 95% intervals",
        "- average_metric_ranks.csv: rank summaries across scenario families",
        "- sampling_comparison.csv: forest-by-forest subsampling comparison",
        "- COMPUTE_TIME.md: selection, final fit and prediction time by algorithm",
        "- manifest.json: provenance, seeds, failures and environment",
        "- benchmark/README.md: full experimental protocol and limitations"
    ])
    external_report=[
        "# Independent shifted-cohort validation: synthetic transport",
        "",
        "Every fitted model also predicted a second **independent** cohort",
        "from the same pinned synthetic generator with a prespecified",
        "proximity change (usually 12 percentage points). Training,",
        "threshold selection, prior assumptions and hyperparameter search",
        "used neither this cohort nor its true labels. This is NOT",
        "validation on an independently collected real-world dataset.",
        "",
        f"The main analysis includes {expected} successful prediction records and",
        "records all 16 metrics on both independent test populations.",
        "",
        "| Scenario | Metric | Best shifted-cohort model | Shifted score | NeuralForest Adaptive |",
        "| --- | --- | --- | ---: | ---: |"]
    for scenario,g in sm.groupby("scenario"):
        q=g[np.isfinite(g.external_primary_mean)].sort_values("external_rank")
        top=q.iloc[0]
        adaptive=q[q.model=="NeuralForest_Adaptive"].iloc[0]
        external_report.append(
            f"| {scenario} | {top.primary_metric} | {top.model} | "
            f"{fmt(top.external_primary_mean,4)} | "
            f"{fmt(adaptive.external_primary_mean,4)} |")
    external_report.extend(["",
      "## Positive-unlabeled variants",
      "",
      "These methods are run only in the three presence/background",
      "scenarios; they must not be credited with winning the 19-scenario",
      "overall rank or compared by overall mean compute time.",
      "",
      "| Scenario | Model | Main test metric | Shifted-cohort metric | Mean total time (s) |",
      "| --- | --- | ---: | ---: | ---: |"])
    for scenario in PU_SCENARIOS:
        g=sm[(sm.scenario==scenario)&sm.model.isin(
            ["NeuralForest_PUNeural","NeuralForest_PUInformed",
             "NeuralForest_PUStable","NeuralForest_PUStableTeacher",
             "NeuralForest_PUSafe","NeuralForest_Adaptive",
             "Logistic","Voting_Heterogeneous"])]
        for row in g.sort_values("model").itertuples():
            external_report.append(
              f"| {scenario} | {row.model} | {fmt(row.primary_mean,4)} | "
              f"{fmt(row.external_primary_mean,4)} | "
              f"{fmt(row.mean_total_seconds,3)} |")
    external_report.extend(["",
       "The native nnPU objective assumes representative observed-positive",
       "and unlabeled distributions and prespecified class-prior values.",
       "The hard-background selection intentionally violates the",
       "representative-unlabeled assumption. An apparent improvement on",
       "this simulated population does not validate those assumptions.",
       "Repeated inspection of these 19 scenario families has also",
       "reduced their value as confirmatory external evidence.",
       "",
       "Source measurements: all_runs.csv and external_validation.csv."])
    (RESULTS/"EXTERNAL_VALIDATION.md").write_text(
         "\n".join(external_report)+"\n",encoding="utf8")
    stability=data[data.model=="NeuralForest_PUStable"].copy()
    safe=data[data.model=="NeuralForest_PUSafe"].copy()
    if len(safe)!=len(PU_SCENARIOS)*len(SEEDS):
        raise RuntimeError("Missing safe PU model comparisons")
    stability["selected_strategy"]=stability.configuration.str.extract(
       r"(?:^|;)selected=([^;]+)",expand=False)
    stability["strength"]=pd.to_numeric(
       stability.configuration.str.extract(
         r"(?:^|;)guidance=([^;]+)",expand=False),errors="coerce")
    stability["ess_fraction"]=pd.to_numeric(
       stability.configuration.str.extract(
         r"(?:^|;)ess=([^;]+)",expand=False),errors="coerce")
    if (len(stability)!=len(PU_SCENARIOS)*len(SEEDS) or
            stability.selected_strategy.isna().any()):
        raise RuntimeError("Missing stability-gate selections")
    safe["selected_strategy"]=safe.configuration.str.extract(
       r"(?:^|;)selected=([^;]+)",expand=False)
    if safe.selected_strategy.isna().any():
        raise RuntimeError("Missing fidelity-safe PU selections")
    safe["strength"]=pd.to_numeric(safe.configuration.str.extract(
         r"(?:^|;)guidance=([^;]+)",expand=False),errors="coerce")
    safe[["scenario","seed","selected_strategy","strength",
          "boyce","external_boyce","roc_auc","external_roc_auc",
          "pr_auc","external_pr_auc","selection_seconds",
          "final_fit_seconds","predict_seconds","total_seconds"]].to_csv(
       RESULTS/"pu_safe_choices.csv",index=False)
    stability[["scenario","seed","selected_strategy",
               "strength","ess_fraction","boyce","external_boyce",
               "roc_auc","external_roc_auc",
               "selection_seconds","final_fit_seconds",
               "predict_seconds","total_seconds"]].to_csv(
        RESULTS/"pu_stability_choices.csv",index=False)
    safety=[
      "# Cross-fitted uncertainty-guided PU forests: stability audit","",
      "Three legacy PU-background scenarios are compared on ten seeds.",
      "Only the PU-specific variants see confirmed-positive versus",
      "unlabeled labels. The 31 comparator methods remain unchanged.",
      "The stability-gated model uses six native C++ nnPU networks",
      "(two initializations at each of three fixed class-prior assumptions).",
      "Its training-example guidance is based on disjoint 2-fold",
      "cross-fitted teacher predictions, never in-sample network outputs.",
      "C++ weighting preserves group mass and ensures unlabeled",
      "effective sample fraction at least 0.96. A candidate forest",
      "must improve PU risk for every predefined prior on both",
      "separate holdout halves by more than 0.004; otherwise it falls",
      "back to the neural-only predictor.",
      "",
      "| Scenario | Legacy nnPU Boyce | PU informed Boyce | Stable Boyce | Stable shifted Boyce | Stable selections |",
      "| --- | ---: | ---: | ---: | ---: | --- |"]
    for scenario in PU_SCENARIOS:
        g=sm[sm.scenario==scenario]
        nn=g[g.model=="NeuralForest_PUNeural"].iloc[0]
        inf=g[g.model=="NeuralForest_PUInformed"].iloc[0]
        stable=g[g.model=="NeuralForest_PUStable"].iloc[0]
        decisions=stability[stability.scenario==scenario]
        counts=decisions.selected_strategy.value_counts().to_dict()
        wording=", ".join(f"{name}: {count}" for name,count in
                          sorted(counts.items()))
        safety.append(
          f"| {scenario} | {fmt(nn.boyce_mean,4)} | "
          f"{fmt(inf.boyce_mean,4)} | {fmt(stable.boyce_mean,4)} | "
          f"{fmt(stable.external_boyce_mean,4)} | {wording} |")
    safety.extend(["","## What the gate can and cannot ensure","",
      "The gate protects ONLY the empirical, worst-case selected-PU",
      "surrogate on two training holdout halves. It does not",
      "guarantee non-inferiority on true outcome labels or on a",
      "feature-selected unrepresentative background (SAR/SNAR).",
      "The score can mis-rank candidate forests if PU priors or",
      "labeling assumptions are misspecified. The benchmark's",
      "independent shifted cohort also comes from the SAME simulator.",
      "Its metrics are test-only; no threshold or model selection",
      "is allowed to consult its labels.",
      "",
      "The 19 synthetic scenarios have been inspected repeatedly.",
      "Out-of-simulator public-data and SAR tests are required",
      "before claiming improvement beyond this simulator.",
      "The time to train *all* guidance candidates and all",
      "cross-fitted C++ neural ensembles is INCLUDED under",
      "selection_seconds; native final refit is separately timed.",
      "All 31 established forest, boosting, neural, voting and",
      "stacking comparators remain unchanged.",
      "",
      "See pu_stability_choices.csv and all_runs.csv for all outcomes."])
    safety += ["","## Fidelity-constrained native nnPU teacher comparator","",
       "The additional PUSafe model anchors its fallback to the exact",
       "original three-prior native C++ nnPU network. A forest can only",
       "be selected if BOTH holdout halves improve the worst-prior PU",
       "risk and maintain Spearman rank correlation >=0.995 with that",
       "teacher. Candidate forest probabilities are clipped on both",
       "validation and prediction to within 0.02 absolute probability",
       "of the teacher, preventing unbounded downstream drift.",
       "",
       "| Scenario | Original nnPU Boyce | Safe Boyce | Safe shifted Boyce | Safe selections |",
       "| --- | ---: | ---: | ---: | --- |"]
    for scenario in PU_SCENARIOS:
        g=sm[sm.scenario==scenario]
        teacher=g[g.model=="NeuralForest_PUNeural"].iloc[0]
        guarded=g[g.model=="NeuralForest_PUSafe"].iloc[0]
        choices=safe[safe.scenario==scenario].selected_strategy.value_counts()
        description=", ".join(f"{label}: {int(n)}" for label,n in
                               choices.sort_index().items())
        safety.append(
          f"| {scenario} | {fmt(teacher.boyce_mean,4)} | "
          f"{fmt(guarded.boyce_mean,4)} | "
          f"{fmt(guarded.external_boyce_mean,4)} | {description} |")
    safety += ["","The safe option is a new *exploratory* model, motivated by",
       "inspection of v0.8 test performance. The same old ten seeds",
       "cannot establish confirmatory generalization or universal",
       "superiority. The v0.9.1 C++ rank-safe projection strictly",
       "preserves nnPU score order, so PUSafe cannot improve or harm",
       "AUC/AP relative to the matched legacy nnPU model apart from",
       "numerical ties. Calibration and ordinary Boyce can still change.",
       "The unmodified v0.8 PUStable model is retained to make that",
       "failure visible. Detailed decisions: pu_safe_choices.csv."]
    (RESULTS/"PU_STABILITY.md").write_text(
        "\n".join(safety)+"\n",encoding="utf8")
    (RESULTS/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines),flush=True)
if __name__=="__main__":
    summarize()
