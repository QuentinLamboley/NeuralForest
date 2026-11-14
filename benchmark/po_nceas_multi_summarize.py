"""Complete six-region NCEAS survey audit, no favorable-species filtering.

All 6 predeclared regions must contribute a result/manifest, even
when none meet the training-only support criterion. All model variants
must be present for each eligible species. AUC/AP are computed
EXCLUSIVELY on real, independently surveyed presence/non-detection,
not on environmental background as if it were absent.

No use of surveyed outcomes for hyperparameter tuning or inclusion.
Means and contrasts are descriptive; species within a region share
sites and are not independent biological sampling experiments.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"benchmark"/"po_nceas_multi_out"
RESULTS=ROOT/"benchmark"/"po_nceas_multi_results"
REGIONS=("AWT","CAN","NSW","NZ","SA","SWI")
PRIMARY="NeuralForest_POGrad_hidden"
CONTRASTS=(
 "NeuralForest_POGrad_teacher",
 "NeuralForest_POGrad_raw",
 "NeuralForest_POGrad_random",
 "NeuralForest_PO_hidden",
 "NeuralForest_PO_random",
 "maxnet_lqph","GAM_smooth","ranger_RF",
 "RandomForest","BalancedSubsample_RF","XGBoost",
 "XGBoost_weighted","CatBoost","CatBoost_weighted")
METRICS=("survey_auc","survey_ap")

def fmt(value):
    return "NA" if not np.isfinite(value) else f"{value:.4f}"

def matched_effects(df,metric="survey_auc"):
    keys=["region","species"]
    original=df[df.model==PRIMARY].set_index(keys)
    output=[]
    for reference in CONTRASTS:
        control=df[df.model==reference].set_index(keys)
        if set(original.index)!=set(control.index):
            raise RuntimeError(f"Unpaired NCEAS species comparison: {reference}")
        common=original.sort_index()
        compare=control.reindex(common.index)
        for (region,species),a in common.iterrows():
            b=compare.loc[(region,species)]
            both=bool(a.evaluable) and bool(b.evaluable)
            x=float(a[metric]);y=float(b[metric])
            delta=x-y if both and np.isfinite(x) and np.isfinite(y) else np.nan
            output.append(dict(region=region,species=species,
                model=PRIMARY,reference=reference,metric=metric,
                primary=x,control=y,delta=delta,
                evaluable=bool(np.isfinite(delta))))
    return pd.DataFrame(output)

def main():
    for region in REGIONS:
        if not (SOURCE/f"{region}.csv").exists() or not (
            SOURCE/f"{region}.json").exists():
            raise RuntimeError(f"Region {region} missing - no cherry-picking")
    RESULTS.mkdir(parents=True,exist_ok=True)
    data=[]
    region_manifests={}
    for region in REGIONS:
        region_manifests[region]=json.loads((SOURCE/f"{region}.json").read_text())
        frame=pd.read_csv(SOURCE/f"{region}.csv")
        if len(frame):data.append(frame)
    if not data:raise RuntimeError("No model evaluations in any region")
    all_df=pd.concat(data,ignore_index=True)
    unique=["region","species","model"]
    if all_df[unique].duplicated().any():
        raise RuntimeError("Duplicated estimator for real species")
    n_methods=all_df.model.nunique()
    if n_methods!=20:
        raise RuntimeError(f"Unexpected complete model suite: {n_methods}")
    species=all_df[["region","species"]].drop_duplicates()
    for (region,s),group in all_df.groupby(["region","species"]):
        if group.model.nunique()!=20:
            raise RuntimeError(f"Missing model fit for {region}/{s}")
        expected=region_manifests[region]["methods"]
        if set(group.model)!=set(expected):
            raise RuntimeError(f"Method registry drift in {region}")
    # Undefined AUCs are NaN and STAY recorded, not converted to zero.
    all_df.to_csv(RESULTS/"all_species_models.csv",index=False)
    eval_df=all_df[all_df.evaluable==True].copy()
    reg_avgs=eval_df.groupby(["region","model"])[
        ["survey_auc","survey_ap","train_seconds",
         "predict_seconds"]].agg("mean").reset_index()
    reg_avgs.to_csv(RESULTS/"regional_model_means.csv",index=False)
    all_avgs=eval_df.groupby(["model"])[
       ["survey_auc","survey_ap","train_seconds","predict_seconds"]].mean(
          numeric_only=True).reset_index()
    all_avgs.to_csv(RESULTS/"pooled_species_means.csv",index=False)
    # Region-macro averages avoid ranking by whichever region happens
    # to have more species in the pilot.
    macro=reg_avgs.groupby(["model"])[
       ["survey_auc","survey_ap","train_seconds"]].mean(
          numeric_only=True).reset_index()
    macro.to_csv(RESULTS/"macro_region_means.csv",index=False)
    effects=pd.concat([matched_effects(all_df,m) for m in METRICS],
                       ignore_index=True)
    effects.to_csv(RESULTS/"paired_species_effects.csv",index=False)
    summary=[]
    for (ref,metric),g in effects.groupby(["reference","metric"],sort=True):
        defined=g.loc[g.evaluable]
        x=defined.delta.to_numpy(float)
        if not len(x):continue
        # Genuine reporting unit: region level means, NOT species/site
        # bootstrap pseudoreplication. Only 6 regions at most, no
        # confirmatory p values or unadjusted CI declarations.
        rg=defined.groupby("region")["delta"].mean()
        summary.append(dict(model=PRIMARY,reference=ref,metric=metric,
          n_evaluable_species=len(x),n_regions=len(rg),
          mean_species_delta=float(x.mean()),
          mean_region_delta=float(rg.mean()),
          region_sd=float(rg.std(ddof=1)) if len(rg)>1 else np.nan,
          species_wins=int(sum(x>1e-9)),
          species_losses=int(sum(x< -1e-9)),
          species_ties=int(sum(abs(x)<=1e-9)),
          region_wins=int(sum(rg>1e-9)),
          region_losses=int(sum(rg< -1e-9))))
    paired=pd.DataFrame(summary)
    paired.to_csv(RESULTS/"paired_effect_summary.csv",index=False)
    inner=all_df[all_df.model=="NeuralForest_POGrad_hidden"].groupby(
      ["region","inner_validation"]).size().reset_index(name="count")
    inner.to_csv(RESULTS/"gradient_inner_validation_audit.csv",index=False)
    lines=[
       "# NeuralForest presence-only: NCEAS six-region exploratory pilot","",
       "**These are separate-source surveyed species evaluations, not",
       "a preregistered independent publication or proof of superiority.**","",
       "Training uses opportunistic presence records plus independently",
       "sampled environmental availability, not invented true absences.",
       "The training-only x/y geography of opportunistic presences and",
       "background is used to hold out an inner band for selecting",
       "native IPP-gradient stage count, when such support is sufficient.",
       "A fallback to RANDOM inner holdout is explicitly flagged.",
       "",
       "Categorical predictors documented in disdat (ontveg, vegsys,",
       "soilfert, age, toxicats, calc) were one-hot coded with levels",
       "derived exclusively from train PO/background sources. Coordinates",
       "are never supplied as ecological predictors. Each region has a",
       "predeclared minimum presence count and up to four species by",
       "training PO frequency, not by independent survey results.",
       "",
       "Surveyed presence/non-detection outcomes are used to assess",
       "ROC-AUC and average precision ONLY after every model fitted",
       "and all prediction vectors were frozen. A survey non-detection",
       "is not proven ecological unoccupancy when detection is imperfect.",
       "",
       "## Coverage (six regions including those with zero eligible species)","",
       "| Region | Species above PO minimum | Evaluated species | Models run |",
       "| --- | ---: | ---: | ---: |"]
    for region in REGIONS:
        manifest=region_manifests[region]
        lines.append(f"| {region} | "
          f"{manifest['available_species_above_support']} | "
          f"{manifest['species_examined']} | "
          f"{manifest['prediction_records']} |")
    lines+=["","## Macro-regional average on evaluable species","",
      "Each represented region receives one equal weight, regardless",
      "of number of species. The species-weighted averages and all",
      "raw species-level outcomes are in the accompanying CSV files.","",
      "| Model | Survey ROC-AUC | Average precision | Mean fit seconds |",
      "| --- | ---: | ---: | ---: |"]
    for r in macro.sort_values("survey_auc",ascending=False).itertuples():
        lines.append(f"| {r.model} | {fmt(r.survey_auc)} | "
          f"{fmt(r.survey_ap)} | {r.train_seconds:.3f} |")
    lines+=["","## Primary candidate versus controls, paired on the SAME species","",
      "| Reference | Mean region AUC gain | Mean region AP gain | Species wins (AUC) |",
      "| --- | ---: | ---: | ---: |"]
    for ref in CONTRASTS:
        aa=paired[(paired.reference==ref)&(paired.metric=="survey_auc")]
        ap=paired[(paired.reference==ref)&(paired.metric=="survey_ap")]
        if len(aa)==1 and len(ap)==1:
            a=aa.iloc[0];b=ap.iloc[0]
            lines.append(f"| {ref} | {a.mean_region_delta:+.4f} | "
               f"{b.mean_region_delta:+.4f} | "
               f"{int(a.species_wins)}/{int(a.n_evaluable_species)} |")
    lines+=["","## Per-region primary and comparator survey AUC","",
      "| Region | Native IPP-gradient hidden | Matched teacher | MaxEnt LQPH | CatBoost weighted |",
      "| --- | ---: | ---: | ---: | ---: |"]
    for reg in REGIONS:
        rows=reg_avgs[reg_avgs.region==reg].set_index("model")
        def get(m):return fmt(float(rows.loc[m,"survey_auc"])) if m in rows.index else "NA"
        lines.append(f"| {reg} | {get(PRIMARY)} | "
                     f"{get('NeuralForest_POGrad_teacher')} | "
                     f"{get('maxnet_lqph')} | "
                     f"{get('CatBoost_weighted')} |")
    lines+=["","## Methodological caveats and next studies","",
      "- Species were selected by TRAINING occurrence count, so this",
      "  *pilot* overrepresents relatively well-observed species.",
      "- Test survey outcomes come from separate collection protocols;",
      "  locations can still overlap opportunistic occurrence regions.",
      "- Species sharing regional survey sites are NOT independent",
      "  observational units. Six regions are too few for robust",
      "  confirmatory region-level significance tests.",
      "- Native IPP-gradient stages are inner-selected, whereas",
      "  competitor hyperparameters are predeclared and fixed, not",
      "  matched nested tuning. Comparability of compute budgets is",
      "  not established.",
      "- Some categories are unseen at survey sites and are encoded",
      "  as all-zero training-known one-hot levels; categorical",
      "  transfer/extrapolation needs a dedicated sensitivity audit.",
      "- Maximum four species per region and 2,500 available points",
      "  are preliminary runtime budgets, not full 226-species data.",
      "- DeepMaxent is not yet included: a validated implementation",
      "  must be pinned and evaluated on the identical splits.",
      "- Because the pilot's outcomes have now been inspected, any",
      "  subsequent architecture change needs a NEW untouched",
      "  dataset/region/time confirmation. Do not claim universal best.",
      "- NeuralForest intensity outputs are NOT calibrated occupancy",
      "  probabilities or pathogen risk.",
      "",
      "All individual outcomes (including losses and undefined scores),",
      "inner spatial/random fallback frequencies, timings, paired",
      "contrasts and counts are available in the CSV artifacts."
    ]
    (RESULTS/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    manifest=dict(complete_regions=list(REGIONS),
      regions_with_evaluable_species=sorted(eval_df.region.unique().tolist()),
      n_species=int(len(species)),
      n_evaluable_species=int(eval_df[["region","species"]].drop_duplicates().shape[0]),
      n_model_variants=n_methods,n_prediction_records=int(len(all_df)),
      independently_fitted_models=int(len(all_df)-len(species)),
      research_status="exploratory_not_preregistered",
      geographic_transfer_guaranteed=False,
      training_background_is_not_verified_absence=True,
      separate_survey_test_nondetection_is_not_perfect_occupancy_truth=True,
      all_six_regions_required_to_report=True,
      github_sha=os.getenv("GITHUB_SHA","local"))
    (RESULTS/"manifest.json").write_text(
                  json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"NCEAS six-region pilot: {manifest['n_species']} species, "
          f"{manifest['n_prediction_records']} model predictions",
          flush=True)

if __name__=="__main__":
    main()
