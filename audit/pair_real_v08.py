"""Matched v0.8 vs v0.9 real-record SCAR and SAR comparison."""
from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE/"baseline-v08-real-pu-all_runs.csv"
NEW=HERE/"real_pu_results"/"all_runs.csv"
OUT=HERE/"real_pu_results"/"PAIRED_V08.md"
RAW=HERE/"real_pu_results"/"paired_v08_scar_sar.csv"
METRICS=("roc_auc","pr_auc","brier","logloss",
         "external_roc_auc","external_pr_auc",
         "external_brier","external_logloss")
LOWER={"brier","logloss","external_brier","external_logloss"}
def read(p):
    with p.open(newline="",encoding="utf8") as f:
        return list(csv.DictReader(f))
def number(r,key):
    try:return float(r[key])
    except (KeyError,TypeError,ValueError):return float("nan")
def main():
    old=read(OLD)
    new=read(NEW)
    old_index={(r["dataset"],r["regime"],int(r["seed"]),r["method"]):r for r in old}
    rows=[]
    for new_row in new:
        key=(new_row["dataset"],new_row["regime"],int(new_row["seed"]),new_row["method"])
        baseline=old_index.get(key)
        if baseline is None: continue
        r={"dataset":key[0],"regime":key[1],"seed":key[2],"model":key[3]}
        for metric in METRICS:
            direction=-1 if metric in LOWER else 1
            r["improvement_"+metric]=direction*(
                number(new_row,metric)-number(baseline,metric))
        r["selection_delta_seconds"]=number(
            new_row,"selection_seconds")-number(baseline,"selection_seconds")
        r["fit_delta_seconds"]=number(
            new_row,"final_fit_seconds")-number(baseline,"final_fit_seconds")
        r["inference_delta_seconds"]=number(
            new_row,"predict_seconds")-number(baseline,"predict_seconds")
        rows.append(r)
    if len(rows)!=3*3*34*2:
        raise RuntimeError(
            f"Expected 612 old/new SCAR/SAR pairs; found {len(rows)}")
    with RAW.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    vals=defaultdict(list)
    for r in rows:
        if r["model"].startswith("NeuralForest"):
            vals[r["model"]].append(r["improvement_roc_auc"])
    lines=["# Independent real-record v0.8 versus v0.9 paired audit","",
     "Three public datasets x three seeds x two regimes x 34 available",
     "methods produce 612 matched comparisons. No test outcome",
     "labels were used for model selection. The",
     "new fidelity-constrained method is excluded from historical pairs",
     "because it has no old counterpart.",
     "",
     "| Previously available method | Mean AUC change |",
     "| --- | ---: |"]
    for name,values in sorted(vals.items()):
        good=[v for v in values if math.isfinite(v)]
        if good:
            lines.append(f"| {name} | {sum(good)/len(good):+.6f} |")
    lines += ["","Timings may vary between GitHub-hosted machines.",
      "Neither repeatedly inspected SCAR datasets nor in-simulator",
      "transfer cohorts constitute independent field validation.",
      "Detailed paired metrics/timings: paired_v08_scar_sar.csv."]
    OUT.write_text("\n".join(lines)+"\n",encoding="utf8")
if __name__=="__main__": main()
