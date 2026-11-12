"""Compare original v0.2 measurements with the same-seed audited run."""
from __future__ import annotations
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/"audit"/"baseline-v0.2-all_runs.csv"
NEW=ROOT/"benchmark"/"results"/"all_runs.csv"
OUTPUT=ROOT/"benchmark"/"results"/"paired_v02_comparison.csv"
METRICS=("roc_auc","pr_auc","accuracy","balanced_accuracy","precision",
         "recall","specificity","f1","mcc","brier","logloss","boyce")
LOWER={"brier","logloss"}
def read_rows(path):
    with path.open(newline="",encoding="utf8") as f:
        return list(csv.DictReader(f))
def compare():
    original=read_rows(OLD)
    updated=read_rows(NEW)
    baseline={(r["scenario"],int(r["seed"]),r["model"]):r
              for r in original}
    data=[]
    for row in updated:
        key=(row["scenario"],int(row["seed"]),row["model"])
        if key not in baseline or int(row["seed"]) not in (17,43):
            continue
        prior=baseline[key]
        metric=row["selected_metric"]
        metric_column="roc_auc" if metric=="auc" else metric
        value_old=float(prior[metric_column])
        value_new=float(row[metric_column])
        diff=value_new-value_old
        if metric in LOWER: diff=-diff
        record=dict(
            scenario=row["scenario"],seed=row["seed"],model=row["model"],
            primary_metric=metric,baseline_primary=value_old,
            audited_primary=value_new,
            primary_improvement=diff,
            baseline_train_seconds=prior["train_seconds"],
            audited_train_seconds=row["train_seconds"],
            baseline_predict_seconds=prior["predict_seconds"],
            audited_predict_seconds=row["predict_seconds"],
            audited_selection_seconds=row["selection_seconds"],
            audited_final_fit_seconds=row["final_fit_seconds"])
        for name in METRICS:
            record["baseline_"+name]=prior[name]
            record["audited_"+name]=row[name]
        data.append(record)
    expected=16*2*15
    if len(data)!=expected:
        raise RuntimeError(f"Expected {expected} matched runs, got {len(data)}")
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with OUTPUT.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    print(f"Published {len(data)} paired scenario-seed-model comparisons.")
if __name__=="__main__":
    compare()
