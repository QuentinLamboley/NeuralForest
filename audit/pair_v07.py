"""Paired v0.7 to v0.8 evaluation on identical simulator seeds."""
from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/"audit"/"baseline-v07-all_runs.csv"
NEW=ROOT/"benchmark"/"results"/"all_runs.csv"
OUT=ROOT/"benchmark"/"results"/"paired_v07_comparison.csv"
REPORT=ROOT/"benchmark"/"results"/"PAIRED_V07.md"
METRIC_MAP={"auc":"roc_auc"}
LOWER={"brier","logloss"}

def read(path):
    with path.open(newline="",encoding="utf8") as f:
        return list(csv.DictReader(f))

def value(x):
    try: return float(x)
    except (TypeError,ValueError): return float("nan")

def main():
    old=read(OLD)
    new=read(NEW)
    lookup={(r["scenario"],int(r["seed"]),r["model"]):r for r in old}
    rows=[]
    for current in new:
        key=(current["scenario"],int(current["seed"]),current["model"])
        historical=lookup.get(key)
        if historical is None: continue
        metric=current["selected_metric"]
        source=METRIC_MAP.get(metric,metric)
        a=value(historical[source])
        b=value(current[source])
        direction=-1 if metric in LOWER else 1
        rows.append({
            "scenario":key[0],
            "seed":key[1],
            "model":key[2],
            "selected_metric":metric,
            "baseline_v07":a,
            "updated_v08":b,
            "improvement":direction*(b-a),
            "baseline_selection_seconds":value(historical["selection_seconds"]),
            "updated_selection_seconds":value(current["selection_seconds"]),
            "baseline_fit_seconds":value(historical["final_fit_seconds"]),
            "updated_fit_seconds":value(current["final_fit_seconds"]),
            "baseline_predict_seconds":value(historical["predict_seconds"]),
            "updated_predict_seconds":value(current["predict_seconds"]),
            "baseline_total_seconds":value(historical["total_seconds"]),
            "updated_total_seconds":value(current["total_seconds"])
        })
    if len(rows)!=19*10*31+3*10*2:
        raise RuntimeError(f"Expected 5950 matched v0.7 observations, got {len(rows)}")
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # Includes all changes, even those that lower the held-out objective.
    nf=defaultdict(list)
    for item in rows:
        if item["model"].startswith("NeuralForest") and math.isfinite(item["improvement"]):
            nf[(item["scenario"],item["model"])].append(item["improvement"])
    lines=[
        "# NeuralForest v0.7 to v0.8 paired comparison",
        "",
        "These observations pair 19 simulator cases, ten original seeds",
        "and 31 common model variants. Both evaluations use untouched",
        "independently generated test samples; higher improvement is better.",
        "Changes in software version and CPU runner can affect timings.",
        "",
        "| Scenario | Model | Mean objective improvement |",
        "| --- | --- | ---: |"
    ]
    for (case,model),improvement in sorted(nf.items()):
        lines.append(f"| {case} | {model} | {sum(improvement)/len(improvement):+.5f} |")
    lines.extend(["","## Caveats","",
        "This is a matched-seed engineering evaluation, not an independent",
        "confirmatory trial: the old scenario families were already inspected.",
        "All ten original seeds are included. An independent shifted cohort",
        "is reported separately in EXTERNAL_VALIDATION.md.",
        "Version 0.8 adds one stable, OOF nnPU-guided experiment on the three",
        "presence/background cases, so it has no v0.7 counterpart.",
        "A positive comparison for a hybrid model cannot be attributed",
        "to one component without an explicit ablation.",
        "",
        "Times include algorithm-specific search and model fitting.",
        "The full per-run paired metric and timing records are in",
        "paired_v07_comparison.csv."])
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf8")
    print(f"Paired {len(rows)} v0.7 vs v0.8 observations")

if __name__=="__main__":
    main()
