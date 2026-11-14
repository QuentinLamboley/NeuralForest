# Six-region NCEAS numerical convergence policy

The first development workflow 43649536955 showed a native `mgcv::gam`
REML convergence failure on NZ/nz36: "inner loop 3; can't correct step size".
All other models on that species were fitted, so dropping the species
would produce an incomplete favorable-subset comparison.

Frozen engineering policy for the full rerun:

1. Try the same ecological GAM formula with REML.
2. On a TRAINING-only numerical error, retry the identical formula
   with ML and outer BFGS optimization, without using survey scores.
3. If both optimizers fail, keep an explicit GAM_smooth row with
   `fit_failed=true`, missing ROC-AUC and AP, complete error details
   and training time; do NOT invent predictions or substitute GLM.
4. Record the optimizer route for successful fallback cases.
5. Use matched species only for paired outcome differences.

The protocol is preliminary and post-pilot; the test results have
been inspected. No post hoc retry justifies confirmatory evidence.
