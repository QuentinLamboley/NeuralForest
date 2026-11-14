# Presence-only NeuralForest: spatial benchmark (EXPLORATORY)

The observed-positive records are not labels of the whole landscape.
Background locations describe environmental availability, NOT true
negatives. Therefore AUC_pb is presence-background discrimination,
not occupancy accuracy, and proper binary calibration scores
must not be computed against background-as-absence labels.

Three virtual species with known relative habitat intensity are
sampled under unbiased and geographically biased detection effort.
Both spatially uniform and effort-matched target-group background
selection are studied. Strict geographical block holdouts are used.
Oracle suitability scores are used ONLY AFTER model predictions
have been frozen, never to train the learned representation.

No algorithm selection or hyperparameter tuning uses test outcomes.
Methods include native C++ IPP neural, raw/learned/random-tanh forests,
MaxEnt/maxnet (LQ and LQPH), GAM, GLM, ranger RF, ordinary and balanced
RF, ExtraTrees, XGBoost, CatBoost and a fixed voting ensemble.

## All methods and scenarios: spatially held-out means

| Scenario | Observation effort | Background policy | Model | AUC_pb | Boyce | Oracle suitability rho | Oracle IPP log-skill | Train seconds |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| linear | effort_biased | effort_matched | BalancedSubsample_RF | 0.5815 | 0.5427 | 0.4602 | 0.0279 | 0.495 |
| linear | effort_biased | effort_matched | CatBoost | 0.6054 | 0.6762 | 0.6544 | 0.0288 | 0.192 |
| linear | effort_biased | effort_matched | Ensemble_RF_XGBoost_CatBoost | 0.5980 | 0.5566 | 0.5941 | 0.0092 | 0.624 |
| linear | effort_biased | effort_matched | ExtraTrees | 0.6245 | 0.7714 | 0.7053 | 0.0533 | 0.191 |
| linear | effort_biased | effort_matched | GAM_smooth | 0.6480 | 0.8114 | 0.8941 | 0.0844 | 0.220 |
| linear | effort_biased | effort_matched | GLM_logistic | 0.6492 | 0.8149 | 0.8971 | 0.0849 | 0.005 |
| linear | effort_biased | effort_matched | NeuralForest_PO_hidden | 0.6378 | 0.6869 | 0.7970 | 0.0550 | 0.172 |
| linear | effort_biased | effort_matched | NeuralForest_PO_neural | 0.6428 | 0.7609 | 0.8403 | 0.0675 | 0.078 |
| linear | effort_biased | effort_matched | NeuralForest_PO_random | 0.6299 | 0.6890 | 0.7727 | 0.0550 | 0.174 |
| linear | effort_biased | effort_matched | NeuralForest_PO_raw | 0.6332 | 0.6723 | 0.7841 | 0.0541 | 0.153 |
| linear | effort_biased | effort_matched | RandomForest | 0.5823 | 0.4828 | 0.5662 | 0.0157 | 0.381 |
| linear | effort_biased | effort_matched | XGBoost | 0.5914 | 0.6464 | 0.5122 | -0.0889 | 0.050 |
| linear | effort_biased | effort_matched | maxnet_lq | 0.6508 | 0.8035 | 0.8971 | 0.0747 | 1.960 |
| linear | effort_biased | effort_matched | maxnet_lqph | 0.6407 | 0.7699 | 0.8510 | 0.0707 | 1.624 |
| linear | effort_biased | effort_matched | ranger_RF | 0.5835 | 0.6026 | 0.4970 | -0.0481 | 0.260 |
| linear | effort_biased | uniform | BalancedSubsample_RF | 0.5732 | 0.4685 | 0.4249 | -0.0017 | 0.487 |
| linear | effort_biased | uniform | CatBoost | 0.6122 | 0.5887 | 0.6723 | 0.0186 | 0.126 |
| linear | effort_biased | uniform | Ensemble_RF_XGBoost_CatBoost | 0.6051 | 0.6631 | 0.6257 | 0.0034 | 0.553 |
| linear | effort_biased | uniform | ExtraTrees | 0.6242 | 0.6449 | 0.7344 | 0.0406 | 0.189 |
| linear | effort_biased | uniform | GAM_smooth | 0.6579 | 0.8295 | 0.9374 | 0.0611 | 0.244 |
| linear | effort_biased | uniform | GLM_logistic | 0.6583 | 0.8169 | 0.9366 | 0.0621 | 0.006 |
| linear | effort_biased | uniform | NeuralForest_PO_hidden | 0.6433 | 0.7188 | 0.8143 | 0.0437 | 0.178 |
| linear | effort_biased | uniform | NeuralForest_PO_neural | 0.6498 | 0.7665 | 0.8706 | 0.0458 | 0.078 |
| linear | effort_biased | uniform | NeuralForest_PO_random | 0.6400 | 0.7716 | 0.8192 | 0.0425 | 0.176 |
| linear | effort_biased | uniform | NeuralForest_PO_raw | 0.6273 | 0.7406 | 0.7934 | 0.0410 | 0.157 |
| linear | effort_biased | uniform | RandomForest | 0.5812 | 0.6316 | 0.5561 | -0.0017 | 0.375 |
| linear | effort_biased | uniform | XGBoost | 0.6006 | 0.7045 | 0.5696 | -0.0725 | 0.052 |
| linear | effort_biased | uniform | maxnet_lq | 0.6533 | 0.8447 | 0.9124 | 0.0789 | 1.896 |
| linear | effort_biased | uniform | maxnet_lqph | 0.6403 | 0.7240 | 0.7852 | 0.0608 | 1.610 |
| linear | effort_biased | uniform | ranger_RF | 0.5579 | 0.6006 | 0.3989 | -0.0954 | 0.236 |
| linear | unbiased | effort_matched | BalancedSubsample_RF | 0.5457 | 0.4122 | 0.3928 | 0.0343 | 0.490 |
| linear | unbiased | effort_matched | CatBoost | 0.5656 | 0.4096 | 0.5934 | 0.0577 | 0.153 |
| linear | unbiased | effort_matched | Ensemble_RF_XGBoost_CatBoost | 0.5520 | 0.2590 | 0.5102 | 0.0395 | 0.585 |
| linear | unbiased | effort_matched | ExtraTrees | 0.5662 | 0.4341 | 0.6051 | 0.0643 | 0.188 |
| linear | unbiased | effort_matched | GAM_smooth | 0.5782 | 0.5587 | 0.7713 | 0.0845 | 0.230 |
| linear | unbiased | effort_matched | GLM_logistic | 0.5814 | 0.6064 | 0.8028 | 0.0838 | 0.020 |
| linear | unbiased | effort_matched | NeuralForest_PO_hidden | 0.5693 | 0.4921 | 0.7187 | 0.0660 | 0.176 |
| linear | unbiased | effort_matched | NeuralForest_PO_neural | 0.5712 | 0.4813 | 0.6719 | 0.0701 | 0.080 |
| linear | unbiased | effort_matched | NeuralForest_PO_random | 0.5674 | 0.4649 | 0.7262 | 0.0658 | 0.176 |
| linear | unbiased | effort_matched | NeuralForest_PO_raw | 0.5641 | 0.4639 | 0.6839 | 0.0650 | 0.160 |
| linear | unbiased | effort_matched | RandomForest | 0.5479 | 0.3501 | 0.5221 | 0.0367 | 0.381 |
| linear | unbiased | effort_matched | XGBoost | 0.5436 | 0.3981 | 0.4181 | -0.0307 | 0.050 |
| linear | unbiased | effort_matched | maxnet_lq | 0.5807 | 0.6801 | 0.8091 | 0.0740 | 1.821 |
| linear | unbiased | effort_matched | maxnet_lqph | 0.5778 | 0.5930 | 0.7988 | 0.0790 | 1.589 |
| linear | unbiased | effort_matched | ranger_RF | 0.5406 | 0.4582 | 0.3741 | -0.0423 | 0.257 |
| linear | unbiased | uniform | BalancedSubsample_RF | 0.5457 | 0.4122 | 0.3928 | 0.0343 | 0.491 |
| linear | unbiased | uniform | CatBoost | 0.5656 | 0.4096 | 0.5934 | 0.0577 | 0.134 |
| linear | unbiased | uniform | Ensemble_RF_XGBoost_CatBoost | 0.5520 | 0.2590 | 0.5102 | 0.0395 | 0.566 |
| linear | unbiased | uniform | ExtraTrees | 0.5662 | 0.4341 | 0.6051 | 0.0643 | 0.189 |
| linear | unbiased | uniform | GAM_smooth | 0.5782 | 0.5587 | 0.7713 | 0.0845 | 0.232 |
| linear | unbiased | uniform | GLM_logistic | 0.5814 | 0.6064 | 0.8028 | 0.0838 | 0.018 |
| linear | unbiased | uniform | NeuralForest_PO_hidden | 0.5693 | 0.4921 | 0.7187 | 0.0660 | 0.175 |
| linear | unbiased | uniform | NeuralForest_PO_neural | 0.5712 | 0.4813 | 0.6719 | 0.0701 | 0.079 |
| linear | unbiased | uniform | NeuralForest_PO_random | 0.5674 | 0.4649 | 0.7262 | 0.0658 | 0.173 |
| linear | unbiased | uniform | NeuralForest_PO_raw | 0.5641 | 0.4639 | 0.6839 | 0.0650 | 0.157 |
| linear | unbiased | uniform | RandomForest | 0.5479 | 0.3501 | 0.5221 | 0.0367 | 0.381 |
| linear | unbiased | uniform | XGBoost | 0.5436 | 0.3981 | 0.4181 | -0.0307 | 0.051 |
| linear | unbiased | uniform | maxnet_lq | 0.5807 | 0.6801 | 0.8091 | 0.0740 | 1.884 |
| linear | unbiased | uniform | maxnet_lqph | 0.5778 | 0.5930 | 0.7988 | 0.0790 | 1.583 |
| linear | unbiased | uniform | ranger_RF | 0.5406 | 0.4582 | 0.3741 | -0.0423 | 0.258 |
| multimodal | effort_biased | effort_matched | BalancedSubsample_RF | 0.6219 | 0.7847 | 0.6198 | 0.1648 | 0.488 |
| multimodal | effort_biased | effort_matched | CatBoost | 0.6316 | 0.7990 | 0.7109 | 0.2170 | 0.109 |
| multimodal | effort_biased | effort_matched | Ensemble_RF_XGBoost_CatBoost | 0.6209 | 0.8072 | 0.6425 | 0.1796 | 0.539 |
| multimodal | effort_biased | effort_matched | ExtraTrees | 0.6485 | 0.7956 | 0.7915 | 0.2310 | 0.192 |
| multimodal | effort_biased | effort_matched | GAM_smooth | 0.6430 | 0.7142 | 0.7652 | 0.2039 | 0.348 |
| multimodal | effort_biased | effort_matched | GLM_logistic | 0.6396 | 0.6816 | 0.8080 | 0.1677 | 0.006 |
| multimodal | effort_biased | effort_matched | NeuralForest_PO_hidden | 0.6360 | 0.7525 | 0.7353 | 0.1688 | 0.178 |
| multimodal | effort_biased | effort_matched | NeuralForest_PO_neural | 0.6341 | 0.6929 | 0.7362 | 0.1775 | 0.079 |
| multimodal | effort_biased | effort_matched | NeuralForest_PO_random | 0.6348 | 0.7888 | 0.7209 | 0.1721 | 0.178 |
| multimodal | effort_biased | effort_matched | NeuralForest_PO_raw | 0.6350 | 0.7536 | 0.7351 | 0.1777 | 0.159 |
| multimodal | effort_biased | effort_matched | RandomForest | 0.6115 | 0.8226 | 0.5804 | 0.1563 | 0.379 |
| multimodal | effort_biased | effort_matched | XGBoost | 0.6118 | 0.7015 | 0.5933 | 0.0984 | 0.051 |
| multimodal | effort_biased | effort_matched | maxnet_lq | 0.6480 | 0.7893 | 0.7538 | 0.1965 | 1.836 |
| multimodal | effort_biased | effort_matched | maxnet_lqph | 0.6269 | 0.7197 | 0.7107 | 0.1759 | 1.569 |
| multimodal | effort_biased | effort_matched | ranger_RF | 0.5794 | 0.6729 | 0.3505 | 0.0414 | 0.238 |
| multimodal | effort_biased | uniform | BalancedSubsample_RF | 0.6114 | 0.7254 | 0.5026 | 0.1556 | 0.489 |
| multimodal | effort_biased | uniform | CatBoost | 0.6168 | 0.8095 | 0.5690 | 0.1788 | 0.109 |
| multimodal | effort_biased | uniform | Ensemble_RF_XGBoost_CatBoost | 0.6094 | 0.8167 | 0.5310 | 0.1578 | 0.539 |
| multimodal | effort_biased | uniform | ExtraTrees | 0.6343 | 0.8304 | 0.6341 | 0.1926 | 0.189 |
| multimodal | effort_biased | uniform | GAM_smooth | 0.6354 | 0.6945 | 0.6446 | 0.1809 | 0.353 |
| multimodal | effort_biased | uniform | GLM_logistic | 0.6343 | 0.6569 | 0.7350 | 0.1482 | 0.005 |
| multimodal | effort_biased | uniform | NeuralForest_PO_hidden | 0.6248 | 0.6662 | 0.6103 | 0.1454 | 0.179 |
| multimodal | effort_biased | uniform | NeuralForest_PO_neural | 0.6374 | 0.6665 | 0.6951 | 0.1680 | 0.080 |
| multimodal | effort_biased | uniform | NeuralForest_PO_random | 0.6268 | 0.7376 | 0.6284 | 0.1551 | 0.178 |
| multimodal | effort_biased | uniform | NeuralForest_PO_raw | 0.6266 | 0.7455 | 0.6375 | 0.1649 | 0.160 |
| multimodal | effort_biased | uniform | RandomForest | 0.6038 | 0.7638 | 0.4832 | 0.1516 | 0.379 |
| multimodal | effort_biased | uniform | XGBoost | 0.5980 | 0.6474 | 0.4784 | 0.0748 | 0.051 |
| multimodal | effort_biased | uniform | maxnet_lq | 0.6384 | 0.7769 | 0.6177 | 0.1730 | 1.833 |
| multimodal | effort_biased | uniform | maxnet_lqph | 0.6116 | 0.6894 | 0.5890 | 0.1495 | 1.575 |
| multimodal | effort_biased | uniform | ranger_RF | 0.5610 | 0.6723 | 0.2350 | 0.0197 | 0.241 |
| multimodal | unbiased | effort_matched | BalancedSubsample_RF | 0.6505 | 0.8283 | 0.5982 | 0.2011 | 0.489 |
| multimodal | unbiased | effort_matched | CatBoost | 0.6593 | 0.8037 | 0.6728 | 0.2478 | 0.108 |
| multimodal | unbiased | effort_matched | Ensemble_RF_XGBoost_CatBoost | 0.6522 | 0.7755 | 0.5988 | 0.2127 | 0.538 |
| multimodal | unbiased | effort_matched | ExtraTrees | 0.6635 | 0.8997 | 0.7465 | 0.2389 | 0.189 |
| multimodal | unbiased | effort_matched | GAM_smooth | 0.6515 | 0.6814 | 0.7141 | 0.2165 | 0.290 |
| multimodal | unbiased | effort_matched | GLM_logistic | 0.6431 | 0.7505 | 0.8244 | 0.1673 | 0.005 |
| multimodal | unbiased | effort_matched | NeuralForest_PO_hidden | 0.6541 | 0.6974 | 0.7289 | 0.1972 | 0.180 |
| multimodal | unbiased | effort_matched | NeuralForest_PO_neural | 0.6511 | 0.7409 | 0.7652 | 0.2044 | 0.079 |
| multimodal | unbiased | effort_matched | NeuralForest_PO_random | 0.6536 | 0.7748 | 0.7513 | 0.2058 | 0.177 |
| multimodal | unbiased | effort_matched | NeuralForest_PO_raw | 0.6593 | 0.7491 | 0.7606 | 0.2093 | 0.159 |
| multimodal | unbiased | effort_matched | RandomForest | 0.6483 | 0.8249 | 0.5739 | 0.1962 | 0.380 |
| multimodal | unbiased | effort_matched | XGBoost | 0.6371 | 0.6714 | 0.5359 | 0.1282 | 0.051 |
| multimodal | unbiased | effort_matched | maxnet_lq | 0.6534 | 0.7430 | 0.7115 | 0.1991 | 1.838 |
| multimodal | unbiased | effort_matched | maxnet_lqph | 0.6524 | 0.8240 | 0.7339 | 0.2176 | 1.546 |
| multimodal | unbiased | effort_matched | ranger_RF | 0.6229 | 0.7154 | 0.4056 | 0.0841 | 0.243 |
| multimodal | unbiased | uniform | BalancedSubsample_RF | 0.6505 | 0.8283 | 0.5982 | 0.2011 | 0.489 |
| multimodal | unbiased | uniform | CatBoost | 0.6593 | 0.8037 | 0.6728 | 0.2478 | 0.114 |
| multimodal | unbiased | uniform | Ensemble_RF_XGBoost_CatBoost | 0.6522 | 0.7755 | 0.5988 | 0.2127 | 0.545 |
| multimodal | unbiased | uniform | ExtraTrees | 0.6635 | 0.8997 | 0.7465 | 0.2389 | 0.189 |
| multimodal | unbiased | uniform | GAM_smooth | 0.6515 | 0.6814 | 0.7141 | 0.2165 | 0.290 |
| multimodal | unbiased | uniform | GLM_logistic | 0.6431 | 0.7505 | 0.8244 | 0.1673 | 0.005 |
| multimodal | unbiased | uniform | NeuralForest_PO_hidden | 0.6541 | 0.6974 | 0.7289 | 0.1972 | 0.180 |
| multimodal | unbiased | uniform | NeuralForest_PO_neural | 0.6511 | 0.7409 | 0.7652 | 0.2044 | 0.079 |
| multimodal | unbiased | uniform | NeuralForest_PO_random | 0.6536 | 0.7748 | 0.7513 | 0.2058 | 0.178 |
| multimodal | unbiased | uniform | NeuralForest_PO_raw | 0.6593 | 0.7491 | 0.7606 | 0.2093 | 0.159 |
| multimodal | unbiased | uniform | RandomForest | 0.6483 | 0.8249 | 0.5739 | 0.1962 | 0.380 |
| multimodal | unbiased | uniform | XGBoost | 0.6371 | 0.6714 | 0.5359 | 0.1282 | 0.051 |
| multimodal | unbiased | uniform | maxnet_lq | 0.6534 | 0.7430 | 0.7115 | 0.1991 | 1.831 |
| multimodal | unbiased | uniform | maxnet_lqph | 0.6524 | 0.8240 | 0.7339 | 0.2176 | 1.550 |
| multimodal | unbiased | uniform | ranger_RF | 0.6229 | 0.7154 | 0.4056 | 0.0841 | 0.240 |
| nonlinear_interaction | effort_biased | effort_matched | BalancedSubsample_RF | 0.7226 | 0.6802 | 0.8105 | 0.2988 | 0.560 |
| nonlinear_interaction | effort_biased | effort_matched | CatBoost | 0.7479 | 0.7234 | 0.8944 | 0.4076 | 0.139 |
| nonlinear_interaction | effort_biased | effort_matched | Ensemble_RF_XGBoost_CatBoost | 0.7379 | 0.6645 | 0.8521 | 0.3588 | 0.629 |
| nonlinear_interaction | effort_biased | effort_matched | ExtraTrees | 0.7163 | 0.8050 | 0.7308 | 0.2637 | 0.214 |
| nonlinear_interaction | effort_biased | effort_matched | GAM_smooth | 0.5321 | 0.3829 | 0.0325 | -0.3278 | 0.380 |
| nonlinear_interaction | effort_biased | effort_matched | GLM_logistic | 0.4263 | -0.0376 | -0.2008 | -0.2352 | 0.006 |
| nonlinear_interaction | effort_biased | effort_matched | NeuralForest_PO_hidden | 0.6972 | 0.8871 | 0.6470 | 0.2117 | 0.245 |
| nonlinear_interaction | effort_biased | effort_matched | NeuralForest_PO_neural | 0.5799 | 0.5606 | 0.2813 | 0.0970 | 0.100 |
| nonlinear_interaction | effort_biased | effort_matched | NeuralForest_PO_random | 0.6974 | 0.8805 | 0.6652 | 0.2091 | 0.243 |
| nonlinear_interaction | effort_biased | effort_matched | NeuralForest_PO_raw | 0.7037 | 0.8785 | 0.6829 | 0.2087 | 0.215 |
| nonlinear_interaction | effort_biased | effort_matched | RandomForest | 0.7275 | 0.6846 | 0.7758 | 0.3087 | 0.429 |
| nonlinear_interaction | effort_biased | effort_matched | XGBoost | 0.7190 | 0.7288 | 0.8022 | 0.2663 | 0.061 |
| nonlinear_interaction | effort_biased | effort_matched | maxnet_lq | 0.4306 | -0.0646 | -0.3299 | -0.2534 | 2.154 |
| nonlinear_interaction | effort_biased | effort_matched | maxnet_lqph | 0.6963 | 0.4269 | 0.7488 | 0.1914 | 1.802 |
| nonlinear_interaction | effort_biased | effort_matched | ranger_RF | 0.7066 | 0.7577 | 0.7299 | 0.2433 | 0.280 |
| nonlinear_interaction | effort_biased | uniform | BalancedSubsample_RF | 0.7204 | 0.7787 | 0.7962 | 0.2840 | 0.558 |
| nonlinear_interaction | effort_biased | uniform | CatBoost | 0.7518 | 0.8303 | 0.8614 | 0.3853 | 0.138 |
| nonlinear_interaction | effort_biased | uniform | Ensemble_RF_XGBoost_CatBoost | 0.7392 | 0.8508 | 0.8313 | 0.3442 | 0.625 |
| nonlinear_interaction | effort_biased | uniform | ExtraTrees | 0.7289 | 0.8443 | 0.7357 | 0.2648 | 0.209 |
| nonlinear_interaction | effort_biased | uniform | GAM_smooth | 0.5381 | 0.3695 | 0.0701 | -0.4378 | 0.425 |
| nonlinear_interaction | effort_biased | uniform | GLM_logistic | 0.3871 | -0.0798 | -0.4179 | -0.2857 | 0.005 |
| nonlinear_interaction | effort_biased | uniform | NeuralForest_PO_hidden | 0.6964 | 0.8209 | 0.6079 | 0.1578 | 0.244 |
| nonlinear_interaction | effort_biased | uniform | NeuralForest_PO_neural | 0.5343 | 0.4391 | 0.0943 | -0.0338 | 0.100 |
| nonlinear_interaction | effort_biased | uniform | NeuralForest_PO_random | 0.6820 | 0.8170 | 0.5433 | 0.1392 | 0.242 |
| nonlinear_interaction | effort_biased | uniform | NeuralForest_PO_raw | 0.6885 | 0.8368 | 0.5878 | 0.1447 | 0.213 |
| nonlinear_interaction | effort_biased | uniform | RandomForest | 0.7289 | 0.7958 | 0.7630 | 0.2922 | 0.426 |
| nonlinear_interaction | effort_biased | uniform | XGBoost | 0.7185 | 0.8252 | 0.7929 | 0.2554 | 0.061 |
| nonlinear_interaction | effort_biased | uniform | maxnet_lq | 0.4275 | -0.1830 | -0.3762 | -0.2884 | 2.140 |
| nonlinear_interaction | effort_biased | uniform | maxnet_lqph | 0.7055 | 0.5663 | 0.7380 | 0.1987 | 1.774 |
| nonlinear_interaction | effort_biased | uniform | ranger_RF | 0.7054 | 0.7164 | 0.7183 | 0.2199 | 0.272 |
| nonlinear_interaction | unbiased | effort_matched | BalancedSubsample_RF | 0.6923 | 0.7336 | 0.7571 | 0.2612 | 0.562 |
| nonlinear_interaction | unbiased | effort_matched | CatBoost | 0.7300 | 0.7489 | 0.8850 | 0.4018 | 0.137 |
| nonlinear_interaction | unbiased | effort_matched | Ensemble_RF_XGBoost_CatBoost | 0.7169 | 0.8188 | 0.8311 | 0.3502 | 0.627 |
| nonlinear_interaction | unbiased | effort_matched | ExtraTrees | 0.6915 | 0.9154 | 0.7144 | 0.2454 | 0.211 |
| nonlinear_interaction | unbiased | effort_matched | GAM_smooth | 0.5338 | 0.2071 | 0.0977 | -0.2348 | 0.467 |
| nonlinear_interaction | unbiased | effort_matched | GLM_logistic | 0.4131 | -0.1937 | -0.3287 | -0.2099 | 0.027 |
| nonlinear_interaction | unbiased | effort_matched | NeuralForest_PO_hidden | 0.6764 | 0.8743 | 0.6309 | 0.1934 | 0.246 |
| nonlinear_interaction | unbiased | effort_matched | NeuralForest_PO_neural | 0.6358 | 0.7890 | 0.4386 | 0.1137 | 0.100 |
| nonlinear_interaction | unbiased | effort_matched | NeuralForest_PO_random | 0.6662 | 0.8304 | 0.5691 | 0.1711 | 0.245 |
| nonlinear_interaction | unbiased | effort_matched | NeuralForest_PO_raw | 0.6678 | 0.8073 | 0.5840 | 0.1648 | 0.215 |
| nonlinear_interaction | unbiased | effort_matched | RandomForest | 0.6963 | 0.8053 | 0.7431 | 0.2774 | 0.429 |
| nonlinear_interaction | unbiased | effort_matched | XGBoost | 0.7060 | 0.7711 | 0.7866 | 0.2862 | 0.061 |
| nonlinear_interaction | unbiased | effort_matched | maxnet_lq | 0.4192 | -0.0198 | -0.3867 | -0.2355 | 2.128 |
| nonlinear_interaction | unbiased | effort_matched | maxnet_lqph | 0.6896 | 0.6290 | 0.7290 | 0.2329 | 1.748 |
| nonlinear_interaction | unbiased | effort_matched | ranger_RF | 0.6885 | 0.8005 | 0.7070 | 0.2127 | 0.279 |
| nonlinear_interaction | unbiased | uniform | BalancedSubsample_RF | 0.6923 | 0.7336 | 0.7571 | 0.2612 | 0.562 |
| nonlinear_interaction | unbiased | uniform | CatBoost | 0.7300 | 0.7489 | 0.8850 | 0.4018 | 0.144 |
| nonlinear_interaction | unbiased | uniform | Ensemble_RF_XGBoost_CatBoost | 0.7169 | 0.8188 | 0.8311 | 0.3502 | 0.637 |
| nonlinear_interaction | unbiased | uniform | ExtraTrees | 0.6915 | 0.9154 | 0.7144 | 0.2454 | 0.210 |
| nonlinear_interaction | unbiased | uniform | GAM_smooth | 0.5338 | 0.2071 | 0.0977 | -0.2348 | 0.466 |
| nonlinear_interaction | unbiased | uniform | GLM_logistic | 0.4131 | -0.1937 | -0.3287 | -0.2099 | 0.027 |
| nonlinear_interaction | unbiased | uniform | NeuralForest_PO_hidden | 0.6764 | 0.8743 | 0.6309 | 0.1934 | 0.246 |
| nonlinear_interaction | unbiased | uniform | NeuralForest_PO_neural | 0.6358 | 0.7890 | 0.4386 | 0.1137 | 0.100 |
| nonlinear_interaction | unbiased | uniform | NeuralForest_PO_random | 0.6662 | 0.8304 | 0.5691 | 0.1711 | 0.244 |
| nonlinear_interaction | unbiased | uniform | NeuralForest_PO_raw | 0.6678 | 0.8073 | 0.5840 | 0.1648 | 0.215 |
| nonlinear_interaction | unbiased | uniform | RandomForest | 0.6963 | 0.8053 | 0.7431 | 0.2774 | 0.432 |
| nonlinear_interaction | unbiased | uniform | XGBoost | 0.7060 | 0.7711 | 0.7866 | 0.2862 | 0.061 |
| nonlinear_interaction | unbiased | uniform | maxnet_lq | 0.4192 | -0.0198 | -0.3867 | -0.2355 | 2.127 |
| nonlinear_interaction | unbiased | uniform | maxnet_lqph | 0.6896 | 0.6290 | 0.7290 | 0.2329 | 1.753 |
| nonlinear_interaction | unbiased | uniform | ranger_RF | 0.6885 | 0.8005 | 0.7070 | 0.2127 | 0.281 |

## Paired native hidden-feature vs reference methods

| Species | Effort | Background | Comparator | Oracle rho change | AUC_pb change | Win folds / comparisons |
| --- | --- | --- | --- | ---: | ---: | ---: |
| linear | effort_biased | effort_matched | CatBoost | +0.1426 | +0.0323 | 10/10 |
| linear | effort_biased | effort_matched | Ensemble_RF_XGBoost_CatBoost | +0.2029 | +0.0397 | 10/10 |
| linear | effort_biased | effort_matched | NeuralForest_PO_neural | -0.0434 | -0.0051 | 4/10 |
| linear | effort_biased | effort_matched | NeuralForest_PO_random | +0.0242 | +0.0079 | 6/10 |
| linear | effort_biased | effort_matched | NeuralForest_PO_raw | +0.0128 | +0.0046 | 4/10 |
| linear | effort_biased | effort_matched | XGBoost | +0.2848 | +0.0464 | 10/10 |
| linear | effort_biased | effort_matched | maxnet_lqph | -0.0541 | -0.0030 | 1/10 |
| linear | effort_biased | effort_matched | ranger_RF | +0.3000 | +0.0542 | 10/10 |
| linear | effort_biased | uniform | CatBoost | +0.1420 | +0.0312 | 9/10 |
| linear | effort_biased | uniform | Ensemble_RF_XGBoost_CatBoost | +0.1886 | +0.0383 | 10/10 |
| linear | effort_biased | uniform | NeuralForest_PO_neural | -0.0563 | -0.0064 | 3/10 |
| linear | effort_biased | uniform | NeuralForest_PO_random | -0.0049 | +0.0033 | 2/10 |
| linear | effort_biased | uniform | NeuralForest_PO_raw | +0.0209 | +0.0160 | 6/10 |
| linear | effort_biased | uniform | XGBoost | +0.2447 | +0.0427 | 9/10 |
| linear | effort_biased | uniform | maxnet_lqph | +0.0291 | +0.0030 | 1/10 |
| linear | effort_biased | uniform | ranger_RF | +0.4155 | +0.0854 | 10/10 |
| linear | unbiased | effort_matched | CatBoost | +0.1252 | +0.0038 | 7/10 |
| linear | unbiased | effort_matched | Ensemble_RF_XGBoost_CatBoost | +0.2085 | +0.0174 | 9/10 |
| linear | unbiased | effort_matched | NeuralForest_PO_neural | +0.0468 | -0.0019 | 2/10 |
| linear | unbiased | effort_matched | NeuralForest_PO_random | -0.0075 | +0.0020 | 6/10 |
| linear | unbiased | effort_matched | NeuralForest_PO_raw | +0.0348 | +0.0052 | 6/10 |
| linear | unbiased | effort_matched | XGBoost | +0.3006 | +0.0257 | 10/10 |
| linear | unbiased | effort_matched | maxnet_lqph | -0.0801 | -0.0084 | 2/10 |
| linear | unbiased | effort_matched | ranger_RF | +0.3446 | +0.0287 | 10/10 |
| linear | unbiased | uniform | CatBoost | +0.1252 | +0.0038 | 7/10 |
| linear | unbiased | uniform | Ensemble_RF_XGBoost_CatBoost | +0.2085 | +0.0174 | 9/10 |
| linear | unbiased | uniform | NeuralForest_PO_neural | +0.0468 | -0.0019 | 2/10 |
| linear | unbiased | uniform | NeuralForest_PO_random | -0.0075 | +0.0020 | 6/10 |
| linear | unbiased | uniform | NeuralForest_PO_raw | +0.0348 | +0.0052 | 6/10 |
| linear | unbiased | uniform | XGBoost | +0.3006 | +0.0257 | 10/10 |
| linear | unbiased | uniform | maxnet_lqph | -0.0801 | -0.0084 | 2/10 |
| linear | unbiased | uniform | ranger_RF | +0.3446 | +0.0287 | 10/10 |
| multimodal | effort_biased | effort_matched | CatBoost | +0.0244 | +0.0044 | 8/10 |
| multimodal | effort_biased | effort_matched | Ensemble_RF_XGBoost_CatBoost | +0.0928 | +0.0151 | 9/10 |
| multimodal | effort_biased | effort_matched | NeuralForest_PO_neural | -0.0009 | +0.0019 | 5/10 |
| multimodal | effort_biased | effort_matched | NeuralForest_PO_random | +0.0144 | +0.0011 | 6/10 |
| multimodal | effort_biased | effort_matched | NeuralForest_PO_raw | +0.0002 | +0.0010 | 4/10 |
| multimodal | effort_biased | effort_matched | XGBoost | +0.1420 | +0.0242 | 9/10 |
| multimodal | effort_biased | effort_matched | maxnet_lqph | +0.0246 | +0.0091 | 6/10 |
| multimodal | effort_biased | effort_matched | ranger_RF | +0.3848 | +0.0565 | 10/10 |
| multimodal | effort_biased | uniform | CatBoost | +0.0413 | +0.0079 | 6/10 |
| multimodal | effort_biased | uniform | Ensemble_RF_XGBoost_CatBoost | +0.0793 | +0.0154 | 8/10 |
| multimodal | effort_biased | uniform | NeuralForest_PO_neural | -0.0847 | -0.0127 | 3/10 |
| multimodal | effort_biased | uniform | NeuralForest_PO_random | -0.0181 | -0.0020 | 4/10 |
| multimodal | effort_biased | uniform | NeuralForest_PO_raw | -0.0272 | -0.0018 | 3/10 |
| multimodal | effort_biased | uniform | XGBoost | +0.1319 | +0.0268 | 9/10 |
| multimodal | effort_biased | uniform | maxnet_lqph | +0.0213 | +0.0132 | 5/10 |
| multimodal | effort_biased | uniform | ranger_RF | +0.3753 | +0.0637 | 10/10 |
| multimodal | unbiased | effort_matched | CatBoost | +0.0561 | -0.0052 | 9/10 |
| multimodal | unbiased | effort_matched | Ensemble_RF_XGBoost_CatBoost | +0.1302 | +0.0019 | 9/10 |
| multimodal | unbiased | effort_matched | NeuralForest_PO_neural | -0.0362 | +0.0030 | 4/10 |
| multimodal | unbiased | effort_matched | NeuralForest_PO_random | -0.0224 | +0.0005 | 4/10 |
| multimodal | unbiased | effort_matched | NeuralForest_PO_raw | -0.0317 | -0.0052 | 3/10 |
| multimodal | unbiased | effort_matched | XGBoost | +0.1931 | +0.0170 | 9/10 |
| multimodal | unbiased | effort_matched | maxnet_lqph | -0.0049 | +0.0017 | 5/10 |
| multimodal | unbiased | effort_matched | ranger_RF | +0.3234 | +0.0312 | 9/10 |
| multimodal | unbiased | uniform | CatBoost | +0.0561 | -0.0052 | 9/10 |
| multimodal | unbiased | uniform | Ensemble_RF_XGBoost_CatBoost | +0.1302 | +0.0019 | 9/10 |
| multimodal | unbiased | uniform | NeuralForest_PO_neural | -0.0362 | +0.0030 | 4/10 |
| multimodal | unbiased | uniform | NeuralForest_PO_random | -0.0224 | +0.0005 | 4/10 |
| multimodal | unbiased | uniform | NeuralForest_PO_raw | -0.0317 | -0.0052 | 3/10 |
| multimodal | unbiased | uniform | XGBoost | +0.1931 | +0.0170 | 9/10 |
| multimodal | unbiased | uniform | maxnet_lqph | -0.0049 | +0.0017 | 5/10 |
| multimodal | unbiased | uniform | ranger_RF | +0.3234 | +0.0312 | 9/10 |
| nonlinear_interaction | effort_biased | effort_matched | CatBoost | -0.2473 | -0.0507 | 1/10 |
| nonlinear_interaction | effort_biased | effort_matched | Ensemble_RF_XGBoost_CatBoost | -0.2051 | -0.0406 | 1/10 |
| nonlinear_interaction | effort_biased | effort_matched | NeuralForest_PO_neural | +0.3657 | +0.1174 | 10/10 |
| nonlinear_interaction | effort_biased | effort_matched | NeuralForest_PO_random | -0.0182 | -0.0002 | 4/10 |
| nonlinear_interaction | effort_biased | effort_matched | NeuralForest_PO_raw | -0.0358 | -0.0064 | 5/10 |
| nonlinear_interaction | effort_biased | effort_matched | XGBoost | -0.1552 | -0.0218 | 2/10 |
| nonlinear_interaction | effort_biased | effort_matched | maxnet_lqph | -0.1017 | +0.0009 | 2/10 |
| nonlinear_interaction | effort_biased | effort_matched | ranger_RF | -0.0828 | -0.0093 | 5/10 |
| nonlinear_interaction | effort_biased | uniform | CatBoost | -0.2535 | -0.0554 | 0/10 |
| nonlinear_interaction | effort_biased | uniform | Ensemble_RF_XGBoost_CatBoost | -0.2234 | -0.0427 | 0/10 |
| nonlinear_interaction | effort_biased | uniform | NeuralForest_PO_neural | +0.5136 | +0.1622 | 10/10 |
| nonlinear_interaction | effort_biased | uniform | NeuralForest_PO_random | +0.0646 | +0.0145 | 7/10 |
| nonlinear_interaction | effort_biased | uniform | NeuralForest_PO_raw | +0.0201 | +0.0079 | 6/10 |
| nonlinear_interaction | effort_biased | uniform | XGBoost | -0.1850 | -0.0221 | 1/10 |
| nonlinear_interaction | effort_biased | uniform | maxnet_lqph | -0.1301 | -0.0090 | 3/10 |
| nonlinear_interaction | effort_biased | uniform | ranger_RF | -0.1104 | -0.0090 | 4/10 |
| nonlinear_interaction | unbiased | effort_matched | CatBoost | -0.2541 | -0.0537 | 0/10 |
| nonlinear_interaction | unbiased | effort_matched | Ensemble_RF_XGBoost_CatBoost | -0.2002 | -0.0405 | 0/10 |
| nonlinear_interaction | unbiased | effort_matched | NeuralForest_PO_neural | +0.1924 | +0.0406 | 10/10 |
| nonlinear_interaction | unbiased | effort_matched | NeuralForest_PO_random | +0.0619 | +0.0102 | 7/10 |
| nonlinear_interaction | unbiased | effort_matched | NeuralForest_PO_raw | +0.0469 | +0.0085 | 7/10 |
| nonlinear_interaction | unbiased | effort_matched | XGBoost | -0.1557 | -0.0297 | 0/10 |
| nonlinear_interaction | unbiased | effort_matched | maxnet_lqph | -0.0981 | -0.0132 | 3/10 |
| nonlinear_interaction | unbiased | effort_matched | ranger_RF | -0.0761 | -0.0121 | 3/10 |
| nonlinear_interaction | unbiased | uniform | CatBoost | -0.2541 | -0.0537 | 0/10 |
| nonlinear_interaction | unbiased | uniform | Ensemble_RF_XGBoost_CatBoost | -0.2002 | -0.0405 | 0/10 |
| nonlinear_interaction | unbiased | uniform | NeuralForest_PO_neural | +0.1924 | +0.0406 | 10/10 |
| nonlinear_interaction | unbiased | uniform | NeuralForest_PO_random | +0.0619 | +0.0102 | 7/10 |
| nonlinear_interaction | unbiased | uniform | NeuralForest_PO_raw | +0.0469 | +0.0085 | 7/10 |
| nonlinear_interaction | unbiased | uniform | XGBoost | -0.1557 | -0.0297 | 0/10 |
| nonlinear_interaction | unbiased | uniform | maxnet_lqph | -0.0981 | -0.0132 | 3/10 |
| nonlinear_interaction | unbiased | uniform | ranger_RF | -0.0761 | -0.0121 | 3/10 |

## Critical publication and interpretation limits

- The simulator has already been inspected while developing the
  estimator. These outcomes are exploratory and cannot prove generality.
- Spatial folds share much of their training areas, so fold-level
  wins and variance are NOT independent biological replications.
- AUC_pb and Boyce depend on the accessible study region and
  how available environment is sampled.
- Observation effort is known in simulation only; target-group
  matching could fail if the target group has different biases.
- Oracle correlation is only possible for virtual species.
- No fully independent real-species datasets or held-out
  external ecological surveys were evaluated here.
- Learned hidden features require random-projection controls
  of identical dimensions and fixed feature-sampling budgets.
- Comparing two MaxEnt feature sets and one predeclared set of
  tuning parameters does not replace equal-budget nested spatial
  tuning across all methods in a publishable confirmatory study.
- Never claim universal superiority, occupancy probability,
  true absence accuracy, or significance from cherry-picked strata.

See all_runs.csv, group_means.csv, paired_differences.csv
and paired_summary.csv for machine-readable results.
