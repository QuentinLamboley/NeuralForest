# Nested-spatial point-process gradient forest benchmark

**This is an exploratory simulation, NOT an external ecological
validation or a demonstration of universal superiority.**

All model predictions are paired on 72 geographically held-out
cases: three virtual species × unbiased/biased observation effort
× uniform/effort-matched background × two independent seeds ×
three outer geographic blocks. A *different* geographic block
inside each outer training set selects the number of C++ IPP
gradient forest stages; after selection, fits use all outer
training blocks. The outer block is never used in model selection.
Each of nine model predictions is recorded per case (648 rows).

Background is available environment and can contain presences.
Neither AUC_pb nor Boyce identifies absolute occupancy. Synthetic
oracle Spearman checks whether relative scores reflect the TRUE
virtual habitat intensity without exposing it to training.
No hyperparameter search used evaluation outcomes.

## All algorithms and conditions: no favorable subset omitted

| Species | Effort | Background | Model | Oracle Spearman | Oracle IPP log skill | AUC_pb | Boyce | Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| linear | effort_biased | effort_matched | CatBoost | 0.5356 | 0.0143 | 0.5989 | 0.5173 | 0.113 |
| linear | effort_biased | effort_matched | GAM_smooth | 0.7214 | 0.0854 | 0.6295 | 0.6276 | 0.258 |
| linear | effort_biased | effort_matched | NeuralForest_POGrad_hidden | 0.5643 | 0.0520 | 0.6086 | 0.5693 | 0.180 |
| linear | effort_biased | effort_matched | NeuralForest_POGrad_random | 0.5507 | 0.0489 | 0.6061 | 0.5090 | 0.182 |
| linear | effort_biased | effort_matched | NeuralForest_POGrad_raw | 0.5766 | 0.0535 | 0.6098 | 0.5847 | 0.175 |
| linear | effort_biased | effort_matched | NeuralForest_POGrad_teacher | 0.5766 | 0.0535 | 0.6098 | 0.5847 | 0.175 |
| linear | effort_biased | effort_matched | RandomForest | 0.3473 | -0.0168 | 0.5470 | 0.3513 | 0.401 |
| linear | effort_biased | effort_matched | XGBoost | 0.3724 | -0.1340 | 0.5618 | 0.3981 | 0.054 |
| linear | effort_biased | effort_matched | maxnet_lqph | 0.5368 | 0.0197 | 0.6001 | 0.4182 | 2.245 |
| linear | effort_biased | uniform | CatBoost | 0.5680 | -0.0013 | 0.5893 | 0.6449 | 0.115 |
| linear | effort_biased | uniform | GAM_smooth | 0.9571 | 0.0757 | 0.6522 | 0.7801 | 0.319 |
| linear | effort_biased | uniform | NeuralForest_POGrad_hidden | 0.8077 | 0.0488 | 0.6262 | 0.8008 | 0.169 |
| linear | effort_biased | uniform | NeuralForest_POGrad_random | 0.8118 | 0.0497 | 0.6253 | 0.7902 | 0.171 |
| linear | effort_biased | uniform | NeuralForest_POGrad_raw | 0.8070 | 0.0490 | 0.6259 | 0.7710 | 0.173 |
| linear | effort_biased | uniform | NeuralForest_POGrad_teacher | 0.8077 | 0.0488 | 0.6262 | 0.8008 | 0.173 |
| linear | effort_biased | uniform | RandomForest | 0.3483 | -0.0548 | 0.5618 | 0.3010 | 0.396 |
| linear | effort_biased | uniform | XGBoost | 0.4049 | -0.1312 | 0.5738 | 0.4963 | 0.056 |
| linear | effort_biased | uniform | maxnet_lqph | 0.6234 | 0.0449 | 0.6039 | 0.5695 | 2.302 |
| linear | unbiased | effort_matched | CatBoost | 0.3453 | -0.0137 | 0.5434 | 0.5905 | 0.115 |
| linear | unbiased | effort_matched | GAM_smooth | 0.9234 | 0.0913 | 0.5987 | 0.6068 | 0.318 |
| linear | unbiased | effort_matched | NeuralForest_POGrad_hidden | 0.5631 | 0.0561 | 0.5734 | 0.4685 | 0.169 |
| linear | unbiased | effort_matched | NeuralForest_POGrad_random | 0.5541 | 0.0547 | 0.5734 | 0.4685 | 0.170 |
| linear | unbiased | effort_matched | NeuralForest_POGrad_raw | 0.5720 | 0.0548 | 0.5744 | 0.4897 | 0.175 |
| linear | unbiased | effort_matched | NeuralForest_POGrad_teacher | 0.5631 | 0.0561 | 0.5734 | 0.4685 | 0.175 |
| linear | unbiased | effort_matched | RandomForest | 0.2156 | -0.0389 | 0.5175 | 0.5192 | 0.404 |
| linear | unbiased | effort_matched | XGBoost | 0.2080 | -0.1594 | 0.5248 | 0.7556 | 0.056 |
| linear | unbiased | effort_matched | maxnet_lqph | 0.6469 | 0.0638 | 0.5786 | 0.6634 | 2.318 |
| linear | unbiased | uniform | CatBoost | 0.3453 | -0.0137 | 0.5434 | 0.5905 | 0.126 |
| linear | unbiased | uniform | GAM_smooth | 0.9234 | 0.0913 | 0.5987 | 0.6068 | 0.310 |
| linear | unbiased | uniform | NeuralForest_POGrad_hidden | 0.5631 | 0.0561 | 0.5734 | 0.4685 | 0.169 |
| linear | unbiased | uniform | NeuralForest_POGrad_random | 0.5541 | 0.0547 | 0.5734 | 0.4685 | 0.170 |
| linear | unbiased | uniform | NeuralForest_POGrad_raw | 0.5720 | 0.0548 | 0.5744 | 0.4897 | 0.174 |
| linear | unbiased | uniform | NeuralForest_POGrad_teacher | 0.5631 | 0.0561 | 0.5734 | 0.4685 | 0.174 |
| linear | unbiased | uniform | RandomForest | 0.2156 | -0.0389 | 0.5175 | 0.5192 | 0.407 |
| linear | unbiased | uniform | XGBoost | 0.2080 | -0.1594 | 0.5248 | 0.7556 | 0.056 |
| linear | unbiased | uniform | maxnet_lqph | 0.6469 | 0.0638 | 0.5786 | 0.6634 | 2.227 |
| multimodal | effort_biased | effort_matched | CatBoost | 0.6529 | 0.1430 | 0.6144 | 0.6304 | 0.115 |
| multimodal | effort_biased | effort_matched | GAM_smooth | 0.7142 | 0.0755 | 0.6149 | 0.5828 | 0.394 |
| multimodal | effort_biased | effort_matched | NeuralForest_POGrad_hidden | 0.8291 | 0.1629 | 0.6263 | 0.6401 | 0.179 |
| multimodal | effort_biased | effort_matched | NeuralForest_POGrad_random | 0.8315 | 0.1581 | 0.6274 | 0.7439 | 0.177 |
| multimodal | effort_biased | effort_matched | NeuralForest_POGrad_raw | 0.8266 | 0.1609 | 0.6262 | 0.6973 | 0.181 |
| multimodal | effort_biased | effort_matched | NeuralForest_POGrad_teacher | 0.8306 | 0.1611 | 0.6265 | 0.6941 | 0.181 |
| multimodal | effort_biased | effort_matched | RandomForest | 0.4836 | 0.1085 | 0.6015 | 0.7468 | 0.405 |
| multimodal | effort_biased | effort_matched | XGBoost | 0.4932 | 0.0270 | 0.5925 | 0.7045 | 0.056 |
| multimodal | effort_biased | effort_matched | maxnet_lqph | 0.6673 | 0.1420 | 0.6048 | 0.6367 | 2.234 |
| multimodal | effort_biased | uniform | CatBoost | 0.5015 | 0.1201 | 0.6128 | 0.8249 | 0.116 |
| multimodal | effort_biased | uniform | GAM_smooth | 0.6537 | 0.1088 | 0.6219 | 0.7208 | 0.467 |
| multimodal | effort_biased | uniform | NeuralForest_POGrad_hidden | 0.7426 | 0.1211 | 0.6292 | 0.7331 | 0.170 |
| multimodal | effort_biased | uniform | NeuralForest_POGrad_random | 0.7436 | 0.1147 | 0.6291 | 0.7649 | 0.174 |
| multimodal | effort_biased | uniform | NeuralForest_POGrad_raw | 0.7423 | 0.1206 | 0.6292 | 0.7657 | 0.174 |
| multimodal | effort_biased | uniform | NeuralForest_POGrad_teacher | 0.7426 | 0.1211 | 0.6292 | 0.7331 | 0.174 |
| multimodal | effort_biased | uniform | RandomForest | 0.3634 | 0.0579 | 0.5982 | 0.7780 | 0.405 |
| multimodal | effort_biased | uniform | XGBoost | 0.3025 | -0.0198 | 0.5893 | 0.6271 | 0.056 |
| multimodal | effort_biased | uniform | maxnet_lqph | 0.5723 | 0.1019 | 0.6136 | 0.6663 | 2.237 |
| multimodal | unbiased | effort_matched | CatBoost | 0.6462 | 0.1232 | 0.6083 | 0.7852 | 0.115 |
| multimodal | unbiased | effort_matched | GAM_smooth | 0.7432 | 0.1156 | 0.6101 | 0.6334 | 0.355 |
| multimodal | unbiased | effort_matched | NeuralForest_POGrad_hidden | 0.7639 | 0.1384 | 0.6225 | 0.7159 | 0.169 |
| multimodal | unbiased | effort_matched | NeuralForest_POGrad_random | 0.7696 | 0.1401 | 0.6226 | 0.7136 | 0.167 |
| multimodal | unbiased | effort_matched | NeuralForest_POGrad_raw | 0.7695 | 0.1407 | 0.6228 | 0.7054 | 0.169 |
| multimodal | unbiased | effort_matched | NeuralForest_POGrad_teacher | 0.7693 | 0.1402 | 0.6226 | 0.7089 | 0.169 |
| multimodal | unbiased | effort_matched | RandomForest | 0.4316 | 0.0684 | 0.5710 | 0.7472 | 0.407 |
| multimodal | unbiased | effort_matched | XGBoost | 0.4121 | -0.0191 | 0.5616 | 0.6429 | 0.056 |
| multimodal | unbiased | effort_matched | maxnet_lqph | 0.7043 | 0.1353 | 0.6136 | 0.6224 | 2.244 |
| multimodal | unbiased | uniform | CatBoost | 0.6462 | 0.1232 | 0.6083 | 0.7852 | 0.125 |
| multimodal | unbiased | uniform | GAM_smooth | 0.7432 | 0.1156 | 0.6101 | 0.6334 | 0.356 |
| multimodal | unbiased | uniform | NeuralForest_POGrad_hidden | 0.7639 | 0.1384 | 0.6225 | 0.7159 | 0.169 |
| multimodal | unbiased | uniform | NeuralForest_POGrad_random | 0.7696 | 0.1401 | 0.6226 | 0.7136 | 0.166 |
| multimodal | unbiased | uniform | NeuralForest_POGrad_raw | 0.7695 | 0.1407 | 0.6228 | 0.7054 | 0.168 |
| multimodal | unbiased | uniform | NeuralForest_POGrad_teacher | 0.7693 | 0.1402 | 0.6226 | 0.7089 | 0.168 |
| multimodal | unbiased | uniform | RandomForest | 0.4316 | 0.0684 | 0.5710 | 0.7472 | 0.407 |
| multimodal | unbiased | uniform | XGBoost | 0.4121 | -0.0191 | 0.5616 | 0.6429 | 0.056 |
| multimodal | unbiased | uniform | maxnet_lqph | 0.7043 | 0.1353 | 0.6136 | 0.6224 | 2.239 |
| nonlinear_interaction | effort_biased | effort_matched | CatBoost | 0.8605 | 0.4604 | 0.7463 | 0.8387 | 0.080 |
| nonlinear_interaction | effort_biased | effort_matched | GAM_smooth | 0.0873 | -0.1372 | 0.5553 | 0.4118 | 0.320 |
| nonlinear_interaction | effort_biased | effort_matched | NeuralForest_POGrad_hidden | 0.5961 | 0.2351 | 0.6871 | 0.7103 | 0.121 |
| nonlinear_interaction | effort_biased | effort_matched | NeuralForest_POGrad_random | 0.5891 | 0.2278 | 0.6848 | 0.7383 | 0.119 |
| nonlinear_interaction | effort_biased | effort_matched | NeuralForest_POGrad_raw | 0.5630 | 0.2183 | 0.6779 | 0.7447 | 0.120 |
| nonlinear_interaction | effort_biased | effort_matched | NeuralForest_POGrad_teacher | 0.5019 | 0.1993 | 0.6595 | 0.7532 | 0.120 |
| nonlinear_interaction | effort_biased | effort_matched | RandomForest | 0.8077 | 0.3830 | 0.7418 | 0.7915 | 0.262 |
| nonlinear_interaction | effort_biased | effort_matched | XGBoost | 0.8599 | 0.4261 | 0.7377 | 0.7344 | 0.036 |
| nonlinear_interaction | effort_biased | effort_matched | maxnet_lqph | 0.8720 | 0.3142 | 0.7434 | 0.5296 | 1.468 |
| nonlinear_interaction | effort_biased | uniform | CatBoost | 0.8906 | 0.4710 | 0.7578 | 0.8401 | 0.080 |
| nonlinear_interaction | effort_biased | uniform | GAM_smooth | 0.0776 | -0.1991 | 0.5500 | 0.1801 | 0.270 |
| nonlinear_interaction | effort_biased | uniform | NeuralForest_POGrad_hidden | 0.5177 | 0.2223 | 0.6689 | 0.6919 | 0.122 |
| nonlinear_interaction | effort_biased | uniform | NeuralForest_POGrad_random | 0.5147 | 0.2111 | 0.6667 | 0.7258 | 0.118 |
| nonlinear_interaction | effort_biased | uniform | NeuralForest_POGrad_raw | 0.5091 | 0.1998 | 0.6670 | 0.7441 | 0.116 |
| nonlinear_interaction | effort_biased | uniform | NeuralForest_POGrad_teacher | 0.4482 | 0.1811 | 0.6442 | 0.6493 | 0.116 |
| nonlinear_interaction | effort_biased | uniform | RandomForest | 0.8301 | 0.3881 | 0.7479 | 0.8573 | 0.258 |
| nonlinear_interaction | effort_biased | uniform | XGBoost | 0.8535 | 0.4169 | 0.7364 | 0.8121 | 0.037 |
| nonlinear_interaction | effort_biased | uniform | maxnet_lqph | 0.8399 | 0.3308 | 0.7331 | 0.4132 | 1.481 |
| nonlinear_interaction | unbiased | effort_matched | CatBoost | 0.9169 | 0.4975 | 0.7540 | 0.8368 | 0.141 |
| nonlinear_interaction | unbiased | effort_matched | GAM_smooth | 0.0934 | -0.0782 | 0.5475 | 0.0797 | 0.366 |
| nonlinear_interaction | unbiased | effort_matched | NeuralForest_POGrad_hidden | 0.6424 | 0.3000 | 0.7040 | 0.8849 | 0.115 |
| nonlinear_interaction | unbiased | effort_matched | NeuralForest_POGrad_random | 0.6514 | 0.3063 | 0.7057 | 0.8681 | 0.116 |
| nonlinear_interaction | unbiased | effort_matched | NeuralForest_POGrad_raw | 0.6387 | 0.3090 | 0.7033 | 0.8711 | 0.114 |
| nonlinear_interaction | unbiased | effort_matched | NeuralForest_POGrad_teacher | 0.6378 | 0.2864 | 0.7035 | 0.8886 | 0.114 |
| nonlinear_interaction | unbiased | effort_matched | RandomForest | 0.8113 | 0.3997 | 0.7359 | 0.8199 | 0.264 |
| nonlinear_interaction | unbiased | effort_matched | XGBoost | 0.8535 | 0.4455 | 0.7345 | 0.7552 | 0.038 |
| nonlinear_interaction | unbiased | effort_matched | maxnet_lqph | 0.8388 | 0.3351 | 0.7312 | 0.5105 | 1.495 |
| nonlinear_interaction | unbiased | uniform | CatBoost | 0.9169 | 0.4975 | 0.7540 | 0.8368 | 0.101 |
| nonlinear_interaction | unbiased | uniform | GAM_smooth | 0.0934 | -0.0782 | 0.5475 | 0.0797 | 0.361 |
| nonlinear_interaction | unbiased | uniform | NeuralForest_POGrad_hidden | 0.6424 | 0.3000 | 0.7040 | 0.8849 | 0.116 |
| nonlinear_interaction | unbiased | uniform | NeuralForest_POGrad_random | 0.6514 | 0.3063 | 0.7057 | 0.8681 | 0.116 |
| nonlinear_interaction | unbiased | uniform | NeuralForest_POGrad_raw | 0.6387 | 0.3090 | 0.7033 | 0.8711 | 0.116 |
| nonlinear_interaction | unbiased | uniform | NeuralForest_POGrad_teacher | 0.6378 | 0.2864 | 0.7035 | 0.8886 | 0.116 |
| nonlinear_interaction | unbiased | uniform | RandomForest | 0.8113 | 0.3997 | 0.7359 | 0.8199 | 0.263 |
| nonlinear_interaction | unbiased | uniform | XGBoost | 0.8535 | 0.4455 | 0.7345 | 0.7552 | 0.037 |
| nonlinear_interaction | unbiased | uniform | maxnet_lqph | 0.8388 | 0.3351 | 0.7312 | 0.5105 | 1.482 |

## Gradient forest vs its EXACT neural teacher, random-feature control and strong competitors

| Species | Effort | Background | Reference | Oracle rho change | AUC_pb change |
| --- | --- | --- | --- | ---: | ---: |
| linear | effort_biased | effort_matched | CatBoost | +0.0287 | +0.0096 |
| linear | effort_biased | effort_matched | GAM_smooth | -0.1571 | -0.0209 |
| linear | effort_biased | effort_matched | NeuralForest_POGrad_random | +0.0136 | +0.0025 |
| linear | effort_biased | effort_matched | NeuralForest_POGrad_raw | -0.0123 | -0.0012 |
| linear | effort_biased | effort_matched | NeuralForest_POGrad_teacher | -0.0123 | -0.0012 |
| linear | effort_biased | effort_matched | RandomForest | +0.2170 | +0.0615 |
| linear | effort_biased | effort_matched | XGBoost | +0.1918 | +0.0467 |
| linear | effort_biased | effort_matched | maxnet_lqph | +0.0275 | +0.0085 |
| linear | effort_biased | uniform | CatBoost | +0.2396 | +0.0368 |
| linear | effort_biased | uniform | GAM_smooth | -0.1494 | -0.0260 |
| linear | effort_biased | uniform | NeuralForest_POGrad_random | -0.0041 | +0.0009 |
| linear | effort_biased | uniform | NeuralForest_POGrad_raw | +0.0006 | +0.0003 |
| linear | effort_biased | uniform | NeuralForest_POGrad_teacher | +0.0000 | +0.0000 |
| linear | effort_biased | uniform | RandomForest | +0.4594 | +0.0644 |
| linear | effort_biased | uniform | XGBoost | +0.4027 | +0.0523 |
| linear | effort_biased | uniform | maxnet_lqph | +0.1843 | +0.0222 |
| linear | unbiased | effort_matched | CatBoost | +0.2178 | +0.0300 |
| linear | unbiased | effort_matched | GAM_smooth | -0.3603 | -0.0253 |
| linear | unbiased | effort_matched | NeuralForest_POGrad_random | +0.0090 | -0.0000 |
| linear | unbiased | effort_matched | NeuralForest_POGrad_raw | -0.0089 | -0.0010 |
| linear | unbiased | effort_matched | NeuralForest_POGrad_teacher | +0.0000 | +0.0000 |
| linear | unbiased | effort_matched | RandomForest | +0.3475 | +0.0559 |
| linear | unbiased | effort_matched | XGBoost | +0.3552 | +0.0486 |
| linear | unbiased | effort_matched | maxnet_lqph | -0.0838 | -0.0052 |
| linear | unbiased | uniform | CatBoost | +0.2178 | +0.0300 |
| linear | unbiased | uniform | GAM_smooth | -0.3603 | -0.0253 |
| linear | unbiased | uniform | NeuralForest_POGrad_random | +0.0090 | -0.0000 |
| linear | unbiased | uniform | NeuralForest_POGrad_raw | -0.0089 | -0.0010 |
| linear | unbiased | uniform | NeuralForest_POGrad_teacher | +0.0000 | +0.0000 |
| linear | unbiased | uniform | RandomForest | +0.3475 | +0.0559 |
| linear | unbiased | uniform | XGBoost | +0.3552 | +0.0486 |
| linear | unbiased | uniform | maxnet_lqph | -0.0838 | -0.0052 |
| multimodal | effort_biased | effort_matched | CatBoost | +0.1762 | +0.0119 |
| multimodal | effort_biased | effort_matched | GAM_smooth | +0.1149 | +0.0114 |
| multimodal | effort_biased | effort_matched | NeuralForest_POGrad_random | -0.0025 | -0.0011 |
| multimodal | effort_biased | effort_matched | NeuralForest_POGrad_raw | +0.0024 | +0.0001 |
| multimodal | effort_biased | effort_matched | NeuralForest_POGrad_teacher | -0.0015 | -0.0002 |
| multimodal | effort_biased | effort_matched | RandomForest | +0.3455 | +0.0248 |
| multimodal | effort_biased | effort_matched | XGBoost | +0.3359 | +0.0338 |
| multimodal | effort_biased | effort_matched | maxnet_lqph | +0.1617 | +0.0215 |
| multimodal | effort_biased | uniform | CatBoost | +0.2410 | +0.0164 |
| multimodal | effort_biased | uniform | GAM_smooth | +0.0888 | +0.0074 |
| multimodal | effort_biased | uniform | NeuralForest_POGrad_random | -0.0011 | +0.0002 |
| multimodal | effort_biased | uniform | NeuralForest_POGrad_raw | +0.0003 | +0.0000 |
| multimodal | effort_biased | uniform | NeuralForest_POGrad_teacher | +0.0000 | +0.0000 |
| multimodal | effort_biased | uniform | RandomForest | +0.3792 | +0.0310 |
| multimodal | effort_biased | uniform | XGBoost | +0.4400 | +0.0399 |
| multimodal | effort_biased | uniform | maxnet_lqph | +0.1703 | +0.0156 |
| multimodal | unbiased | effort_matched | CatBoost | +0.1178 | +0.0141 |
| multimodal | unbiased | effort_matched | GAM_smooth | +0.0207 | +0.0123 |
| multimodal | unbiased | effort_matched | NeuralForest_POGrad_random | -0.0057 | -0.0002 |
| multimodal | unbiased | effort_matched | NeuralForest_POGrad_raw | -0.0056 | -0.0003 |
| multimodal | unbiased | effort_matched | NeuralForest_POGrad_teacher | -0.0053 | -0.0001 |
| multimodal | unbiased | effort_matched | RandomForest | +0.3323 | +0.0515 |
| multimodal | unbiased | effort_matched | XGBoost | +0.3518 | +0.0609 |
| multimodal | unbiased | effort_matched | maxnet_lqph | +0.0596 | +0.0088 |
| multimodal | unbiased | uniform | CatBoost | +0.1178 | +0.0141 |
| multimodal | unbiased | uniform | GAM_smooth | +0.0207 | +0.0123 |
| multimodal | unbiased | uniform | NeuralForest_POGrad_random | -0.0057 | -0.0002 |
| multimodal | unbiased | uniform | NeuralForest_POGrad_raw | -0.0056 | -0.0003 |
| multimodal | unbiased | uniform | NeuralForest_POGrad_teacher | -0.0053 | -0.0001 |
| multimodal | unbiased | uniform | RandomForest | +0.3323 | +0.0515 |
| multimodal | unbiased | uniform | XGBoost | +0.3518 | +0.0609 |
| multimodal | unbiased | uniform | maxnet_lqph | +0.0596 | +0.0088 |
| nonlinear_interaction | effort_biased | effort_matched | CatBoost | -0.2644 | -0.0592 |
| nonlinear_interaction | effort_biased | effort_matched | GAM_smooth | +0.5088 | +0.1319 |
| nonlinear_interaction | effort_biased | effort_matched | NeuralForest_POGrad_random | +0.0070 | +0.0024 |
| nonlinear_interaction | effort_biased | effort_matched | NeuralForest_POGrad_raw | +0.0331 | +0.0093 |
| nonlinear_interaction | effort_biased | effort_matched | NeuralForest_POGrad_teacher | +0.0942 | +0.0276 |
| nonlinear_interaction | effort_biased | effort_matched | RandomForest | -0.2116 | -0.0546 |
| nonlinear_interaction | effort_biased | effort_matched | XGBoost | -0.2638 | -0.0505 |
| nonlinear_interaction | effort_biased | effort_matched | maxnet_lqph | -0.2759 | -0.0562 |
| nonlinear_interaction | effort_biased | uniform | CatBoost | -0.3729 | -0.0889 |
| nonlinear_interaction | effort_biased | uniform | GAM_smooth | +0.4401 | +0.1189 |
| nonlinear_interaction | effort_biased | uniform | NeuralForest_POGrad_random | +0.0029 | +0.0021 |
| nonlinear_interaction | effort_biased | uniform | NeuralForest_POGrad_raw | +0.0086 | +0.0019 |
| nonlinear_interaction | effort_biased | uniform | NeuralForest_POGrad_teacher | +0.0694 | +0.0247 |
| nonlinear_interaction | effort_biased | uniform | RandomForest | -0.3124 | -0.0790 |
| nonlinear_interaction | effort_biased | uniform | XGBoost | -0.3359 | -0.0675 |
| nonlinear_interaction | effort_biased | uniform | maxnet_lqph | -0.3222 | -0.0642 |
| nonlinear_interaction | unbiased | effort_matched | CatBoost | -0.2744 | -0.0500 |
| nonlinear_interaction | unbiased | effort_matched | GAM_smooth | +0.5490 | +0.1566 |
| nonlinear_interaction | unbiased | effort_matched | NeuralForest_POGrad_random | -0.0089 | -0.0017 |
| nonlinear_interaction | unbiased | effort_matched | NeuralForest_POGrad_raw | +0.0037 | +0.0007 |
| nonlinear_interaction | unbiased | effort_matched | NeuralForest_POGrad_teacher | +0.0047 | +0.0005 |
| nonlinear_interaction | unbiased | effort_matched | RandomForest | -0.1689 | -0.0319 |
| nonlinear_interaction | unbiased | effort_matched | XGBoost | -0.2110 | -0.0305 |
| nonlinear_interaction | unbiased | effort_matched | maxnet_lqph | -0.1964 | -0.0271 |
| nonlinear_interaction | unbiased | uniform | CatBoost | -0.2744 | -0.0500 |
| nonlinear_interaction | unbiased | uniform | GAM_smooth | +0.5490 | +0.1566 |
| nonlinear_interaction | unbiased | uniform | NeuralForest_POGrad_random | -0.0089 | -0.0017 |
| nonlinear_interaction | unbiased | uniform | NeuralForest_POGrad_raw | +0.0037 | +0.0007 |
| nonlinear_interaction | unbiased | uniform | NeuralForest_POGrad_teacher | +0.0047 | +0.0005 |
| nonlinear_interaction | unbiased | uniform | RandomForest | -0.1689 | -0.0319 |
| nonlinear_interaction | unbiased | uniform | XGBoost | -0.2110 | -0.0305 |
| nonlinear_interaction | unbiased | uniform | maxnet_lqph | -0.1964 | -0.0271 |

## Guardrails and unresolved questions

- The native model fits an IPP-negative-gradient regression forest.
  It does NOT fit false binary biological absences.
- Spatial outer blocks and a separate spatial inner block mitigate
  spatially optimistic selection but do NOT create 72 independent
  real-world replicate species. Inner-fold selection is a single
  fixed 1-of-4 training-block rule, not equal-budget nested tuning.
- MaxEnt LQPH, GAM, and CatBoost are substantive competitors;
  superiority should be examined across ALL relevant metrics.
- Variation in sampling/detection effort is a separate causal
  source of bias; an effort-matched background is not a guarantee
  of unbiased occurrence observation.
- Oracle correlation and oracle IPP skill are known only for
  virtual species. Independent empirical field validation
  still requires new true ecological survey evidence.
- Existing Ixodes hard pseudoabsences sampled outside an occupied
  FAMD convex hull are NOT an unbiased accessible-background
  sample. They are a separate sensitivity design.
- No true occurrence prevalence, disease risk, abundance or
  calibrated occupancy can be inferred from these scores.
- Fold-level dispersion is exploratory because fold training
  sets overlap and the simulator families were previously
  examined during method development.

See all_runs.csv, group_means.csv, stage_selection.csv,
paired_effect.csv and paired_summary.csv for exhaustive results.
