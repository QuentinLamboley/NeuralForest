# Comparaison des temps de calcul, NeuralForest v0.4

Mesures réelles sur six graines, 19 scénarios et 17 méthodes. Valeurs moyennes en secondes, calculées sur les durées consignées dans summary.csv.

La sélection couvre l'entraînement des candidats, la recherche d'hyperparamètres, la sélection des experts et la calibration lorsqu'elles sont activées. L'ajustement final correspond au modèle réentraîné après la sélection. La prédiction est mesurée sur les observations de test indépendantes. Les durées de génération des données et d'installation des dépendances sont exclues.

Les bibliothèques numériques sont limitées à un thread. Les groupes ont toutefois été exécutés sur des runners GitHub distincts. Ces durées ne sont donc pas des mesures matériellement normalisées.

## Durées moyennes sur les 19 scénarios

| Algorithme | Sélection (s) | Ajustement final (s) | Prédiction (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: |
| Dummy | 0.013 | 0.002 | 0.001 | 0.016 |
| Logistic | 0.031 | 0.004 | 0.001 | 0.037 |
| MLP | 0.112 | 0.050 | 0.002 | 0.164 |
| XGBoost | 0.156 | 0.055 | 0.003 | 0.215 |
| LightGBM | 0.162 | 0.073 | 0.015 | 0.249 |
| SVM_RBF | 0.146 | 0.083 | 0.028 | 0.257 |
| ranger | 0.221 | 0.121 | 0.035 | 0.377 |
| HistGradientBoosting | 0.278 | 0.132 | 0.013 | 0.423 |
| NeuralForest_CART | 0.299 | 0.160 | 0.014 | 0.473 |
| ExtraTrees | 0.326 | 0.154 | 0.025 | 0.506 |
| CatBoost | 0.524 | 0.203 | 0.002 | 0.729 |
| NeuralForest_Differentiable | 0.584 | 0.159 | 0.016 | 0.760 |
| AdaBoost | 0.502 | 0.285 | 0.016 | 0.804 |
| NeuralForest_Calibrated | 0.633 | 0.159 | 0.017 | 0.808 |
| NeuralForest_Adaptive | 0.639 | 0.159 | 0.017 | 0.814 |
| GradientBoosting | 0.537 | 0.301 | 0.003 | 0.841 |
| RandomForest | 0.632 | 0.359 | 0.021 | 1.012 |

## Détails par situation

Le tableau ci-dessous juxtapose des méthodes représentatives. Les 323 combinaisons algorithme-scénario figurent dans summary.csv, ainsi que les 1 938 évaluations individuelles et leurs durées dans all_runs.csv.

| Situation | Algorithme | Score principal | Sélection (s) | Ajustement (s) | Prédiction (s) | Total (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rare_1pct | NeuralForest_Calibrated | 0.186 | 2.896 | 0.737 | 0.022 | 3.655 |
| rare_1pct | NeuralForest_Adaptive | 0.186 | 2.901 | 0.735 | 0.021 | 3.658 |
| rare_1pct | NeuralForest_Differentiable | 0.166 | 2.807 | 0.728 | 0.021 | 3.555 |
| rare_1pct | NeuralForest_CART | 0.157 | 1.306 | 0.728 | 0.017 | 2.050 |
| rare_1pct | CatBoost | 0.151 | 0.569 | 0.222 | 0.002 | 0.794 |
| rare_1pct | XGBoost | 0.146 | 0.172 | 0.082 | 0.004 | 0.259 |
| rare_1pct | Logistic | 0.196 | 0.037 | 0.008 | 0.001 | 0.046 |
| rare_1pct | ranger | 0.161 | 0.589 | 0.368 | 0.038 | 0.996 |
| rare_3pct | NeuralForest_Calibrated | 0.267 | 0.863 | 0.207 | 0.015 | 1.085 |
| rare_3pct | NeuralForest_Adaptive | 0.267 | 0.867 | 0.207 | 0.016 | 1.090 |
| rare_3pct | NeuralForest_Differentiable | 0.249 | 0.845 | 0.232 | 0.015 | 1.092 |
| rare_3pct | NeuralForest_CART | 0.235 | 0.379 | 0.204 | 0.012 | 0.596 |
| rare_3pct | CatBoost | 0.238 | 0.319 | 0.152 | 0.002 | 0.474 |
| rare_3pct | XGBoost | 0.205 | 0.100 | 0.046 | 0.004 | 0.149 |
| rare_3pct | Logistic | 0.290 | 0.024 | 0.004 | 0.001 | 0.028 |
| rare_3pct | ranger | 0.218 | 0.196 | 0.097 | 0.032 | 0.325 |
| presence_bg_1to5 | NeuralForest_Calibrated | 0.985 | 0.664 | 0.198 | 0.023 | 0.884 |
| presence_bg_1to5 | NeuralForest_Adaptive | 0.985 | 0.703 | 0.198 | 0.023 | 0.923 |
| presence_bg_1to5 | NeuralForest_Differentiable | 0.973 | 0.615 | 0.196 | 0.022 | 0.834 |
| presence_bg_1to5 | NeuralForest_CART | 0.973 | 0.366 | 0.195 | 0.021 | 0.582 |
| presence_bg_1to5 | CatBoost | 0.950 | 0.429 | 0.167 | 0.002 | 0.598 |
| presence_bg_1to5 | XGBoost | 0.907 | 0.146 | 0.070 | 0.004 | 0.220 |
| presence_bg_1to5 | Logistic | 0.963 | 0.030 | 0.004 | 0.001 | 0.036 |
| presence_bg_1to5 | ranger | 0.955 | 0.264 | 0.155 | 0.050 | 0.468 |
| large_n | NeuralForest_Calibrated | 0.771 | 1.519 | 0.346 | 0.020 | 1.885 |
| large_n | NeuralForest_Adaptive | 0.771 | 1.526 | 0.344 | 0.020 | 1.890 |
| large_n | NeuralForest_Differentiable | 0.771 | 1.529 | 0.346 | 0.020 | 1.894 |
| large_n | NeuralForest_CART | 0.770 | 0.615 | 0.343 | 0.017 | 0.976 |
| large_n | CatBoost | 0.779 | 0.344 | 0.098 | 0.002 | 0.444 |
| large_n | XGBoost | 0.774 | 0.123 | 0.036 | 0.003 | 0.163 |
| large_n | Logistic | 0.743 | 0.023 | 0.003 | 0.001 | 0.027 |
| large_n | ranger | 0.765 | 0.543 | 0.322 | 0.058 | 0.924 |
| nonlinear_xor | NeuralForest_Calibrated | 0.984 | 0.340 | 0.103 | 0.013 | 0.457 |
| nonlinear_xor | NeuralForest_Adaptive | 0.984 | 0.381 | 0.104 | 0.013 | 0.498 |
| nonlinear_xor | NeuralForest_Differentiable | 0.984 | 0.314 | 0.104 | 0.013 | 0.431 |
| nonlinear_xor | NeuralForest_CART | 0.983 | 0.208 | 0.106 | 0.012 | 0.326 |
| nonlinear_xor | CatBoost | 0.997 | 0.302 | 0.086 | 0.002 | 0.390 |
| nonlinear_xor | XGBoost | 0.993 | 0.098 | 0.040 | 0.003 | 0.141 |
| nonlinear_xor | Logistic | 0.504 | 0.022 | 0.003 | 0.001 | 0.026 |
| nonlinear_xor | ranger | 0.989 | 0.159 | 0.072 | 0.030 | 0.261 |
| identical_100 | NeuralForest_Calibrated | 0.698 | 0.344 | 0.092 | 0.015 | 0.443 |
| identical_100 | NeuralForest_Adaptive | 0.698 | 0.292 | 0.092 | 0.016 | 0.401 |
| identical_100 | NeuralForest_Differentiable | 0.698 | 0.247 | 0.090 | 0.015 | 0.352 |
| identical_100 | NeuralForest_CART | 0.706 | 0.185 | 0.091 | 0.012 | 0.289 |
| identical_100 | CatBoost | 0.718 | 0.513 | 0.142 | 0.002 | 0.657 |
| identical_100 | XGBoost | 0.759 | 0.182 | 0.050 | 0.003 | 0.235 |
| identical_100 | Logistic | 0.701 | 0.044 | 0.005 | 0.001 | 0.051 |
| identical_100 | ranger | 0.719 | 0.181 | 0.086 | 0.033 | 0.301 |

## Limites

NeuralForest v0.4 a été entraîné avec deux tours neuronaux dans le benchmark principal. Le mécanisme de transport des poids entre réseaux successive ne s'active que lorsque des tours supplémentaires sont utilisés. Il est vérifié par un test numérique, mais le benchmark principal ne constitue pas une ablation de ce mécanisme. Les critères AUC, Boyce et average precision ne réagissent pas nécessairement à une calibration monotone.

Le temps ne suffit pas à définir un modèle optimal. Une performance supérieure sur une seule métrique ou un seul environnement de calcul ne vaut pas supériorité générale.
