# Evaluation Results

## Mean ± Std Metrics

| Dataset | Model | Tier | RMSE | MAE | R^2 | Train(s) | PeakMem(MB) | Refresh | Cond(kappa) |
|---------|-------|------|------|-----|-----|----------|-------------|---------|-------------|
| WineQuality | Kernos | Small | 0.827752 ± 0.004126 | 0.649977 ± 0.011225 | 0.283901 ± 0.023760 | 1.0583 ± 0.0023 | 126.58 ± 0.00 | 1.0 ± 0.0 | 1.51e+04 ± 1.23e+04 |
| WineQuality | Ridge | Small | 0.827993 ± 0.005697 | 0.650705 ± 0.009002 | 0.283585 ± 0.021058 | 0.0001 ± 0.0000 | 0.01 ± 0.00 | 0.0 ± 0.0 | nan ± nan |
| WineQuality | Nystrom | Small | 0.906074 ± 0.022055 | 0.709006 ± 0.020006 | 0.142922 ± 0.004740 | 0.0380 ± 0.0007 | 102.27 ± 0.00 | 0.0 ± 0.0 | nan ± nan |
| WineQuality | RFF | Small | 0.923235 ± 0.037713 | 0.732162 ± 0.041200 | 0.110458 ± 0.034303 | 0.0320 ± 0.0002 | 53.33 ± 0.00 | 0.0 ± 0.0 | nan ± nan |
| WineQuality | K-NoRefresh | Small | 0.795432 ± 0.009486 | 0.623952 ± 0.011449 | 0.339049 ± 0.012760 | 1.0053 ± 0.0302 | 126.58 ± 0.00 | 0.0 ± 0.0 | 4.18e+05 ± 3.90e+05 |
| WineQuality | K-NoHysteresis | Small | 0.827752 ± 0.004126 | 0.649977 ± 0.011225 | 0.283901 ± 0.023760 | 1.0690 ± 0.0285 | 126.58 ± 0.00 | 1.0 ± 0.0 | 1.51e+04 ± 1.23e+04 |
| WineQuality | K-NoCooldown | Small | 0.827752 ± 0.004126 | 0.649977 ± 0.011225 | 0.283901 ± 0.023760 | 1.0409 ± 0.0079 | 126.58 ± 0.00 | 1.0 ± 0.0 | 1.51e+04 ± 1.23e+04 |
| WineQuality | K-NoResidAnchors | Small | 0.836082 ± 0.001245 | 0.656203 ± 0.006100 | 0.269213 ± 0.029349 | 0.8907 ± 0.0294 | 126.58 ± 0.00 | 1.0 ± 0.0 | 2.77e+04 ± 1.64e+02 |
| WineQuality | K-NoOrthog | Small | 0.827820 ± 0.006321 | 0.651531 ± 0.008653 | 0.283923 ± 0.019967 | 0.7520 ± 0.0218 | 126.58 ± 0.00 | 1.0 ± 0.0 | 1.99e+04 ± 1.62e+04 |
| WineQuality | K-NoDivPenalty | Small | 0.827752 ± 0.004126 | 0.649977 ± 0.011225 | 0.283901 ± 0.023760 | 1.0725 ± 0.0253 | 126.58 ± 0.00 | 1.0 ± 0.0 | 1.51e+04 ± 1.23e+04 |
| WineQuality | K-NoFreeze | Small | 0.827752 ± 0.004126 | 0.649977 ± 0.011225 | 0.283901 ± 0.023760 | 1.0114 ± 0.0422 | 126.58 ± 0.00 | 1.0 ± 0.0 | 1.51e+04 ± 1.23e+04 |

## Stability-over-Refits

| Dataset | Model | Tier | Var(RMSE) | Var(Time) | Var(Cond) |
|---------|-------|------|-----------|-----------|-----------|
| WineQuality | Kernos | Small | 4.126009e-03 | 2.342541e-03 | 1.227685e+04 |
| WineQuality | Ridge | Small | 5.696559e-03 | 4.687492e-06 | nan |
| WineQuality | Nystrom | Small | 2.205510e-02 | 6.745835e-04 | nan |
| WineQuality | RFF | Small | 3.771334e-02 | 1.814170e-04 | nan |
| WineQuality | K-NoRefresh | Small | 9.486356e-03 | 3.019492e-02 | 3.901114e+05 |
| WineQuality | K-NoHysteresis | Small | 4.126009e-03 | 2.847069e-02 | 1.227685e+04 |
| WineQuality | K-NoCooldown | Small | 4.126009e-03 | 7.875750e-03 | 1.227685e+04 |
| WineQuality | K-NoResidAnchors | Small | 1.245406e-03 | 2.939554e-02 | 1.643054e+02 |
| WineQuality | K-NoOrthog | Small | 6.320666e-03 | 2.179208e-02 | 1.620523e+04 |
| WineQuality | K-NoDivPenalty | Small | 4.126009e-03 | 2.534942e-02 | 1.227685e+04 |
| WineQuality | K-NoFreeze | Small | 4.126009e-03 | 4.221679e-02 | 1.227685e+04 |