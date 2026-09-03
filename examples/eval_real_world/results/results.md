# Evaluation Results

## Mean ± Std Metrics

| Dataset | Model | Tier | RMSE | MAE | R^2 | Train(s) | PeakMem(MB) | Refresh | Cond(kappa) |
|---------|-------|------|------|-----|-----|----------|-------------|---------|-------------|
| WineQuality | Kernos | Small | 0.823626 ± 0.000000 | 0.638752 ± 0.000000 | 0.260141 ± 0.000000 | 1.1107 ± 0.0000 | 126.58 ± 0.00 | 1.0 ± 0.0 | 2.74e+04 ± 0.00e+00 |
| WineQuality | Ridge | Small | 0.822297 ± 0.000000 | 0.641703 ± 0.000000 | 0.262528 ± 0.000000 | 0.0001 ± 0.0000 | 0.01 ± 0.00 | 0.0 ± 0.0 | nan ± nan |
| WineQuality | Nystrom | Small | 0.884019 ± 0.000000 | 0.688999 ± 0.000000 | 0.147662 ± 0.000000 | 0.0378 ± 0.0000 | 102.27 ± 0.00 | 0.0 ± 0.0 | nan ± nan |
| WineQuality | RFF | Small | 0.885522 ± 0.000000 | 0.690962 ± 0.000000 | 0.144761 ± 0.000000 | 0.0312 ± 0.0000 | 53.33 ± 0.00 | 0.0 ± 0.0 | nan ± nan |
| WineQuality | K-NoRefresh | Small | 0.785946 ± 0.000000 | 0.612502 ± 0.000000 | 0.326288 ± 0.000000 | 1.1940 ± 0.0000 | 126.58 ± 0.00 | 0.0 ± 0.0 | 2.82e+04 ± 0.00e+00 |
| WineQuality | K-NoHysteresis | Small | 0.823626 ± 0.000000 | 0.638752 ± 0.000000 | 0.260141 ± 0.000000 | 0.9802 ± 0.0000 | 126.58 ± 0.00 | 1.0 ± 0.0 | 2.74e+04 ± 0.00e+00 |
| WineQuality | K-NoCooldown | Small | 0.823626 ± 0.000000 | 0.638752 ± 0.000000 | 0.260141 ± 0.000000 | 1.0328 ± 0.0000 | 126.58 ± 0.00 | 1.0 ± 0.0 | 2.74e+04 ± 0.00e+00 |
| WineQuality | K-NoResidAnchors | Small | 0.834837 ± 0.000000 | 0.650104 ± 0.000000 | 0.239863 ± 0.000000 | 0.9356 ± 0.0000 | 126.58 ± 0.00 | 1.0 ± 0.0 | 2.76e+04 ± 0.00e+00 |
| WineQuality | K-NoOrthog | Small | 0.821500 ± 0.000000 | 0.642878 ± 0.000000 | 0.263957 ± 0.000000 | 0.7917 ± 0.0000 | 126.58 ± 0.00 | 1.0 ± 0.0 | 3.61e+04 ± 0.00e+00 |
| WineQuality | K-NoDivPenalty | Small | 0.823626 ± 0.000000 | 0.638752 ± 0.000000 | 0.260141 ± 0.000000 | 1.1209 ± 0.0000 | 126.58 ± 0.00 | 1.0 ± 0.0 | 2.74e+04 ± 0.00e+00 |
| WineQuality | K-NoFreeze | Small | 0.823626 ± 0.000000 | 0.638752 ± 0.000000 | 0.260141 ± 0.000000 | 1.0302 ± 0.0000 | 126.58 ± 0.00 | 1.0 ± 0.0 | 2.74e+04 ± 0.00e+00 |

## Stability-over-Refits

| Dataset | Model | Tier | Var(RMSE) | Var(Time) | Var(Cond) |
|---------|-------|------|-----------|-----------|-----------|
| WineQuality | Kernos | Small | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| WineQuality | Ridge | Small | 0.000000e+00 | 0.000000e+00 | nan |
| WineQuality | Nystrom | Small | 0.000000e+00 | 0.000000e+00 | nan |
| WineQuality | RFF | Small | 0.000000e+00 | 0.000000e+00 | nan |
| WineQuality | K-NoRefresh | Small | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| WineQuality | K-NoHysteresis | Small | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| WineQuality | K-NoCooldown | Small | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| WineQuality | K-NoResidAnchors | Small | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| WineQuality | K-NoOrthog | Small | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| WineQuality | K-NoDivPenalty | Small | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| WineQuality | K-NoFreeze | Small | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |