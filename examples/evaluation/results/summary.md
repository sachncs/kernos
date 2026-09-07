# Kernos evaluation summary

> Sweep over 9 dataset(s) × tier(s) Small.

## Headline

| Metric | Value |
|---|---|
| Kernos mean RMSE | 0.558518 |
| Best baseline mean RMSE | 0.607019 (Ridge) |
| Win rate vs Ridge | 6 / 9 |
| Win rate vs Nystrom | 8 / 9 |
| Win rate vs RFF | 8 / 9 |
| Mean ΔRMSE vs best baseline | -7.99% |

## Per-dataset ranking (1 = best)

| Dataset | Kernos rank | Top model | Worst model |
|---|---|---|---|
| Abalone | 6 / 11 | K-NoRefresh | K-NoOrthog |
| CPUActivity | 2 / 11 | K-NoRefresh | RFF |
| CaliforniaHousing | 3 / 11 | K-NoRefresh | Nystrom |
| Elevators | 5 / 11 | Ridge | RFF |
| HouseSales | 3 / 11 | K-NoRefresh | RFF |
| Kin8nm | 4 / 11 | K-NoRefresh | RFF |
| Superconduct | 3 / 11 | K-NoRefresh | Nystrom |
| WineQuality | 3 / 11 | K-NoRefresh | RFF |
| YearPredictionMSD | 3 / 11 | K-NoRefresh | RFF |

## Ablation impact (mean ΔRMSE % vs full Kernos)

| Ablation | Mean ΔRMSE % |
|---|---|
| K-NoRefresh | -6.29% |
| K-NoHysteresis | +0.00% |
| K-NoCooldown | +0.00% |
| K-NoResidAnchors | +0.29% |
| K-NoOrthog | +0.37% |
| K-NoDivPenalty | +0.00% |
| K-NoFreeze | +0.00% |
