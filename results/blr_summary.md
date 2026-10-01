# Bayesian linear regression: test results

Reproduce with `python -m src.experiments.blr`. Splits from data/checkpoints/first_innings/: train <= 2022, validation 2023 (picks tau2), test >= 2024.

`ecdf_*`: the ECDF baseline, i.e. the empirical distribution of training totals bucketed like the model's forecasts; `skill` is relative to it, and `ecdf_n_zero_prob` counts test innings whose bucket it gives probability 0. `*_opt`: model-implied optimum. `marg_*`: no-features optimum, the entropy of the bucketed ECDF of test totals.

|                   |   Before ball 1 |   After over 6 |   After over 10 |   After over 15 |
|:------------------|----------------:|---------------:|----------------:|----------------:|
| n_train           |           1,515 |          1,515 |           1,515 |           1,495 |
| n_val             |             331 |            331 |             329 |             324 |
| n_test            |           1,473 |          1,473 |           1,471 |           1,433 |
| tau2              |          0.1000 |         0.1000 |          0.1000 |          0.1000 |
| n_features        |              19 |             31 |              31 |              31 |
| nu                |       1515.0200 |      1515.0200 |       1515.0200 |       1495.0200 |
| ecdf_rps          |         26.5680 |        26.5680 |         26.4628 |         25.2174 |
| skill             |          0.2006 |         0.4304 |          0.5540 |          0.6978 |
| ecdf_n_zero_prob  |              19 |             19 |              19 |              27 |
| rps_se            |          0.4380 |         0.3063 |          0.2272 |          0.1368 |
| rps               |         21.2396 |        15.1324 |         11.8015 |          7.6200 |
| rps_opt           |         20.1514 |        15.2733 |         12.4472 |          8.1365 |
| log               |          2.7609 |         2.4218 |          2.1669 |          1.7306 |
| log_opt           |          2.6882 |         2.4079 |          2.1986 |          1.7604 |
| mse               |       1447.4009 |       737.6592 |        444.3201 |        181.7092 |
| mse_opt           |       1260.2004 |       717.7818 |        472.6818 |        193.4016 |
| hit               |          0.1134 |         0.1656 |          0.1931 |          0.2715 |
| hit_opt           |          0.1117 |         0.1473 |          0.1804 |          0.2752 |
| bias              |          4.6048 |         0.6576 |         -1.1148 |         -0.9568 |
| cov90             |          0.8832 |         0.8988 |          0.9137 |          0.9246 |
| log_max           |         10.4502 |        12.5939 |          9.6221 |          6.8885 |
| n_above_train_max |              15 |             15 |              15 |              15 |
| marg_rps          |         26.3522 |        26.3522 |         26.2592 |         25.0795 |
| marg_log          |          2.9479 |         2.9479 |          2.9440 |          2.8955 |
| marg_mse          |       2210.2261 |      2210.2261 |       2192.5024 |       1991.9501 |

## Demo: match 1415755

| checkpoint    | score   |   best estimate | bucket   |   P(bucket) |   actual |   P(actual bucket) |
|:--------------|:--------|----------------:|:---------|------------:|---------:|-------------------:|
| Before ball 1 | 0/0     |        177.4476 | 170-179  |      0.1117 | 176.0000 |             0.1117 |
| After over 6  | 45/3    |        164.2324 | 160-169  |      0.1478 | 176.0000 |             0.1375 |
| After over 10 | 75/3    |        168.7444 | 160-169  |      0.1788 | 176.0000 |             0.1760 |
| After over 15 | 118/4   |        171.5494 | 170-179  |      0.2750 | 176.0000 |             0.2750 |
