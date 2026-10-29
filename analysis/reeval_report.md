# Selection-rule re-evaluation (ISAAC, 65 runs)

Checkpoints scored through the trainer's INTERNAL eval path; per-run control = best_model.pt re-scored and compared to results.json exactly.

## Problems (0)

- none: every control reproduced results.json exactly and every rule checkpoint loaded.


## Per-cell aggregate (mean / fold spread of test BA)

| group | method | source | target | n | oracle_target_val mean | oracle_target_val spread | source_val mean | source_val spread | last_epoch mean | last_epoch spread | min_target_entropy mean | min_target_entropy spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | cdan | broad573 | 3mdad | 3 | 0.4224 | 0.0555 | 0.2000 | 0.0000 | 0.2000 | 0.0000 | 0.2000 | 0.0000 |
| E1 | cdan | broad573 | dmd | 3 | 0.5130 | 0.1880 | 0.2000 | 0.0000 | 0.1335 | 0.1995 | 0.2000 | 0.0000 |
| E1 | cdan | da573_random | 3mdad | 3 | 0.2061 | 0.0125 | 0.2000 | 0.0000 | 0.2000 | 0.0000 | 0.2000 | 0.0000 |
| E1 | cdan | da573_random | dmd | 3 | 0.5380 | 0.0378 | 0.3146 | 0.3439 | 0.2220 | 0.0661 | 0.2000 | 0.0000 |
| E1 | cdan | narrow573 | 3mdad | 3 | 0.2593 | 0.0515 | 0.2338 | 0.0765 | 0.2000 | 0.0000 | 0.2000 | 0.0000 |
| E1 | cdan | narrow573 | dmd | 3 | 0.4922 | 0.1245 | 0.2028 | 0.0075 | 0.2000 | 0.0000 | 0.2026 | 0.0075 |
| E1 | cdan | obc | 3mdad | 3 | 0.2000 | 0.0000 | 0.2000 | 0.0000 | 0.1333 | 0.2000 | 0.2000 | 0.0000 |
| E1 | cdan | obc | dmd | 3 | 0.2709 | 0.0782 | 0.2000 | 0.0000 | 0.2000 | 0.0000 | 0.2000 | 0.0000 |
| E1 | dann | broad573 | 3mdad | 3 | 0.4025 | 0.0882 | 0.2000 | 0.0000 | 0.2021 | 0.0062 | 0.2103 | 0.0250 |
| E1 | dann | broad573 | dmd | 3 | 0.5127 | 0.0486 | 0.2003 | 0.0008 | 0.2001 | 0.0003 | 0.2023 | 0.0070 |
| E1 | dann | da573_random | 3mdad | 3 | 0.3862 | 0.0588 | 0.2000 | 0.0000 | 0.2000 | 0.0000 | 0.2000 | 0.0000 |
| E1 | dann | da573_random | dmd | 3 | 0.5503 | 0.0400 | 0.2006 | 0.0004 | 0.1584 | 0.1274 | 0.2013 | 0.0038 |
| E1 | dann | narrow573 | 3mdad | 3 | 0.2725 | 0.1471 | 0.2000 | 0.0000 | 0.2196 | 0.0588 | 0.2000 | 0.0000 |
| E1 | dann | narrow573 | dmd | 3 | 0.5494 | 0.0672 | 0.3958 | 0.4739 | 0.2002 | 0.0005 | 0.2769 | 0.2289 |
| E1 | dann | obc | 3mdad | 3 | 0.3243 | 0.0706 | 0.2235 | 0.0706 | 0.2118 | 0.0353 | 0.2000 | 0.0000 |
| E1 | dann | obc | dmd | 3 | 0.3120 | 0.0675 | 0.2041 | 0.0124 | 0.2013 | 0.0026 | 0.2000 | 0.0000 |
| E1 | source_only | broad573 | 3mdad | 3 | 0.5020 | 0.0176 | 0.4738 | 0.0199 | 0.4705 | 0.1474 | 0.4020 | 0.2176 |
| E1 | source_only | broad573 | dmd | 2 | 0.5543 | 0.0617 | 0.4918 | 0.0048 | 0.3971 | 0.0696 | 0.3714 | 0.0089 |
| E1 | source_only | da573_random | 3mdad | 3 | 0.4821 | 0.1654 | 0.4898 | 0.0471 | 0.3157 | 0.0706 | 0.2745 | 0.0471 |
| E1 | source_only | da573_random | dmd | 3 | 0.5318 | 0.0553 | 0.4114 | 0.0955 | 0.3116 | 0.0825 | 0.2842 | 0.0672 |
| E1 | source_only | obc | 3mdad | 3 | 0.3163 | 0.0765 | 0.2718 | 0.0706 | 0.2594 | 0.0746 | 0.2314 | 0.0882 |
| E1 | source_only | obc | dmd | 3 | 0.3436 | 0.0531 | 0.3022 | 0.0646 | 0.3029 | 0.0433 | 0.3011 | 0.0885 |
