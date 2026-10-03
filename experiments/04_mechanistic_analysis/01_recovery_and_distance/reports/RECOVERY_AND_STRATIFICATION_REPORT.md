# GERT Mechanistic Recovery & Stratification Analysis (Paper Section 4.6)

Evaluated with candidate budget Top-K = 15 across SpiderUnion, BirdUnion, and SynLink.

## 1. Overall Table CR Transition & Recovery (Table 6 & Table 7)

| Dataset | DR Table CR | GERT Table CR | ΔCR | Table Recovery | Query Gain | Query Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SpiderUnion | 92.25% | 94.98% | +2.74% | 46.03% | 52.94% | 1.48% |
| BirdUnion | 80.64% | 90.55% | +9.91% | 52.75% | 57.24% | 1.46% |
| SynLink | 64.40% | 78.80% | +14.40% | 49.16% | 49.44% | 4.97% |


## 2. SpiderUnion Recovery by FK Distance (Table 8)

| Dist. | Missed | Recovered | Rate (%) |
| :--- | :--- | :--- | :--- |
| 1-hop | 33 | 29 | 87.88% |
| 2-hop | 1 | 0 | 0.00% |
| 3+ hops | 0 | 0 | 0.00% |
| N/A | 0 | 0 | 0.00% |


## 2. BirdUnion Recovery by FK Distance (Table 8)

| Dist. | Missed | Recovered | Rate (%) |
| :--- | :--- | :--- | :--- |
| 1-hop | 216 | 178 | 82.41% |
| 2-hop | 10 | 5 | 50.00% |
| 3+ hops | 0 | 0 | 0.00% |
| N/A | 0 | 0 | 0.00% |


## 2. SynLink Recovery by FK Distance (Table 8)

| Dist. | Missed | Recovered | Rate (%) |
| :--- | :--- | :--- | :--- |
| 1-hop | 324 | 232 | 71.60% |
| 2-hop | 65 | 27 | 41.54% |
| 3+ hops | 0 | 0 | 0.00% |
| N/A | 0 | 0 | 0.00% |
