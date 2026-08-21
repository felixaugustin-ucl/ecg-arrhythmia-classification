# Figures

Generated during the original analysis and reproduced here at 1600 px wide.
Six of these are embedded in the top-level [README](../../README.md); the rest
are the per-model detail behind the summary tables.

## Data and problem framing

| File | Shows |
|---|---|
| `01-condition-frequency.png` | Frequency of every condition — the long tail |
| `02-condition-categories.png` | Conditions grouped into five categories |
| `03-conditional-probability.png` | P(condition B \| condition A) for the top 25 |
| `04-twelve-lead-example.png` | One record across leads V2–V5 |
| `05-demographics.png` | ECG patterns compared across age and sex |
| `06-feature-correlation.png` | Correlation matrix of the engineered features |

## Method

| File | Shows |
|---|---|
| `07-architecture-comparison.png` | CNN_250Hz ResNet1D beside [Weimann & Conrad's ResNet](https://doi.org/10.1038/s41598-021-84374-8) |
| `08-decision-algorithm.png` | Tuning → metric ranking → recall-constrained cutoff |
| `09-pca-feature-variance.png` | 13 components retain 96.0% of *feature* variance |
| `10-pca-signal-variance.png` | 100 components retain only 52.0% of *signal* variance |

## Per-model results

Each model has a tuning history (search metric across trials) and a cutoff
sweep (all six metrics against probability threshold, chosen cutoff dashed).

| Model | Tuning | Cutoff sweep |
|---|---|---|
| SGD_18F | `11-tuning-sgd-18f.png` | `12-cutoff-sgd-18f.png` |
| SGD_13C | `13-tuning-sgd-13c.png` | `14-cutoff-sgd-13c.png` |
| XGB_18F | `15-tuning-xgb-18f.png` | `16-cutoff-xgb-18f.png` |
| XGB_100F | `17-tuning-xgb-100f.png` | `18-cutoff-xgb-100f.png` |
| CNN_250Hz | — (not Ray-tuned) | `19-cutoff-cnn-250hz.png` |
