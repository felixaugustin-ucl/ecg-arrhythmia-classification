# ECG Arrhythmia Classification

**Multi-label prediction of cardiac conditions from 12-lead ECG, comparing handcrafted signal features against learned representations.**

Five models on 45,152 clinical ECG recordings from Shaoxing People's Hospital,
labelled across 94 diagnostic categories: three built on handcrafted
signal-processing features, two on the raw waveform.

A compact 1D ResNet wins on every metric, reaching 0.71 micro F1 on held-out
data. The more interesting number is its macro F1 of 0.21 — the gap between
the two is the whole story of this dataset, and it is the reason the project
concludes with a screening tool rather than a diagnostic one.

---

## The problem

Each recording carries zero or more SNOMED-CT diagnosis codes, so this is
multi-label, not multi-class: a patient can be in atrial fibrillation *and*
have a bundle branch block. The label distribution is severely long-tailed —
a handful of rhythms account for most of the mass, while many conditions
appear in a few dozen records out of 45,000.

That shape drives three decisions that run through the whole project:

**Micro and macro F1 are always reported together.** Micro is dominated by the
common rhythms; macro weights every condition equally. A model can improve one
while degrading the other, so neither is quoted alone.

**The decision threshold is tuned, not left at 0.5.** With labels this
imbalanced, the default cutoff is arbitrary. Each model's operating point is
selected by sweeping cutoffs against cross-validated predictions, breaking
ties toward recall — the safer direction to err when the cost of a missed
arrhythmia exceeds the cost of a false alarm.

**Thresholds are never selected on the test set.** The sweep runs on training
folds (or, for the CNN, on a validation split carved out of the training
portion). Selecting a cutoff on test data would leak it and inflate the score.

---

## Results

94 diagnostic labels across 45,152 recordings, split 36,121 train / 9,031 test.

### Model comparison at each model's selected cutoff

| Model | Features | Cutoff | F1 micro | F1 macro | Prec. micro | Rec. micro | Prec. macro | Rec. macro |
|---|---|---|---|---|---|---|---|---|
| SGD_18F | 18 handcrafted | 0.45 | 0.1153 | 0.0733 | 0.0621 | 0.8085 | 0.0472 | 0.6098 |
| SGD_13C | 13 PCA of features | 0.45 | 0.1030 | 0.0672 | 0.0550 | 0.7965 | 0.0430 | 0.6217 |
| XGB_18F | 18 handcrafted | 0.30 | 0.5489 | 0.1045 | 0.6270 | 0.4881 | 0.1576 | 0.0934 |
| XGB_100F | 100 signal PCA | 0.30 | 0.5051 | 0.0574 | 0.6286 | 0.4222 | 0.1121 | 0.0515 |
| **CNN_250Hz** | raw 12-lead signal | 0.35 | **0.7196** | **0.2174** | 0.7476 | 0.6937 | 0.2742 | 0.2149 |

These are cross-validated (for the CNN, validation-split) scores — the basis on
which the models were compared.

### CNN_250Hz on the held-out test set

| Cutoff | F1 micro | F1 macro | Prec. micro | Rec. micro | Prec. macro | Rec. macro |
|---|---|---|---|---|---|---|
| 0.35 | 0.7147 | 0.2078 | 0.7478 | 0.6845 | 0.2879 | 0.2078 |

The CNN was the selected final model and the only one carried through to the
held-out test set. Validation-to-test drop is 0.005 in micro F1, so the model
generalises and the chosen cutoff transfers.

### What the numbers show

**Linear models fail outright.** One-vs-rest SGD reaches micro F1 of 0.12 at a
cutoff tuned for ~0.80 recall. A linear decision boundary cannot separate
arrhythmias whose signature is waveform morphology, and compressing the
features first (SGD_13C) makes it slightly worse rather than better.

**Handcrafted features beat an unsupervised basis over the raw signal.**
XGB_18F edges out XGB_100F by 0.044 micro F1 and 0.047 macro. The reason is
visible in the variance: the 13 components in SGD_13C retain 96.0% of feature
variance, but the 100 Incremental-PCA components in XGB_100F capture only
**52.0%** of raw waveform variance. PCA is the wrong tool for this signal —
diagnostic information lives in localised morphology, not in the directions of
greatest global variance.

**The CNN wins because it learns temporal filters.** At 250 Hz each sample is
4 ms, so the stem's kernel of 15 spans ~60 ms and the residual blocks' kernel
of 7 spans ~28 ms — both on the scale of a QRS complex, which typically runs
60–120 ms. The architecture is matched to the physiology it needs to detect.

**Every model collapses on macro metrics, and that is the real finding.** The
best macro F1 is 0.22 against a micro F1 of 0.72. The label distribution is
severely long-tailed — some of the 94 conditions appear fewer than ten times —
so aggregate performance is carried by a handful of common rhythms. Rare
classes were deliberately retained rather than filtered, since they are often
the clinically significant ones, but predictions for them should be treated as
unreliable. `outputs/tables/*_per_label.csv` shows exactly which conditions the
model never predicts.

The practical reading: this is a **screening** tool for well-represented
arrhythmias, not a diagnostic replacement.

**On reproducibility.** Re-running will not reproduce these figures exactly.
Ray Tune's search is seeded but its trial scheduling is not fully
deterministic under parallel execution, and some CUDA kernels are
nondeterministic. Expect variation in the third decimal place.

## Repository layout

```
src/ecg/
  config.py            Paths, constants, seeding, YAML config loading
  data/
    wfdb.py            WFDB .mat/.hea reading into a padded signal tensor
    labels.py          SNOMED-CT code normalisation and condition lookup
    leads.py           Lead names parsed from headers, not assumed
    resample.py        Polyphase 500 Hz to 250 Hz resampling
  features/
    waveform.py        R-peak detection, PR/QRS/QT intervals, HRV statistics
    spectral.py        Welch PSD, LF/HF band powers, spectral entropy
    complexity.py      Approximate/sample entropy, DFA, Higuchi dimension
    extract.py         Assembles the 22-feature vector per record
  preprocessing.py     Imputation, label binarisation, train/test splitting
  decomposition.py     PCA on features; Incremental PCA on raw signal
  models/
    registry.py        Maps model names to build/tune/feature-source specs
    sklearn_models.py  SGD and XGBoost specs (all four registered here)
    cnn.py             1D ResNet, defined once
  tuning/ray_search.py One Ray Tune harness for every model
  evaluation/
    metrics.py         Multi-label metrics, micro and macro
    thresholds.py      Cutoff sweeps and operating-point selection
  viz/                 One chart theme, one set of result charts

scripts/               Runnable pipeline stages, in order
configs/               Data settings and per-model frozen hyperparameters
tests/                 pytest suite (31 tests, no dataset required)
```

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then, in order:

```bash
python scripts/01_build_dataset.py --limit 500
```

```bash
python scripts/02_extract_features.py
```

```bash
python scripts/03_train.py --model xgb_18f
```

`--limit` caps how many records are read; drop it for the full run. Every
model goes through the same script:

```bash
python scripts/03_train.py --model sgd_18f
```

The raw-signal models need the resampling and PCA stage first:

```bash
python scripts/04_signal_pca.py --components 100
```

```bash
python scripts/05_train_cnn.py --epochs 8
```

```bash
python scripts/06_compare_models.py
```

### Hyperparameters are frozen by default

`configs/models/*.yaml` holds the winning configuration from each search, so a
run reproduces in minutes instead of re-searching for hours. To search again:

```bash
python scripts/03_train.py --model xgb_18f --retune --trials 50
```

That prints the configuration it found and tells you to update the YAML —
deliberately, rather than by pasting values back into the code.

### Tests

```bash
python -m pytest
```

The suite runs without the dataset: it uses synthetic two-rhythm signals to
exercise the full path and to check that the feature extractor recovers a
known heart rate.

---

## Getting the data

The **ECG Arrhythmia Dataset** from PhysioNet — about 5.3 GB uncompressed, and
not included here.

<https://physionet.org/content/ecg-arrhythmia/1.0.0/>

```bash
wget -r -N -c -np https://physionet.org/files/ecg-arrhythmia/1.0.0/
```

Place it at `./ecg_data/`, or point `ECG_DATA_ROOT` at an existing copy:

```bash
export ECG_DATA_ROOT=/Volumes/external/ecg_data
```

The pipeline expects `ecg_data/WFDBRecords/`, `ConditionNames_SNOMED-CT.csv`
and `Remaining_DX_Codes_SNOMED_Labels.csv`.

---

## Method notes and limitations

**Fiducial points are approximated, not delineated.** P, QRS and T boundaries
are located by fixed physiological offsets from each detected R peak rather
than by a proper delineation algorithm. This is fast enough for 45,000 records
and robust to noise, but the intervals should be read as population-level
descriptors, not as clinical measurements on any individual trace.

**Features come from one lead.** Handcrafted extraction uses lead II only
(index 1), where the P wave is typically clearest. The raw-signal models use
all twelve.

**250 Hz for the raw-signal models.** Halves memory for the Incremental PCA
pass and the CNN, and the diagnostic content of a surface ECG sits well below
the 125 Hz Nyquist limit this affords. Resampling is polyphase, so decimation
anti-aliases rather than folding high-frequency noise into the band of
interest.

**Two features are dropped before modelling.** `higuchi_fractal_dimension` and
`dfa_scaling_exponent` are both RR-derived and undefined for short records, so
they carry the most missingness of the 22. The rest are median-imputed.

**Not a clinical tool.** This is a methods study on a public research dataset.
Nothing here is validated for diagnostic use.

---

## Licence

MIT — see [LICENSE](LICENSE).

The PhysioNet dataset carries its own licence; consult the source before
redistributing any part of it.
