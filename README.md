# Weibo Interaction Prediction Baseline

This workspace contains a reproducible baseline for the Tianchi Weibo interaction prediction task.

## What the task is

For each post, predict three targets:

- `forward_count`
- `comment_count`
- `like_count`

The local files in this project do not match the dates written in `instruction.md`.

- `instruction.md` says training data is from `2014-07-01` to `2014-12-31` and prediction data is from `2015-01-01` to `2015-01-31`
- the actual local training file spans `2015-02-01` to `2015-07-31`
- the actual local prediction file spans `2015-08-01` to `2015-08-31`

The baseline uses the real local files as the source of truth.

## Data layout in this repository

GitHub does not accept files larger than `100 MB`, so the training dataset is stored in split parts:

- `Weibo Data/weibo_train_data.partaa`
- `Weibo Data/weibo_train_data.partab`
- `Weibo Data/weibo_train_data.partac`
- `Weibo Data/weibo_train_data.partad`
- `Weibo Data/weibo_train_data.partae`

The loader reads either:

- the original monolithic `Weibo Data/weibo_train_data.txt` if it exists locally
- or the split `part*` files automatically when the large file is absent

## What is implemented

- time-based validation split
- user history mean baseline
- user history features
- simple text and metadata features
- character n-gram hashing for Chinese text
- optional three separate `SGDRegressor` models blended with history baseline
- submission export in Tianchi format
- two local score variants for the ambiguous `sign()` definition

## Environment

Create the environment:

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -r requirements.txt
```

Packages were already installed once in this workspace during setup.

## Run

Full run:

```bash
.venv/bin/python train_baseline.py
```

Run with the optional text model:

```bash
.venv/bin/python train_baseline.py --enable-text-model
```

Faster smoke run:

```bash
.venv/bin/python train_baseline.py --max-train-rows 200000 --max-predict-rows 50000
```

Outputs:

- `artifacts/submission_baseline.txt`
- `artifacts/validation_report.json`

## Notes on the metric

The archived formula image was incomplete, so the local evaluator uses the most likely interpretation from public writeups:

- `deviation = abs(pred - real) / (real + 5)`
- `precision_i = 1 - 0.5 * deviation_f - 0.25 * deviation_c - 0.25 * deviation_l`

Two variants are reported:

- `indicator`: counts a post only when `precision_i > 0.8`
- `math_sign`: uses the mathematical sign `-1/0/1`

## Baseline choice

The default model is intentionally conservative:

- it predicts each user's average historical interactions
- this is fast, stable, and already aligns with public writeups of the task
- the text model is optional because it is slower and may or may not help on a given split

## Where Alibaba Cloud may help

The Alibaba Cloud coupon and Tianchi compute resources are useful if you want to:

- run larger models on the full dataset repeatedly
- add heavier NLP pipelines
- tune multiple experiments in parallel

Tongyi Lingma can also help if you want IDE-side coding assistance, but it is not required for this baseline.
