# [Embodied Perspective: Evaluating Observer-Relative Spatial Reasoning in Vision-Language Models]

**[Authors: Anonymous]**  
[ARR, May 2026 Round]  
[[Paper](#)] &nbsp;|&nbsp; [[Dataset](#)]

---

## Overview

This repository contains the evaluation code for our benchmark that tests whether vision-language models (VLMs) can perform **perspective-taking** — reasoning about spatial relations from a described viewpoint rather than their own.

We evaluate models under three conditions:

| Type | Input | Description |
|------|-------|-------------|
| **A** | 30 exploration images | Observer activity described *without* naming the observer's furniture |
| **B** | 30 exploration images | Observer activity described *with* the furniture named |
| **C** | 1 observer image     | Factual spatial statement only (no viewpoint inference required) |

The metric is the normalised L2 distance between the predicted pixel coordinate and the ground-truth projection of the target object.

---

## Repository structure

```
evaluation/
├── run_vlm_eval.py          # Main evaluation runner
├── run_vlm_eval_ab0.py      # Ablation base (helper for ab4)
├── run_vlm_eval_ab4.py      # Ablation: chain-of-thought prompting
├── eval_results.py          # Compute metrics from predictions
├── add_final_prediction.py  # Re-parse raw responses (final strategy)
└── aggregate_runs.py        # Aggregate multiple runs into CSV
dataset/
└── README.md                # Dataset download instructions
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
```

For local Qwen inference, install the model weights separately (see [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)) and pass `--qwen-model-dir <path>`.

---

## Dataset

Download the dataset from Google Drive (see [`dataset/README.md`](dataset/README.md)) and extract it as:

```
dataset/
└── 0519_house54/
    ├── train_house_00000.json
    ├── ...
    └── artifacts/
        ├── train_house_00000/
        │   ├── exploration_images/
        │   └── observers/
        └── ...
```

---

## Running evaluation

### Step 1 — Run VLM predictions

```bash
python evaluation/run_vlm_eval.py \
    --dataset-dir  dataset/0519_house54 \
    --output-dir   eval_results/gemini_run1 \
    --vlm          gemini \
    --types        a b c
```

Supported values for `--vlm`: `gemini`, `gemini-robotics`, `gemma`, `gpt`, `qwen`

To run with chain-of-thought prompting (Type A/B only):

```bash
python evaluation/run_vlm_eval_ab4.py \
    --dataset-dir  dataset/0519_house54 \
    --output-dir   eval_results/qwen_cot_run1 \
    --vlm          qwen \
    --types        a b
```

Parallel workers:

```bash
# Worker 0 of 4
python evaluation/run_vlm_eval.py ... --worker-id 0 --num-workers 4
# Worker 1 of 4
python evaluation/run_vlm_eval.py ... --worker-id 1 --num-workers 4
# After all workers finish, merge:
python evaluation/run_vlm_eval.py ... --merge --num-workers 4
```

### Step 2 — Compute metrics

```bash
python evaluation/eval_results.py \
    --predictions  eval_results/gemini_run1 \
    --dataset-dir  dataset/0519_house54 \
    --output-dir   eval_results/gemini_run1
```

This produces `eval_results.json` with per-item L2 errors and summary statistics.

### Step 3 — Aggregate multiple runs

```bash
python evaluation/aggregate_runs.py \
    --run gemini:1:eval_results/gemini_run1 \
    --run gemini:2:eval_results/gemini_run2 \
    --run gemini:3:eval_results/gemini_run3 \
    --output-dir analysis/
```

Output: `analysis/all_runs_long.csv`

### Step 4 — Re-parse with final strategy (optional)

```bash
python evaluation/add_final_prediction.py \
    --csv    analysis/all_runs_long.csv \
    --output analysis/all_runs_long_with_final.csv
```

---

## API keys

Copy `.env.example` to `.env` and fill in the keys you need:

```
GEMINI_API_KEY=...
OPENAI_API_KEY=...
```

Alternatively, pass them as environment variables inline:

```bash
GEMINI_API_KEY=<key> python evaluation/run_vlm_eval.py ...
```

<!-- ---

## Citation

```bibtex
@inproceedings{[cite_key],
  title   = {[Paper Title]},
  author  = {[Authors]},
  booktitle = {[Venue]},
  year    = {[Year]},
}
``` -->
