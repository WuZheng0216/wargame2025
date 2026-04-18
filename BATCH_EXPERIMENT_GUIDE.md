# Batch Experiment Guide

This guide covers the recommended workflow for unattended experiments in this project.

## Goal

Use one batch directory as the unit of record for a single experiment setting.

- You can start with a small number of runs.
- You can stop at any time.
- You can resume the same batch later and grow it to a larger target such as 50 runs.
- Each run keeps its own logs, traces, battle logs, LLM outputs, and lessons snapshots.

## Main Entry Points

- New batch: `experiment_runner.py`
- Resume latest batch: `resume_latest_batch.ps1`
- Batch analysis: `analyze_batch.py`

## Environment Prerequisites

Batch experiments are expected to run with:

- `C:\Users\Tk\.conda\envs\scene\python.exe`

Minimal experiment dependencies are recorded in:

- `requirements-experiment.txt`

This includes the current long-term memory retrieval dependencies:

- `jieba`
- `scikit-learn`
- `python-dotenv`
- `httpx`

## Recommended Workflow

### 1. Start a new batch

Example: start with 5 runs.

```powershell
& C:\Users\Tk\.conda\envs\scene\python.exe .\experiment_runner.py --runs 6 --end-simtime 1800 --engine-host 127.0.0.1 --step-interval 0.1 --batch-name paper_v2
```

This creates a new directory under `test\batches\`, for example:

```text
test\batches\paper_v2_20260403_153000
```

### 2. Continue the same batch later

If you want to grow the same batch from 5 runs to 20 runs:

```powershell
.\resume_latest_batch.ps1 -Runs 20
```

This resumes the newest batch under `test\batches\`.

If you want to resume a specific batch instead of the newest one:

```powershell
.\resume_latest_batch.ps1 -Runs 20 -BatchRef paper_v2_20260403_153000
```

You can also pass a full path:

```powershell
.\resume_latest_batch.ps1 -Runs 20 -BatchRef test\batches\paper_v2_20260403_153000
```

### 3. Grow to 50 runs

When you are ready to continue:

```powershell
.\resume_latest_batch.ps1 -Runs 50 -BatchRef paper_v2_20260403_153000
```

Important: `-Runs 50` means the target total number of runs for that batch is 50. It does not mean "add 50 more runs".

## What Resume Does

Resume mode keeps the original batch configuration:

- `end_simtime`
- `engine_host`
- `step_interval`
- batch base lessons snapshots

It reads existing `run_summary.json` files and continues from the next missing run index.

Examples:

- If `run_0001` to `run_0005` exist, resume starts from `run_0006`
- If `run_0006` directory exists but has no valid `run_summary.json`, resume re-runs that index

## Where Results Are Stored

Each batch is under:

```text
test\batches\<batch_id>\
```

Batch-level files:

- `batch_config.json`
- `batch_status.json`
- `run_manifest.jsonl`
- `batch_summary.csv`
- `analysis\index.html`
- `analysis\summary.json`

Each run has its own directory:

```text
run_0001\
run_0002\
...
```

Each run keeps:

- `logs\`
- `battle_logs\`
- `llm_outputs\`
- `knowledge\lessons_before.jsonl`
- `knowledge\lessons_after.jsonl`
- `run_summary.json`
- `stdout.log`
- `stderr.log`
- `env_snapshot.json`

## How To Check Progress

Open:

- `batch_status.json`

It shows:

- total runs requested
- completed runs
- succeeded runs
- failed runs
- current run id

Example:

```powershell
Get-Content test\batches\paper_v2_20260403_153000\batch_status.json
```

## How To Check Results

Open:

- `analysis\index.html`
- `analysis\summary.json`
- `batch_summary.csv`

Example:

```powershell
Get-Content test\batches\paper_v2_20260403_153000\analysis\summary.json
```

## Suggested Experiment Rhythm

For a target of 50 runs, a practical schedule is:

1. Start with 5 runs
2. Resume to 15 runs
3. Resume to 30 runs
4. Resume to 50 runs

This lets you inspect stability and quality after each stage without losing prior data.

## Notes About Long-Term Memory

Within one batch, the base lessons snapshots are frozen for comparability.

That means:

- runs inside the same batch do not change the batch base knowledge for later runs
- each run still writes its own `lessons_after.jsonl`

This is good for controlled experiments.

If you later want a true self-improving setting, create a new batch from updated lessons.

## Recommended Commands

### Start a fresh batch

```powershell
& C:\Users\Tk\.conda\envs\scene\python.exe .\experiment_runner.py --runs 5 --end-simtime 1800 --engine-host 127.0.0.1 --step-interval 0.1 --batch-name paper_v2
```

### Resume the newest batch to 20 total runs

```powershell
.\resume_latest_batch.ps1 -Runs 20
```

### Resume a named batch to 50 total runs

```powershell
.\resume_latest_batch.ps1 -Runs 50 -BatchRef paper_v2_20260403_153000
```

### Rebuild analysis for an existing batch

```powershell
& C:\Users\Tk\.conda\envs\scene\python.exe .\analyze_batch.py --batch-root test\batches\paper_v2_20260403_153000
```
