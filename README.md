# LangGraph Multi-Agent Wargame Decision System

This repository contains a multi-agent decision system for complex rule-constrained adversarial simulation. The project focuses on turning large-model agents into a full decision pipeline rather than a single prompt call, with structured planning, memory, reflection, observability, and batch evaluation.

The current implementation centers on:

- A LangGraph-based multi-agent workflow for RED-side System2 decision making
- Short-term memory for within-match continuity
- Long-term lesson retrieval for cross-match experience reuse
- Post-battle reflection and structured lesson writeback
- Trace logging and a lightweight dashboard for decision-chain inspection
- Batch experiment orchestration and analysis

## Main Capabilities

- Multi-agent workflow: `analyst -> commander -> allocator -> operator -> critic`
- Structured action generation with schema-constrained outputs
- Rule and semantic validation before action submission
- Action freshness guard to avoid stale plans in a dynamic environment
- Long-term lesson retrieval with hybrid scoring
- Reflection pipeline to extract reusable lessons from battle logs
- Batch experiment runner with resumable execution and analysis outputs
- Trace dashboard for inspecting intermediate reasoning steps, LLM calls, guard results, and submitted actions

## Repository Structure

Key files and directories:

- [`main.py`](./main.py): main entry point for running the simulation
- [`base_commander.py`](./base_commander.py): shared commander runtime, memory wiring, graph scheduling, reflection integration
- [`graph.py`](./graph.py): LangGraph workflow definition and node logic
- [`red_commander.py`](./red_commander.py): RED-side decision logic and graph context
- [`blue_commander.py`](./blue_commander.py): BLUE-side logic
- [`memory_manager.py`](./memory_manager.py): short-term memory implementation
- [`ltm_retriever.py`](./ltm_retriever.py): long-term memory retrieval and structured lesson utilities
- [`reflection_agent.py`](./reflection_agent.py): post-battle reflection and lesson writeback
- [`trace_dashboard.py`](./trace_dashboard.py): local dashboard backend for RED System2 traces
- [`experiment_runner.py`](./experiment_runner.py): batch experiment runner
- [`analyze_batch.py`](./analyze_batch.py): batch summary and visualization generation
- [`test/`](./test): local experiment artifacts, lessons, diagnostics, logs, and batches

## Architecture Overview

The RED-side System2 decision chain is organized as a staged LangGraph workflow:

1. `analyst`
   - reads current battlefield summary, short-term memory, and target context
   - produces compact findings for the next stage
2. `commander`
   - forms high-level intent from findings, engagement context, and long-term lessons
3. `allocator`
   - turns intent into a structured allocation plan for fire, guidance, scouting, reserve, and withheld units
4. `operator`
   - translates allocation plans into executable high-level action JSON
   - falls back to LLM generation when deterministic translation is insufficient
5. `critic`
   - validates the output
   - routes the workflow back to `allocator` or `operator` when correction is needed

This design reduces long-chain reasoning brittleness and makes failures easier to localize and repair.

## Requirements

This project assumes a local simulation environment and external dependencies that are not bundled in this repository.

Minimum Python-side dependencies currently tracked here:

- `jieba`
- `python-dotenv`
- `httpx`
- `scikit-learn`

They are listed in [`requirements-experiment.txt`](./requirements-experiment.txt).

The runtime also depends on:

- `jsqlsim`
- a reachable simulation engine / scenario environment
- local environment variables in `.env`

Because this repository targets a local/private experiment setup, you may need to adapt imports, scenario paths, or engine endpoints before it can run on another machine.

## Quick Start

### 1. Create a Python environment

Example:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-experiment.txt
```

### 2. Prepare local configuration

Create a local `.env` file with your model endpoint, simulation endpoint, and other runtime configuration values.

Important:

- `.env` is ignored by Git and should not be committed
- review the environment variables referenced by [`main.py`](./main.py), [`llm_manager.py`](./llm_manager.py), and [`experiment_runner.py`](./experiment_runner.py)

### 3. Run a local simulation

Example:

```powershell
python .\main.py --end-simtime 1800 --engine-host 127.0.0.1 --step-interval 0.1
```

### 4. Run a batch experiment

Example:

```powershell
python .\experiment_runner.py --runs 5 --end-simtime 1800 --engine-host 127.0.0.1 --step-interval 0.1 --batch-name paper_v1
```

See [`BATCH_EXPERIMENT_GUIDE.md`](./BATCH_EXPERIMENT_GUIDE.md) for the resumable workflow and output layout.

## Trace Dashboard

The project includes a lightweight dashboard for inspecting RED System2 traces.

Run:

```powershell
python .\trace_dashboard.py --host 127.0.0.1 --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

The dashboard can display:

- trace overview
- node execution order and duration
- LLM call details
- guard / submit summaries
- parsed actions
- recorded errors

## Generated Outputs

During local runs and batch experiments, the project generates artifacts such as:

- battle logs
- run logs
- RED trace logs
- LLM outputs
- lesson snapshots
- diagnostics
- batch summaries and HTML reports

Most generated runtime artifacts are ignored by the default `.gitignore` provided for public repository packaging.

## Notes Before Publishing

Before pushing this repository to GitHub, you should review:

- `.env` and any local secrets
- model endpoints and access tokens
- large runtime outputs under `llm_outputs/` and `test/`
- proprietary or internal scenario documents
- any machine-specific paths in scripts or guides

This repository has been packaged to favor source code and documentation over generated artifacts.

## Suggested Public Positioning

If you want to describe this project publicly, a good summary is:

> A LangGraph-based multi-agent decision system for adversarial simulation, with structured planning, memory-enhanced reasoning, reflection-driven lesson accumulation, trace observability, and batch experiment support.

