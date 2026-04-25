# Agent Evaluation Framework

> **Acknowledgement.** This work is built on top of the **REAL** paper repository
> ([AGI SDK](https://github.com/agi-inc/agisdk) — *paper:* [arxiv.org/abs/2504.11543](https://arxiv.org/abs/2504.11543)).
> The agent run mechanics (browser harness, task loop, action/observation interface) are reused directly from the REAL repo.
> The original upstream README is preserved as [`REAL-README.md`](./REAL-README.md).
> Everything in this repository — the custom evaluation framework, batch runner, objective and LLM-as-a-judge evaluators,
> and the new task suites — is my own work built on those run mechanics.

## Setup

```bash
# Install dependencies
poetry install

# Install playwright browsers
poetry run playwright install

# Create .env file for API keys
echo "OPENAI_API_KEY=your_key_here" > .env

# OR for OpenRouter:
echo "OPENROUTER_API_KEY=your_key_here" >> .env
```

## Usage

```bash
# Run a specific task
poetry run python main.py --task crm-1

# Run all tasks in an application
poetry run python main.py --application crm

# Run tasks with options
poetry run python main.py --run-all --concurrent --headless
poetry run python main.py --run-random -n 5 --iterations 3

# Run a batch of experiments from JSON
poetry run python run_experiments.py --config example/batch_experiments.example.json

# Skip experiments whose results already exist
poetry run python run_experiments.py --config example/batch_experiments.example.json --skip-existing-results
```

### Key Options
- `--model`: Model to use (default: gpt-4.1)
- `--concurrent`: Run tasks in parallel
- `--headless`: Run browser in headless mode
- `--iterations`: Number of times to run each task
- `--ignore-completed`: Skip tasks already in results.json
- `--rerun-errors`: Only run tasks that previously failed

### Batch Config

Use `run_experiments.py` to launch many `main.py` runs at once from JSON. The launcher reuses the same CLI under the hood, writes each experiment's combined stdout/stderr to `<results_dir>/batch_runner.log`, and when you stop the launcher it terminates every running experiment process group so browser children do not linger.

Pass `--skip-existing-results` to ignore experiments whose `results_dir` already contains prior run artifacts such as `run_manifests/run_*.json` or task `summary_info.json` files.
You can also set `"skip_existing_results": true` at the root of a batch config JSON, on an individual model block, or on a specific testcase. The precedence is: CLI flag, then testcase, then model, then root config.

```json
{
  "max_parallel": 2,
  "defaults": {
    "run_all": true,
    "headless": true,
    "max_steps": 40
  },
  "models": {
    "openrouter/x-ai/grok-4": {
      "skip_existing_results": true,
      "defaults": {
        "url": "http://localhost:3100/"
      },
      "testcases": [
        {
          "name": "grok4-flightradar8",
          "task_file": "example/tasks/flightradar8.yaml",
          "results_dir": "results/results_grok4_frad"
        }
      ]
    }
  }
}
```

Each model key can define shared `defaults` plus a `testcases` list. Each testcase can override any supported `main.py` option, including `results_dir`, `max_steps`, `post_run_url`, `post_run_js_snippet_path` (or the legacy alias `js_snippet_file`), `system_prompt`, and `extra_args`.

## Evaluation

After running experiments, evaluate agent performance using two complementary methods:

### Objective Evaluation

Automated pass/fail scoring by comparing agent outputs against ground truth answers. Each benchmark app has a dedicated evaluator:

- **Circuit** — builds component graphs from circuit exports and computes graph edit distance (requires `networkx`)
- **Flightradar** — extracts JSON from agent responses and compares field-by-field against expected values
- **Voidcut** — validates video editor timeline exports block-by-block against scenario rules

```bash
# Evaluate all result directories under a root
poetry run python -m evaluation.objective.batch_evaluate clean_results/open_source

# Multiple roots, parallel workers
poetry run python -m evaluation.objective.batch_evaluate \
  clean_results/open_source clean_results/closed_source --workers 8
```

Result directories are auto-discovered by naming convention (`_circuit`, `_frad`, `_video`/`_voidcut`). Each directory gets an `objective_evaluation.json` file with `{test_id: 0|1}` scores.

You can also run individual evaluators directly:

```bash
# Circuit
poetry run python -m evaluation.objective.evaluate_circuit_scheme \
  --test-cases tasks/circuit.yaml --responses results/results_gpt54_circuit

# Flightradar
poetry run python -m evaluation.objective.evaluate \
  --test-cases tasks/flightradar.yaml --responses results/results_gpt54_frad

# Voidcut
poetry run python -m evaluation.objective.eval_voidcut \
  results/results_gpt54_video gt_all.json --tolerance_ms 1000 --verbose
```

### LLM-as-a-Judge Evaluation

A 3-stage pipeline that uses LLM models to assess agent behavior from screenshots and action traces:

1. **Stage 1** — Screenshot diff filtering: compares consecutive screenshots using ImageMagick (`compare`) to detect visually significant changes via RMSE, perceptual hash, and changed pixel fraction
2. **Stage 2** — Per-change judgment: sends flagged screenshot pairs to an LLM (default: `gpt-4o-mini`) to classify the visible change, task relevance, and progress
3. **Stage 3** — Final outcome: sends the full trajectory and final screenshot to an LLM (default: `gpt-5.1`) to produce a 1-5 score with reasoning

**Requirements:** `OPENAI_API_KEY` env variable, ImageMagick installed.

```bash
# Basic batch run
poetry run python -m evaluation.llm_judge.batch_run_llm_as_judge results/

# Parallel with task YAML overrides
poetry run python -m evaluation.llm_judge.batch_run_llm_as_judge results/ \
  --parallel 4 \
  --task-yaml circuit=tasks/circuit.yaml \
  --task-yaml frad=tasks/flightradar.yaml

# Generate summary CSVs
poetry run python -m evaluation.llm_judge.batch_run_llm_as_judge results/ \
  --summary-csv llm_summary.csv --runs-csv llm_runs.csv

# Skip already-evaluated directories
poetry run python -m evaluation.llm_judge.batch_run_llm_as_judge results/ \
  --skip-existing --match "closed_source"

# Custom models and reasoning
poetry run python -m evaluation.llm_judge.batch_run_llm_as_judge results/ \
  --stage2-model gpt-4o-mini --stage3-model gpt-5.1 \
  --stage3-reasoning-effort high

# Feed objective results into Stage 3 for consistency checks
poetry run python -m evaluation.llm_judge.batch_run_llm_as_judge results/ \
  --objective-evaluation-json results/results_gpt54_circuit/objective_evaluation.json

# Dry run (no API calls)
poetry run python -m evaluation.llm_judge.batch_run_llm_as_judge results/ --dry-run
```

Each result directory gets an `llm_judgments.json` file. The optional summary CSVs aggregate scores across models and apps.

**Key options:**
- `--parallel N` — concurrent worker threads
- `--rmse-threshold`, `--phash-threshold`, `--changed-fraction-threshold` — Stage 1 sensitivity
- `--stage2-image-detail` / `--stage3-image-detail` — image detail level (`low`/`auto`/`high`)
- `--max-judgments` — cap Stage 2 API calls per directory
- `--base-url` — custom OpenAI-compatible API endpoint

## Adding New Tasks

Create a JSON file in `tasks/eval/<application>/<task-id>.json`:

```json
{
  "id": "app-1",
  "goal": "Task description for the agent",
  "website": {
    "id": "app",
    "name": "Application Name",
    "url": "http://localhost:8901"
  },
  "difficulty": "easy|medium|hard",
  "challengeType": "retrieval|interaction|form",
  "evals": [
    {
      "type": "llm_boolean",
      "expected_value": true,
      "rubric": "Evaluation criteria"
    }
  ],
  "points": 1
}
```

Tasks are automatically discovered from the `tasks/eval/` directory.
