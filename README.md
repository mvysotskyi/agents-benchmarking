# Agent Evaluation Framework

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
