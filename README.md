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
```

### Key Options
- `--model`: Model to use (default: gpt-4.1)
- `--concurrent`: Run tasks in parallel
- `--headless`: Run browser in headless mode
- `--iterations`: Number of times to run each task
- `--ignore-completed`: Skip tasks already in results.json
- `--rerun-errors`: Only run tasks that previously failed

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