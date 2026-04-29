# Claude Computer Use Agent

An interactive CLI tool that enables Claude AI to control a web browser through natural language instructions using Anthropic's [Computer Use API](https://docs.anthropic.com/en/docs/agents-and-tools/computer-use).

## Overview

This project demonstrates how to build an agent that leverages Claude's computer use capabilities to automate browser interactions. Users describe tasks in plain English, and Claude executes them by taking screenshots, analyzing the page, and performing mouse/keyboard actions.

```
┌─────────────────────────────────────────────────────────────┐
│                     User Prompt                              │
│            "Search for Claude AI on Google"                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ComputerUseAgent                           │
│  1. Takes screenshot of current browser state               │
│  2. Sends to Claude with computer use tools                 │
│  3. Claude analyzes and returns actions                     │
│  4. Agent executes actions (click, type, scroll)            │
│  5. Repeat until task complete                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Browser (Playwright)                      │
│           Visible window showing all actions                 │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Natural Language Control** - Describe tasks in plain English
- **Visual Feedback** - Watch Claude navigate the browser in real-time
- **Multi-Model Support** - Use Haiku, Sonnet, or Opus based on task complexity
- **Multi-Provider Support** - Anthropic API or AWS Bedrock via LiteLLM
- **Multi-Monitor Support** - Launch browser on any connected display
- **Resolution Presets** - Quick presets (square, hd, fhd) or custom dimensions
- **Interactive CLI** - Rich terminal interface with multi-line input
- **Token Tracking** - Real-time display of token usage, context %, and cost estimates
- **Auto-Summarization** - Automatic context compression when nearing capacity
- **Session Context** - Task summaries preserved across tasks in a session

## Supported Actions

| Action | Description |
|--------|-------------|
| `screenshot` | Capture current viewport |
| `left_click` | Click at coordinates |
| `right_click` | Right-click at coordinates |
| `double_click` | Double-click at coordinates |
| `triple_click` | Triple-click to select line |
| `type` | Type text string |
| `key` | Press key combination (e.g., `Enter`, `Control+a`) |
| `scroll` | Scroll in any direction |
| `mouse_move` | Move cursor to position |
| `left_click_drag` | Click and drag operation |
| `wait` | Pause for specified duration |

## Installation

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- Anthropic API key (or AWS credentials for Bedrock)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/computer-use.git
cd computer-use

# Install dependencies
poetry install

# Install Playwright browsers
poetry run playwright install chromium

# Configure environment
cp .env.example .env
# Edit .env and add your API key
```

### Environment Variables

```bash
# Required for Anthropic provider
ANTHROPIC_API_KEY=your-api-key

# Optional: Use AWS Bedrock instead
LLM_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-west-2
```

## Usage

### Basic Usage

```bash
# Start interactive mode
poetry run computer-use

# You'll be prompted to:
# 1. Select a model (Haiku, Sonnet, or Opus)
# 2. Enter a starting URL
# 3. Type your task
```

### Command Line Options

```bash
# Open directly to a URL with HD resolution
poetry run computer-use --start-url https://google.com --resolution hd

# Execute a task immediately
poetry run computer-use --start-url https://example.com --prompt "Click the login button"

# Open on second monitor with custom resolution
poetry run computer-use --screen 2 --resolution 1920x1080

# Enable verbose element info for debugging
poetry run computer-use --verbose-selectors
```

### Resolution Presets

| Preset | Dimensions |
|--------|------------|
| `square` | 1000x1000 |
| `hd` | 1280x720 |
| `fhd` | 1920x1080 |

### Interactive Controls

- **Esc + Enter** - Submit multi-line task
- **Ctrl+C** (once) - Cancel current action
- **Ctrl+C** (twice) - Exit the program
- Type `quit`, `exit`, or `q` - Exit gracefully

## Example Tasks

```
Task: Go to google.com and search for "Claude AI computer use"

Task: Find the login button and click it

Task: Fill out the contact form with:
      Name: John Doe
      Email: john@example.com
      Message: Hello, this is a test message

Task: Scroll down to the footer and click the "About" link

Task: Take a screenshot of the current page
```

## Project Structure

```
computer-use/
├── src/computer_use/
│   ├── __init__.py
│   ├── main.py          # CLI entry point
│   ├── agent.py         # ComputerUseAgent with action loop
│   ├── browser.py       # Playwright browser management
│   ├── config.py        # Configuration models
│   └── providers.py     # LLM provider abstraction
├── tests/               # pytest test suite
├── docs/                # MkDocs documentation
└── pyproject.toml       # Poetry configuration
```

## How It Works

### Agent Loop

1. User submits a task in natural language
2. Agent sends the prompt to Claude with computer use tools enabled
3. Claude responds with either:
   - **Text**: Task complete or status update
   - **Tool Use**: Action to execute (click, type, etc.)
4. If tool use, agent executes the action via Playwright
5. Agent takes a screenshot and sends it back as tool result
6. Repeat steps 3-5 until Claude responds with text only

### Tool Configuration

The computer use tool is configured based on the selected model:

```python
# Haiku/Sonnet (computer_20250124)
{
    "type": "computer_20250124",
    "name": "computer",
    "display_width_px": 1024,
    "display_height_px": 768,
}

# Opus (computer_20251124) - includes zoom capability
{
    "type": "computer_20251124",
    "name": "computer",
    "display_width_px": 1024,
    "display_height_px": 768,
    "enable_zoom": True,
}
```

## Development

```bash
# Run tests
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=computer_use

# Serve documentation locally
poetry run mkdocs serve

# Lint code
poetry run ruff check src/
```

## Token Optimization

The agent includes several optimizations to reduce API token consumption:

### Token Consumption Tracking
After each iteration, the agent displays:
- Token counts (input/output)
- Context fill percentage
- Estimated cost

At task completion, a session summary shows total tokens and cost:
```
[dim]Tokens: 15,234 in / 156 out | Context: 7.6% of 200K | Cost: $0.0069[/dim]

Session Stats: 12 iterations | 45,678 tokens (38,234 in / 7,444 out)
Context Peak: 19.1% | Estimated Cost: $0.2265
```

### Auto-Summarization
When context usage reaches 70%, the agent automatically summarizes older messages to free up space. This allows long-running tasks to continue without hitting context limits.

Disable via environment variable:
```bash
AUTO_SUMMARIZE=false poetry run computer-use
```

### Session Context
Task summaries are preserved between tasks in the same CLI session. When you start a new task, the agent includes context from previous tasks, enabling multi-step workflows across separate prompts.

### Screenshot History Pruning
Only the last 2 screenshots are kept in conversation history. Older screenshots are replaced with text placeholders, dramatically reducing token usage for long-running tasks.

### Lazy Screenshots
Non-visual actions (`wait`, `mouse_move`) return text descriptions instead of screenshots, saving ~25-50k tokens per skipped screenshot.

### Prompt Caching (Anthropic Provider)
When using the native Anthropic provider, system prompts and tool definitions are cached using Anthropic's prompt caching feature, reducing repeated token costs by 75%.

### Configuration
Token optimization settings can be customized via `TokenOptimizationConfig`:

```python
from computer_use.config import AgentConfig, TokenOptimizationConfig

config = AgentConfig(
    token_optimization=TokenOptimizationConfig(
        max_screenshots_in_history=2,    # Keep last N screenshots
        prune_history=True,              # Enable history pruning
        lazy_screenshots=True,           # Skip screenshots for wait/mouse_move
        enable_prompt_caching=True,      # Anthropic prompt caching
        auto_summarize=True,             # Enable auto-summarization
        summarize_threshold_percent=70,  # Trigger at 70% context fill
        summarize_preserve_recent=4,     # Keep last 4 message pairs
    )
)
```

## Limitations

- Browser runs in non-headless mode (visible window required)
- Maximum 50 iterations per task to prevent runaway API costs
- Coordinate-based clicking may be sensitive to viewport size
- Some websites may detect automation and block access

## Resources

- [Anthropic Computer Use Documentation](https://docs.anthropic.com/en/docs/agents-and-tools/computer-use)
- [Computer Use API Reference](https://docs.anthropic.com/en/docs/agents-and-tools/computer-use#computer-tool)
- [Playwright Documentation](https://playwright.dev/python/docs/intro)
