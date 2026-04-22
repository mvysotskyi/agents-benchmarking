#!/usr/bin/env python3
"""
Reformat human evaluation results into the directory structure expected
by the evaluation scripts in evaluation/llm_judge/.

Human results layout (input):
    human_evaluation_results/{app}/{testcase_number}/{participant_id}/{uuid}_testcase_{N}_results.html
    human_evaluation_results/{app}/{participant_id}/attempts/{N}.json

Clean results layout (output):
    clean_results/{output_group}/{participant_id}/results_{participant_id}_{app}/
        {datetime}_DemoAgentArgs_on_eval.tc_{app}_{NNN}_{rand}_{uuid}/
            screenshot_step_0.png
            screenshot_step_1.png
            ...
            summary_info.json

The evaluation scripts (llm_as_judge.py, batch_run_llm_as_judge.py) expect:
  - A results_* directory discovered via results_{model}_{app} naming pattern
  - Each run subdirectory contains screenshot_step_N.png files
  - Each run subdirectory contains summary_info.json with task_name for ID detection
  - step_N.pkl.gz files are optional (load_action_text_for_step returns "" if absent)
"""

import argparse
import base64
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


# Maps app directory name to the prefix used in test case IDs (e.g. "frad" for flightradar)
APP_ID_PREFIX: dict[str, str] = {
    "flightradar": "frad",
    "video": "vid",
}

SHOT_META_RE = re.compile(
    r'<div class="shot-meta"><span>(.*?)</span><span>(.*?)</span><span>(.*?)</span></div>'
)
IMG_RE = re.compile(r'<img src="data:image/png;base64,([A-Za-z0-9+/=]+)"')
PROMPT_RE = re.compile(r'<div class="prompt">(.*?)</div>', re.DOTALL)
DURATION_RE = re.compile(r"Duration:\s*([\d.]+)s")
SCREENSHOTS_COUNT_RE = re.compile(r"Screenshots:\s*(\d+)")
PRE_RE = re.compile(r"<pre>(.*?)</pre>", re.DOTALL)


def unescape_html(text: str) -> str:
    """Unescape common HTML entities."""
    return (
        text.replace("&quot;", '"')
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&apos;", "'")
    )


def strip_html_tags(text: str) -> str:
    """Remove HTML tags, keeping text content."""
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def parse_final_answer(prompt_div: str) -> str:
    """Extract the final answer from the 2nd prompt div (Result Format Answers).

    The answer is inside a <pre> block.
    """
    pre_match = PRE_RE.search(prompt_div)
    if not pre_match:
        return ""
    return unescape_html(pre_match.group(1).strip())


def parse_app_export(prompt_div: str) -> str:
    """Extract the app export from the 3rd prompt div (Final State).

    Content follows the <strong>Final State:</strong><br/> prefix.
    The HTML wrapper tags (<strong>, <br/>) must be removed first,
    then entities are unescaped to recover the raw export text (which
    may contain XML/angle brackets that should be preserved).
    """
    # Strip only the known HTML wrapper: <strong>Final State:</strong><br/>
    text = re.sub(r"<strong>.*?</strong>", "", prompt_div)
    text = re.sub(r"<br\s*/?>", "\n", text)
    # Now unescape HTML entities to recover the actual export content
    text = unescape_html(text)
    return text.strip()


def parse_html_results(html_path: Path) -> dict:
    """Parse an HTML results file and extract screenshots, metadata, answer, and export."""
    content = html_path.read_text()

    # Extract base64 images
    images = IMG_RE.findall(content)

    # Extract shot metadata
    metas = SHOT_META_RE.findall(content)

    # Extract duration
    duration_match = DURATION_RE.search(content)
    duration = float(duration_match.group(1)) if duration_match else 0.0

    # Extract all prompt divs (1=task prompt, 2=final answer, 3=app export)
    prompt_divs = PROMPT_RE.findall(content)

    task_prompt = ""
    final_answer = ""
    app_export = ""

    if len(prompt_divs) >= 1:
        task_prompt = unescape_html(prompt_divs[0].strip())
    if len(prompt_divs) >= 2:
        final_answer = parse_final_answer(prompt_divs[1])
    if len(prompt_divs) >= 3:
        app_export = parse_app_export(prompt_divs[2])

    return {
        "images": images,
        "metas": metas,
        "duration": duration,
        "task_prompt": task_prompt,
        "final_answer": final_answer,
        "app_export": app_export,
        "n_screenshots": len(images),
    }


def build_summary_info(
    testcase_number: int,
    app: str,
    participant_id: str,
    parsed: dict,
) -> dict:
    """Build a summary_info.json compatible with the evaluation scripts."""
    app_prefix = APP_ID_PREFIX.get(app, app)
    task_id_str = f"tc_{app_prefix}_{testcase_number:03d}"
    n_steps = parsed["n_screenshots"] - 1  # last screenshot is "finish"

    final_answer = parsed.get("final_answer", "")
    app_export = parsed.get("app_export", "")

    return {
        "task_name": f"eval.{task_id_str}",
        "agent_type": "HumanAgent",
        "model_name": f"human/{participant_id}",
        "max_steps": 100,
        "leaderboard": False,
        "experiment_status": "completed",
        "run_uuid": str(uuid.uuid4()),
        "n_steps": n_steps,
        "cum_reward": 0.0,
        "cum_raw_reward": 0,
        "err_msg": None,
        "stack_trace": None,
        "completed": True,
        "success": False,
        "error": False,
        "score": 0.0,
        "task_id": f"eval.{task_id_str}_{testcase_number}",
        "agent_response": final_answer,
        "raw_agent_response": final_answer,
        "finish_state": {"env_state": {}},
        "finish_page_content": None,
        "finish_page_html": None,
        "finish_page_axtree": None,
        "post_run_url": None,
        "post_run_js_snippet_path": None,
        "post_run_js_result": app_export or None,
        "post_run_js_error": None,
        "post_run_page_url": None,
        "post_run_page_content": None,
        "post_run_page_html": None,
        "post_run_page_axtree": None,
        "post_run_page_error": None,
        "eval_results": [],
        "env_setup_error": None,
        "stats.cum_step_elapsed": parsed["duration"],
        "stats.max_step_elapsed": parsed["duration"] / max(n_steps, 1),
        "stats.cum_agent_elapsed": parsed["duration"],
        "stats.max_agent_elapsed": parsed["duration"] / max(n_steps, 1),
        "terminated": True,
        "truncated": False,
    }


def build_agent_outputs(parsed: dict) -> dict:
    """Build an agent_outputs.json compatible with the evaluation scripts."""
    final_answer = parsed.get("final_answer", "")
    app_export = parsed.get("app_export", "")

    return {
        "primary_output": final_answer,
        "agent_response": final_answer,
        "raw_agent_response": final_answer,
        "post_run_url": None,
        "post_run_js_result": app_export or None,
        "post_run_js_error": None,
        "post_run_js_snippet_path": None,
        "post_run_page_url": None,
        "post_run_page_content": None,
        "post_run_page_html": None,
        "post_run_page_axtree": None,
        "post_run_page_error": None,
    }


def reformat_human_results(
    input_dir: Path,
    output_dir: Path,
    output_group: str = "human",
) -> None:
    """Convert human evaluation results to clean_results format."""

    # Discover apps (e.g., "circuit")
    for app_dir in sorted(input_dir.iterdir()):
        if not app_dir.is_dir():
            continue
        app = app_dir.name

        # Find all HTML result files: {app}/{testcase_number}/{participant_id}/*.html
        html_files: list[tuple[int, str, Path]] = []
        for child in sorted(app_dir.iterdir()):
            if not child.is_dir():
                continue
            # Check if this is a testcase number directory (digits only)
            if not child.name.isdigit():
                continue
            testcase_number = int(child.name)

            for participant_dir in sorted(child.iterdir()):
                if not participant_dir.is_dir():
                    continue
                participant_id = participant_dir.name

                for html_file in sorted(participant_dir.glob("*.html")):
                    html_files.append((testcase_number, participant_id, html_file))

        if not html_files:
            print(f"No HTML results found for app '{app}' in {app_dir}")
            continue

        # Group by participant
        participants: dict[str, list[tuple[int, Path]]] = {}
        for testcase_number, participant_id, html_file in html_files:
            participants.setdefault(participant_id, []).append((testcase_number, html_file))

        for participant_id, testcases in sorted(participants.items()):
            results_dir_name = f"results_{participant_id}_{app}"
            results_dir = output_dir / output_group / participant_id / results_dir_name
            results_dir.mkdir(parents=True, exist_ok=True)

            print(f"Processing participant {participant_id} for {app} ({len(testcases)} testcases)")

            for testcase_number, html_file in sorted(testcases):
                parsed = parse_html_results(html_file)

                if not parsed["images"]:
                    print(f"  WARNING: No images found in {html_file}")
                    continue

                # Build run directory name matching the expected pattern
                app_prefix = APP_ID_PREFIX.get(app, app)
                task_id_str = f"tc_{app_prefix}_{testcase_number:03d}"
                run_uuid = uuid.uuid4().hex
                rand_num = testcase_number  # deterministic
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
                run_dir_name = (
                    f"{timestamp}_DemoAgentArgs_on_eval.{task_id_str}_{rand_num}_{run_uuid}"
                )
                run_dir = results_dir / run_dir_name
                run_dir.mkdir(parents=True, exist_ok=True)

                # Save screenshots
                for i, img_b64 in enumerate(parsed["images"]):
                    img_data = base64.b64decode(img_b64)
                    screenshot_path = run_dir / f"screenshot_step_{i}.png"
                    screenshot_path.write_bytes(img_data)

                # Save summary_info.json
                summary_info = build_summary_info(
                    testcase_number=testcase_number,
                    app=app,
                    participant_id=participant_id,
                    parsed=parsed,
                )
                summary_path = run_dir / "summary_info.json"
                summary_path.write_text(json.dumps(summary_info, indent=4))

                # Save agent_outputs.json
                agent_outputs = build_agent_outputs(parsed)
                agent_outputs_path = run_dir / "agent_outputs.json"
                agent_outputs_path.write_text(json.dumps(agent_outputs, indent=4))

                # Save agent_output.txt (app export or final answer)
                app_export = parsed.get("app_export", "")
                agent_output_text = app_export if app_export else parsed.get("final_answer", "")
                agent_output_path = run_dir / "agent_output.txt"
                agent_output_path.write_text(agent_output_text)

                n_screenshots = len(parsed["images"])
                print(f"  tc_{app_prefix}_{testcase_number:03d}: {n_screenshots} screenshots -> {run_dir.name}")

    print(f"\nDone. Output written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reformat human evaluation results into clean_results format."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path("human_evaluation_results"),
        help="Path to human_evaluation_results/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("clean_results"),
        help="Output directory (defaults to clean_results/).",
    )
    parser.add_argument(
        "--output-group",
        default="human",
        help="Subdirectory group name under output_dir (defaults to 'human').",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    reformat_human_results(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_group=args.output_group,
    )


if __name__ == "__main__":
    main()
