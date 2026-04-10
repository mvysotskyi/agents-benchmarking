#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path


def is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text


def escape_yaml_double_quoted(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def escape_json_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def yaml_block(text: str, indent: int = 2) -> str:
    pad = " " * indent
    if not text:
        return pad + '|-\n' + pad + "  "
    lines = text.split("\n")
    out = [pad + "|-"]
    for line in lines:
        out.append(pad + "  " + line)
    return "\n".join(out)


def normalize_gt(gt_raw: str):
    """
    Rules:
    - if empty or '-' => return None, meaning emit gt: {}
    - if it already looks like JSON, keep it as a JSON string
    - otherwise wrap as {"answer": "<text>"}
    """
    gt_raw = clean_text(gt_raw)

    if gt_raw in {"", "-"}:
        return None

    if gt_raw.startswith("{") or gt_raw.startswith("["):
        return gt_raw

    return f'{{"answer": "{escape_json_string(gt_raw)}"}}'


def build_testcase_id(prefix: str, row_number: int) -> str:
    return f"{prefix}_{row_number:03d}"


def read_csv_rows(csv_path: Path):
    encodings = ["utf-8", "utf-8-sig", "cp1251", "cp1252", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                return rows
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not read CSV with tested encodings. Last error: {last_error}")


def convert_csv_to_yaml(csv_path: Path, yaml_path: Path, prefix: str, gt_field: str = "Ground Truth"):
    rows = read_csv_rows(csv_path)

    out_lines = ["test_cases:"]

    for idx, row in enumerate(rows, start=1):
        raw_num = clean_text(row.get("#", ""))
        try:
            row_num = int(float(raw_num)) if raw_num else idx
        except ValueError:
            row_num = idx

        testcase_id = build_testcase_id(prefix, row_num)

        prompt = clean_text(row.get("Prompt to agent", ""))

        gt_value = normalize_gt(row.get(gt_field, ""))

        out_lines.append(f'- id: "{testcase_id}"')
        out_lines.append("  prompt: " + yaml_block(prompt, indent=2).lstrip())

        if gt_value is None:
            out_lines.append("  gt: {}")
        else:
            out_lines.append(
                f'  gt: {{answer: "{escape_yaml_double_quoted(gt_value)}"}}'
            )

    yaml_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Convert benchmark CSV into YAML testcases file."
    )
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument("yaml_path", help="Path to output YAML file")
    parser.add_argument(
        "--prefix",
        default="tc_graph",
        help='Testcase ID prefix, e.g. "tc_graph" -> tc_graph_001',
    )
    parser.add_argument(
        "--gt-field",
        default="Ground Truth",
        help='Name of the CSV column containing the ground truth value. Defaults to "Ground Truth".',
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    yaml_path = Path(args.yaml_path)

    convert_csv_to_yaml(csv_path, yaml_path, args.prefix, args.gt_field)
    print(f"Saved YAML to: {yaml_path}")


if __name__ == "__main__":
    main()