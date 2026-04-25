#!/usr/bin/env python3
"""Evaluate graph (workflow) editor exports against ground-truth workflow files.

Compares agent results from benchmark run directories against per-task
ground-truth JSON files. The predicted workflow is read from the
``post_run_page_content`` field in ``agent_outputs.json``. Ground-truth
files live under ``assets/graph_ground_truth/taskN.json``.

Because node IDs are unstable (agents create their own UUIDs), nodes are
matched by content (type + ``data`` dict) using a greedy cost-minimising
pairing. Edges are then verified using the resulting GT-to-pred ID
mapping.

Usage:
    python -m evaluation.objective.eval_graph <results_dir> <gt_dir> [--verbose]

Examples:
    python -m evaluation.objective.eval_graph seed_runs/run1/graph/gemini31pro assets/graph_ground_truth
    python -m evaluation.objective.eval_graph seed_runs/run1/graph/opus46 assets/graph_ground_truth --verbose
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fire


TESTCASE_ID_PATTERN = re.compile(r"tc_graph_(\d+)")
TEST_ID_PATTERN = re.compile(r"(tc_graph_\d+)")


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class NodeResult:
    gt_id: str
    gt_type: str
    matched: bool
    pred_id: str | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class EdgeResult:
    gt_edge: dict
    matched: bool
    error: str | None = None


@dataclass
class GraphResult:
    workflow_name: str
    found: bool
    name_matches: bool = False
    node_results: list[NodeResult] = field(default_factory=list)
    edge_results: list[EdgeResult] = field(default_factory=list)
    structural_errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if not self.found:
            return False
        if self.structural_errors:
            return False
        if not all(n.matched for n in self.node_results):
            return False
        if not all(e.matched for e in self.edge_results):
            return False
        return True


@dataclass
class EvalResult:
    test_id: str
    task_number: int
    graph_result: GraphResult | None
    error: str | None = None

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        if self.graph_result is None:
            return False
        return self.graph_result.passed


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_task_number(test_id: str) -> int | None:
    match = TESTCASE_ID_PATTERN.search(test_id)
    return int(match.group(1)) if match else None


def _extract_test_id(value: str) -> str | None:
    match = TEST_ID_PATTERN.search(value)
    return match.group(1) if match else None


def _normalise_value(v: Any) -> Any:
    """Lightweight normalisation: strip strings, leave numbers/bools as-is."""
    if isinstance(v, str):
        return v.strip()
    return v


def _values_match(gt_val: Any, pred_val: Any) -> bool:
    """Compare two data values with light normalisation.

    Strings are compared after stripping; numbers are compared by value
    (so ``1`` and ``1.0`` match); dicts/lists are compared recursively.
    """
    gt_val = _normalise_value(gt_val)
    pred_val = _normalise_value(pred_val)

    if isinstance(gt_val, dict) and isinstance(pred_val, dict):
        if set(gt_val.keys()) != set(pred_val.keys()):
            return False
        return all(_values_match(gt_val[k], pred_val[k]) for k in gt_val)

    if isinstance(gt_val, list) and isinstance(pred_val, list):
        if len(gt_val) != len(pred_val):
            return False
        return all(_values_match(g, p) for g, p in zip(gt_val, pred_val))

    if isinstance(gt_val, bool) or isinstance(pred_val, bool):
        return gt_val == pred_val

    if isinstance(gt_val, (int, float)) and isinstance(pred_val, (int, float)):
        return float(gt_val) == float(pred_val)

    return gt_val == pred_val


def _list_unordered_match(gt_list: list, pred_list: list) -> bool:
    """Check whether two lists of dicts/values match irrespective of order.

    Used for fields like switch ``cases`` whose order may legitimately
    differ between GT and predicted.
    """
    if len(gt_list) != len(pred_list):
        return False
    used = set()
    for g in gt_list:
        for i, p in enumerate(pred_list):
            if i in used:
                continue
            if _values_match(g, p):
                used.add(i)
                break
        else:
            return False
    return True


# ── Node matching ────────────────────────────────────────────────────────────


# Fields whose list ordering should be ignored when comparing.
UNORDERED_LIST_FIELDS = {"cases"}


def _data_mismatches(gt_data: dict, pred_data: dict) -> list[str]:
    """Return mismatch descriptions for keys present in GT data."""
    errors: list[str] = []
    for key, gt_val in gt_data.items():
        pred_val = pred_data.get(key, "<missing>") if isinstance(pred_data, dict) else "<missing>"
        if pred_val == "<missing>" and key not in pred_data:
            errors.append("data.%s missing in predicted" % key)
            continue

        if (
            key in UNORDERED_LIST_FIELDS
            and isinstance(gt_val, list)
            and isinstance(pred_val, list)
        ):
            if not _list_unordered_match(gt_val, pred_val):
                errors.append("data.%s: gt=%r pred=%r" % (key, gt_val, pred_val))
            continue

        if not _values_match(gt_val, pred_val):
            errors.append("data.%s: gt=%r pred=%r" % (key, gt_val, pred_val))
    return errors


def _pair_cost(gt_node: dict, pred_node: dict) -> int | None:
    """Match cost between a GT and predicted node.

    Returns ``None`` if the types differ (ineligible). Otherwise returns
    the number of mismatching/missing data fields.
    """
    if str(gt_node.get("type", "")).strip() != str(pred_node.get("type", "")).strip():
        return None
    gt_data = gt_node.get("data") or {}
    pred_data = pred_node.get("data") or {}
    return len(_data_mismatches(gt_data, pred_data))


def match_nodes(
    gt_nodes: list[dict],
    pred_nodes: list[dict],
) -> tuple[dict[int, int], list[NodeResult], list[str]]:
    """Greedy cost-minimising assignment of GT nodes to predicted nodes.

    Returns ``(gt_index_to_pred_index, node_results, structural_errors)``.
    Only pairs with the same type are eligible. Ties are broken by GT
    order then predicted order so results are deterministic.
    """
    structural_errors: list[str] = []

    pair_costs: dict[tuple[int, int], int] = {}
    for gi, gt_node in enumerate(gt_nodes):
        for pi, pred_node in enumerate(pred_nodes):
            cost = _pair_cost(gt_node, pred_node)
            if cost is not None:
                pair_costs[(gi, pi)] = cost

    sorted_pairs = sorted(
        pair_costs.items(),
        key=lambda kv: (kv[1], kv[0][0], kv[0][1]),
    )

    gt_to_pred: dict[int, int] = {}
    pred_used: set[int] = set()
    for (gi, pi), _ in sorted_pairs:
        if gi in gt_to_pred or pi in pred_used:
            continue
        gt_to_pred[gi] = pi
        pred_used.add(pi)

    node_results: list[NodeResult] = []
    for gi, gt_node in enumerate(gt_nodes):
        gt_id = str(gt_node.get("id", ""))
        gt_type = str(gt_node.get("type", ""))
        if gi in gt_to_pred:
            pi = gt_to_pred[gi]
            pred_node = pred_nodes[pi]
            errors = _data_mismatches(
                gt_node.get("data") or {},
                pred_node.get("data") or {},
            )
            node_results.append(
                NodeResult(
                    gt_id=gt_id,
                    gt_type=gt_type,
                    matched=len(errors) == 0,
                    pred_id=str(pred_node.get("id", "")),
                    errors=errors,
                )
            )
        else:
            node_results.append(
                NodeResult(
                    gt_id=gt_id,
                    gt_type=gt_type,
                    matched=False,
                    errors=["no matching predicted node of type '%s'" % gt_type],
                )
            )

    extra_pred = [pi for pi in range(len(pred_nodes)) if pi not in pred_used]
    for pi in extra_pred:
        pred_type = pred_nodes[pi].get("type", "<unknown>")
        structural_errors.append(
            "extra predicted node of type '%s'" % pred_type
        )
    if len(gt_nodes) != len(pred_nodes):
        structural_errors.append(
            "node count mismatch (gt=%d, pred=%d)"
            % (len(gt_nodes), len(pred_nodes))
        )

    return gt_to_pred, node_results, structural_errors


# ── Edge matching ────────────────────────────────────────────────────────────


def _normalise_handle(h: Any) -> str | None:
    if h is None:
        return None
    s = str(h).strip()
    return s if s else None


def _edge_key(source: str, target: str, source_handle: str | None) -> tuple:
    return (source, target, source_handle)


def match_edges(
    gt_edges: list[dict],
    pred_edges: list[dict],
    gt_node_id_to_pred: dict[str, str],
) -> tuple[list[EdgeResult], list[str]]:
    """Verify each GT edge has a corresponding edge in the predicted graph.

    Edges are compared by ``(source, target, sourceHandle)`` after
    remapping GT node IDs to their predicted counterparts. Extra
    predicted edges that don't correspond to any GT edge become
    structural errors.
    """
    structural_errors: list[str] = []

    pred_edge_keys: dict[tuple, int] = {}
    for e in pred_edges:
        source = str(e.get("source", ""))
        target = str(e.get("target", ""))
        sh = _normalise_handle(e.get("sourceHandle"))
        key = _edge_key(source, target, sh)
        pred_edge_keys[key] = pred_edge_keys.get(key, 0) + 1

    consumed: dict[tuple, int] = {}

    edge_results: list[EdgeResult] = []
    for gt_edge in gt_edges:
        gt_source = str(gt_edge.get("source", ""))
        gt_target = str(gt_edge.get("target", ""))
        gt_sh = _normalise_handle(gt_edge.get("sourceHandle"))

        if gt_source not in gt_node_id_to_pred:
            edge_results.append(
                EdgeResult(
                    gt_edge=gt_edge,
                    matched=False,
                    error="GT source node '%s' has no predicted match" % gt_source,
                )
            )
            continue
        if gt_target not in gt_node_id_to_pred:
            edge_results.append(
                EdgeResult(
                    gt_edge=gt_edge,
                    matched=False,
                    error="GT target node '%s' has no predicted match" % gt_target,
                )
            )
            continue

        expected_key = _edge_key(
            gt_node_id_to_pred[gt_source],
            gt_node_id_to_pred[gt_target],
            gt_sh,
        )
        available = pred_edge_keys.get(expected_key, 0) - consumed.get(expected_key, 0)
        if available > 0:
            consumed[expected_key] = consumed.get(expected_key, 0) + 1
            edge_results.append(EdgeResult(gt_edge=gt_edge, matched=True))
        else:
            edge_results.append(
                EdgeResult(
                    gt_edge=gt_edge,
                    matched=False,
                    error="missing edge %s --(%s)--> %s"
                    % (gt_source, gt_sh or "default", gt_target),
                )
            )

    pred_to_gt_id = {v: k for k, v in gt_node_id_to_pred.items()}
    for e in pred_edges:
        source = str(e.get("source", ""))
        target = str(e.get("target", ""))
        sh = _normalise_handle(e.get("sourceHandle"))
        key = _edge_key(source, target, sh)
        total = pred_edge_keys.get(key, 0)
        used = consumed.get(key, 0)
        if used < total:
            consumed[key] = used + 1
            label_src = pred_to_gt_id.get(source, source)
            label_tgt = pred_to_gt_id.get(target, target)
            structural_errors.append(
                "extra predicted edge %s --(%s)--> %s"
                % (label_src, sh or "default", label_tgt)
            )

    return edge_results, structural_errors


# ── Graph comparison ─────────────────────────────────────────────────────────


def compare_graphs(gt_workflow: dict, pred_workflow: dict) -> GraphResult:
    workflow_name = gt_workflow.get("name", "unknown")

    gt_graph = gt_workflow.get("graph") or {}
    pred_graph = pred_workflow.get("graph") or {}

    gt_nodes = gt_graph.get("nodes") or []
    pred_nodes = pred_graph.get("nodes") or []
    gt_edges = gt_graph.get("edges") or []
    pred_edges = pred_graph.get("edges") or []

    gt_to_pred_idx, node_results, struct_errs_nodes = match_nodes(gt_nodes, pred_nodes)

    gt_id_to_pred_id: dict[str, str] = {}
    for gi, pi in gt_to_pred_idx.items():
        gt_id_to_pred_id[str(gt_nodes[gi].get("id", ""))] = str(
            pred_nodes[pi].get("id", "")
        )

    edge_results, struct_errs_edges = match_edges(gt_edges, pred_edges, gt_id_to_pred_id)

    name_matches = (
        str(gt_workflow.get("name", "")).strip().lower()
        == str(pred_workflow.get("name", "")).strip().lower()
    )

    return GraphResult(
        workflow_name=workflow_name,
        found=True,
        name_matches=name_matches,
        node_results=node_results,
        edge_results=edge_results,
        structural_errors=struct_errs_nodes + struct_errs_edges,
    )


# ── Data loading ─────────────────────────────────────────────────────────────


def load_predicted_content(run_dir: Path) -> str | None:
    """Load post_run_page_content from a run directory."""
    for filename in ("agent_outputs.json", "summary_info.json"):
        filepath = run_dir / filename
        if not filepath.exists():
            continue
        data = json.loads(filepath.read_text(encoding="utf-8"))
        content = data.get("post_run_page_content")
        if content and isinstance(content, str) and content.strip():
            return content
    return None


def load_gt_workflow(gt_dir: Path, task_number: int) -> dict | None:
    gt_path = gt_dir / ("task%d.json" % task_number)
    if not gt_path.exists():
        return None
    return json.loads(gt_path.read_text(encoding="utf-8"))


def load_agent_responses(results_dir: Path) -> list[dict[str, Any]]:
    """Discover all run directories and extract test IDs + content."""
    candidates: dict[str, dict[str, Any]] = {}

    run_dirs = [
        p for p in sorted(results_dir.iterdir())
        if p.is_dir() and p.name != "run_manifests"
    ]

    for run_dir in run_dirs:
        test_id = _extract_test_id(run_dir.name)
        if not test_id:
            summary_path = run_dir / "summary_info.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                test_id = _extract_test_id(
                    summary.get("task_name", "")
                ) or _extract_test_id(summary.get("task_id", ""))
            if not test_id:
                continue

        content = load_predicted_content(run_dir)
        entry = {
            "test_id": test_id,
            "run_dir": str(run_dir),
            "content": content,
        }

        prev = candidates.get(test_id)
        if prev is None or (content and not prev.get("content")):
            candidates[test_id] = entry

    return list(candidates.values())


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_single(
    test_id: str,
    content_str: str | None,
    gt_dir: Path,
) -> EvalResult:
    task_number = _extract_task_number(test_id)
    if task_number is None:
        return EvalResult(
            test_id=test_id,
            task_number=0,
            graph_result=None,
            error="could not extract task number from test_id",
        )

    gt_workflow = load_gt_workflow(gt_dir, task_number)
    if gt_workflow is None:
        return EvalResult(
            test_id=test_id,
            task_number=task_number,
            graph_result=None,
            error="ground truth file task%d.json not found" % task_number,
        )

    if not content_str:
        return EvalResult(
            test_id=test_id,
            task_number=task_number,
            graph_result=None,
            error="no post_run_page_content available",
        )

    try:
        pred_workflow = json.loads(content_str)
    except json.JSONDecodeError as exc:
        return EvalResult(
            test_id=test_id,
            task_number=task_number,
            graph_result=None,
            error="invalid JSON in post_run_page_content: %s" % exc,
        )

    if not isinstance(pred_workflow, dict) or "graph" not in pred_workflow:
        return EvalResult(
            test_id=test_id,
            task_number=task_number,
            graph_result=GraphResult(
                workflow_name=gt_workflow.get("name", ""), found=False
            ),
        )

    graph_result = compare_graphs(gt_workflow, pred_workflow)

    return EvalResult(
        test_id=test_id,
        task_number=task_number,
        graph_result=graph_result,
    )


def evaluate_all(
    results_dir: str,
    gt_dir: str,
    verbose: bool = False,
) -> None:
    """Evaluate all graph workflow test cases from a results directory."""
    results_path = Path(results_dir)
    gt_path = Path(gt_dir)

    if not results_path.exists():
        print("ERROR: results directory not found: %s" % results_path, file=sys.stderr)
        sys.exit(1)
    if not gt_path.exists():
        print("ERROR: ground truth directory not found: %s" % gt_path, file=sys.stderr)
        sys.exit(1)

    responses = load_agent_responses(results_path)
    if not responses:
        print("ERROR: no test case responses found in %s" % results_path, file=sys.stderr)
        sys.exit(1)

    print("Evaluating %d test case(s)" % len(responses))
    print("-" * 60)

    total = 0
    passed = 0
    failed_ids: list[str] = []
    objective_evaluation: dict[str, int] = {}

    for response in sorted(responses, key=lambda r: r["test_id"]):
        test_id = response["test_id"]
        content = response["content"]

        result = evaluate_single(test_id, content, gt_path)
        total += 1

        if result.passed:
            passed += 1
            objective_evaluation[test_id] = 1
            print("\n[PASS] %s  (task%d)" % (test_id, result.task_number))
        else:
            failed_ids.append(test_id)
            objective_evaluation[test_id] = 0
            print("\n[FAIL] %s  (task%d)" % (test_id, result.task_number))

        if result.error:
            print("       ERROR: %s" % result.error)
            continue

        gr = result.graph_result
        if gr is None:
            continue

        if not gr.found:
            print("       workflow '%s' not found in predicted output" % gr.workflow_name)
            continue

        node_pass = sum(1 for n in gr.node_results if n.matched)
        node_total = len(gr.node_results)
        edge_pass = sum(1 for e in gr.edge_results if e.matched)
        edge_total = len(gr.edge_results)
        print("       nodes: %d/%d matched" % (node_pass, node_total))
        print("       edges: %d/%d matched" % (edge_pass, edge_total))
        if not gr.name_matches:
            print("       [name] gt=%r pred-name differs" % gr.workflow_name)

        for err in gr.structural_errors:
            print("       [structural] %s" % err)

        if verbose or not result.passed:
            for nr in gr.node_results:
                if nr.matched and not verbose:
                    continue
                status = "OK" if nr.matched else "FAIL"
                print("         [node %s] %s (%s)" % (status, nr.gt_id, nr.gt_type))
                for e in nr.errors:
                    print("                  %s" % e)
            for er in gr.edge_results:
                if er.matched and not verbose:
                    continue
                status = "OK" if er.matched else "FAIL"
                e = er.gt_edge
                label = "%s --(%s)--> %s" % (
                    e.get("source"),
                    e.get("sourceHandle") or "default",
                    e.get("target"),
                )
                print("         [edge %s] %s" % (status, label))
                if er.error:
                    print("                  %s" % er.error)

    output_dir = results_path
    output_dir.mkdir(parents=True, exist_ok=True)
    objective_path = output_dir / "objective_evaluation.json"
    objective_path.write_text(
        json.dumps(objective_evaluation, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("TOTAL: %d/%d passed" % (passed, total))
    if failed_ids:
        print("FAILED: %s" % ", ".join(failed_ids))
    print("Saved objective evaluation to: %s" % objective_path)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    fire.Fire(evaluate_all)
