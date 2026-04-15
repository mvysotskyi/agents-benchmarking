import json

import networkx as nx

try:
    from .build_graph import build_component_graph
    from .truth_table import compare_circuit_truth_tables
except ImportError:
    from build_graph import build_component_graph
    from truth_table import compare_circuit_truth_tables


def node_subst_cost(left, right):
    """
    Prefer exact component-code matches, then fall back to coarse type matches.
    """
    left_code = left.get("code")
    right_code = right.get("code")
    if left_code == right_code:
        return 0.0

    left_type = left.get("type")
    right_type = right.get("type")
    if left_type == right_type:
        return 0.5

    similar_pairs = {
        ("AND", "NAND"),
        ("OR", "NOR"),
        ("XOR", "XNOR"),
    }
    if (left_type, right_type) in similar_pairs or (right_type, left_type) in similar_pairs:
        return 1.0

    return 2.0


def node_del_cost(node):
    if node.get("type") in {"INPUT", "OUTPUT", "GROUND", "RAIL"}:
        return 2.0
    return 2.0


def node_ins_cost(node):
    if node.get("type") in {"INPUT", "OUTPUT", "GROUND", "RAIL"}:
        return 2.0
    return 2.0


def edge_subst_cost(left, right):
    return abs(left.get("shared_nets", 1) - right.get("shared_nets", 1)) * 0.25


def edge_del_cost(_edge):
    return 1.0


def edge_ins_cost(_edge):
    return 1.0


def _graphs_are_identical(left: nx.Graph, right: nx.Graph) -> bool:
    return (
        sorted(left.nodes(data=True)) == sorted(right.nodes(data=True))
        and sorted(left.edges(data=True)) == sorted(right.edges(data=True))
    )


def compute_circuit_ged(G_gt: nx.Graph, G_pred: nx.Graph) -> float:
    """
    Exact GED using NetworkX with a fast identical-graph shortcut.
    """
    if _graphs_are_identical(G_gt, G_pred):
        return 0.0

    ged = nx.graph_edit_distance(
        G_gt,
        G_pred,
        node_subst_cost=node_subst_cost,
        node_del_cost=node_del_cost,
        node_ins_cost=node_ins_cost,
        edge_subst_cost=edge_subst_cost,
        edge_del_cost=edge_del_cost,
        edge_ins_cost=edge_ins_cost,
    )

    if ged is None:
        raise RuntimeError("graph_edit_distance returned None")

    return float(ged)


def compute_circuit_similarity(G_gt: nx.Graph, G_pred: nx.Graph) -> float:
    """
    Return a 0-1 similarity score using GED normalized by GT graph size.
    """
    ged = compute_circuit_ged(G_gt, G_pred)
    denom = G_gt.number_of_nodes() + G_gt.number_of_edges()

    if denom == 0:
        pred_size = G_pred.number_of_nodes() + G_pred.number_of_edges()
        return 1.0 if pred_size == 0 else 0.0

    similarity = 1.0 - ged / denom
    return max(0.0, min(1.0, similarity))


def compare_circuit_exports(export_gt: str, export_pred: str):
    """
    Parse exports, build structural graphs, and compute GED plus similarity.
    """
    G_gt = build_component_graph(export_gt)
    G_pred = build_component_graph(export_pred)

    ged = compute_circuit_ged(G_gt, G_pred)
    structural_similarity = compute_circuit_similarity(G_gt, G_pred)
    truth_table = compare_circuit_truth_tables(export_gt, export_pred)
    similarity = structural_similarity
    if truth_table["applicable"] and truth_table["equivalent"]:
        similarity = 1.0

    return {
        "gt_nodes": G_gt.number_of_nodes(),
        "gt_edges": G_gt.number_of_edges(),
        "pred_nodes": G_pred.number_of_nodes(),
        "pred_edges": G_pred.number_of_edges(),
        "ged": ged,
        "similarity": similarity,
        "structural_similarity": structural_similarity,
        "truth_table_applicable": truth_table["applicable"],
        "truth_table_equivalent": truth_table["equivalent"],
        "truth_table_similarity": truth_table["similarity"],
        "truth_table_reason": truth_table["reason"],
        "truth_table_matched_input_order": truth_table["matched_input_order"],
    }


if __name__ == "__main__":
    export_text = r'''
"$ 1 0.000005 10.20027730826997 50 5 50 5e-11
L 688 624 560 624 0 0 false 5 0
M 1120 624 1264 624 0 2.5
I 688 624 1120 624 0 0.5 5
"
'''

    result = compare_circuit_exports(export_text, export_text)
    print(json.dumps(result, indent=2))
