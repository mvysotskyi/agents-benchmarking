from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import permutations, product
from typing import Dict, Iterable

from .build_graph import build_circuit_graph


LOGIC_COMPONENT_CODES = {"L", "M", "I", "150", "151", "152", "153", "154", "155"}
BINARY_GATE_CODES = {"150", "151", "152", "153", "154", "155"}
MAX_TRUTH_TABLE_INPUTS = 6

DRIVER_PINS = {
    "L": {"out"},
    "I": {"out"},
    "150": {"out"},
    "151": {"out"},
    "152": {"out"},
    "153": {"out"},
    "154": {"out"},
    "155": {"out"},
}

REQUIRED_PINS = {
    "L": {"out"},
    "M": {"in1"},
    "I": {"in1", "out"},
    "150": {"in1", "in2", "out"},
    "151": {"in1", "in2", "out"},
    "152": {"in1", "in2", "out"},
    "153": {"in1", "in2", "out"},
    "154": {"in1", "in2", "out"},
    "155": {"in1", "in2", "out"},
}


class TruthTableError(ValueError):
    """Raised when a circuit cannot be simulated as a pure logic scheme."""


@dataclass
class LogicCircuit:
    component_codes: dict[str, str]
    pin_to_net: dict[str, dict[str, str]]
    net_drivers: dict[str, list[str]]
    input_ids: list[str]
    output_ids: list[str]

    def evaluate_outputs(
        self,
        ordered_input_ids: list[str],
        values: tuple[bool, ...],
    ) -> tuple[bool, ...]:
        assignment = dict(zip(ordered_input_ids, values))
        component_cache: dict[str, bool] = {}
        net_cache: dict[str, bool] = {}
        visiting_components: set[str] = set()
        visiting_nets: set[str] = set()

        def required_net(cid: str, pin_name: str) -> str:
            try:
                return self.pin_to_net[cid][pin_name]
            except KeyError as exc:
                raise TruthTableError(f"component {cid} missing pin {pin_name}") from exc

        def eval_component(cid: str) -> bool:
            if cid in component_cache:
                return component_cache[cid]
            if cid in visiting_components:
                raise TruthTableError("logic cycle detected")

            visiting_components.add(cid)
            code = self.component_codes[cid]

            if code == "L":
                if cid not in assignment:
                    raise TruthTableError(f"missing input assignment for {cid}")
                value = assignment[cid]
            elif code == "I":
                value = not eval_net(required_net(cid, "in1"))
            elif code in BINARY_GATE_CODES:
                left = eval_net(required_net(cid, "in1"))
                right = eval_net(required_net(cid, "in2"))
                if code == "150":
                    value = left and right
                elif code == "151":
                    value = not (left and right)
                elif code == "152":
                    value = left or right
                elif code == "153":
                    value = not (left or right)
                elif code == "154":
                    value = left ^ right
                elif code == "155":
                    value = not (left ^ right)
                else:
                    raise TruthTableError(f"unsupported binary gate {code}")
            elif code == "M":
                value = eval_net(required_net(cid, "in1"))
            else:
                raise TruthTableError(f"unsupported logic component {code}")

            visiting_components.remove(cid)
            component_cache[cid] = value
            return value

        def eval_net(net_id: str) -> bool:
            if net_id in net_cache:
                return net_cache[net_id]
            if net_id in visiting_nets:
                raise TruthTableError("logic cycle detected")

            visiting_nets.add(net_id)
            drivers = self.net_drivers.get(net_id, [])
            if not drivers:
                raise TruthTableError(f"floating net {net_id}")

            driver_values = {eval_component(cid) for cid in drivers}
            if len(driver_values) > 1:
                raise TruthTableError(f"conflicting drivers on {net_id}")

            value = next(iter(driver_values))
            visiting_nets.remove(net_id)
            net_cache[net_id] = value
            return value

        return tuple(eval_component(cid) for cid in self.output_ids)


def _build_logic_circuit(export_text: str) -> LogicCircuit:
    bipartite = build_circuit_graph(export_text)
    component_codes: dict[str, str] = {}
    pin_to_net: dict[str, dict[str, str]] = {}
    net_drivers: dict[str, list[str]] = {}

    for node, data in bipartite.nodes(data=True):
        if data["kind"] == "component":
            component_codes[node] = data["code"]
        elif data["kind"] == "net":
            net_drivers[node] = []

    unsupported_codes = sorted({
        code for code in component_codes.values()
        if code not in LOGIC_COMPONENT_CODES
    })
    if unsupported_codes:
        raise TruthTableError(
            "unsupported logic components: " + ", ".join(unsupported_codes)
        )

    input_ids = sorted(cid for cid, code in component_codes.items() if code == "L")
    output_ids = sorted(cid for cid, code in component_codes.items() if code == "M")
    if not input_ids:
        raise TruthTableError("no logic inputs found")
    if not output_ids:
        raise TruthTableError("no logic outputs found")
    if len(input_ids) > MAX_TRUTH_TABLE_INPUTS:
        raise TruthTableError(
            f"too many logic inputs ({len(input_ids)} > {MAX_TRUTH_TABLE_INPUTS})"
        )

    for cid in component_codes:
        comp_pins: dict[str, str] = {}
        for net_id in bipartite.neighbors(cid):
            pins = bipartite[cid][net_id].get("pins", [])
            for pin_name in pins:
                comp_pins[pin_name] = net_id
        pin_to_net[cid] = comp_pins

        required_pins = REQUIRED_PINS.get(component_codes[cid], set())
        missing = sorted(required_pins - set(comp_pins))
        if missing:
            raise TruthTableError(
                f"component {cid} ({component_codes[cid]}) missing pins: {', '.join(missing)}"
            )

        for pin_name, net_id in comp_pins.items():
            if pin_name in DRIVER_PINS.get(component_codes[cid], set()):
                net_drivers[net_id].append(cid)

    for net_id in net_drivers:
        net_drivers[net_id].sort()

    return LogicCircuit(
        component_codes=component_codes,
        pin_to_net=pin_to_net,
        net_drivers=net_drivers,
        input_ids=input_ids,
        output_ids=output_ids,
    )


def _column_signature_multiset(rows: Iterable[tuple[bool, ...]]) -> Counter[str]:
    rows = list(rows)
    if not rows:
        return Counter()

    output_count = len(rows[0])
    return Counter(
        "".join("1" if row[col_idx] else "0" for row in rows)
        for col_idx in range(output_count)
    )


def compare_circuit_truth_tables(export_gt: str, export_pred: str) -> dict[str, object]:
    """Compare logic-simulable circuits by truth table, ignoring input/output ordering."""
    try:
        gt_circuit = _build_logic_circuit(export_gt)
    except TruthTableError as exc:
        return {
            "applicable": False,
            "equivalent": False,
            "similarity": 0.0,
            "reason": f"GT not logic-simulable: {exc}",
            "gt_inputs": 0,
            "gt_outputs": 0,
            "pred_inputs": 0,
            "pred_outputs": 0,
            "matched_input_order": None,
        }

    try:
        pred_circuit = _build_logic_circuit(export_pred)
    except TruthTableError as exc:
        return {
            "applicable": False,
            "equivalent": False,
            "similarity": 0.0,
            "reason": f"Prediction not logic-simulable: {exc}",
            "gt_inputs": len(gt_circuit.input_ids),
            "gt_outputs": len(gt_circuit.output_ids),
            "pred_inputs": 0,
            "pred_outputs": 0,
            "matched_input_order": None,
        }

    result = {
        "applicable": True,
        "equivalent": False,
        "similarity": 0.0,
        "reason": None,
        "gt_inputs": len(gt_circuit.input_ids),
        "gt_outputs": len(gt_circuit.output_ids),
        "pred_inputs": len(pred_circuit.input_ids),
        "pred_outputs": len(pred_circuit.output_ids),
        "matched_input_order": None,
    }

    if len(gt_circuit.input_ids) != len(pred_circuit.input_ids):
        result["reason"] = "input count mismatch"
        return result

    if len(gt_circuit.output_ids) != len(pred_circuit.output_ids):
        result["reason"] = "output count mismatch"
        return result

    assignments = list(product((False, True), repeat=len(gt_circuit.input_ids)))

    try:
        gt_rows = [
            gt_circuit.evaluate_outputs(gt_circuit.input_ids, values)
            for values in assignments
        ]
        gt_signatures = _column_signature_multiset(gt_rows)
    except TruthTableError as exc:
        result["applicable"] = False
        result["reason"] = f"GT simulation failed: {exc}"
        return result

    best_similarity = 0.0
    best_order: list[str] | None = None

    for input_order in permutations(pred_circuit.input_ids):
        try:
            pred_rows = [
                pred_circuit.evaluate_outputs(list(input_order), values)
                for values in assignments
            ]
        except TruthTableError as exc:
            result["applicable"] = False
            result["reason"] = f"Prediction simulation failed: {exc}"
            return result

        pred_signatures = _column_signature_multiset(pred_rows)
        matched_outputs = sum((gt_signatures & pred_signatures).values())
        similarity = matched_outputs / max(len(gt_circuit.output_ids), 1)

        if similarity > best_similarity:
            best_similarity = similarity
            best_order = list(input_order)

        if pred_signatures == gt_signatures:
            result["equivalent"] = True
            result["similarity"] = 1.0
            result["matched_input_order"] = list(input_order)
            return result

    result["similarity"] = best_similarity
    result["matched_input_order"] = best_order
    result["reason"] = "truth tables differ"
    return result
