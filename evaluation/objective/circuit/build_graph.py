import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Tuple

import networkx as nx


Point = Tuple[int, int]

IGNORED_CODES = {"o"}
TWO_INPUT_GATE_CODES = {"150", "151", "152", "153", "154", "155"}
TWO_TERMINAL_CODES = {"v", "i", "r", "l", "c"}
SINGLE_TERMINAL_CODES = {"L", "M", "g", "R"}

# Mapping from XML tag names to legacy CircuitJS element codes.
XML_TAG_TO_CODE: Dict[str, str] = {
    "And": "150",
    "Nand": "151",
    "Or": "152",
    "Nor": "153",
    "Xor": "154",
    "Xnor": "155",
}


@dataclass
class Component:
    cid: str
    code: str
    x1: int
    y1: int
    x2: int
    y2: int
    raw: str

    def _is_horizontal(self) -> bool:
        return abs(self.x2 - self.x1) >= abs(self.y2 - self.y1)

    def pin_points(self) -> Dict[str, Point]:
        """
        Return electrical connection points for supported CircuitJS elements.

        The GT set contains a mix of digital gates and analog components.
        We use explicit geometry for known multi-pin elements and a conservative
        fallback for ordinary two-terminal parts.
        """
        if self.code == "L":
            return {"out": (self.x1, self.y1)}

        if self.code == "M":
            return {"in1": (self.x1, self.y1)}

        if self.code == "g":
            return {"ref": (self.x1, self.y1)}

        if self.code == "R":
            # CircuitJS rails expose a single electrical node at the first point.
            return {"out": (self.x1, self.y1)}

        if self.code == "I":
            return {
                "in1": (self.x1, self.y1),
                "out": (self.x2, self.y2),
            }

        if self.code in TWO_INPUT_GATE_CODES:
            if self._is_horizontal():
                return {
                    "in1": (self.x1, self.y1 - 16),
                    "in2": (self.x1, self.y1 + 16),
                    "out": (self.x2, self.y2),
                }
            return {
                "in1": (self.x1 - 16, self.y1),
                "in2": (self.x1 + 16, self.y1),
                "out": (self.x2, self.y2),
            }

        if self.code == "a":
            # CircuitJS op-amps expose two adjacent inputs at the first anchor
            # and one output at the second anchor; the adjacent pair rotates
            # with the symbol orientation.
            if self._is_horizontal():
                return {
                    "in1": (self.x1, self.y1 - 16),
                    "in2": (self.x1, self.y1 + 16),
                    "out": (self.x2, self.y2),
                }
            return {
                "in1": (self.x1 - 16, self.y1),
                "in2": (self.x1 + 16, self.y1),
                "out": (self.x2, self.y2),
            }

        if self.code in TWO_TERMINAL_CODES:
            return {
                "p1": (self.x1, self.y1),
                "p2": (self.x2, self.y2),
            }

        # Fallback for future components that still expose endpoint geometry.
        if (self.x1, self.y1) != (self.x2, self.y2):
            return {
                "p1": (self.x1, self.y1),
                "p2": (self.x2, self.y2),
            }

        return {"p1": (self.x1, self.y1)}

    def type_label(self) -> str:
        mapping = {
            "L": "INPUT",
            "M": "OUTPUT",
            "g": "GROUND",
            "R": "RAIL",
            "I": "NOT",
            "150": "AND",
            "151": "NAND",
            "152": "OR",
            "153": "NOR",
            "154": "XOR",
            "155": "XNOR",
            "a": "AMPLIFIER",
            "c": "CAPACITOR",
            "i": "CURRENT_SOURCE",
            "l": "INDUCTOR",
            "r": "RESISTOR",
            "v": "VOLTAGE_SOURCE",
        }
        return mapping.get(self.code, f"CODE_{self.code}")


class UnionFind:
    def __init__(self):
        self.parent: Dict[Any, Any] = {}
        self.rank: Dict[Any, int] = {}

    def add(self, x: Any) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: Any) -> Any:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: Any, b: Any) -> None:
        self.add(a)
        self.add(b)
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def clean_export_text(export_text: str) -> List[str]:
    """
    Remove surrounding quotes and split into non-empty lines.
    """
    text = export_text.strip()

    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    return [line.strip() for line in text.splitlines() if line.strip()]


def _extract_xml_export_block(text: str) -> str | None:
    """Return the ``<cir>...</cir>`` block even when wrapped in banner text."""
    stripped = text.strip()

    if stripped.startswith('"') and stripped.endswith('"'):
        stripped = stripped[1:-1].strip()

    start = stripped.find("<cir")
    if start == -1:
        return None

    end = stripped.rfind("</cir>")
    if end == -1 or end < start:
        return stripped[start:]

    return stripped[start:end + len("</cir>")]


def _is_xml_format(text: str) -> bool:
    """Detect whether the export text uses the XML ``<cir>`` format."""
    return _extract_xml_export_block(text) is not None


_XML_ELEMENT_RE = re.compile(
    r'<(?P<tag>\w+)\s+x="(?P<coords>[^"]+)"[^/]*/>', re.DOTALL
)


def _parse_xml_export(
    export_text: str,
) -> Tuple[List[Component], List[Tuple[Point, Point]]]:
    """Parse the XML ``<cir>`` export format into components and wires."""
    components: List[Component] = []
    wires: List[Tuple[Point, Point]] = []
    comp_idx = 0

    xml_export = _extract_xml_export_block(export_text)
    if xml_export is None:
        return components, wires

    for match in _XML_ELEMENT_RE.finditer(xml_export):
        tag = match.group("tag")
        coords_str = match.group("coords")

        try:
            coords = list(map(int, coords_str.split()))
        except ValueError:
            continue
        if len(coords) < 4:
            continue
        x1, y1, x2, y2 = coords[:4]

        # Map multi-char XML tag names to legacy numeric codes.
        code = XML_TAG_TO_CODE.get(tag, tag)

        if code in IGNORED_CODES:
            continue

        if code == "w":
            wires.append(((x1, y1), (x2, y2)))
            continue

        raw = match.group(0)
        components.append(
            Component(
                cid=f"c{comp_idx}",
                code=code,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                raw=raw,
            )
        )
        comp_idx += 1

    return components, wires


def parse_circuit_export(
    export_text: str,
) -> Tuple[List[Component], List[Tuple[Point, Point]]]:
    """
    Parse the CircuitJS export into:
      - component list
      - wire segments as point pairs

    Supports both the legacy plain-text format and the newer XML
    ``<cir>`` format.
    """
    if _is_xml_format(export_text):
        return _parse_xml_export(export_text)

    lines = clean_export_text(export_text)

    components: List[Component] = []
    wires: List[Tuple[Point, Point]] = []
    comp_idx = 0

    for line in lines:
        if line.startswith("$"):
            continue

        parts = line.split()
        code = parts[0]

        if code in IGNORED_CODES:
            continue

        if code == "w":
            x1, y1, x2, y2 = map(int, parts[1:5])
            wires.append(((x1, y1), (x2, y2)))
            continue

        if len(parts) < 5:
            continue

        try:
            x1, y1, x2, y2 = map(int, parts[1:5])
        except ValueError:
            continue

        components.append(
            Component(
                cid=f"c{comp_idx}",
                code=code,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                raw=line,
            )
        )
        comp_idx += 1

    return components, wires


def build_circuit_graph(export_text: str) -> nx.Graph:
    """
    Build an undirected bipartite graph of components and electrical nets.

    This representation is layout-agnostic and works for both the digital and
    analog GT samples.
    """
    components, wires = parse_circuit_export(export_text)
    uf = UnionFind()

    for p1, p2 in wires:
        uf.add(p1)
        uf.add(p2)
        uf.union(p1, p2)

    pin_map: Dict[str, Dict[str, Point]] = {}
    for comp in components:
        pins = comp.pin_points()
        pin_map[comp.cid] = pins
        for pt in pins.values():
            uf.add(pt)

    root_to_net: Dict[Point, str] = {}
    net_counter = 0

    def net_id_for_point(pt: Point) -> str:
        nonlocal net_counter
        root = uf.find(pt)
        if root not in root_to_net:
            root_to_net[root] = f"n{net_counter}"
            net_counter += 1
        return root_to_net[root]

    graph = nx.Graph()

    for comp in components:
        graph.add_node(
            comp.cid,
            kind="component",
            type=comp.type_label(),
            code=comp.code,
            raw=comp.raw,
        )

    for comp in components:
        for pin_name, pt in pin_map[comp.cid].items():
            net_id = net_id_for_point(pt)
            if net_id not in graph:
                graph.add_node(net_id, kind="net", type="NET")

            if graph.has_edge(comp.cid, net_id):
                graph[comp.cid][net_id]["pins"].append(pin_name)
            else:
                graph.add_edge(comp.cid, net_id, pins=[pin_name])

    return graph


def build_component_graph(export_text: str) -> nx.Graph:
    """
    Collapse the component-net graph into an undirected component graph.

    Two components are adjacent when they share an electrical net.
    """
    bipartite = build_circuit_graph(export_text)
    graph = nx.Graph()

    for node, data in bipartite.nodes(data=True):
        if data["kind"] == "component":
            graph.add_node(node, **data)

    for net, net_data in bipartite.nodes(data=True):
        if net_data["kind"] != "net":
            continue

        attached_components = sorted(
            neighbor
            for neighbor in bipartite.neighbors(net)
            if bipartite.nodes[neighbor]["kind"] == "component"
        )

        for left, right in combinations(attached_components, 2):
            if graph.has_edge(left, right):
                graph[left][right]["shared_nets"] += 1
            else:
                graph.add_edge(left, right, shared_nets=1)

    return graph


def build_gate_only_graph(export_text: str) -> nx.Graph:
    """
    Backwards-compatible alias for the structural component graph.

    The older name is kept because `calc_similarity.py` already uses it.
    """
    return build_component_graph(export_text)


if __name__ == "__main__":
    export_text = r'''
"$ 1 0.000005 10.20027730826997 50 5 50 5e-11
L 160 144 64 144 0 0 false 5 0
L 160 224 64 224 0 1 false 5 0
I 160 224 288 224 0 0.5 5
M 288 224 384 224 0 2.5
w 160 144 160 224 0
"
'''

    bipartite_graph = build_circuit_graph(export_text)
    component_graph = build_component_graph(export_text)

    print("Bipartite graph:", bipartite_graph.number_of_nodes(), bipartite_graph.number_of_edges())
    print("Component graph:", component_graph.number_of_nodes(), component_graph.number_of_edges())
