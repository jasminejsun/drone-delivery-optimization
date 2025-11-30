from __future__ import annotations
import importlib.util
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

NodeCoords = Dict[int, Tuple[float, float]]
ARC_ACTIVE_THRESHOLD = 0.5
SOLVER_FILENAME = "drone-delivery-optimization.py"
SOLVER_MODULE_NAME = "drone_delivery_optimization"
SOLVER_PATH = Path(__file__).resolve().parent / SOLVER_FILENAME
_SOLVER_MODULE = None


def _load_solver_module():
    """load the solver script that contains the optimization results."""
    global _SOLVER_MODULE
    if _SOLVER_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            SOLVER_MODULE_NAME,
            SOLVER_PATH,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to locate solver module at {SOLVER_PATH}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SOLVER_MODULE = module
    return _SOLVER_MODULE


def _get_solver_attr(attr: str):
    module = _load_solver_module()
    if not hasattr(module, attr):
        raise AttributeError(f"Solver module is missing required attribute '{attr}'")
    return getattr(module, attr)


def get_solver_context():
    """
    loads the reusable artifacts from the optimization script. Cached so we only solve once per session
    """
    module = _load_solver_module()
    required_attrs = ("node_coords", "no_fly_polygon", "X_opt", "V_opt", "d", "P_const")
    missing = [attr for attr in required_attrs if not hasattr(module, attr)]
    if missing:
        raise AttributeError(
            f"Solver module is missing required attributes: {', '.join(missing)}"
        )

    return {attr: getattr(module, attr) for attr in required_attrs}


def extract_tour_from_X(X: np.ndarray, start: int = 0, tol: float = ARC_ACTIVE_THRESHOLD):
    """
    proxy to the solver's tour extraction helper.
    """
    solver_fn = _get_solver_attr("extract_tour_from_X")
    return solver_fn(X, start=start, tol=tol)


def _active_arcs(X: np.ndarray) -> Iterable[Tuple[int, int]]:
    """
    return a list of arc indices with active routing decisions.
    """
    indices = np.argwhere(X > ARC_ACTIVE_THRESHOLD)
    return [tuple(map(int, ij)) for ij in indices]


import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable

def plot_route_with_order(
    node_coords,
    no_fly_polygon: np.ndarray,
    tour: Iterable[int],
    title: str = "Optimal Route (Ordered)",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """shows node layout and the recovered route in order with leg numbers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ordered_nodes = sorted(node_coords.items())
    depot = node_coords.get(0)
    customers = np.array([coords for idx, coords in ordered_nodes if idx != 0])

    if customers.size:
        ax.scatter(customers[:, 0], customers[:, 1], c="royalblue", s=80, label="Customers")

    if depot:
        ax.scatter(depot[0], depot[1], c="black", s=120, label="Depot")

    polygon = np.vstack([no_fly_polygon, no_fly_polygon[0]])
    ax.plot(polygon[:, 0], polygon[:, 1], "r--", label="No-Fly Zone")

    tour_list = list(tour)
    for step, (i, j) in enumerate(zip(tour_list, tour_list[1:]), start=1):
        x1, y1 = node_coords[i]
        x2, y2 = node_coords[j]
        ax.plot([x1, x2], [y1, y2], "g-", lw=2.2, alpha=0.9)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.annotate(
            str(step),
            xy=(mid_x, mid_y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=10,
            color="darkred",
            weight="bold",
        )

    for idx, (x, y) in node_coords.items():
        ax.text(x + 0.15, y + 0.15, str(idx), fontsize=11)

    # ----- Direction arrow (bearing from north) -----
    dir_deg = 40  # hard-coded, can be 0–360

    # Get data bounds to place arrow nicely
    all_x = [x for x, y in node_coords.values()]
    all_y = [y for x, y in node_coords.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Arrow origin near top-left of the data bounds
    x0 = min_x + 0.1 * (max_x - min_x)
    y0 = max_y - 0.1 * (max_y - min_y)

    # Convert bearing-from-north to dx, dy in plot coordinates (x = east, y = north)
    theta = np.deg2rad(dir_deg)
    L = 0.2 * max(max_x - min_x, max_y - min_y)  # arrow length
    dx = np.sin(theta) * L
    dy = np.cos(theta) * L

    ax.arrow(
        x0, y0,
        dx, dy,
        length_includes_head=True,
        head_width=0.03 * L,
        head_length=0.05 * L,
        fc="orange",
        ec="orange",
    )
    ax.text(
        x0, y0,
        f"{dir_deg:.0f}°",
        fontsize=9,
        color="orange",
        ha="right",
        va="bottom",
    )

    ax.set_title(title)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    return ax



def plot_arc_energy_breakdown(
    distance_matrix: np.ndarray,
    X: np.ndarray,
    V: np.ndarray,
    P_const: float,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    energy consumption per active arc.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    labels, energies = [], []
    for i, j in _active_arcs(X):
        leg_km = distance_matrix[i, j]
        time_seconds = (leg_km * 1000) / V[i, j]
        energy_kwh = (P_const * time_seconds) / 3.6e6
        energies.append(energy_kwh)
        labels.append(f"{i}→{j}")

    ax.bar(labels, energies, color="mediumseagreen")
    ax.set_title("Energy Consumption per Arc")
    ax.set_ylabel("Energy (kWh)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    return ax


def main() -> None:
    ctx = get_solver_context()
    tour = extract_tour_from_X(ctx["X_opt"])
    plot_route_with_order(ctx["node_coords"], ctx["no_fly_polygon"], tour)
    plot_arc_energy_breakdown(ctx["d"], ctx["X_opt"], ctx["V_opt"], ctx["P_const"])
    plt.show()


if __name__ == "__main__":
    main()
