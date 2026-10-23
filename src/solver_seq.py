"""
- Дискретизація рівняння теплопровідності на графі полігонів:
    dT_i / dt = alpha * sum_{j in neighbors(i)} (A_ij / d_ij) * (T_j - T_i)
- Цикл моделювання (Одноядерний / послідовний прототип) за явною схемою Ейлера:
    T_new[i] = T_old[i] + dt * Flux[i]
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .mesh_loader import MeshTopology


@dataclass
class SimulationResult:
    """Результати моделювання теплопровідності."""

    T_final: np.ndarray
    history: List[np.ndarray]
    history_steps: List[int]
    elapsed_time: float
    num_steps: int
    dt: float
    alpha: float


def compute_flux_reference(topology: MeshTopology, T: np.ndarray, alpha: float) -> np.ndarray:
    """
    Класична (поелементна) реалізація розрахунку потоку тепла Flux_i
    через списки суміжності adj[i] (для верифікації на малих мешах).
    """
    N = topology.num_faces
    flux = np.zeros(N, dtype=np.float64)
    for i in range(N):
        ti = T[i]
        s = 0.0
        nbrs = topology.adj[i]
        weights = topology.adj_weights[i]
        for k in range(len(nbrs)):
            j = nbrs[k]
            s += weights[k] * (T[j] - ti)
        flux[i] = alpha * s
    return flux


def run_sequential_simulation(
    topology: MeshTopology,
    T0: np.ndarray,
    alpha: float,
    dt: float,
    num_steps: int,
    fixed_indices: Optional[np.ndarray] = None,
    fixed_values: Optional[np.ndarray | float] = None,
    snapshot_interval: int = 0,
) -> SimulationResult:
    """
    Виконує послідовне моделювання поширення тепла на поверхні меша
    за явною схемою Ейлера:
        T_new_i = T_old_i + dt * alpha * sum_{j in neighbors(i)} (A_ij / d_ij) * (T_j - T_i)

    Parameters
    ----------
    topology : MeshTopology
        Топологія трикутного меша.
    T0 : np.ndarray
        Початковий вектор температур (N,).
    alpha : float
        Коефіцієнт теплопровідності.
    dt : float
        Крок за часом.
    num_steps : int
        Кількість часових кроків.
    fixed_indices : Optional[np.ndarray]
        Індекси полігонів із фіксованою температурою (гранична умова Діріхле), якщо задано.
    fixed_values : Optional[np.ndarray | float]
        Значення фіксованої температури для fixed_indices.
    snapshot_interval : int
        Якщо > 0, зберігає стан температури кожні snapshot_interval кроків (для анімації).
    """
    T_old = np.array(T0, dtype=np.float64, copy=True)
    T_new = np.empty_like(T_old)

    N, max_deg = topology.neighbors_padded.shape
    safe_neighbors = topology.neighbors_padded.copy()
    mask_invalid = safe_neighbors < 0
    row_indices = np.broadcast_to(np.arange(N, dtype=np.int32)[:, None], (N, max_deg))
    safe_neighbors[mask_invalid] = row_indices[mask_invalid]
    weights = topology.weights_padded
    dt_alpha = dt * alpha

    history: List[np.ndarray] = []
    history_steps: List[int] = []

    if snapshot_interval > 0:
        history.append(T_old.copy())
        history_steps.append(0)

    t_start = time.perf_counter()

    for step in range(1, num_steps + 1):
        T_nbrs = T_old[safe_neighbors]
        diff = T_nbrs - T_old[:, None]
        flux = np.sum(weights * diff, axis=1)
        np.add(T_old, dt_alpha * flux, out=T_new)

        if fixed_indices is not None and fixed_values is not None:
            T_new[fixed_indices] = fixed_values

        T_old, T_new = T_new, T_old

        if snapshot_interval > 0 and (step % snapshot_interval == 0 or step == num_steps):
            history.append(T_old.copy())
            history_steps.append(step)

    elapsed = time.perf_counter() - t_start

    return SimulationResult(
        T_final=T_old,
        history=history,
        history_steps=history_steps,
        elapsed_time=elapsed,
        num_steps=num_steps,
        dt=dt,
        alpha=alpha,
    )
