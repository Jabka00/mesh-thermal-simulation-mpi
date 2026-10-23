"""
- Розбиття меша на P піддоменів (Partitioning):
  * 'contiguous' — просте розбиття масиву граней N на рівні шматки N / P
  * 'geometric' — рекурсивна геометрична бісекція за координатами центроїдів
    для мінімізації довжини межі між процесами (гало-зонами)
- Визначення гало-осередків (Ghost / Halo Cells), внутрішніх (Interior)
  та граничних (Boundary) полігонів, а також буферів обміну для кожного сусіднього рангу.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mesh_loader import MeshTopology


@dataclass
class SubdomainData:
    """Локальна структура піддомену для одного MPI-процесу (rank)."""

    rank: int
    num_procs: int
    local_global_ids: np.ndarray        # (N_local,) глобальні індекси локальних полігонів
    halo_global_ids: np.ndarray         # (N_halo,) глобальні індекси гало-полігонів
    interior_local_ids: np.ndarray      # (N_interior,) локальні індекси внутрішніх полігонів [0..N_local-1]
    boundary_local_ids: np.ndarray      # (N_boundary,) локальні індекси граничних полігонів [0..N_local-1]
    local_neighbors_padded: np.ndarray  # (N_local, max_deg) індекси сусідів у розширеному масиві [0..N_local+N_halo-1]
    local_weights_padded: np.ndarray    # (N_local, max_deg) ваги A_ij / d_ij
    neighbor_ranks: List[int]           # список сусідніх MPI-рангів
    send_local_indices: Dict[int, np.ndarray]  # rank_nbr -> локальні індекси полігонів для відправки
    recv_halo_slices: Dict[int, Tuple[int, int]]  # rank_nbr -> (start, end) у масиві T_ext[N_local : N_local + N_halo]
    local_fixed_indices: Optional[np.ndarray]  # локальні індекси фіксованих джерел тепла
    local_fixed_values: Optional[np.ndarray]   # значення температур фіксованих джерел


def _recursive_coordinate_partition(
    centroids: np.ndarray,
    face_indices: np.ndarray,
    num_parts: int,
    part_offset: int,
    part_labels: np.ndarray,
) -> None:
    """
    Рекурсивно розбиває множину полігонів face_indices на num_parts збалансованих
    піддоменів вздовж координатної осі з найбільшим розкидом центроїдів.
    """
    if num_parts <= 1 or len(face_indices) == 0:
        part_labels[face_indices] = part_offset
        return

    coords = centroids[face_indices]
    spans = np.ptp(coords, axis=0)
    axis = int(np.argmax(spans))

    order = np.argsort(coords[:, axis], kind="mergesort")
    sorted_faces = face_indices[order]

    left_parts = num_parts // 2
    right_parts = num_parts - left_parts
    split_idx = (len(sorted_faces) * left_parts) // num_parts

    _recursive_coordinate_partition(
        centroids, sorted_faces[:split_idx], left_parts, part_offset, part_labels
    )
    _recursive_coordinate_partition(
        centroids, sorted_faces[split_idx:], right_parts, part_offset + left_parts, part_labels
    )


def partition_mesh(
    topology: MeshTopology,
    num_procs: int,
    method: str = "geometric",
) -> np.ndarray:
    """
    Розбиває меш із N граней на num_procs піддоменів.

    Parameters
    ----------
    topology : MeshTopology
        Топологія меша.
    num_procs : int
        Кількість MPI-процесів P.
    method : str
        Метод декомпозиції:
        - 'geometric': рекурсивна координатна бісекція за центроїдами (мінімізує гало-межу)
        - 'contiguous': лінійне розбиття масиву індексів на рівні блоки N / P

    Returns
    -------
    part_labels : np.ndarray
        Масив розміру (N,) типу int32, де part_labels[i] in [0, num_procs - 1] — ранг власника полігона i.
    """
    N = topology.num_faces
    if num_procs <= 1:
        return np.zeros(N, dtype=np.int32)

    part_labels = np.zeros(N, dtype=np.int32)
    if method == "contiguous":
        splits = np.array_split(np.arange(N, dtype=np.int32), num_procs)
        for r, idx in enumerate(splits):
            part_labels[idx] = r
    elif method == "geometric":
        all_faces = np.arange(N, dtype=np.int32)
        _recursive_coordinate_partition(
            topology.centroids, all_faces, num_procs, 0, part_labels
        )
    else:
        raise ValueError(f"Невідомий метод декомпозиції: {method}. Доступні: 'geometric', 'contiguous'.")

    return part_labels


def compute_partition_metrics(topology: MeshTopology, part_labels: np.ndarray) -> Dict[str, float]:
    """
    Обчислює метрики якості розбиття меша:
    - кількість розрізаних ребер (edge cut)
    - загальну довжину межі між піддоменами
    - дисбаланс навантаження (max_size / avg_size)
    """
    f0 = topology.face_adjacency[:, 0]
    f1 = topology.face_adjacency[:, 1]
    cut_mask = part_labels[f0] != part_labels[f1]
    edge_cut = int(np.sum(cut_mask))
    boundary_length = float(np.sum(topology.edge_lengths[cut_mask]))

    num_procs = int(np.max(part_labels)) + 1
    counts = np.bincount(part_labels, minlength=num_procs)
    imbalance = float(np.max(counts) / np.mean(counts)) if len(counts) > 0 else 1.0

    return {
        "edge_cut": edge_cut,
        "total_edges": len(topology.face_adjacency),
        "cut_ratio": edge_cut / max(1, len(topology.face_adjacency)),
        "boundary_length": boundary_length,
        "load_imbalance": imbalance,
        "min_cells_per_proc": int(np.min(counts)),
        "max_cells_per_proc": int(np.max(counts)),
    }


def build_subdomain(
    topology: MeshTopology,
    part_labels: np.ndarray,
    rank: int,
    num_procs: int,
    fixed_global_indices: Optional[np.ndarray] = None,
    fixed_global_values: Optional[np.ndarray | float] = None,
) -> SubdomainData:
    """
    Етап 3.2: Побудова локального піддомену, списку гало-полігонів (Ghost / Halo Cells)
    та таблиць комунікації для заданого MPI-рангу.
    """
    local_global_ids = np.flatnonzero(part_labels == rank).astype(np.int32)
    N_local = len(local_global_ids)

    N_total = topology.num_faces
    global_to_ext = np.full(N_total, -1, dtype=np.int32)
    global_to_ext[local_global_ids] = np.arange(N_local, dtype=np.int32)

    halo_by_rank: Dict[int, set[int]] = {}
    export_by_rank: Dict[int, set[int]] = {}

    interior_list: List[int] = []
    boundary_list: List[int] = []

    for local_i, global_i in enumerate(local_global_ids):
        nbrs = topology.adj[global_i]
        is_boundary = False
        for global_j in nbrs:
            nbr_rank = int(part_labels[global_j])
            if nbr_rank != rank:
                is_boundary = True
                if nbr_rank not in halo_by_rank:
                    halo_by_rank[nbr_rank] = set()
                    export_by_rank[nbr_rank] = set()
                halo_by_rank[nbr_rank].add(int(global_j))
                export_by_rank[nbr_rank].add(int(global_i))
        if is_boundary:
            boundary_list.append(local_i)
        else:
            interior_list.append(local_i)

    neighbor_ranks = sorted(halo_by_rank.keys())

    halo_global_list: List[int] = []
    recv_halo_slices: Dict[int, Tuple[int, int]] = {}
    send_local_indices: Dict[int, np.ndarray] = {}

    current_offset = N_local
    for nbr_rank in neighbor_ranks:
        sorted_halo_globals = sorted(halo_by_rank[nbr_rank])
        count = len(sorted_halo_globals)
        recv_halo_slices[nbr_rank] = (current_offset, current_offset + count)
        for g_id in sorted_halo_globals:
            global_to_ext[g_id] = current_offset
            halo_global_list.append(g_id)
            current_offset += 1

        sorted_export_globals = sorted(export_by_rank[nbr_rank])
        send_local_indices[nbr_rank] = global_to_ext[np.asarray(sorted_export_globals, dtype=np.int32)]

    halo_global_ids = np.asarray(halo_global_list, dtype=np.int32)

    max_deg = topology.max_neighbors
    local_neighbors_padded = np.zeros((N_local, max_deg), dtype=np.int32)
    local_weights_padded = topology.weights_padded[local_global_ids].copy()

    global_nbrs = topology.neighbors_padded[local_global_ids]
    for local_i in range(N_local):
        for k in range(max_deg):
            g_nbr = int(global_nbrs[local_i, k])
            if g_nbr < 0:
                local_neighbors_padded[local_i, k] = local_i
            else:
                local_neighbors_padded[local_i, k] = int(global_to_ext[g_nbr])

    local_fixed_indices = None
    local_fixed_values = None
    if fixed_global_indices is not None and fixed_global_values is not None:
        fixed_global_indices = np.asarray(fixed_global_indices, dtype=np.int32)
        if np.ndim(fixed_global_values) == 0:
            fixed_vals_arr = np.full(len(fixed_global_indices), float(fixed_global_values), dtype=np.float64)
        else:
            fixed_vals_arr = np.asarray(fixed_global_values, dtype=np.float64)

        mapped = global_to_ext[fixed_global_indices]
        valid_mask = (mapped >= 0) & (mapped < N_local)
        if np.any(valid_mask):
            local_fixed_indices = mapped[valid_mask].astype(np.int32)
            local_fixed_values = fixed_vals_arr[valid_mask].astype(np.float64)

    return SubdomainData(
        rank=rank,
        num_procs=num_procs,
        local_global_ids=local_global_ids,
        halo_global_ids=halo_global_ids,
        interior_local_ids=np.asarray(interior_list, dtype=np.int32),
        boundary_local_ids=np.asarray(boundary_list, dtype=np.int32),
        local_neighbors_padded=local_neighbors_padded,
        local_weights_padded=local_weights_padded,
        neighbor_ranks=neighbor_ranks,
        send_local_indices=send_local_indices,
        recv_halo_slices=recv_halo_slices,
        local_fixed_indices=local_fixed_indices,
        local_fixed_values=local_fixed_values,
    )
