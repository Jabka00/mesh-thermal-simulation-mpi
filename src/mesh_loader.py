"""
- Завантаження або генерація STL-файлу за допомогою trimesh
- Обчислення площ граней та центроїдів трикутників
- Побудова структури сусідства (Adjacency Graph), довжин спільних ребер A_ij та відстаней d_ij
- Прив'язка атрибутів (температура T) та початкових/граничних умов
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh


@dataclass
class MeshTopology:
    """Зберігає геометрію та топологію трикутного меша для задачі теплопровідності."""

    mesh: trimesh.Trimesh
    num_faces: int
    centroids: np.ndarray          # (N, 3) координати центроїдів трикутників
    areas: np.ndarray              # (N,) площі трикутників
    normals: np.ndarray            # (N, 3) нормалі трикутників
    adj: List[List[int]]           # adj[i] = [j, k, l] — список сусідів полігона i
    adj_weights: List[List[float]] # adj_weights[i] = [A_ij / d_ij, ...] для кожного сусіда
    face_adjacency: np.ndarray     # (M, 2) пари сусідніх граней
    edge_lengths: np.ndarray       # (M,) довжини спільних ребер A_ij
    centroid_distances: np.ndarray # (M,) відстані між центроїдами d_ij
    conductance: np.ndarray        # (M,) коефіцієнти провідності A_ij / d_ij
    max_neighbors: int             # максимальна кількість сусідів (зазвичай 3 для замкненого трикутного меша)
    neighbors_padded: np.ndarray   # (N, max_neighbors) індекси сусідів (-1 якщо немає)
    weights_padded: np.ndarray     # (N, max_neighbors) ваги A_ij / d_ij (0.0 якщо немає)


def generate_stl_mesh(
    kind: str = "sphere",
    subdivisions: int = 3,
    radius: float = 1.0,
    output_path: Optional[str | Path] = None,
) -> trimesh.Trimesh:
    """
    Генерує тестовий меш (ікосаедр, сфера або тор) та за потреби зберігає у формат STL.

    Parameters
    ----------
    kind : str
        Тип поверхні: 'icosahedron', 'sphere', 'torus'.
    subdivisions : int
        Рівень деталізації (кількість поділів граней ікосаедра для сфери).
        subdivisions=0 -> 20 граней (ікосаедр)
        subdivisions=3 -> 1280 граней
        subdivisions=4 -> 5120 граней
        subdivisions=5 -> 20480 граней
        subdivisions=6 -> 81920 граней
    radius : float
        Радіус моделі.
    output_path : Optional[str | Path]
        Якщо вказано, зберігає згенерований меш у STL-файл.
    """
    if kind == "icosahedron":
        mesh = trimesh.creation.icosahedron()
        mesh.vertices *= radius
    elif kind == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    elif kind == "torus":
        major_sections = max(32, 16 * (2 ** max(0, subdivisions - 1)))
        minor_sections = max(16, 8 * (2 ** max(0, subdivisions - 1)))
        mesh = trimesh.creation.torus(
            major_radius=radius,
            minor_radius=0.35 * radius,
            major_sections=major_sections,
            minor_sections=minor_sections,
        )
    else:
        raise ValueError(f"Невідомий тип меша: {kind}. Доступні: 'icosahedron', 'sphere', 'torus'.")

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(path), file_type="stl")

    return mesh


def load_stl_mesh(stl_path: str | Path) -> trimesh.Trimesh:
    """
    Завантажує STL-файл за допомогою trimesh та об'єднує дубльовані вершини,
    щоб коректно побудувати граф сусідства полігонів.
    """
    mesh = trimesh.load_mesh(str(stl_path), file_type="stl", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Очікувався trimesh.Trimesh, отримано {type(mesh)}")
    mesh.merge_vertices()
    return mesh


def build_mesh_topology(mesh: trimesh.Trimesh) -> MeshTopology:
    """
    - Обчислює площі кожної грані та центроїди трикутників.
    - Будує структуру сусідства adj[i] = [j, k, l] за спільними ребрами.
    - Обчислює довжини спільних ребер A_ij та відстані між центроїдами d_ij.
    """
    num_faces = len(mesh.faces)
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)

    face_adj = np.asarray(mesh.face_adjacency, dtype=np.int32)
    adj_edges = mesh.face_adjacency_edges

    v0 = mesh.vertices[adj_edges[:, 0]]
    v1 = mesh.vertices[adj_edges[:, 1]]
    edge_lengths = np.linalg.norm(v1 - v0, axis=1).astype(np.float64)

    c0 = centroids[face_adj[:, 0]]
    c1 = centroids[face_adj[:, 1]]
    centroid_distances = np.linalg.norm(c1 - c0, axis=1).astype(np.float64)
    centroid_distances = np.maximum(centroid_distances, 1e-12)

    conductance = edge_lengths / centroid_distances

    adj: List[List[int]] = [[] for _ in range(num_faces)]
    adj_weights: List[List[float]] = [[] for _ in range(num_faces)]

    for pair_idx in range(len(face_adj)):
        i = int(face_adj[pair_idx, 0])
        j = int(face_adj[pair_idx, 1])
        w = float(conductance[pair_idx])
        adj[i].append(j)
        adj_weights[i].append(w)
        adj[j].append(i)
        adj_weights[j].append(w)

    max_neighbors = max((len(nbrs) for nbrs in adj), default=3)
    max_neighbors = max(max_neighbors, 3)

    neighbors_padded = np.full((num_faces, max_neighbors), -1, dtype=np.int32)
    weights_padded = np.zeros((num_faces, max_neighbors), dtype=np.float64)

    for i in range(num_faces):
        deg = len(adj[i])
        if deg > 0:
            neighbors_padded[i, :deg] = adj[i]
            weights_padded[i, :deg] = adj_weights[i]

    return MeshTopology(
        mesh=mesh,
        num_faces=num_faces,
        centroids=centroids,
        areas=areas,
        normals=normals,
        adj=adj,
        adj_weights=adj_weights,
        face_adjacency=face_adj,
        edge_lengths=edge_lengths,
        centroid_distances=centroid_distances,
        conductance=conductance,
        max_neighbors=max_neighbors,
        neighbors_padded=neighbors_padded,
        weights_padded=weights_padded,
    )


def initialize_temperature(
    topology: MeshTopology,
    hot_temp: float = 100.0,
    cold_temp: float = 0.0,
    mode: str = "polar_caps",
    hot_fraction: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Етап 1.3: Прив'язка атрибутів та встановлення початкових/граничних умов.

    Parameters
    ----------
    topology : MeshTopology
        Топологія меша.
    hot_temp : float
        Початкова температура гарячих полігонів (наприклад, 100.0).
    cold_temp : float
        Початкова фонова температура (0.0).
    mode : str
        Спосіб вибору початкових гарячих полігонів:
        - 'single': один полігон на верхньому полюсі (max Z)
        - 'polar_caps': група полігонів біля полюса / кількох точок нагріву
        - 'two_sources': два джерела тепла на протилежних кінцях осі Z
    hot_fraction : float
        Частка полігонів, що отримують початковий нагрів (для 'polar_caps' або 'two_sources').

    Returns
    -------
    T0 : np.ndarray
        Початковий розподіл температури (N,).
    hot_indices : np.ndarray
        Індекси початково нагрітих полігонів.
    """
    N = topology.num_faces
    T = np.full(N, cold_temp, dtype=np.float64)
    z = topology.centroids[:, 2]

    if mode == "single":
        hot_idx = np.array([int(np.argmax(z))], dtype=np.int32)
    elif mode == "two_sources":
        k = max(1, int(N * hot_fraction * 0.5))
        order = np.argsort(z)
        hot_idx = np.concatenate([order[-k:], order[:k]]).astype(np.int32)
    else:  # 'polar_caps' або за замовчуванням — шапка на верхньому полюсі + бічний імпульс
        k = max(1, int(N * hot_fraction))
        top_idx = np.argsort(z)[-k:]
        hot_idx = np.asarray(top_idx, dtype=np.int32)

    T[hot_idx] = hot_temp
    return T, hot_idx


def compute_stable_dt(topology: MeshTopology, alpha: float, safety_factor: float = 0.45) -> float:
    """
    Обчислює максимально допустимий крок часу dt за критерієм стійкості
    Куранта-Фрідріхса-Леві (CFL) для явної схеми Ейлера:

        dt_max = 1 / (alpha * max_i sum_j (A_ij / d_ij))
    """
    row_sums = np.sum(topology.weights_padded, axis=1)
    max_rate = float(np.max(row_sums)) * alpha
    if max_rate <= 0.0:
        return 0.01
    dt_crit = 1.0 / max_rate
    return safety_factor * dt_crit
