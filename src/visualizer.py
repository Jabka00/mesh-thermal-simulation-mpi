"""
- Збереження результатів у формат VTK/VTU за допомогою pyvista (температура, ранг MPI, граничні клітинки)
- Експорт 3D-анімації (GIF) та статичних кадрів (PNG), де колір полігона відповідає його температурі
- Візуалізація декомпозиції меша на піддомени MPI та гало-меж
- Побудова графіків прискорення (Speedup) та ефективності при збільшенні кількості MPI-процесів (1, 2, 4, 8 cores)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import trimesh


def trimesh_to_pyvista(mesh: trimesh.Trimesh) -> pv.PolyData:
    """Конвертує trimesh.Trimesh у pyvista.PolyData."""
    faces = mesh.faces
    n_faces = len(faces)
    pv_faces = np.column_stack([np.full(n_faces, 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), pv_faces)


def compute_boundary_mask(face_adjacency: np.ndarray, part_labels: np.ndarray, num_faces: int) -> np.ndarray:
    """Повертає масив (N,), де 1 позначає граничні полігони піддоменів (що мають сусідів у гало-зоні), 0 — внутрішні."""
    mask = np.zeros(num_faces, dtype=np.int32)
    f0 = face_adjacency[:, 0]
    f1 = face_adjacency[:, 1]
    cut = part_labels[f0] != part_labels[f1]
    mask[f0[cut]] = 1
    mask[f1[cut]] = 1
    return mask


def export_to_vtk(
    mesh: trimesh.Trimesh,
    temperature: np.ndarray,
    output_path: str | Path,
    initial_temperature: Optional[np.ndarray] = None,
    part_labels: Optional[np.ndarray] = None,
    boundary_mask: Optional[np.ndarray] = None,
) -> Path:
    """
    Зберігає трикутний меш разом з атрибутами полігонів (cell_data) у формат VTK/VTU.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pv_mesh = trimesh_to_pyvista(mesh)
    pv_mesh.cell_data["Temperature"] = np.asarray(temperature, dtype=np.float64)

    if initial_temperature is not None:
        pv_mesh.cell_data["Initial_Temperature"] = np.asarray(initial_temperature, dtype=np.float64)
    if part_labels is not None:
        pv_mesh.cell_data["MPI_Rank"] = np.asarray(part_labels, dtype=np.int32)
    if boundary_mask is not None:
        pv_mesh.cell_data["Is_Boundary_Cell"] = np.asarray(boundary_mask, dtype=np.int32)

    if out.suffix.lower() == ".vtu":
        ugrid = pv_mesh.cast_to_unstructured_grid()
        ugrid.save(str(out))
    else:
        pv_mesh.save(str(out))

    return out


def create_thermal_gif_pyvista(
    mesh: trimesh.Trimesh,
    history: List[np.ndarray],
    history_steps: List[int],
    dt: float,
    gif_path: str | Path,
    clim: Optional[ tuple[float, float] ] = None,
    fps: int = 12,
    show_edges: bool = True,
) -> Path:
    """
    Створює 3D-анімацію (GIF) поширення тепла на поверхні меша за допомогою PyVista.
    """
    out = Path(gif_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pv_mesh = trimesh_to_pyvista(mesh)
    if clim is None:
        vmin = float(min(np.min(f) for f in history))
        vmax = float(max(np.max(f) for f in history))
    else:
        vmin, vmax = clim

    pv_mesh.cell_data["Temperature"] = np.asarray(history[0], dtype=np.float64)

    plotter = pv.Plotter(off_screen=True, window_size=[800, 700])
    plotter.set_background("#12161f")
    plotter.open_gif(str(out), fps=fps)

    plotter.add_mesh(
        pv_mesh,
        scalars="Temperature",
        cmap="inferno",
        clim=[vmin, vmax],
        show_edges=show_edges,
        edge_color="#2a2f3a",
        line_width=0.5,
        preference="cell",
        scalar_bar_args={
            "title": "Temperature T (°C)",
            "color": "white",
            "vertical": True,
            "fmt": "%.1f",
        },
    )
    plotter.view_isometric()
    plotter.camera.zoom(1.15)

    for frame_idx, (T_frame, step) in enumerate(zip(history, history_steps)):
        pv_mesh.cell_data["Temperature"] = np.asarray(T_frame, dtype=np.float64)
        t_val = step * dt
        plotter.add_text(
            f"Mesh Thermal Diffusion (MPI)\nStep: {step} | t = {t_val:.4f} s\n"
            f"T_max = {np.max(T_frame):.2f} °C | T_min = {np.min(T_frame):.2f} °C",
            position="upper_left",
            font_size=11,
            color="white",
            name="status_text",
        )
        plotter.write_frame()

    plotter.close()
    return out


def render_thermal_snapshots_panel(
    mesh: trimesh.Trimesh,
    history: List[np.ndarray],
    history_steps: List[int],
    dt: float,
    output_path: str | Path,
    part_labels: Optional[np.ndarray] = None,
    boundary_mask: Optional[np.ndarray] = None,
) -> Path:
    """
    Створює багатопанельне зображення (PNG) ключових етапів поширення тепла
    та розбиття меша на піддомени MPI з виділенням гало-меж.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_hist = len(history)
    indices = [0, max(0, n_hist // 6), max(0, n_hist // 2), n_hist - 1]
    vmin = float(np.min(history[0]))
    vmax = float(np.max(history[0]))

    pv_mesh = trimesh_to_pyvista(mesh)
    show_edges = len(mesh.faces) <= 6000

    images = []
    titles = []

    for idx in indices:
        T_frame = history[idx]
        step = history_steps[idx]
        p = pv.Plotter(off_screen=True, window_size=[650, 600])
        p.set_background("#12161f")
        m_copy = pv_mesh.copy()
        m_copy.cell_data["Temperature"] = np.asarray(T_frame, dtype=np.float64)
        p.add_mesh(
            m_copy,
            scalars="Temperature",
            cmap="inferno",
            clim=[vmin, vmax],
            show_edges=show_edges,
            edge_color="#2e3440",
            line_width=0.4,
            preference="cell",
            scalar_bar_args={"title": "T (°C)", "color": "white", "fmt": "%.1f"},
        )
        p.view_isometric()
        p.camera.zoom(1.18)
        img = p.screenshot(return_img=True)
        p.close()
        images.append(img)
        titles.append(
            f"Крок {step} (t = {step * dt:.4f} с)\n"
            f"T ∈ [{np.min(T_frame):.1f}, {np.max(T_frame):.1f}] °C"
        )

    extra_imgs = []
    extra_titles = []
    if part_labels is not None:
        num_procs = int(np.max(part_labels)) + 1
        p = pv.Plotter(off_screen=True, window_size=[650, 600])
        p.set_background("#12161f")
        m_copy = pv_mesh.copy()
        m_copy.cell_data["MPI_Rank"] = np.asarray(part_labels, dtype=np.int32)
        p.add_mesh(
            m_copy,
            scalars="MPI_Rank",
            cmap="tab10",
            show_edges=True,
            edge_color="#222222",
            line_width=0.5,
            preference="cell",
            scalar_bar_args={"title": "MPI Rank", "color": "white", "n_labels": num_procs, "fmt": "%.0f"},
        )
        p.view_isometric()
        p.camera.zoom(1.18)
        extra_imgs.append(p.screenshot(return_img=True))
        p.close()
        extra_titles.append(f"Декомпозиція меша (P = {num_procs} процесів)\nКожен колір — окремий MPI Rank")

    if boundary_mask is not None:
        p = pv.Plotter(off_screen=True, window_size=[650, 600])
        p.set_background("#12161f")
        m_copy = pv_mesh.copy()
        m_copy.cell_data["Halo_Boundary"] = np.asarray(boundary_mask, dtype=np.int32)
        p.add_mesh(
            m_copy,
            scalars="Halo_Boundary",
            cmap="coolwarm",
            show_edges=True,
            edge_color="#2a2e38",
            line_width=0.5,
            preference="cell",
            scalar_bar_args={"title": "Boundary (1) / Interior (0)", "color": "white", "n_labels": 2, "fmt": "%.0f"},
        )
        p.view_isometric()
        p.camera.zoom(1.18)
        extra_imgs.append(p.screenshot(return_img=True))
        p.close()
        n_bnd = int(np.sum(boundary_mask))
        extra_titles.append(
            f"Граничні полігони та гало-зони\n"
            f"Граничних: {n_bnd} ({100.0 * n_bnd / len(boundary_mask):.1f}%), Внутрішніх: {len(boundary_mask) - n_bnd}"
        )

    all_imgs = images + extra_imgs
    all_titles = titles + extra_titles

    ncols = 3 if len(all_imgs) > 4 else 2
    nrows = (len(all_imgs) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.2 * nrows), facecolor="#12161f")
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, img, title in zip(axes_flat, all_imgs, all_titles):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    for ax in axes_flat[len(all_imgs):]:
        ax.axis("off")

    fig.suptitle(
        f"Моделювання теплопровідності на поверхні 3D-меша ({len(mesh.faces)} полігонів)",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(out), dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_speedup_analysis(
    benchmark_results: List[Dict[str, float]],
    output_path: str | Path,
    mesh_info: str = "",
) -> Path:
    """
    Будує графіки аналізу продуктивності MPI:
    1. Час виконання (Execution Time vs MPI Processes) + розбиття на Compute та Communication
    2. Прискорення (Speedup S_p = T_1 / T_p) порівняно з ідеальним лінійним
    3. Паралельна ефективність (Efficiency E_p = S_p / P * 100%)
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    procs = [int(r["procs"]) for r in benchmark_results]
    times = [float(r["elapsed_time"]) for r in benchmark_results]
    speedups = [float(r["speedup"]) for r in benchmark_results]
    efficiencies = [float(r["efficiency"]) * 100.0 for r in benchmark_results]
    comp_times = [float(r.get("compute_time", r["elapsed_time"])) for r in benchmark_results]
    comm_times = [float(r.get("comm_time", 0.0)) for r in benchmark_results]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), facecolor="#12161f")
    for ax in axes:
        ax.set_facecolor("#1a202c")
        ax.grid(True, linestyle="--", alpha=0.3)

    x = np.arange(len(procs))
    width = 0.52
    axes[0].bar(x, comp_times, width, label="Обчислення (Compute)", color="#38bdf8")
    axes[0].bar(x, comm_times, width, bottom=comp_times, label="Обмін гало (MPI Comm)", color="#f97316")
    axes[0].plot(x, times, "o-", color="#facc15", linewidth=2, label="Загальний час (Wall Time)")
    for xi, t in zip(x, times):
        axes[0].text(xi, t * 1.03, f"{t:.3f}s", ha="center", va="bottom", color="white", fontsize=9.5, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(p) for p in procs])
    axes[0].set_xlabel("Кількість MPI-процесів (P)", fontsize=11)
    axes[0].set_ylabel("Час виконання (с)", fontsize=11)
    axes[0].set_title("Час виконання та структура витрат", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, max(times) * 1.18)
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].plot(procs, procs, "--", color="#94a3b8", linewidth=1.6, label="Ідеальне прискорення (S = P)")
    axes[1].plot(procs, speedups, "s-", color="#22c55e", linewidth=2.6, markersize=8, label="Фактичне прискорення (MPI)")
    for p, s in zip(procs, speedups):
        axes[1].annotate(
            f"{s:.2f}x",
            (p, s),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="#4ade80",
            fontsize=10,
            fontweight="bold",
        )
    axes[1].set_xticks(procs)
    axes[1].set_xlabel("Кількість MPI-процесів (P)", fontsize=11)
    axes[1].set_ylabel("Прискорення S(P) = T(1) / T(P)", fontsize=11)
    axes[1].set_title("Прискорення (Speedup)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper left", fontsize=9.5)

    axes[2].plot(procs, efficiencies, "D-", color="#a855f7", linewidth=2.5, markersize=8, label="Ефективність E(P)")
    axes[2].axhline(100.0, linestyle="--", color="#94a3b8", linewidth=1.4, label="Ідеал (100%)")
    for p, eff in zip(procs, efficiencies):
        axes[2].annotate(
            f"{eff:.1f}%",
            (p, eff),
            textcoords="offset points",
            xytext=(0, -16 if eff > 85 else 10),
            ha="center",
            color="#d8b4fe",
            fontsize=10,
            fontweight="bold",
        )
    axes[2].set_xticks(procs)
    axes[2].set_ylim(0, 115)
    axes[2].set_xlabel("Кількість MPI-процесів (P)", fontsize=11)
    axes[2].set_ylabel("Ефективність E(P) = S(P)/P (%)", fontsize=11)
    axes[2].set_title("Паралельна ефективність (%)", fontsize=12, fontweight="bold")
    axes[2].legend(loc="lower left", fontsize=9.5)

    title_str = "Аналіз масштабованості паралельного MPI-солвера теплопровідності"
    if mesh_info:
        title_str += f"\n({mesh_info})"
    fig.suptitle(title_str, color="white", fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(str(out), dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out
