"""
Головний скрипт запуску моделювання теплопровідності на поверхні 3D-меша (Послідовно та через MPI).

Приклади запуску:
  1) Перевірка на сфера-моделі / ікосаедрі з повною візуалізацією (4 процеси MPI):
     mpirun -n 4 .venv/bin/python run_simulation.py --mesh sphere --subdivisions 3 --steps 400 --gif

  2) Запуск на власному STL-файлі:
     mpirun -n 4 .venv/bin/python run_simulation.py --stl models/sphere.stl --steps 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from src.mesh_loader import (
    build_mesh_topology,
    compute_stable_dt,
    generate_stl_mesh,
    initialize_temperature,
    load_stl_mesh,
)
from src.partitioner import build_subdomain, compute_partition_metrics, partition_mesh
from src.solver_mpi import run_mpi_simulation
from src.solver_seq import run_sequential_simulation
from src.visualizer import (
    compute_boundary_mask,
    create_thermal_gif_pyvista,
    export_to_vtk,
    render_thermal_snapshots_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Моделювання теплопровідності на поверхні трикутного STL-меша з MPI"
    )
    parser.add_argument(
        "--stl",
        type=str,
        default=None,
        help="Шлях до вхідного STL-файлу (якщо не вказано, генерується автоматично)",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="sphere",
        choices=["icosahedron", "sphere", "torus"],
        help="Тип генерації тестового меша, якщо --stl не задано",
    )
    parser.add_argument(
        "--subdivisions",
        type=int,
        default=3,
        help="Рівень деталізації сфери (0=20 граней, 3=1280, 4=5120, 5=20480, 6=81920)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Коефіцієнт теплопровідності alpha",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Крок за часом dt (якщо не задано, обчислюється автоматично за критерієм CFL)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=400,
        help="Кількість кроків за часом",
    )
    parser.add_argument(
        "--hot-temp",
        type=float,
        default=100.0,
        help="Початкова температура гарячої зони (°C)",
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        default="two_sources",
        choices=["single", "polar_caps", "two_sources"],
        help="Режим початкового нагріву полігонів",
    )
    parser.add_argument(
        "--fixed-source",
        action="store_true",
        help="Підтримувати постійну температуру гарячих полігонів (гранична умова Діріхле)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="geometric",
        choices=["geometric", "contiguous"],
        help="Метод декомпозиції області між MPI-процесами",
    )
    parser.add_argument(
        "--comm-mode",
        type=str,
        default="isend_irecv",
        choices=["isend_irecv", "sendrecv"],
        help="Режим обміну гало-даними в MPI",
    )
    parser.add_argument(
        "--verify-seq",
        action="store_true",
        default=True,
        help="Порівняти результат MPI із послідовним солвером на rank == 0",
    )
    parser.add_argument(
        "--no-verify-seq",
        action="store_false",
        dest="verify_seq",
        help="Вимкнути запуск послідовного солвера (для швидких бенчмарків)",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Згенерувати GIF-анімацію та PNG-панель за допомогою PyVista",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Директорія для збереження VTK/VTU, GIF та звітів",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Якщо вказано, зберегти метрики запуску у JSON-файл",
    )
    return parser.parse_args()


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_args()
    out_dir = Path(args.output_dir)

    if args.stl is not None:
        stl_path = Path(args.stl)
    else:
        stl_name = (
            "icosahedron.stl"
            if args.mesh == "icosahedron"
            else f"{args.mesh}_sub{args.subdivisions}.stl"
        )
        stl_path = Path("models") / stl_name
        if rank == 0:
            generate_stl_mesh(
                kind=args.mesh,
                subdivisions=args.subdivisions,
                radius=1.0,
                output_path=stl_path,
            )
    comm.Barrier()

    mesh = load_stl_mesh(stl_path)
    topology = build_mesh_topology(mesh)

    dt_cfl = compute_stable_dt(topology, args.alpha, safety_factor=0.40)
    dt = args.dt if args.dt is not None else dt_cfl

    T0, hot_indices = initialize_temperature(
        topology,
        hot_temp=args.hot_temp,
        cold_temp=0.0,
        mode=args.init_mode,
        hot_fraction=0.04,
    )
    fixed_idx = hot_indices if args.fixed_source else None
    fixed_val = args.hot_temp if args.fixed_source else None

    part_labels = partition_mesh(topology, num_procs=size, method=args.partition)
    subdomain = build_subdomain(
        topology=topology,
        part_labels=part_labels,
        rank=rank,
        num_procs=size,
        fixed_global_indices=fixed_idx,
        fixed_global_values=fixed_val,
    )

    snapshot_interval = max(1, args.steps // 32) if args.gif else 0

    if rank == 0:
        part_metrics = compute_partition_metrics(topology, part_labels)
        print("=" * 72)
        print(" МОДЕЛЮВАННЯ ТЕПЛОПРОВІДНОСТІ НА ПОВЕРХНІ МЕША (MPI)")
        print("=" * 72)
        print(f" STL-модель            : {stl_path} ({topology.num_faces} трикутників)")
        print(f" Середня площа грані   : {np.mean(topology.areas):.6f}")
        print(f" Сусідніх пар (ребер)  : {len(topology.face_adjacency)}")
        print(f" Коефіцієнт alpha      : {args.alpha}")
        print(f" Крок часу dt          : {dt:.6e} с (CFL безпечний <= {dt_cfl:.6e} с)")
        print(f" Кількість кроків      : {args.steps} (загальний час t = {args.steps * dt:.4f} с)")
        print(f" Початково нагріто     : {len(hot_indices)} полігонів (T0 = {args.hot_temp} °C)")
        print(f" MPI процесів (P)      : {size} | Декомпозиція: {args.partition} | Обмін: {args.comm_mode}")
        print(
            f" Розрізаних ребер      : {part_metrics['edge_cut']} / {part_metrics['total_edges']} "
            f"({100.0 * part_metrics['cut_ratio']:.1f}%), дисбаланс: {part_metrics['load_imbalance']:.3f}"
        )
        print("-" * 72)

    local_info = (
        subdomain.rank,
        len(subdomain.local_global_ids),
        len(subdomain.interior_local_ids),
        len(subdomain.boundary_local_ids),
        len(subdomain.halo_global_ids),
        subdomain.neighbor_ranks,
    )
    all_info = comm.gather(local_info, root=0)
    if rank == 0 and all_info is not None:
        for r_id, n_loc, n_int, n_bnd, n_halo, nbrs in all_info:
            print(
                f" [Rank {r_id}] Local: {n_loc:6d} (Interior: {n_int:6d}, Boundary: {n_bnd:5d}) | "
                f"Halo cells: {n_halo:5d} | Сусіди: {nbrs}"
            )
        print("-" * 72)

    mpi_res = run_mpi_simulation(
        comm=comm,
        subdomain=subdomain,
        T0_global=T0,
        alpha=args.alpha,
        dt=dt,
        num_steps=args.steps,
        comm_mode=args.comm_mode,
        snapshot_interval=snapshot_interval,
    )

    if rank == 0 and mpi_res.T_final_global is not None:
        T_mpi = mpi_res.T_final_global
        print(f" Час виконання MPI     : {mpi_res.elapsed_time:.5f} с")
        print(
            f"   - Обчислення (rank0): {mpi_res.compute_time:.5f} с | "
            f"Обмін гало (rank0): {mpi_res.comm_time:.5f} с"
        )
        print(
            f" Фінальна температура  : min = {np.min(T_mpi):.4f} °C, "
            f"max = {np.max(T_mpi):.4f} °C, mean = {np.mean(T_mpi):.4f} °C"
        )

        max_abs_err = 0.0
        seq_time = None
        if args.verify_seq:
            seq_res = run_sequential_simulation(
                topology=topology,
                T0=T0,
                alpha=args.alpha,
                dt=dt,
                num_steps=args.steps,
                fixed_indices=fixed_idx,
                fixed_values=fixed_val,
                snapshot_interval=0,
            )
            seq_time = seq_res.elapsed_time
            max_abs_err = float(np.max(np.abs(T_mpi - seq_res.T_final)))
            print(f" Час послідовної версії: {seq_time:.5f} с")
            print(f" Різниця |T_mpi - T_seq|: max = {max_abs_err:.3e} (чітка числова відповідність)")

        boundary_mask = compute_boundary_mask(topology.face_adjacency, part_labels, topology.num_faces)
        vtu_path = export_to_vtk(
            mesh=mesh,
            temperature=T_mpi,
            output_path=out_dir / "thermal_result.vtu",
            initial_temperature=T0,
            part_labels=part_labels,
            boundary_mask=boundary_mask,
        )
        vtk_path = export_to_vtk(
            mesh=mesh,
            temperature=T_mpi,
            output_path=out_dir / "thermal_result.vtk",
            initial_temperature=T0,
            part_labels=part_labels,
            boundary_mask=boundary_mask,
        )
        print(f" Збережено VTU/VTK     : {vtu_path}, {vtk_path}")

        if args.gif and len(mpi_res.history_global) > 1:
            gif_path = create_thermal_gif_pyvista(
                mesh=mesh,
                history=mpi_res.history_global,
                history_steps=mpi_res.history_steps,
                dt=dt,
                gif_path=out_dir / "thermal_diffusion.gif",
                clim=(0.0, args.hot_temp),
                show_edges=(topology.num_faces <= 5500),
            )
            panel_path = render_thermal_snapshots_panel(
                mesh=mesh,
                history=mpi_res.history_global,
                history_steps=mpi_res.history_steps,
                dt=dt,
                output_path=out_dir / "thermal_overview.png",
                part_labels=part_labels,
                boundary_mask=boundary_mask,
            )
            print(f" Збережено GIF-анімацію: {gif_path}")
            print(f" Збережено огляд PNG   : {panel_path}")

        if args.json_out:
            json_path = Path(args.json_out)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "procs": size,
                "num_faces": topology.num_faces,
                "steps": args.steps,
                "dt": dt,
                "alpha": args.alpha,
                "partition": args.partition,
                "comm_mode": args.comm_mode,
                "elapsed_time": mpi_res.elapsed_time,
                "compute_time": mpi_res.compute_time,
                "comm_time": mpi_res.comm_time,
                "seq_time": seq_time,
                "max_abs_error": max_abs_err,
                "T_min": float(np.min(T_mpi)),
                "T_max": float(np.max(T_mpi)),
                "T_mean": float(np.mean(T_mpi)),
                "edge_cut": part_metrics["edge_cut"],
                "cut_ratio": part_metrics["cut_ratio"],
            }
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("=" * 72)


if __name__ == "__main__":
    main()
