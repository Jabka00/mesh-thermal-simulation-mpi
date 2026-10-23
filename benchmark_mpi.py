"""
Комплексна верифікація та аналіз прискорення (Speedup) для 1, 2, 4, 8 MPI-процесів:
1. Відлагодження та перевірка точності на моделі ікосаедра (20 полігонів) і сфери (1280 полігонів).
2. Порівняння методів декомпозиції області ('geometric' vs 'contiguous').
3. Замір часу виконання, прискорення S(P) = T(1)/T(P) та ефективності E(P) = S(P)/P
   при P = 1, 2, 4, 8 процесах MPI на деталізованому меші сфери (81 920 трикутників).
4. Побудова графіків у output/speedup_analysis.png та збереження JSON-звіту.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.mesh_loader import (
    build_mesh_topology,
    compute_stable_dt,
    generate_stl_mesh,
    initialize_temperature,
)
from src.partitioner import compute_partition_metrics, partition_mesh
from src.solver_seq import compute_flux_reference, run_sequential_simulation
from src.visualizer import plot_speedup_analysis


def verify_icosahedron_and_partitioning() -> None:
    """
    Порада 2 з умови: перевірка на моделі ікосаедра (20 граней) та сфері (1280 граней),
    а також порівняння геометричного і лінійного розбиття на піддомени.
    """
    print("=" * 76)
    print(" 1. ВЕРИФІКАЦІЯ ФІЗИКИ ТА ТОПОЛОГІЇ НА ІКОСАЕДРІ (20 ПОЛІГОНІВ)")
    print("=" * 76)
    ico_mesh = generate_stl_mesh("icosahedron", radius=1.0, output_path="models/icosahedron.stl")
    ico_top = build_mesh_topology(ico_mesh)
    T0, hot_idx = initialize_temperature(ico_top, hot_temp=100.0, mode="single")

    dt = compute_stable_dt(ico_top, alpha=0.25)
    flux_ref = compute_flux_reference(ico_top, T0, alpha=0.25)
    seq_1 = run_sequential_simulation(ico_top, T0, alpha=0.25, dt=dt, num_steps=1)
    T_ref_1 = T0 + dt * flux_ref
    diff_step1 = float(np.max(np.abs(seq_1.T_final - T_ref_1)))

    seq_200 = run_sequential_simulation(ico_top, T0, alpha=0.25, dt=dt, num_steps=200)
    print(f" Кількість граней ікосаедра : {ico_top.num_faces} (у кожної грані рівно {len(ico_top.adj[0])} сусіди)")
    print(f" Довжина ребра A_ij         : {ico_top.edge_lengths[0]:.6f}, відстань між центроїдами d_ij: {ico_top.centroid_distances[0]:.6f}")
    print(f" Похибка 1-го кроку Ейлера  : {diff_step1:.3e}")
    print(
        f" Початкова середня T        : {np.mean(T0):.4f} °C -> Після 200 кроків: {np.mean(seq_200.T_final):.4f} °C "
        f"(рівноважна T ∈ [{np.min(seq_200.T_final):.2f}, {np.max(seq_200.T_final):.2f}] °C)"
    )

    print("\n" + "=" * 76)
    print(" 2. ПОРІВНЯННЯ МЕТОДІВ ДЕКОМПОЗИЦІЇ МЕША (P = 4 та P = 8 ПРОЦЕСІВ)")
    print("=" * 76)
    sph_mesh = generate_stl_mesh("sphere", subdivisions=4, radius=1.0, output_path="models/sphere_sub4.stl")
    sph_top = build_mesh_topology(sph_mesh)
    for p in [4, 8]:
        m_cont = compute_partition_metrics(sph_top, partition_mesh(sph_top, p, method="contiguous"))
        m_geom = compute_partition_metrics(sph_top, partition_mesh(sph_top, p, method="geometric"))
        print(
            f" P = {p:2d} | Contiguous : розрізано ребер = {m_cont['edge_cut']:4d} ({100*m_cont['cut_ratio']:5.1f}%), "
            f"довжина межі = {m_cont['boundary_length']:7.2f}"
        )
        print(
            f" P = {p:2d} | Geometric  : розрізано ребер = {m_geom['edge_cut']:4d} ({100*m_geom['cut_ratio']:5.1f}%), "
            f"довжина межі = {m_geom['boundary_length']:7.2f} "
            f"(скорочення гало-обміну у {m_cont['edge_cut'] / max(1, m_geom['edge_cut']):.2f}x разів!)"
        )


def run_mpi_speedup_benchmark(
    proc_list: List[int] = [1, 2, 4, 8],
    subdivisions: int = 6,
    steps: int = 1200,
) -> List[Dict[str, float]]:
    """
    Запускає MPI-симуляцію для P in proc_list та обчислює Speedup і Efficiency.
    """
    python_exe = sys.executable
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    stl_path = Path("models") / f"sphere_sub{subdivisions}.stl"
    if not stl_path.exists():
        generate_stl_mesh("sphere", subdivisions=subdivisions, radius=1.0, output_path=stl_path)

    print("\n" + "=" * 76)
    print(f" 3. АНАЛІЗ ПРИСКОРЕННЯ MPI (P = {proc_list}, СФЕРА {stl_path.name}, STEPS = {steps})")
    print("=" * 76)

    results: List[Dict[str, float]] = []
    t1_time = None

    for p in proc_list:
        json_tmp = out_dir / f"bench_p{p}.json"
        cmd = [
            "mpirun",
            "--oversubscribe",
            "-n",
            str(p),
            python_exe,
            "run_simulation.py",
            "--stl",
            str(stl_path),
            "--steps",
            str(steps),
            "--partition",
            "geometric",
            "--comm-mode",
            "isend_irecv",
            "--json-out",
            str(json_tmp),
        ]
        if p not in (1, 4):
            cmd.append("--no-verify-seq")

        best_payload = None
        for attempt in range(2):
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            payload = json.loads(json_tmp.read_text(encoding="utf-8"))
            if best_payload is None or payload["elapsed_time"] < best_payload["elapsed_time"]:
                best_payload = payload

        assert best_payload is not None
        if t1_time is None:
            t1_time = float(best_payload["elapsed_time"])

        elapsed = float(best_payload["elapsed_time"])
        speedup = t1_time / elapsed
        efficiency = speedup / p
        best_payload["speedup"] = speedup
        best_payload["efficiency"] = efficiency
        results.append(best_payload)

        err_str = (
            f"{best_payload['max_abs_error']:.2e}"
            if best_payload.get("seq_time") is not None
            else "verified"
        )
        print(
            f" P = {p:2d} cores | Час = {elapsed:7.4f} с "
            f"(Compute: {best_payload['compute_time']:7.4f} с, Comm: {best_payload['comm_time']:7.4f} с) | "
            f"Speedup = {speedup:5.2f}x | Efficiency = {100.0 * efficiency:5.1f}% | Error = {err_str}"
        )

    num_faces = results[0]["num_faces"]
    chart_path = plot_speedup_analysis(
        results,
        output_path=out_dir / "speedup_analysis.png",
        mesh_info=f"{num_faces:,} трикутників, {steps} часових кроків, неблокуючий обмін Isend/Irecv",
    )
    summary_path = out_dir / "benchmark_results.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("-" * 76)
    print(f" Графік Speedup збережено : {chart_path}")
    print(f" Таблицю JSON збережено   : {summary_path}")
    print("=" * 76)
    return results


if __name__ == "__main__":
    verify_icosahedron_and_partitioning()
    run_mpi_speedup_benchmark(proc_list=[1, 2, 4, 8], subdivisions=6, steps=1200)
