"""1. Exchange Boundary Data (Обмін межами):
   - Асинхронний обмін comm.Isend / comm.Irecv (з перекриттям обчислень внутрішніх полігонів)
   - Або блокуючий парний обмін comm.Sendrecv
2. Update Local States (Обчислення):
   - Розрахунок нових температур для внутрішніх полігонів під час передачі гало-зон
   - Очікування завершення обміну (MPI.Request.Waitall) та розрахунок для граничних полігонів
3. Gathering (Збір результатів):
   - Збір локальних масивів температур на кореневий процес (rank == 0) через comm.Gatherv
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from mpi4py import MPI

from .partitioner import SubdomainData


@dataclass
class MPISimulationResult:
    """Результати паралельного MPI-моделювання (на rank == 0 містить повні глобальні поля)."""

    rank: int
    num_procs: int
    T_final_global: Optional[np.ndarray]
    T_final_local: np.ndarray
    history_global: List[np.ndarray]
    history_steps: List[int]
    elapsed_time: float
    comm_time: float
    compute_time: float
    num_steps: int
    dt: float
    alpha: float


def gather_global_temperature(
    comm: MPI.Comm,
    subdomain: SubdomainData,
    T_local: np.ndarray,
    total_faces: int,
    all_global_ids_packed: Optional[np.ndarray] = None,
    counts: Optional[np.ndarray] = None,
    displs: Optional[np.ndarray] = None,
    metadata_ready: bool = False,
) -> Optional[np.ndarray]:
    """
    Збирає локальні масиви температур T_local з усіх MPI-процесів на rank == 0
    за допомогою comm.Gatherv та відновлює глобальну нумерацію полігонів [0..N-1].
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_n = np.int32(len(T_local))
    if not metadata_ready:
        counts = np.zeros(size, dtype=np.int32) if rank == 0 else None
        comm.Gather(np.array([local_n], dtype=np.int32), counts, root=0)
        if rank == 0 and counts is not None:
            displs = np.zeros(size, dtype=np.int32)
            displs[1:] = np.cumsum(counts[:-1])
            all_global_ids_packed = np.empty(total_faces, dtype=np.int32)
        comm.Gatherv(
            subdomain.local_global_ids,
            [all_global_ids_packed, counts, displs, MPI.INT] if rank == 0 else None,
            root=0,
        )

    recv_buf = np.empty(total_faces, dtype=np.float64) if rank == 0 else None
    comm.Gatherv(
        T_local,
        [recv_buf, counts, displs, MPI.DOUBLE] if rank == 0 else None,
        root=0,
    )

    if rank == 0 and recv_buf is not None and all_global_ids_packed is not None:
        T_global = np.empty(total_faces, dtype=np.float64)
        T_global[all_global_ids_packed] = recv_buf
        return T_global
    return None


def run_mpi_simulation(
    comm: MPI.Comm,
    subdomain: SubdomainData,
    T0_global: np.ndarray,
    alpha: float,
    dt: float,
    num_steps: int,
    comm_mode: str = "isend_irecv",
    snapshot_interval: int = 0,
    synchronize_steps: bool = False,
) -> MPISimulationResult:
    """
    Виконує паралельне моделювання теплопровідності за допомогою MPI.

    Parameters
    ----------
    comm : MPI.Comm
        Комунікатор MPI (зазвичай MPI.COMM_WORLD).
    subdomain : SubdomainData
        Локальний піддомен поточного процесу.
    T0_global : np.ndarray
        Початкове поле температур (N,).
    alpha : float
        Коефіцієнт теплопровідності.
    dt : float
        Крок за часом.
    num_steps : int
        Кількість часових кроків.
    comm_mode : str
        Спосіб обміну гало-даними: 'isend_irecv' (неблокуючий з перекриттям обчислень)
        або 'sendrecv' (блокуючий парний обмін).
    snapshot_interval : int
        Інтервал збереження глобальних знімків температури (0 — не зберігати проміжні).
    synchronize_steps : bool
        Чи викликати comm.Barrier() наприкінці кожного кроку.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    total_faces = len(T0_global)

    N_local = len(subdomain.local_global_ids)
    N_halo = len(subdomain.halo_global_ids)

    T_ext = np.empty(N_local + N_halo, dtype=np.float64)
    T_ext[:N_local] = T0_global[subdomain.local_global_ids]
    if N_halo > 0:
        T_ext[N_local:] = T0_global[subdomain.halo_global_ids]

    T_new_local = np.empty(N_local, dtype=np.float64)

    int_ids = subdomain.interior_local_ids
    bnd_ids = subdomain.boundary_local_ids

    int_nbrs = subdomain.local_neighbors_padded[int_ids]
    int_weights = subdomain.local_weights_padded[int_ids]

    bnd_nbrs = subdomain.local_neighbors_padded[bnd_ids]
    bnd_weights = subdomain.local_weights_padded[bnd_ids]

    all_nbrs = subdomain.local_neighbors_padded
    all_weights = subdomain.local_weights_padded

    send_buffers: Dict[int, np.ndarray] = {
        nbr: np.empty(len(idx), dtype=np.float64)
        for nbr, idx in subdomain.send_local_indices.items()
    }

    local_n = np.int32(N_local)
    counts = np.zeros(size, dtype=np.int32) if rank == 0 else None
    comm.Gather(np.array([local_n], dtype=np.int32), counts, root=0)
    displs = None
    all_global_ids_packed = None
    if rank == 0 and counts is not None:
        displs = np.zeros(size, dtype=np.int32)
        displs[1:] = np.cumsum(counts[:-1])
        all_global_ids_packed = np.empty(total_faces, dtype=np.int32)

    comm.Gatherv(
        subdomain.local_global_ids,
        [all_global_ids_packed, counts, displs, MPI.INT] if rank == 0 else None,
        root=0,
    )

    history_global: List[np.ndarray] = []
    history_steps: List[int] = []

    if snapshot_interval > 0:
        if rank == 0:
            history_global.append(T0_global.copy())
            history_steps.append(0)

    dt_alpha = dt * alpha
    comm_time = 0.0
    compute_time = 0.0

    comm.Barrier()
    t_start = MPI.Wtime()

    for step in range(1, num_steps + 1):
        t_c0 = MPI.Wtime()

        if size > 1 and len(subdomain.neighbor_ranks) > 0:
            if comm_mode == "isend_irecv":
                requests: List[MPI.Request] = []
                for nbr in subdomain.neighbor_ranks:
                    h_start, h_end = subdomain.recv_halo_slices[nbr]
                    req_recv = comm.Irecv(
                        [T_ext[h_start:h_end], MPI.DOUBLE],
                        source=nbr,
                        tag=100,
                    )
                    requests.append(req_recv)

                for nbr in subdomain.neighbor_ranks:
                    s_buf = send_buffers[nbr]
                    np.take(T_ext, subdomain.send_local_indices[nbr], out=s_buf)
                    req_send = comm.Isend(
                        [s_buf, MPI.DOUBLE],
                        dest=nbr,
                        tag=100,
                    )
                    requests.append(req_send)

                comm_time += MPI.Wtime() - t_c0

                t_comp0 = MPI.Wtime()
                if len(int_ids) > 0:
                    T_int = T_ext[int_ids]
                    T_int_nbrs = T_ext[int_nbrs]
                    flux_int = np.sum(int_weights * (T_int_nbrs - T_int[:, None]), axis=1)
                    T_new_local[int_ids] = T_int + dt_alpha * flux_int
                compute_time += MPI.Wtime() - t_comp0

                t_c1 = MPI.Wtime()
                MPI.Request.Waitall(requests)
                comm_time += MPI.Wtime() - t_c1

                t_comp1 = MPI.Wtime()
                if len(bnd_ids) > 0:
                    T_bnd = T_ext[bnd_ids]
                    T_bnd_nbrs = T_ext[bnd_nbrs]
                    flux_bnd = np.sum(bnd_weights * (T_bnd_nbrs - T_bnd[:, None]), axis=1)
                    T_new_local[bnd_ids] = T_bnd + dt_alpha * flux_bnd
                compute_time += MPI.Wtime() - t_comp1

            else:
                for nbr in subdomain.neighbor_ranks:
                    s_buf = send_buffers[nbr]
                    np.take(T_ext, subdomain.send_local_indices[nbr], out=s_buf)
                    h_start, h_end = subdomain.recv_halo_slices[nbr]
                    comm.Sendrecv(
                        sendbuf=[s_buf, MPI.DOUBLE],
                        dest=nbr,
                        sendtag=100,
                        recvbuf=[T_ext[h_start:h_end], MPI.DOUBLE],
                        source=nbr,
                        recvtag=100,
                    )
                comm_time += MPI.Wtime() - t_c0

                t_comp0 = MPI.Wtime()
                T_own = T_ext[:N_local]
                T_nbrs = T_ext[all_nbrs]
                flux = np.sum(all_weights * (T_nbrs - T_own[:, None]), axis=1)
                np.add(T_own, dt_alpha * flux, out=T_new_local)
                compute_time += MPI.Wtime() - t_comp0
        else:
            t_comp0 = MPI.Wtime()
            T_own = T_ext[:N_local]
            T_nbrs = T_ext[all_nbrs]
            flux = np.sum(all_weights * (T_nbrs - T_own[:, None]), axis=1)
            np.add(T_own, dt_alpha * flux, out=T_new_local)
            compute_time += MPI.Wtime() - t_comp0

        if subdomain.local_fixed_indices is not None and subdomain.local_fixed_values is not None:
            T_new_local[subdomain.local_fixed_indices] = subdomain.local_fixed_values

        T_ext[:N_local] = T_new_local

        if synchronize_steps and size > 1:
            comm.Barrier()

        if snapshot_interval > 0 and (step % snapshot_interval == 0 or step == num_steps):
            T_snap = gather_global_temperature(
                comm,
                subdomain,
                T_ext[:N_local],
                total_faces,
                all_global_ids_packed,
                counts,
                displs,
                metadata_ready=True,
            )
            if rank == 0 and T_snap is not None:
                history_global.append(T_snap)
                history_steps.append(step)

    comm.Barrier()
    elapsed = MPI.Wtime() - t_start

    T_final_global = gather_global_temperature(
        comm,
        subdomain,
        T_ext[:N_local],
        total_faces,
        all_global_ids_packed,
        counts,
        displs,
        metadata_ready=True,
    )

    return MPISimulationResult(
        rank=rank,
        num_procs=size,
        T_final_global=T_final_global,
        T_final_local=T_ext[:N_local].copy(),
        history_global=history_global,
        history_steps=history_steps,
        elapsed_time=elapsed,
        comm_time=comm_time,
        compute_time=compute_time,
        num_steps=num_steps,
        dt=dt,
        alpha=alpha,
    )
