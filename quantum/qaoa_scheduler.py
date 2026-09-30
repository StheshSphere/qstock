"""
QAOA Payout Rotation Scheduler
===============================

Problem: In a stokvel, deciding who gets paid out "early" vs "late" in the
rotation should balance urgency (how much a member needs the funds soon)
against fairness (avoiding always favouring the same members). As the group
grows, finding the best 2-way split becomes a combinatorial optimization
problem — a natural fit for QAOA.

Model (simplified, MaxCut-style QUBO):
  - Each member i gets a binary variable x_i: 1 = "early" payout slot,
    0 = "late" payout slot.
  - We build a pairwise "incompatibility" weight w_ij between members i, j
    based on how different their urgency scores are — members with very
    different urgency should land in *different* slots (an early/late split)
    so urgent members aren't stuck waiting behind less-urgent ones.
  - This is exactly the MaxCut objective: maximize sum(w_ij) over pairs
    placed in different slots.

This follows the Qiskit Pattern:
  1. MAP    -> build the cost Hamiltonian from member urgency data
  2. OPTIMIZE -> transpile the QAOA ansatz for the backend
  3. EXECUTE -> run the variational loop with a classical optimizer
  4. POST-PROCESS -> decode the best bitstring into an early/late schedule
"""

import numpy as np
from scipy.optimize import minimize
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer import AerSimulator
from qiskit import transpile


def build_cost_hamiltonian(urgency_scores: list[float]) -> SparsePauliOp:
    """
    STEP 1 - MAP: Convert member urgency scores into a MaxCut-style cost
    Hamiltonian. Edge weight w_ij = |urgency_i - urgency_j|, so members with
    very different urgency are rewarded for landing in different slots.
    """
    n = len(urgency_scores)
    pauli_list = []
    for i in range(n):
        for j in range(i + 1, n):
            weight = abs(urgency_scores[i] - urgency_scores[j])
            if weight == 0:
                continue
            z_string = ["I"] * n
            z_string[i] = "Z"
            z_string[j] = "Z"
            # MaxCut-style cost operator: H = sum_{i<j} w_ij * Z_i Z_j.
            # A pair in different slots contributes -w_ij (a reward), so
            # *minimizing* <H> maximizes the weighted cut. Note the sign: with
            # -w_ij coefficients the true ground state becomes the degenerate
            # all-same-slot solution, which is the trap QAOA was falling into.
            pauli_list.append(("".join(z_string), weight))
    return SparsePauliOp.from_list(pauli_list)


def run_qaoa_scheduler(
    urgency_scores: list[float],
    reps: int = 3,
    seed: int = 42,
    n_restarts: int = 8,
):
    """
    STEP 2 (OPTIMIZE for hardware) + STEP 3 (EXECUTE): Build the QAOA
    ansatz, transpile it, and run the classical-quantum optimization loop
    using a local Aer simulator.

    Shallow QAOA landscapes are riddled with local minima, so a single COBYLA
    run from one random start can stall at a degenerate solution (e.g. every
    member in the same slot). We therefore optimize from several random
    initial parameter sets and keep the run with the lowest cost.
    """
    cost_hamiltonian = build_cost_hamiltonian(urgency_scores)
    n = len(urgency_scores)

    ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    backend = AerSimulator()
    transpiled_ansatz = transpile(ansatz, backend, optimization_level=1)

    # The Hamiltonian's qubit layout must match the transpiled circuit
    isa_hamiltonian = cost_hamiltonian.apply_layout(transpiled_ansatz.layout)

    estimator = AerEstimator()
    rng = np.random.default_rng(seed)

    def cost_fn(params):
        job = estimator.run([(transpiled_ansatz, isa_hamiltonian, params)])
        return float(job.result()[0].data.evs)

    best_result = None
    best_cost = np.inf
    restart_costs = []

    for restart in range(n_restarts):
        initial_params = rng.uniform(0, 2 * np.pi, transpiled_ansatz.num_parameters)
        result = minimize(
            cost_fn, initial_params, method="COBYLA", options={"maxiter": 100}
        )
        restart_costs.append(float(result.fun))
        if result.fun < best_cost:
            best_cost = float(result.fun)
            best_result = result
            marker = "  <- new best"
        else:
            marker = ""
        print(
            f"  restart {restart + 1:>2}/{n_restarts}: "
            f"cost = {result.fun:+.4f}{marker}"
        )

    print(f"  best cost after {n_restarts} restarts: {best_cost:+.4f}")

    return {
        "optimal_params": best_result.x,
        "optimal_cost": best_cost,
        "restart_costs": restart_costs,
        "transpiled_ansatz": transpiled_ansatz,
        "backend": backend,
        "n": n,
    }


def decode_schedule(run_result: dict, member_names: list[str], shots: int = 2048):
    """
    STEP 4 - POST-PROCESS: Sample the optimized circuit and decode the most
    frequent bitstring into an actual early/late payout schedule.
    """
    from qiskit_aer.primitives import SamplerV2 as AerSampler

    circuit = run_result["transpiled_ansatz"].assign_parameters(run_result["optimal_params"])
    circuit.measure_all()

    sampler = AerSampler()
    job = sampler.run([circuit], shots=shots)
    counts = job.result()[0].data.meas.get_counts()

    best_bitstring = max(counts, key=counts.get)
    early = [name for bit, name in zip(reversed(best_bitstring), member_names) if bit == "1"]
    late = [name for bit, name in zip(reversed(best_bitstring), member_names) if bit == "0"]
    return {"early_payout": early, "late_payout": late, "bitstring": best_bitstring, "counts": counts}


def classical_baseline(urgency_scores: list[float], member_names: list[str]):
    """
    Simple classical baseline for comparison: rank members by urgency and
    split top half into "early", bottom half into "late". Useful to show
    judges whether QAOA actually improves on a naive rule.
    """
    ranked = sorted(zip(member_names, urgency_scores), key=lambda x: -x[1])
    half = len(ranked) // 2
    early = [name for name, _ in ranked[:half]]
    late = [name for name, _ in ranked[half:]]
    return {"early_payout": early, "late_payout": late}


if __name__ == "__main__":
    # --- Demo with synthetic data ---
    member_names = ["Thabo", "Naledi", "Sipho", "Amahle", "Katlego", "Zanele"]
    urgency_scores = [0.9, 0.2, 0.7, 0.1, 0.8, 0.3]  # higher = more urgent need

    print("Running QAOA payout scheduler (local simulator)...")
    result = run_qaoa_scheduler(urgency_scores, reps=3)
    schedule = decode_schedule(result, member_names)

    print("\n--- QAOA schedule ---")
    print("Early payout:", schedule["early_payout"])
    print("Late payout: ", schedule["late_payout"])

    print("\n--- Classical baseline (rank-split) ---")
    baseline = classical_baseline(urgency_scores, member_names)
    print("Early payout:", baseline["early_payout"])
    print("Late payout: ", baseline["late_payout"])
