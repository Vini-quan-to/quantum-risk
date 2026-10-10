
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector

from quantum_cvar_optimizer import apply_xy_gate


def feasible_masks(n, k):
    """Return bitmasks selecting exactly k of n assets."""
    masks = []

    for selected in combinations(range(n), k):
        mask = 0
        for bit in selected:
            mask |= 1 << bit
        masks.append(mask)

    return np.asarray(masks, dtype=int)


def make_synthetic_costs(n, masks):
    """Create deterministic toy costs for circuit validation only."""
    costs = np.zeros(1 << n, dtype=float)

    for mask in masks:
        selected = [i for i in range(n) if (mask >> i) & 1]

        raw_cost = sum((i + 1) ** 2 for i in selected)

        interaction = sum(
            (selected[a] + 1) * (selected[b] + 1)
            for a in range(len(selected))
            for b in range(a + 1, len(selected))
        )

        costs[mask] = raw_cost + 0.25 * interaction

    feasible_values = costs[masks]
    mean = float(np.mean(feasible_values))
    std = float(np.std(feasible_values))

    if std > 0:
        costs[masks] = (feasible_values - mean) / std

    return costs


def custom_cost_phase(state, costs, gamma):
    """Apply exp(-i * gamma * cost) to each basis state."""
    return state * np.exp(-1j * gamma * costs)


def custom_xy_layer(state, n, beta):
    """Apply the custom adjacent-ring XY mixer."""
    state = state.copy()

    for qa in range(n):
        qb = (qa + 1) % n

        i01 = np.array(
            [
                i
                for i in range(1 << n)
                if ((i >> qa) & 1) == 0
                and ((i >> qb) & 1) == 1
            ],
            dtype=int,
        )

        i10 = i01 ^ (1 << qa) ^ (1 << qb)

        apply_xy_gate(state, beta, i01, i10)

    return state


def custom_full_circuit(initial_state, n, costs, gammas, betas):
    """Apply cost-phase and XY-mixer layers in sequence."""
    state = initial_state.copy()

    for gamma, beta in zip(gammas, betas):
        state = custom_cost_phase(state, costs, gamma)
        state = custom_xy_layer(state, n, beta)

    return state


def qiskit_full_circuit(initial_state, n, costs, gammas, betas):
    """Build and simulate the Qiskit reference circuit."""
    circuit = QuantumCircuit(n)

    for gamma, beta in zip(gammas, betas):
        diagonal = np.exp(-1j * gamma * costs)
        circuit.append(DiagonalGate(diagonal), range(n))

        for qa in range(n):
            qb = (qa + 1) % n
            circuit.rxx(beta, qa, qb)
            circuit.ryy(beta, qa, qb)

    return np.asarray(Statevector(initial_state).evolve(circuit).data)


def hamming_weight(index):
    return int(index).bit_count()


def probability_outside_feasible_sector(probabilities, n, k):
    return float(
        sum(
            probabilities[index]
            for index in range(1 << n)
            if hamming_weight(index) != k
        )
    )


def main():
    # Small test problem; this is not the 20-asset experiment.
    n = 6
    k = 2

    gammas = (0.31, -0.17)
    betas = (0.23, 0.41)
    tolerance = 1e-10

    masks = feasible_masks(n, k)
    costs = make_synthetic_costs(n, masks)

    # Uniform superposition over all feasible portfolios.
    initial_state = np.zeros(1 << n, dtype=complex)
    initial_state[masks] = 1.0 / np.sqrt(len(masks))

    custom_state = custom_full_circuit(
        initial_state, n, costs, gammas, betas
    )

    qiskit_state = qiskit_full_circuit(
        initial_state, n, costs, gammas, betas
    )

    custom_probabilities = np.abs(custom_state) ** 2
    qiskit_probabilities = np.abs(qiskit_state) ** 2

    custom_norm = float(np.vdot(custom_state, custom_state).real)
    qiskit_norm = float(np.vdot(qiskit_state, qiskit_state).real)

    max_probability_difference = float(
        np.max(np.abs(custom_probabilities - qiskit_probabilities))
    )

    max_amplitude_difference = float(
        np.max(np.abs(custom_state - qiskit_state))
    )

    custom_outside = probability_outside_feasible_sector(
        custom_probabilities, n, k
    )

    qiskit_outside = probability_outside_feasible_sector(
        qiskit_probabilities, n, k
    )

    probabilities_match = np.allclose(
        custom_probabilities,
        qiskit_probabilities,
        atol=tolerance,
        rtol=tolerance,
    )

    norms_ok = (
        abs(custom_norm - 1.0) < tolerance
        and abs(qiskit_norm - 1.0) < tolerance
    )

    feasible_sector_ok = (
        custom_outside < tolerance
        and qiskit_outside < tolerance
    )

    print("=== Full Cost-Phase + XY Circuit Validation ===")
    print(f"Assets (qubits): {n}")
    print(f"Selected assets per portfolio: {k}")
    print(f"Feasible portfolios: {len(masks)}")
    print(f"Circuit layers: {len(gammas)}")
    print(f"Gammas: {gammas}")
    print(f"Betas: {betas}")
    print()

    print(f"Custom norm squared: {custom_norm:.16f}")
    print(f"Qiskit norm squared: {qiskit_norm:.16f}")
    print(
        "Maximum probability difference: "
        f"{max_probability_difference:.3e}"
    )
    print(
        "Maximum amplitude difference: "
        f"{max_amplitude_difference:.3e}"
    )
    print(f"Custom probability outside k={k}: {custom_outside:.3e}")
    print(f"Qiskit probability outside k={k}: {qiskit_outside:.3e}")
    print()

    print("Probabilities match:", probabilities_match)
    print("Both states normalized:", norms_ok)
    print("Feasible sector preserved:", feasible_sector_ok)

    if probabilities_match and norms_ok and feasible_sector_ok:
        print("\nPASS: Full toy circuit validation passed.")
    else:
        print("\nFAIL: Validation failed. Do not modify the main optimizer yet.")


if __name__ == "__main__":
    main()
