
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_cvar_optimizer import apply_xy_gate


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


def qiskit_xy_layer(state, n, beta):
    """Apply the equivalent XY mixer using Qiskit."""
    circuit = QuantumCircuit(n)

    for qa in range(n):
        qb = (qa + 1) % n
        circuit.rxx(beta, qa, qb)
        circuit.ryy(beta, qa, qb)

    return np.asarray(Statevector(state).evolve(circuit).data)


def hamming_weight(index):
    """Count the number of set bits in a basis-state index."""
    return int(index).bit_count()


def probability_outside_weight(probabilities, n, target_weight):
    """Calculate probability outside the target particle-number sector."""
    return float(
        sum(
            probabilities[i]
            for i in range(1 << n)
            if hamming_weight(i) != target_weight
        )
    )


def run_case(n, basis_index, beta, tolerance=1e-10):
    """Compare both mixers for one initial state and angle."""
    initial = np.zeros(1 << n, dtype=complex)
    initial[basis_index] = 1.0

    custom_state = custom_xy_layer(initial, n, beta)
    qiskit_state = qiskit_xy_layer(initial, n, beta)

    custom_probabilities = np.abs(custom_state) ** 2
    qiskit_probabilities = np.abs(qiskit_state) ** 2

    target_weight = hamming_weight(basis_index)

    custom_outside = probability_outside_weight(
        custom_probabilities, n, target_weight
    )
    qiskit_outside = probability_outside_weight(
        qiskit_probabilities, n, target_weight
    )

    custom_norm = float(np.vdot(custom_state, custom_state).real)
    qiskit_norm = float(np.vdot(qiskit_state, qiskit_state).real)

    max_probability_difference = float(
        np.max(np.abs(custom_probabilities - qiskit_probabilities))
    )
    max_amplitude_difference = float(
        np.max(np.abs(custom_state - qiskit_state))
    )

    passed = (
        np.allclose(
            custom_probabilities,
            qiskit_probabilities,
            atol=tolerance,
            rtol=tolerance,
        )
        and abs(custom_norm - 1.0) < tolerance
        and abs(qiskit_norm - 1.0) < tolerance
        and custom_outside < tolerance
        and qiskit_outside < tolerance
    )

    return {
        "bitstring": format(basis_index, f"0{n}b"),
        "particles": target_weight,
        "beta": beta,
        "custom_norm": custom_norm,
        "qiskit_norm": qiskit_norm,
        "max_probability_difference": max_probability_difference,
        "max_amplitude_difference": max_amplitude_difference,
        "custom_outside_weight": custom_outside,
        "qiskit_outside_weight": qiskit_outside,
        "passed": passed,
    }


def main():
    n = 4
    betas = (0.11, 0.23, 0.47)

    # Different initial configurations for each particle number.
    basis_indices_by_weight = {
        1: (0b0001, 0b0010, 0b0100, 0b1000),
        2: (0b0011, 0b0101, 0b1010, 0b1100),
        3: (0b0111, 0b1011, 0b1101, 0b1110),
    }

    cases = [
        run_case(n, basis_index, beta)
        for beta in betas
        for basis_indices in basis_indices_by_weight.values()
        for basis_index in basis_indices
    ]

    print("=== Expanded XY Mixer Validation ===")
    print(f"Qubits: {n}")
    print(f"Particle numbers tested: {tuple(basis_indices_by_weight)}")
    print(f"Beta values tested: {betas}")
    print(f"Total test cases: {len(cases)}")
    print()

    print(
        f"{'State':<8} {'k':>2} {'beta':>6} "
        f"{'Max prob diff':>16} {'Custom outside':>16} "
        f"{'Qiskit outside':>16} {'Result':>8}"
    )
    print("-" * 90)

    for case in cases:
        result = "PASS" if case["passed"] else "FAIL"
        print(
            f"{case['bitstring']:<8} "
            f"{case['particles']:>2} "
            f"{case['beta']:>6.2f} "
            f"{case['max_probability_difference']:>16.3e} "
            f"{case['custom_outside_weight']:>16.3e} "
            f"{case['qiskit_outside_weight']:>16.3e} "
            f"{result:>8}"
        )

    max_probability_difference = max(
        case["max_probability_difference"] for case in cases
    )
    max_amplitude_difference = max(
        case["max_amplitude_difference"] for case in cases
    )
    max_norm_error = max(
        abs(case[key] - 1.0)
        for case in cases
        for key in ("custom_norm", "qiskit_norm")
    )

    all_passed = all(case["passed"] for case in cases)

    print("\n=== Summary ===")
    print(f"Passed: {sum(c['passed'] for c in cases)} / {len(cases)}")
    print(f"Maximum probability difference: {max_probability_difference:.3e}")
    print(f"Maximum amplitude difference: {max_amplitude_difference:.3e}")
    print(f"Maximum norm error: {max_norm_error:.3e}")

    if all_passed:
        print("\nPASS: All expanded XY mixer checks passed.")
    else:
        print("\nFAIL: One or more checks failed.")
        print("Do not change the main optimizer yet.")


if __name__ == "__main__":
    main()
