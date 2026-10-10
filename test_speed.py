
import time

from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler


def test_simulator():
    sampler = StatevectorSampler(default_shots=128, seed=2026)

    for n in [16, 18, 20]:
        print(f"\nTesting {n} qubits...", flush=True)

        terms = [
            ("ZZ", [i, j], 0.01)
            for i in range(n)
            for j in range(i + 1, n)
        ]

        operator = SparsePauliOp.from_sparse_list(
            terms,
            num_qubits=n,
        )

        circuit = QAOAAnsatz(operator, reps=1)
        circuit.measure_all()

        # Decompose high-level gates into a simpler gate set.
        circuit = transpile(
            circuit,
            basis_gates=["h", "rx", "rz", "cx"],
            optimization_level=0,
        )

        start = time.perf_counter()

        job = sampler.run(
            [(circuit, [0.1, 0.1])],
            shots=128,
        )

        result = job.result()
        elapsed = time.perf_counter() - start

        print(
            f"{n} qubits completed in {elapsed:.3f} seconds",
            flush=True,
        )


if __name__ == "__main__":
    test_simulator()
