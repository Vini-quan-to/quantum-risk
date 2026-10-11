
import time
import numpy as np

from scipy.optimize import OptimizeResult
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import Optimizer
from qiskit.primitives import StatevectorSampler


class OneEvaluationOptimizer(Optimizer):
    def __init__(self):
        super().__init__()

    def get_support_level(self):
        return {
            "gradient": 0,
            "bounds": 0,
            "initial_point": 0,
        }

    def minimize(self, fun, x0, jac=None, bounds=None):
        print("Evaluating QAOA objective once...", flush=True)
        value = fun(x0)
        print("Objective evaluation finished.", flush=True)

        return OptimizeResult(
            x=np.asarray(x0),
            fun=value,
            nfev=1,
            success=True,
            message="One evaluation completed",
        )


def main():
    n = 8
    print("Building minimal QAOA problem...", flush=True)

    qp = QuadraticProgram("minimal_qaoa_test")
    for i in range(n):
        qp.binary_var(name=f"x{i}")

    linear = {f"x{i}": 1.0 for i in range(n)}
    qp.minimize(linear=linear)
    qp.linear_constraint(
        linear=linear,
        sense="==",
        rhs=3,
        name="select_three",
    )

    print("Converting to QUBO...", flush=True)
    qubo = QuadraticProgramToQubo().convert(qp)
    operator, _ = qubo.to_ising()

    qaoa = QAOA(
        sampler=StatevectorSampler(default_shots=128, seed=2026),
        optimizer=OneEvaluationOptimizer(),
        reps=1,
        initial_point=np.array([0.5, 0.5]),
    )

    print("Starting QAOA...", flush=True)
    start = time.perf_counter()
    result = qaoa.compute_minimum_eigenvalue(operator)

    print(f"QAOA finished in {time.perf_counter() - start:.2f} seconds.")
    print(f"Best eigenvalue: {result.eigenvalue.real:.6f}")
    print(f"Best measurement: {result.best_measurement}", flush=True)


if __name__ == "__main__":
    main()
