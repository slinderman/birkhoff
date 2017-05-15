import numpy as np
from scipy.optimize import linear_sum_assignment
from birkhoff.qap import solve_qap

if __name__ == "__main__":
    # Simple test: linear cost sum assignment
    N = 10
    A = np.zeros((1, N, N))
    B = np.zeros((1, N, N))
    C = np.random.randn(1, N, N)
    P_initial = 0.1 / N * np.ones((N, N)) + 0.9 * np.eye(N)

    # Solve with approximate algorithm
    P_hat = solve_qap(A, B, C, P0=P_initial)

    # Solve using Hungarian algorithm
    row, col = linear_sum_assignment(C[0])
    P_test = np.zeros((N, N))
    P_test[row, col] = 1

    # Compare
    print(np.where(P_hat)[1])
    print(np.where(P_test)[1])
    assert(np.allclose(P_hat, P_test))
