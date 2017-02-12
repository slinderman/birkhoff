"""
Forward mode differentiation for stick breaking transformations.
In the stick breaking transformations, all of the computations are
addition and subtraction of scalars.  This makes it pretty easy to
do the forward mode autodiff
"""
import autograd.numpy as np
from autograd import jacobian
from autograd.core import primitive

from birkhoff.utils import psi_to_pi as psi_to_pi_old
from birkhoff.utils import logistic

# Categorical distributions
@primitive
def psi_to_pi(psi, return_intermediates=False):
    """
    Convert psi to a probability vector pi
    :param psi:     Length K-1 vector in [0,1]
    :return:        Length K normalized probability vector
    """
    K = psi.size + 1
    pi = np.zeros(K)

    # Intermediate terms
    ubs = np.zeros(K)
    ubs[0] = 1.0

    # Set pi[1..K-1]
    for k in range(K-1):
        pi[k] = psi[k] * ubs[k]
        ubs[k+1] = ubs[k] - pi[k]

    # Set the last output
    pi[K-1] = ubs[K-1]

    if return_intermediates:
        return pi, ubs
    else:
        return pi

def jacobian_psi_to_pi(psi):
    """
    J = [  -- dpi_1 / dpsi -- ]
        [  -- dpi_2 / dpsi -- ]
        [         ...         ]
        [  -- dpi_K / dpsi -- ]

    Output is K x K-1 matrix
    """
    # Initialize output
    K = psi.size + 1
    J = np.zeros((K, K-1))

    # Run once to get intermediate computations
    pi, ubs = psi_to_pi(psi, return_intermediates=True)

    for ii in range(K-1):
        dpsi = np.zeros(K - 1)
        dpsi[ii] = 1.0

        dpi = np.zeros(K)
        dubs = np.zeros(K)

        # Set pi[1..K-1]
        for k in range(K - 1):
            # Step 1: weight by ubs
            dpi[k] = dpsi[k] * ubs[k] + psi[k] * dubs[k]
            # Step 2: Adjust stick
            dubs[k + 1] = dubs[k] - dpi[k]

        # Set the last output
        dpi[K - 1] = dubs[K - 1]
        J[:, ii] = dpi
    return J

# Make an autograd gradient
def grad_psi_to_pi(g, ans, vs, gvs, psi, return_intermediates=False):
    return np.dot(g, jacobian_psi_to_pi(psi))
psi_to_pi.defvjp(grad_psi_to_pi)

### Now do the same for the permutation
@primitive
def psi_to_birkhoff(Psi, return_intermediates=False):
    """
    Transform a (K-1) x (K-1) matrix Psi into a KxK doubly
    stochastic matrix Pi.  I've introduced some tolerance to
    make it more stable, but I'm not convinced of this approach yet.

    Note that this version uses direct assignment to the matrix P,
    but this is not amenable to automatic differentiation with
    autograd. The next version below uses list comprehension to
    perform the same psi -> pi conversion in an automatically
    differentiable way.
    """
    K = Psi.shape[0] + 1
    assert Psi.shape == (K - 1, K - 1)

    # Initialize intermediate values
    P = np.zeros((K, K), dtype=float)
    ub_rows = np.ones((K, K))
    ub_cols = np.ones((K, K))
    lbs = np.zeros((K, K))

    for i in range(K - 1):
        for j in range(K - 1):
            # Compute lower bound
            lbs[i, j] = ub_rows[i, j] - ub_cols[i, j + 1:].sum()

            # Four cases
            if ub_rows[i, j] < ub_cols[i, j]:
                if lbs[i, j] > 0:
                    P[i, j] = lbs[i, j] + (ub_rows[i, j] - lbs[i, j]) * Psi[i, j]
                else:
                    P[i, j] = ub_rows[i, j] * Psi[i, j]
            else:
                if lbs[i, j] > 0:
                    P[i, j] = lbs[i, j] + (ub_cols[i, j] - lbs[i, j]) * Psi[i, j]
                else:
                    P[i, j] = ub_cols[i, j] * Psi[i, j]

            # Update upper bounds
            ub_rows[i, j+1] = ub_rows[i, j] - P[i, j]
            ub_cols[i+1, j] = ub_cols[i, j] - P[i, j]

        # Finish off the row
        P[i, -1] = ub_rows[i, -1]
        ub_cols[i+1, -1] = ub_cols[i, -1] - P[i, -1]

    # Finish off the columns
    for j in range(K-1):
        P[-1, j] = ub_cols[-1, j]

    # Compute the bottom right entry
    P[-1,-1] = 1 - np.sum(ub_cols[-1,:-1])

    if return_intermediates:
        return P, ub_rows, ub_cols, lbs
    else:
        return P

def jacobian_psi_to_birkhoff(Psi):
    """
    As above, use dual numbers to perform forward mode
    differentiation.
    """
    K = Psi.shape[0] + 1
    assert Psi.shape == (K - 1, K - 1)

    # Run once to get the intermediate values
    P, ub_rows, ub_cols, lbs = psi_to_birkhoff(Psi, return_intermediates=True)

    # Initialize the Jacobian
    J = np.zeros((K, K, K-1, K-1))

    # Initialize intermediate values and derivatives
    for ii in range(K-1):
        for jj in range(K-1):
            dPsi = np.zeros((K - 1, K - 1))
            dPsi[ii, jj] = 1.0

            dP = np.zeros((K, K))
            dub_rows = np.zeros((K, K))
            dub_cols = np.zeros((K, K))
            dlbs = np.zeros((K, K))

            for i in range(K - 1):
                for j in range(K - 1):
                    # Compute lower bound
                    dlbs[i, j] = dub_rows[i, j] - dub_cols[i,j+1:].sum()

                    # The upper bounds are computed recursively
                    if ub_rows[i, j] < ub_cols[i, j]:
                        if lbs[i, j] > 0:
                            dP[i, j] = dlbs[i, j] + \
                                       (dub_rows[i, j] - dlbs[i, j]) * Psi[i, j] + \
                                       (ub_rows[i, j] - lbs[i, j]) * dPsi[i, j]
                        else:
                            dP[i, j] = dub_rows[i, j] * Psi[i, j] + ub_rows[i, j] * dPsi[i, j]
                    else:
                        if lbs[i, j] > 0:
                            dP[i, j] = dlbs[i, j] + \
                                       (dub_cols[i, j] - dlbs[i, j]) * Psi[i, j] + \
                                       (ub_cols[i, j] - lbs[i, j]) * dPsi[i, j]
                        else:
                            dP[i, j] = dub_cols[i, j] * Psi[i, j] + ub_cols[i, j] * dPsi[i, j]


                    # Update upper bounds
                    dub_rows[i, j+1] = dub_rows[i, j] - dP[i, j]
                    dub_cols[i+1, j] = dub_cols[i, j] - dP[i, j]

                # Finish off the row
                dP[i, -1] = dub_rows[i, -1]
                dub_cols[i+1, -1] = dub_cols[i, -1] - dP[i, -1]

            # Finish off the columns
            for j in range(K-1):
                dP[-1, j] = dub_cols[-1, j]

            # Compute the bottom right entry
            dP[-1,-1] = -np.sum(dub_cols[-1,:-1])

            J[:,:,ii,jj] = dP

    return J

def grad_psi_to_pi_birkhoff(g, ans, vs, gvs, psi, return_intermediates=False):
    return np.tensordot(g, jacobian_psi_to_birkhoff(psi), ((0, 1), (0, 1)))
psi_to_birkhoff.defvjp(grad_psi_to_pi_birkhoff)

### Simple tests
def _psi_to_pi_list(psi):
    pi = []
    for i in range(psi.shape[0]):
        pi.append(psi[i] * (1 - np.sum(np.array(pi),0)))
    pi.append(1-np.sum(np.array(pi)))
    return np.array(pi)

def _psi_to_birkhoff_list(Psi, verbose=False):
    """
    Autograd doesn't work with array assignment.  Rewrite with
    list comprehension instead.  This will be a bit slower, but
    not too bad.
    """
    N = Psi.shape[0] + 1
    assert Psi.shape == (N - 1, N - 1)

    # P = np.zeros((N, N), dtype=float)
    P = []
    for i in range(N - 1):
        P_i = []
        for j in range(N - 1):
            # Upper bounded by partial row sum
            ub_row = 1.0
            for jj in range(j):
                ub_row = ub_row - P_i[jj]

            # Upper bounded by partial column sum
            # ub_col = 1 - np.sum(P[:i, j])
            ub_col = 1.0
            for ii in range(i):
                ub_col = ub_col - P[ii][j]

            # Lower bounded (see notes)
            # lb_rem = (1 - np.sum(P[i, :j])) - (N - (j + 1)) + np.sum(P[:i, j + 1:])
            lb_rem = 1.0 - (N - (j + 1))
            for jj in range(j):
                lb_rem = lb_rem - P_i[jj]

            for ii in range(i):
                for jj in range(j+1,N):
                    lb_rem = lb_rem + P[ii][jj]

            # Combine constraints
            ub = min(ub_row, ub_col)
            lb = max(0, lb_rem)

            if verbose:
                print("({0}, {1}): lb_rem: {2} lb: {3}  ub: {4}".format(i, j, lb_rem, lb, ub))

            # P[i, j] = lb + (ub - lb) * logistic(Psi[i, j])
            P_i.append(lb + (ub - lb) * Psi[i, j])

        # Finish off the row
        # P[i, -1] = 1 - np.sum(P[i, :-1])
        PiN = 1.0
        for jj in range(N-1):
            PiN = PiN - P_i[jj]
        P_i.append(PiN)

        # Append
        P.append(P_i)

    # Finish off the columns
    P_N = []
    for j in range(N):
        PjN = 1.0
        for ii in range(N-1):
            PjN = PjN - P[ii][j]
        P_N.append(PjN)
    P.append(P_N)

    return np.array(P)

def test_psi_to_pi():
    K = 10
    psi = np.random.randn(K-1)
    pi1 = psi_to_pi_old(psi)
    pi2 = _psi_to_pi_list(logistic(psi))
    pi3 = psi_to_pi(logistic(psi))
    assert np.allclose(pi1, pi3)
    assert np.allclose(pi2, pi3)

def test_jacobian_psi_to_pi():
    K = 10
    psi = np.random.randn(K-1)
    p = logistic(psi)
    J1 = jacobian(_psi_to_pi_list)(p)
    J2 = jacobian(psi_to_pi)(p)
    J3 = jacobian_psi_to_pi(p)
    assert np.allclose(J1, J2)
    assert np.allclose(J1, J3)

def test_psi_to_birkhoff():
    """
    Make sure everything runs
    """
    K = 10
    mu = np.zeros((K - 1, K - 1))
    sigma = 0.1 * np.ones((K - 1, K - 1))
    Psi = mu + sigma * np.random.randn(K - 1, K - 1)

    # Convert Psi to P via stick breaking raster scan
    from birkhoff.permutation import psi_to_pi_assignment
    P1 = psi_to_pi_assignment(Psi, tol=0)
    P2 = _psi_to_birkhoff_list(logistic(Psi))
    P3 = psi_to_birkhoff(logistic(Psi))

    assert np.allclose(P1, P2)
    assert np.allclose(P1, P3)


def test_jacobian_psi_to_birkhoff():
    K = 10
    Psi = np.random.randn(K-1, K-1)
    print("ag start")
    J1 = jacobian(_psi_to_birkhoff_list)(logistic(Psi))
    print("ag stop")
    print(J1.shape)

    print("manual start")
    J2 = jacobian(psi_to_birkhoff)(logistic(Psi))
    print("manual stop")

    print("manual start")
    J3 = jacobian_psi_to_birkhoff(logistic(Psi))
    print("manual stop")

    assert np.allclose(J2, J3)
    assert np.allclose(J1, J2)
    assert np.allclose(J1, J3)


def test_plot_jacobian():
    K = 4
    Psi = np.random.randn(K - 1, K - 1)
    J1 = jacobian(_psi_to_birkhoff_list)(logistic(Psi))
    J2 = jacobian_psi_to_birkhoff(logistic(Psi))

    J1 = J1.reshape((K ** 2, (K - 1) ** 2))
    J2 = J2.reshape((K ** 2, (K - 1) ** 2))
    lim = max(abs(J1).max(), abs(J2).max())

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.imshow(J1, interpolation="none", vmin=-lim, vmax=lim)
    plt.xlabel("$\\mathrm{vec}(\\Psi)$")
    plt.ylabel("$\\mathrm{vec}(\\Pi)$")
    plt.title("Lower triangular Jacobian")
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(J2, interpolation="none", vmin=-lim, vmax=lim)
    plt.xlabel("$\\mathrm{vec}(\\Psi)$")
    plt.ylabel("$\\mathrm{vec}(\\Pi)$")
    plt.title("Lower triangular Jacobian")
    plt.colorbar()

    # plt.subplot(133)
    # plt.imshow(J1-J2, interpolation="none")
    # plt.xlabel("$\\mathrm{vec}(\\Psi)$")
    # plt.ylabel("$\\mathrm{vec}(\\Pi)$")
    # plt.title("Lower triangular Jacobian")
    # plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_psi_to_pi()
    test_jacobian_psi_to_pi()
    test_psi_to_birkhoff()
    test_jacobian_psi_to_birkhoff()
    test_plot_jacobian()
