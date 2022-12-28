import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import unitary_group
from numpy.typing import ArrayLike
from qiskit import quantum_info as qinfo
import test
import sys


def compute_eigens(rho: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """compute eigenvalues and eigenvectors of a given matrix

    Args:
        rho (ArrayLike): 2D square (n,n) array

    Returns:
        tuple[ArrayLike, ArrayLike]: tuple of eigenvalues (n,1) array and eigenvectors (n,n) array
                                    eigenvalues' [i] is the i-th eigenvalue
                                    eigenvectors' [:,i] is the i-th eigenvector
    """
    w, v = np.linalg.eig(rho)
    w_fix = np.where(np.isclose(w.real, 0., atol=1e-08), 0., w.real)

    return w_fix, v

def compute_entropy(rho: ArrayLike) -> float:
    """compute the von Neumann Entropy of a given density matrix

    Args:
        rho (ArrayLike): 2D complex-valued array that is a density matrix

    Returns:
        float: non-negative real number between 0 and log(dimension)
    """
    w = compute_eigens(rho)[0]
    w_fix = np.where(w==0., 1e-08, w)
    return -np.dot(w_fix, np.log2(w_fix))

def get_standard_purification(rho: ArrayLike) -> ArrayLike:
    """get the standard purification of a given density matrix

    Args:
        rho (ArrayLike): 2D complex-valued array that is a density matrix

    Returns:
        ArrayLike: 2D (size, dim_abcd) array
    """

    w, v = compute_eigens(rho)
    psi_s = (np.sqrt(w[0]) * np.kron(np.kron(v[0], np.array([1., 0., 0., 0.])), np.array([1., 0., 0., 0.]))
            + np.sqrt(w[1]) * np.kron(np.kron(v[1], np.array([1., 0., 0., 0.])), np.array([0., 1., 0., 0.]))
            + np.sqrt(w[2]) * np.kron(np.kron(v[2], np.array([1., 0., 0., 0.])), np.array([0., 0., 1., 0.]))
            + np.sqrt(w[3]) * np.kron(np.kron(v[3], np.array([1., 0., 0., 0.])), np.array([0., 0., 0., 1.])))

    return psi_s


def generate_random_unitary(dims: int, size: int) -> ArrayLike:
    """generate a random unitary matrix U(n) for a given dimenstion n

    Args:
        dims (int): dimension of the unitary matrix (n)
        size (int): the number of random unitrary matrices

    Returns:
        ArrayLike: 3D (size, dims, dims) complex-valued array
    """
    return unitary_group.rvs(dims, size)

def generate_random_densitymatrix(dims: int) -> ArrayLike:
    """generate a random (n x n) density matrix for a given dimension n

    Args:
        dims (int): dimension of the density matrix

    Returns:
        ArrayLike: 2D (n,n) complex-valued array
    """

    return qinfo.random_density_matrix(dims)

def generate_random_quantum_state(size:int, dims: int) -> ArrayLike:
    """generate a normalized random complex vector

    Args:
        dims (int): dimension of the vector

    Returns:
        ArrayLike: normalized vector as a (n,1) array
    """
    
    rand_vec = np.random.uniform(-1, 1, (size, dims)) + 1.j * np.random.uniform(-1, 1, (size, dims))
    normalized = np.r_[[v/np.sqrt(np.dot(v, v.conj())) for v in rand_vec]]

    return normalized

def generate_random_purification(state: ArrayLike, uni_cd: list[ArrayLike]) -> list[ArrayLike]:
    """generate a random purification by using a random unitary_cd and tensor product of (1_ab x U_cd) on the standard purification

    Args:
        psi_s (ArrayLike): 1D (dim_a * dim_b * dim_c * dim_d, 1) array
        uni (list[ArrayLike]): 2D (dim_c * dim_d, dim_c * dim_d) unitary matrix

    Returns:
        list[ArrayLike]: tensor product of (1_ab x U_cd) on |psi_s> as a 1D (dim_a * dim_b * dim_c * dim_d, 1) vector
    """
    identity = np.eye(dim_a * dim_b)
    operators = np.r_[[np.kron(identity, u) for u in uni_cd]]
    if state.shape == (dim_abcd):
        return np.r_[[np.matmul(oper, state) for oper in operators]]
    elif state.shape == (dim_abcd, dim_abcd):
        return np.r_[[np.matmul(np.matmul(oper, state), oper.conj().transpose()) for oper in operators]]

def obtain_entanglement_entropy(psi_abc: ArrayLike):
    # STEP 1: generate a random density matrix (rho_ab), compute its entropy, get its standard purification (psi_s)
    psi_d = np.ones(dim_d)/np.sqrt(dim_d)
    rho = np.kron(np.outer(psi_abc, psi_abc.conj()), np.outer(psi_d, psi_d.conj()))
    rho_abc = np.trace(rho.reshape(dim_a, dim_b, dim_c, dim_d, dim_a, dim_b, dim_c, dim_d).transpose(0,4,1,5,2,6,3,7), axis1=6, axis2=7)
    rho_ab = np.trace(rho_abc, axis1=4, axis2=5).transpose(0,2,1,3).reshape(dim_a * dim_b, dim_a * dim_b)
    entropy_ab = compute_entropy(rho_ab)

    # STEP 2: get a random purification (psi) of (rho_ab) by applying (1_ab x unitary_cd) on (psi_s)
    unitary_cd = generate_random_unitary(dim_c * dim_d, size)
    rho_psi = generate_random_purification(rho, unitary_cd)

    # STEP 3: get the reduced density matrix of (rho_bd) from (rho_psi) by tracing over AC and compute its entropy
    rho_tensor = rho_psi.reshape(size, dim_a, dim_b, dim_c, dim_d, dim_a, dim_b, dim_c, dim_d).transpose(0,1,5,2,6,3,7,4,8)
    rho_bcd = np.trace(rho_tensor, axis1=1, axis2=2)
    rho_bd = np.trace(rho_bcd, axis1=3, axis2=4).transpose(0,1,3,2,4).reshape(size, dim_b * dim_d, dim_b * dim_d)
    entropy_bd = np.r_[[compute_entropy(x) for x in rho_bd]]

    # STEP 4: find the minimum of the (entropy_bd) which is the entanglement of purification E_p(rho)
    entanglement_ab = min(entropy_bd)

    return entropy_ab, entanglement_ab
 

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)

    dim_a, dim_b = 2, 2    # AB is the bipartite system
    dim_c, dim_d = 4, 4    # C and D are auxiliary systems of A and B, respectively
    dim_abcd = dim_a * dim_b * dim_c * dim_d
    size = 1000

    psi_abc = generate_random_quantum_state(size, dim_a * dim_b * dim_c)
    result = np.r_[[obtain_entanglement_entropy(p) for p in psi_abc]]
    
    # STEP 5: plot E_p(rho) against S(rho)
    x = result[:,0]
    y = result[:,1]
    plt.scatter(x, y, s=10, c='b', marker='o')
    plt.axis([0., np.log2(dim_a*dim_b), 0., np.log2(dim_b*dim_d)])
    plt.show()
        
        

