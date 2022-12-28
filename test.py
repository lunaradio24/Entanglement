import numpy as np
from numpy.typing import ArrayLike
from qiskit import quantum_info as qinfo
import sys

dim_a, dim_b = 2, 2    # AB is the bipartite system
dim_c, dim_d = 4, 4    # C and D are auxiliary systems of A and B, respectively
dim_abcd = dim_a * dim_b * dim_c * dim_d


def test_eigenvalues(matrix: ArrayLike, eigens: tuple [np.ndarray, np.ndarray]) -> bool:
        """test the eigenvalue equation with given eigenvalues and eigenvectors for a given matrix

        Args:
            matrix (array_like):
                2D complex-valued array

            eigens (tuple [ndarray, ndarray]):
                tuple of eigenvalues (1D array) and eigenvectors (2D array)
                eigenvalues' [i] is the i-th eigenvalue
                eigenvectors' [:,i] is the i-th eigenvector

        Returns:
            bool: True when eigens are correct for the given matrix, False otherwise
        """
        
        vals, vecs = eigens
        dim = len(vals)
        eigvaltest = 1
        for i in range(dim):
                eigvaltest *= np.allclose(vecs[:,i], np.dot(matrix,vecs[:,i])/vals[i], rtol=1e-08, atol=1e-08)
                if eigvaltest == 0:
                        print(vecs[:,i])
                        print(np.dot(matrix,vecs[:,i])/vals[i])
                        break
        return bool(eigvaltest)


def test_density_matrix(rho: ArrayLike) -> bool:
        """test three conditions of a density matrix for a given matrix
           1. positive semi-definite
           2. hermitian
           3. trace 1

        Args:
            rho (array_like): 2D square complex-valued array
        
        Returns:
            bool: True when all three conditions for a density matrix are satisfied, False otherwise
        """
        w = np.linalg.eigvals(rho)
        psd = bool(np.all(w.real)>=0)
        her = bool(np.all(np.add(rho, np.transpose(rho))).imag < 1e-08)
        tr1 = bool(np.isclose(np.trace(rho).real, 1.0, rtol=1e-08, atol=1e-08) and np.isclose(sum(w.real), 1.0, rtol=1e-08, atol=1e-08))

        return psd and her and tr1


def test_unitary(matrix: ArrayLike) -> bool:
        """test the unitarity condition for a given matrix

        Args:
            matrix (array_like): 2D square complex-valued array

        Returns:
            bool: True when the unitary condition is satisfied, False otherwise
        """
        
        dims = np.size(matrix)[0]
        uu = np.matmul(np.conjugate(np.transpose(matrix)), np.asarray(matrix))

        return np.allclose(uu.real, np.eye(dims)) and np.allclose(uu.imag, np.zeros((dims, dims)))


def test_purification(psi: ArrayLike, rho_target: ArrayLike) -> bool:
        """test if a given vector is a correct purification of a given matrix

        Args:
            psi_s (array_like): 1D array
            rho (_type_): _description_
        """
        psi = psi.reshape(dim_a * dim_b * dim_c * dim_d)
        rho_psi = np.outer(psi, psi.conj())
        rho_tensor = rho_psi.reshape([dim_a, dim_b, dim_c, dim_d, dim_a, dim_b, dim_c, dim_d])
        rho_abc = np.trace(rho_tensor, axis1=3, axis2=7)
        rho_ab = np.trace(rho_abc, axis1=2, axis2=5).reshape([dim_a * dim_b, dim_a * dim_b])
      
        return bool(np.allclose(rho_ab, rho_target, rtol=1e-05, atol=1e-08))


if __name__ == '__main__':
        np.set_printoptions(threshold=sys.maxsize)
        rho = qinfo.random_density_matrix((2,1))
        print(type(rho))
        w, v = np.linalg.eig(rho)
        print(rho)
        print(test_eigenvalues(rho,(w,v)))
        print(test_density_matrix(rho))
        uni = qinfo.random_unitary(2)
        print(test_unitary(uni))

