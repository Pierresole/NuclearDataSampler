import numpy as np

class CovarianceMatrixHandler:
    def __init__(self, covariance_matrix):
        """
        Initialize the handler with a covariance matrix.
        """
        self.covariance_matrix = covariance_matrix
        self.L_matrix = None
        self.is_cholesky = False
        self.decompose_matrix()

    def decompose_matrix(self):
        """
        Process the covariance matrix to compute the decomposition.
        Attempts Cholesky decomposition first; if it fails, falls back to eigenvalue decomposition.
        """
        try:
            # Attempt Cholesky decomposition
            self.L_matrix = np.linalg.cholesky(self.covariance_matrix)
            self.is_cholesky = True  # Indicate that L_matrix is a Cholesky decomposition
        except np.linalg.LinAlgError:
            # Handle non-positive definite covariance matrix
            # Use eigenvalue decomposition as a fallback
            eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix)
            # Ensure all eigenvalues are non-negative
            eigenvalues[eigenvalues < 0] = 0
            # Reconstruct L_matrix
            self.L_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            self.is_cholesky = False  # Indicate that L_matrix is not a Cholesky decomposition

    def get_decomposition(self):
        """
        Get the decomposition matrix and its type (Cholesky or eigenvalue-based).
        """
        return self.L_matrix, self.is_cholesky
