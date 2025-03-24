import numpy as np
from abc import ABC, abstractmethod
import h5py
import scipy.stats as stats

class CovarianceBase(ABC):
    """
    Base class for all covariance classes in the project.
    Provides common functionality for covariance matrix operations.
    """

    def __init__(self):
        """
        Initialize the base covariance class.
        """
        self.covariance_matrix = None
        self.mean_vector = None
        self.std_dev_vector = None
        self.L_matrix = None
        self.is_cholesky = False
        self.sampled_values = None

    def delete_parameters(self, indices_to_delete):
        """
        Deletes parameters by indices and updates the covariance matrix and parameters list.

        Parameters:
        - indices_to_delete: List of indices of parameters to delete.
        """
        # Ensure indices are sorted in descending order to avoid index shifting issues
        indices_to_delete = sorted(indices_to_delete, reverse=True)

        # Delete rows and columns from the covariance matrix
        self.covariance_matrix = np.delete(self.covariance_matrix, indices_to_delete, axis=0)
        self.covariance_matrix = np.delete(self.covariance_matrix, indices_to_delete, axis=1)
        
        # Delete parameters from the list if parameters exist
        if hasattr(self, 'parameters') and self.parameters is not None:
            for idx in indices_to_delete:
                del self.parameters[idx]
            
            # Update indices in parameters
            for idx, param in enumerate(self.parameters):
                if isinstance(param, dict) and 'index' in param:
                    param['index'] = idx
        
        # Update mean vector and standard deviation vector
        if hasattr(self, 'mean_vector') and self.mean_vector is not None:
            self.mean_vector = np.delete(self.mean_vector, indices_to_delete)
            
        if hasattr(self, 'std_dev_vector') and self.std_dev_vector is not None:
            self.std_dev_vector = np.delete(self.std_dev_vector, indices_to_delete)

    def remove_zero_variance_parameters(self):
        """
        Removes parameters with zero variance and updates the covariance matrix accordingly.
        """
        # Identify parameters with non-zero standard deviation
        if hasattr(self, 'std_dev_vector') and self.std_dev_vector is not None:
            non_zero_indices = np.where(self.std_dev_vector != 0.0)[0]
        else:
            non_zero_indices = np.where(np.diag(self.covariance_matrix) != 0.0)[0]

        # Update parameters if they exist
        if hasattr(self, 'parameters') and self.parameters is not None:
            self.parameters = [self.parameters[i] for i in non_zero_indices]
            
        # Update vectors
        if hasattr(self, 'mean_vector') and self.mean_vector is not None:
            self.mean_vector = self.mean_vector[non_zero_indices]
            
        if hasattr(self, 'std_dev_vector') and self.std_dev_vector is not None:
            self.std_dev_vector = self.std_dev_vector[non_zero_indices]
            
        # Update the covariance matrix
        self.covariance_matrix = self.covariance_matrix[np.ix_(non_zero_indices, non_zero_indices)]
            
    def compute_L_matrix(self):
        try:
            self.L_matrix = np.linalg.cholesky(self.covariance_matrix)
            self.is_cholesky = True  
        except np.linalg.LinAlgError:
            # Handle non-positive definite covariance matrix
            eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix)
            eigenvalues[eigenvalues < 0] = 0
            self.L_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            self.is_cholesky = False  

    def write_to_hdf5(self, hdf5_group):
        """
        Writes the covariance data to an HDF5 group.
        """
        # Write L_matrix
        if self.L_matrix is not None:
            hdf5_group.create_dataset('L_matrix', data=self.L_matrix)
            
        # Write mean_vector
        if hasattr(self, 'mean_vector') and self.mean_vector is not None:
            hdf5_group.create_dataset('mean_vector', data=self.mean_vector)
            
        # Write standard deviations if available
        if hasattr(self, 'std_dev_vector') and self.std_dev_vector is not None:
            hdf5_group.create_dataset('std_dev_vector', data=self.std_dev_vector)
            
        # Indicate if L_matrix is a Cholesky decomposition
        hdf5_group.attrs['is_cholesky'] = self.is_cholesky
        
        # Call the derived class method to write format-specific data
        self.write_additional_data_to_hdf5(hdf5_group)
        
    # @abstractmethod
    # def write_additional_data_to_hdf5(self, hdf5_group):
    #     """
    #     Abstract method to write format-specific data to HDF5.
    #     To be implemented by derived classes.
    #     """
    #     pass

    # @abstractmethod
    # def sample_parameters(self):
    #     pass
        
    # @abstractmethod
    # def print_parameters(self):
    #     pass
    
    @abstractmethod
    def update_tape(self):
        pass

    def sample_parameters(self, sampling_method="Simple", mode="stack", use_copula=False, num_samples=1):
        """
        Sample parameters based on the covariance matrix using the specified sampling method.
        
        Parameters:
        -----------
        sampling_method : str
            The sampling method to use. Options are:
            - 'Simple': Standard Monte Carlo sampling
            - 'LHS': Latin Hypercube Sampling
            - 'Sobol': Sobol sequence sampling
        mode : str
            How to apply samples to parameters:
            - 'stack': Append new samples (default)
            - 'replace': Replace existing samples
        use_copula : bool
            Whether to use Gaussian copula for respecting marginal distributions
        num_samples : int
            Number of samples to generate (for LHS and Sobol, all samples are generated at once)
        """
        if self.L_matrix is None:
            raise ValueError("Decomposed covariance matrix is not initialized")
        
        n_params = self.L_matrix.shape[0]
        
        # For LHS and Sobol methods, always generate all samples at once
        batch_size = num_samples if sampling_method in ["LHS", "Sobol"] else 1
        
        # Generate uniform samples based on the chosen method
        if sampling_method == "Simple":
            # Simple Monte Carlo sampling - one sample at a time
            u_uniform = np.random.uniform(size=(1, n_params)) if use_copula else None
            
            # Standard normal random variables for correlation structure
            z = np.random.normal(size=n_params)
            
        elif sampling_method == "LHS":
            # Latin Hypercube Sampling - generate all samples at once
            from pyDOE3 import lhs
            
            # Generate LHS samples in [0, 1]
            u_uniform = lhs(n_params, samples=batch_size)
            
            # If using copula, we'll keep these uniform values
            # Otherwise, transform to normal distribution directly
            if not use_copula:
                from scipy.stats import norm
                z = norm.ppf(u_uniform[0]).flatten() if batch_size == 1 else norm.ppf(u_uniform)
            else:
                z = None
                
        elif sampling_method == "Sobol":
            # Sobol sequence sampling - generate all samples at once
            from scipy.stats import qmc, norm
            
            sampler = qmc.Sobol(d=n_params, scramble=True)
            u_uniform = sampler.random(batch_size)
            
            if not use_copula:
                z = norm.ppf(u_uniform[0]).flatten() if batch_size == 1 else norm.ppf(u_uniform)
            else:
                z = None
                
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        # Apply correlation structure
        if use_copula:
            # Transform uniform marginals to correlated standard normal
            from scipy.stats import norm
            
            if batch_size == 1:
                # Single sample case
                # First convert uniform samples to standard normal
                z_independent = norm.ppf(u_uniform).flatten()
                
                # Apply Cholesky to get correlated standard normal
                z_correlated = self.L_matrix @ z_independent
                
                # Convert back to uniform using the standard normal CDF
                u_correlated = norm.cdf(z_correlated)
                
                # This is handled by the _apply_samples method which needs to check constraints
                samples = u_correlated
                self.sampled_uniform_values = u_correlated
            else:
                # Batch case
                # Transform each uniform sample to independent standard normal
                z_independent = norm.ppf(u_uniform)
                
                # Apply correlation structure to each sample
                z_correlated = np.zeros_like(z_independent)
                for i in range(batch_size):
                    z_correlated[i] = self.L_matrix @ z_independent[i]
                
                # Convert back to uniform using standard normal CDF
                u_correlated = norm.cdf(z_correlated)
                
                samples = u_correlated
                self.sampled_uniform_values = u_correlated
        else:
            # Standard approach - apply correlation structure directly
            if batch_size == 1:
                samples = self.L_matrix @ z
            else:
                # For batches, apply correlation to each sample
                samples = np.zeros((batch_size, n_params))
                for i in range(batch_size):
                    samples[i] = self.L_matrix @ z[i]
        
        self.sampled_values = samples
        
        # Let subclasses handle how these samples are applied to their specific parameters
        self._apply_samples(samples, mode, use_copula=use_copula, batch_size=batch_size, sampling_method=sampling_method)
        
        return samples
    
    def _apply_samples(self, samples, mode="stack", use_copula: bool=False, batch_size: int=1, sampling_method: str="Simple"):
        """
        Apply generated samples to the model parameters.
        This method should be overridden by subclasses to handle specific parameter structures.
        
        Parameters:
        -----------
        samples : numpy.ndarray
            The samples generated by sample_parameters
        mode : str
            How to apply samples:
            - 'stack': Append new samples
            - 'replace': Replace existing samples
        use_copula : bool
            Whether copula transformation was used
        batch_size : int
            Number of samples in the batch (1 for Simple method, >1 for LHS/Sobol)
        sampling_method : str
            The sampling method used ('Simple', 'LHS', or 'Sobol')
        """
        # Default implementation (for simple cases only)
        # Most subclasses will need to override this with their specific implementation
        pass

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads the covariance data from the given HDF5 group and returns an instance.
        
        This is a base implementation that should be called by derived classes.
        """
        # Create an instance without calling __init__
        instance = cls.__new__(cls)
        
        # Set L_matrix
        if 'L_matrix' in hdf5_group:
            instance.L_matrix = hdf5_group['L_matrix'][()]
        
        # Set is_cholesky
        instance.is_cholesky = hdf5_group.attrs.get('is_cholesky', False)
        
        # Set mean_vector if available
        if 'mean_vector' in hdf5_group:
            instance.mean_vector = hdf5_group['mean_vector'][()]
        
        # Set std_dev_vector if available
        if 'std_dev_vector' in hdf5_group:
            instance.std_dev_vector = hdf5_group['std_dev_vector'][()]
        
        # Set covariance_matrix if available
        if 'covariance_matrix' in hdf5_group:
            instance.covariance_matrix = hdf5_group['covariance_matrix'][()]
        
        return instance


