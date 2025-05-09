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
        
            
    def compute_L_matrix(self, method='svd'):
        
        std_devs = np.sqrt(np.diag(self.covariance_matrix))
        n_params = len(std_devs)
        # Create correlation matrix
        corr_matrix = np.zeros_like(self.covariance_matrix)
        for i in range(n_params):
            for j in range(n_params):
                if std_devs[i] > 0 and std_devs[j] > 0:
                    corr_matrix[i, j] = self.covariance_matrix[i, j] / (std_devs[i] * std_devs[j])
                else:
                    # Handle zero standard deviations - set correlation to 0
                    corr_matrix[i, j] = 0.0 if i != j else 1.0
        try:
            if method == 'cholesky':
                # Try Cholesky decomposition
                self.L_matrix = np.linalg.cholesky(corr_matrix)
                self.is_cholesky = True
            elif method == 'svd':
                # Use SVD decomposition: C = U Sigma U^T => L_matrix = U sqrt(Sigma)
                U, s, _ = np.linalg.svd(corr_matrix)
                self.L_matrix = U @ np.diag(np.sqrt(s))
                self.is_cholesky = False  
            elif method == 'eigen':
                # Eigen decomposition fallback (for non-positive definite matrix)
                eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
                eigenvalues[eigenvalues < 0] = 0
                self.L_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))
                self.is_cholesky = False  
            else:
                raise ValueError("Unknown decomposition method.")
        except np.linalg.LinAlgError:
            # Handle failure gracefully
            print("Decomposition failed using method:", method)
            raise
        
        # if debug:
        #     print("Correlation matrix decomposition:")
        #     print(f"Using Cholesky: {np.allclose(self.L_matrix @ self.L_matrix.T, corr_matrix)}")
        #     if not np.allclose(self.L_matrix @ self.L_matrix.T, corr_matrix):
        #         print("Using spectral decomposition instead")


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
        
    @abstractmethod
    def update_tape(self):
        pass


    def calculate_adjusted_mean(self, nominal_mean, a, b=10.0):
        """
        Calculate the adjusted mean for a truncated normal distribution to ensure
        that the truncated distribution has the desired nominal mean.
        
        Parameters:
        -----------
        nominal_mean : float
            The desired mean for the truncated distribution
        a : float
            Lower bound in standard deviations
        b : float
            Upper bound in standard deviations (default: 10.0)
            
        Returns:
        --------
        float
            The adjusted mean parameter (loc) to use in truncnorm
        """
        from scipy.stats import norm
        from scipy.optimize import root_scalar
        
        # For wide bounds (where truncation has minimal effect), return nominal mean
        if a < -5 and b > 5:
            return 0.0  # No adjustment needed for standard normal
            
        # Define the function that computes the mean of truncated normal
        # with given mu (loc) parameter. We want this to equal nominal_mean.
        def mean_difference(mu):
            # Adjust bounds for the new mu
            a_adj = a - mu
            b_adj = b - mu
            
            # Calculate truncated mean for this mu
            # Formula: mu + (pdf(a_adj) - pdf(b_adj))/(cdf(b_adj) - cdf(a_adj))
            pdf_a = norm.pdf(a_adj)
            pdf_b = norm.pdf(b_adj)
            cdf_a = norm.cdf(a_adj)
            cdf_b = norm.cdf(b_adj)
            
            # Avoid division by zero
            if abs(cdf_b - cdf_a) < 1e-10:
                return 0.0
                
            trunc_mean = mu + (pdf_a - pdf_b)/(cdf_b - cdf_a)
            
            # Return difference from desired mean
            return trunc_mean - nominal_mean
        
        # Use root-finding to solve for mu
        try:
            result = root_scalar(mean_difference, bracket=[-5.0, 5.0], method='brentq')
            if result.converged:
                return result.root
        except:
            # If root-finding fails, use approximation
            pass
            
        # Fallback approximation: shift mu in opposite direction of truncation effect
        # For one-sided truncation (b is large), this works reasonably well
        return -a/3.0 if a > -5 else 0.0


    def calculate_adjusted_sigma(self, nominal_sigma, a, b=10.0, adjusted_mean=0.0):
        """
        Calculate the adjusted standard deviation for a truncated normal distribution
        to ensure that the truncated distribution has the desired nominal standard deviation.
        
        The formula is derived from the variance of a truncated normal distribution:
        sigma^2 * [1 + (a*pdf_a - b*pdf_b)/(cdf_b - cdf_a) - ((pdf_a - pdf_b)/(cdf_b - cdf_a))^2]
        
        Parameters:
        -----------
        nominal_sigma : float
            The desired standard deviation for the truncated distribution
        a : float
            Lower bound in original standard deviations
        b : float
            Upper bound in original standard deviations (default: 10.0)
        adjusted_mean : float
            The adjusted mean parameter, if already calculated
            
        Returns:
        --------
        float
            The adjusted scale parameter to use in truncnorm
        """
        from scipy.stats import norm
        from scipy.optimize import root_scalar
        
        # For wide bounds (where truncation has minimal effect), return nominal sigma
        if a < -5 and b > 5:
            return nominal_sigma  # No adjustment needed for mild truncation
        
        # Adjust bounds based on the adjusted mean
        a_adj = a - adjusted_mean
        b_adj = b - adjusted_mean
        
        # Calculate components for the variance formula
        pdf_a = norm.pdf(a_adj)
        pdf_b = norm.pdf(b_adj)
        cdf_a = norm.cdf(a_adj)
        cdf_b = norm.cdf(b_adj)
        
        # Avoid division by zero
        if abs(cdf_b - cdf_a) < 1e-10:
            return nominal_sigma
        
        # Term 1: (a*pdf_a - b*pdf_b)/(cdf_b - cdf_a)
        term1 = (a_adj * pdf_a - b_adj * pdf_b) / (cdf_b - cdf_a)
        
        # Term 2: ((pdf_a - pdf_b)/(cdf_b - cdf_a))^2
        term2 = ((pdf_a - pdf_b) / (cdf_b - cdf_a)) ** 2
        
        # Variance reduction factor due to truncation
        var_factor = 1.0 + term1 - term2
        
        # Avoid negative or very small variance factors
        if var_factor < 0.1:
            var_factor = 0.1  # Set a minimum factor to avoid numerical issues
        
        # Calculate adjusted sigma
        adjusted_sigma = nominal_sigma / np.sqrt(var_factor)
        
        return adjusted_sigma


    def sample_parameters(self, sampling_method="Simple", mode="stack", use_copula=False, num_samples=1, debug=False):
        """
        Sample parameters based on the covariance matrix using the specified sampling method.
        
        Parameters:
        -----------
        sampling_method : str
            The sampling method to use. Options are:
            - 'Simple': Standard Monte Carlo sampling
            - 'LHS': Latin Hypercube Sampling
            - 'Sobol': Sobol sequence sampling
            - 'Halton': Halton sequence sampling
        mode : str
            How to apply samples to parameters:
            - 'stack': Append new samples (default)
            - 'replace': Replace existing samples
        use_copula : bool
            Whether to use Gaussian copula for respecting marginal distributions
        num_samples : int
            Number of samples to generate (all samples are generated at once)
        debug : bool
            If True, prints out the matrix of sampled parameters and returns without updating
        """
        if self.L_matrix is None:
            raise ValueError("Decomposed covariance matrix is not initialized")
        
        n_params = self.L_matrix.shape[0]
        
        # Generate all samples at once
        batch_size = num_samples
        
        scrambling = True # samples are randomly placed within cells of the grid

        # Generate uniform samples based on the chosen method
        if sampling_method == "Simple":
            # Simple Monte Carlo sampling - batch of samples
            if use_copula:
                # Generate uniform values in range (0,1) but avoid extremes (0 and 1)
                # Use a safer range away from extremes
                u_uniform = np.random.uniform(0.001, 0.999, size=(batch_size, n_params))
                z = None
            else:
                z = np.random.normal(size=(batch_size, n_params))
                u_uniform = None
                
        elif sampling_method == "LHS":
            # Latin Hypercube Sampling using scipy.stats.qmc
            from scipy.stats import qmc
            
            # Generate LHS samples in [0, 1]
            sampler = qmc.LatinHypercube(d=n_params, scramble=scrambling)
            u_uniform_raw = sampler.random(batch_size)
            
            # If using copula, scale values to avoid extremes
            if use_copula:
                # Rescale to avoid extreme values more aggressively
                u_uniform = 0.001 + 0.998 * u_uniform_raw
            else:
                from scipy.stats import norm
                z = norm.ppf(u_uniform_raw)
                u_uniform = None  
                    
        elif sampling_method == "Sobol":
            # Sobol sequence sampling
            from scipy.stats import qmc
            
            sampler = qmc.Sobol(d=n_params, scramble=scrambling)
            u_uniform_raw = sampler.random(batch_size)
            
            if use_copula:
                # Rescale to avoid extreme values more aggressively
                u_uniform = 0.001 + 0.998 * u_uniform_raw
            else:
                from scipy.stats import norm
                z = norm.ppf(u_uniform_raw)
                u_uniform = None
                
        elif sampling_method == "Halton":
            # Halton sequence sampling
            from scipy.stats import qmc
            
            sampler = qmc.Halton(d=n_params, scramble=scrambling)
            u_uniform_raw = sampler.random(batch_size)
            
            if use_copula:
                # Rescale to avoid extreme values more aggressively
                u_uniform = 0.001 + 0.998 * u_uniform_raw
            else:
                from scipy.stats import norm
                z = norm.ppf(u_uniform_raw)
                u_uniform = None
                
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        # Apply correlation structure
        if use_copula:
            # Transform uniform marginals to correlated standard normal
            from scipy.stats import norm
            
            # Transform each uniform sample to independent standard normal
            z_independent = norm.ppf(u_uniform)
            
            # Apply correlation structure to each sample using L_matrix instead of L_matrix
            z_correlated = np.zeros_like(z_independent)
            for i in range(batch_size):
                # Use L_matrix (correlation decomposition)
                z_correlated[i] = self.L_matrix @ z_independent[i]
            
            # IMPORTANT FIX: Bound z-values to prevent extreme CDF values
            # Limit to ±8 standard deviations, which gives CDF values around 1e-15 and 1-1e-15
            z_correlated = np.clip(z_correlated, -8.0, 8.0)
            
            # Convert back to uniform using standard normal CDF
            # Apply more aggressive clipping to prevent near-1 values
            u_correlated = norm.cdf(z_correlated)
            
            # Apply final rescaling to ensure safe range (0.001, 0.999)
            # This ensures we don't get values too close to 0 or 1
            u_correlated = 0.001 + 0.998 * u_correlated
            
            # Add diagnostic information for debug mode
            if debug:
                extreme_values_count = np.sum(u_correlated > 0.999)
                if extreme_values_count > 0:
                    print(f"\nDIAGNOSTIC: Found {extreme_values_count} values > 0.999 after correction")
                    print(f"Max value: {np.max(u_correlated)}")
                    print(f"This represents {extreme_values_count/(batch_size*n_params):.2%} of all values")
                    
                    # Find parameters with extreme values
                    param_indices = np.where(np.any(u_correlated > 0.999, axis=0))[0]
                    if len(param_indices) > 0:
                        print(f"Parameters with extreme values: {param_indices}")
                        
                        # Show max z-values before CDF transformation
                        max_z_values = np.max(z_correlated, axis=0)[param_indices]
                        print(f"Max z-values for these parameters: {max_z_values}")
            
            samples = u_correlated
            self.sampled_uniform_values = u_correlated
        else:
            # Standard approach - apply correlation structure directly
            samples = np.zeros((batch_size, n_params))
            for i in range(batch_size):
                samples[i] = self.mean_vector + self.L_matrix @ z[i]
        
        self.sampled_values = samples
        
        # Let subclasses handle how these samples are applied to their specific parameters
        self._apply_samples(samples, mode, use_copula=use_copula, batch_size=batch_size, 
                            sampling_method=sampling_method, debug=debug)
        
        return samples

    def _apply_samples(self, samples, mode="stack", use_copula=False, batch_size=1, 
                  sampling_method="Simple", debug=False):
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
        debug : bool
            If True, print and save the transformed parameter samples
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


