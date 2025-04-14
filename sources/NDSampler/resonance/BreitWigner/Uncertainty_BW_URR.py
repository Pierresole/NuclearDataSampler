import numpy as np
from collections import defaultdict
from ..ResonanceRangeCovariance import ResonanceRangeCovariance
from .Parameters_URR import URREnergyDependent
from ...mathmatrix import CovarianceMatrixHandler
from ENDFtk import tree
from ENDFtk.MF2.MT151 import ResonanceRange, Isotope, Section
from scipy.linalg import block_diag  # Import block_diag function


class Uncertainty_BW_URR(ResonanceRangeCovariance):
    """
    Class for handling uncertainties in the Unresolved Resonance Region (URR) Breit-Wigner parameters.
    NER : int, energy range index
    MPAR : int, number of parameters in the covariance matrix
    LFW : int, fission width flag
    L_matrix : np.ndarray, Cholesky decomposition of the covariance matrix
    urre_data : instance of URREnergyDependent, URR parameters
    """

    def __init__(self, mf2_resonance_range, mf32_resonance_range, NER):
        self.NER = NER
        super().__init__(mf2_resonance_range, NER)
        self.MPAR = mf32_resonance_range.parameters.covariance_matrix.MPAR
        self.LFW = mf32_resonance_range.parameters.LFW

        # Extract parameters and covariance matrices
        self.urre_data = URREnergyDependent.from_endftk(mf2_resonance_range)
        self.extract_covariance_matrix(mf32_resonance_range)
        self.remove_zero_variance_parameters()
        
        # Process the covariance matrix
        handler = CovarianceMatrixHandler(self.covariance_matrix)

        # Get the decomposition
        self.L_matrix, self.is_cholesky = handler.get_decomposition()
        
        # Set mean vector (for relative deviations, the mean is zero)
        n_params = self.L_matrix.shape[0]
        self.mean_vector = np.zeros(n_params)

    def get_covariance_matrix(self):
        """
        Override to return the specific covariance type.
        """
        return np.dot(self.L_matrix, self.L_matrix.T)
    
    def get_covariance_type(self):
        """
        Override to return the specific covariance type.
        """
        return "URR_BreitWigner"
    
    def extract_covariance_matrix(self, mf32_resonance_range):
        covariance_data = mf32_resonance_range.parameters.covariance_matrix
        NPAR_spin = covariance_data.NPAR  # Number of parameters at the spin group level

        # Get the spin-level covariance matrix
        relative_cov_matrix_upper = covariance_data.covariance_matrix
        relative_cov_matrix_spin = np.zeros((NPAR_spin, NPAR_spin))
        idx = 0
        for i in range(NPAR_spin):
            for j in range(i, NPAR_spin):
                relative_cov_matrix_spin[i, j] = relative_cov_matrix_upper[idx]
                relative_cov_matrix_spin[j, i] = relative_cov_matrix_upper[idx]
                idx += 1

        # Build the expanded covariance matrix as a block-diagonal matrix
        param_names = self.get_param_names()
        num_params_per_spin = len(param_names)

        block_matrices = []
        index_list = []
        for l_idx, l_value in enumerate(self.urre_data.Llist):
            for j_idx, j_value in enumerate(l_value.Jlist):
                num_energies = len(j_value.RP)
                for e_idx in range(num_energies):
                    # Add to index mapping
                    for p_idx, param_name in enumerate(param_names):
                        index_list.append(
                            (l_idx, j_idx, e_idx, param_name.encode('utf-8'))
                        )
                    # Append the spin-level covariance matrix for this energy
                    block_matrices.append(relative_cov_matrix_spin)
        
        # Build block-diagonal covariance matrix
        covariance_matrix = block_diag(*block_matrices)
        
        # Set the covariance matrix as an attribute of CovarianceBase
        # This ensures we're using the one from the parent class
        super().__setattr__('covariance_matrix', covariance_matrix)

        # Convert index_list to structured array
        dtype = np.dtype([
            ('l_idx', 'i4'),
            ('j_idx', 'i4'),
            ('e_idx', 'i4'),
            ('param_name', 'S20')  # String of max length 20
        ])
        self.index_mapping = np.array(index_list, dtype=dtype)

    def remove_zero_variance_parameters(self):
        """
        Removes parameters with zero variance and updates the covariance matrix accordingly.
        """
        # Identify zero variance parameters
        zero_variance_indices = np.where(np.diag(self.covariance_matrix) == 0.0)[0]
        if zero_variance_indices.size == 0:
            return

        # Remove parameters with zero variance from index_mapping
        self.index_mapping = np.array([
            self.index_mapping[i]
            for i in range(len(self.index_mapping))
            if i not in zero_variance_indices
        ], dtype=self.index_mapping.dtype)

        # Remove entries from covariance matrix
        self.covariance_matrix = np.delete(self.covariance_matrix, zero_variance_indices, axis=0)
        self.covariance_matrix = np.delete(self.covariance_matrix, zero_variance_indices, axis=1)

    def _apply_samples(self, samples, mode="stack", use_copula=False, batch_size=1, 
                      sampling_method="Simple", debug=False):
        """
        Apply generated samples to the URR resonance parameters.
        
        Parameters:
        -----------
        samples : numpy.ndarray
            The samples generated by sample_parameters. Can be a single sample or a batch of samples.
            If use_copula=True, these are uniform values that need to be transformed.
        mode : str
            How to apply samples:
            - 'stack': Append new samples 
            - 'replace': Replace existing samples
        use_copula : bool
            Whether copula transformation was used. If True, samples contains uniform values
            that need to be transformed to the appropriate distribution.
        batch_size : int
            Number of samples in the batch (1 for Simple method, >1 for LHS/Sobol)
        sampling_method : str
            The sampling method used ('Simple', 'LHS', or 'Sobol')
        debug : bool
            If True, print and save the transformed parameter samples
        """
        from scipy.stats import norm, truncnorm
        
        # Handle single sample vs batch sample format
        if batch_size == 1:
            # Single sample (1D array)
            sample_list = [samples]  # Convert to list with one element for consistent processing
            operation_mode = 'replace'
        else:
            # Batch of samples (2D array - samples[sample_index][parameter_index])
            sample_list = samples
            operation_mode = 'stack'
            
        # For debug mode, collect transformed samples
        if debug:
            n_params = samples.shape[1] if batch_size > 1 else len(samples)
            transformed_samples = np.zeros((batch_size, n_params))
            param_names = []
            
        # Process each batch of samples
        for sample_batch_idx, current_samples in enumerate(sample_list):
            # Apply deviations to parameters
            for idx_in_Y, item in enumerate(self.index_mapping):
                l_idx = item['l_idx']
                j_idx = item['j_idx']
                e_idx = item['e_idx']
                param_name = item['param_name'].decode('utf-8')  # Decode bytes to string

                # Get the relative deviation value
                if use_copula:
                    # For copula, transform uniform value to standard normal
                    u_value = current_samples[idx_in_Y]
                    # Ensure safe range for ppf transformation
                    u_value = np.clip(u_value, 0.001, 0.999)
                    # Transform to standard normal
                    relative_deviation = norm.ppf(u_value)
                    
                    # For parameters that must be positive, ensure reasonable values
                    if param_name in ['D', 'GN', 'GG', 'GF', 'GX']:
                        # Limit extreme negative values to avoid zero or negative parameters
                        relative_deviation = max(relative_deviation, -4.0)
                else:
                    # Standard approach - samples are already relative deviations
                    relative_deviation = current_samples[idx_in_Y]

                l_value = self.urre_data.Llist[l_idx]
                j_value = l_value.Jlist[j_idx]
                rp = j_value.RP[e_idx]

                # Get the original value
                original_value = getattr(rp, param_name)[0]

                # Compute the sampled value (applying relative deviation)
                sampled_value = original_value * (1.0 + relative_deviation)

                # Ensure positive values for parameters that must be positive
                if param_name in ['D', 'GN', 'GG', 'GF', 'GX'] and sampled_value <= 0:
                    # Set a minimum value as a small fraction of the original
                    sampled_value = max(sampled_value, original_value * 0.01)
                
                # Store transformed sample for debug output
                if debug:
                    transformed_samples[sample_batch_idx, idx_in_Y] = sampled_value
                    if sample_batch_idx == 0:  # Collect names only once
                        param_names.append(f"{param_name}_L{l_idx}_J{j_idx}_E{e_idx}")

                # Determine the effective sample index
                effective_sample_idx = sample_batch_idx + 1  # +1 because index 0 is the original value
                
                # Apply the value based on the mode
                param_list = getattr(rp, param_name)
                if operation_mode == 'stack':
                    if effective_sample_idx < len(param_list):
                        param_list[effective_sample_idx] = sampled_value
                    else:
                        param_list.append(sampled_value)
                elif operation_mode == 'replace':
                    # Clear all existing samples except the original if this is first parameter
                    if sample_batch_idx == 0 and idx_in_Y == 0 and len(param_list) > 1:
                        param_list = [param_list[0]]
                        setattr(rp, param_name, param_list)
                    
                    # Add the new sampled value
                    if effective_sample_idx < len(param_list):
                        param_list[effective_sample_idx] = sampled_value
                    else:
                        param_list.append(sampled_value)
        
        # Debug output of transformed samples
        if debug:
            print(f"\n=== Debug Output for URR Breit-Wigner (Transformed Samples) ===")
            print(f"Number of parameters: {len(self.index_mapping)}")
            print(f"Number of samples: {batch_size}")
            print(f"Sampling method: {sampling_method}")
            
            # Print sample matrix
            print("\nTransformed sample matrix (first 5 samples, first 10 parameters):")
            display_samples = transformed_samples[:min(5, batch_size), :min(10, len(self.index_mapping))]
            for i, sample in enumerate(display_samples):
                print(f"Sample {i+1}: {sample}")
            
            # Calculate and print sample correlations
            if batch_size > 1:
                print("\nTransformed sample correlation matrix (first 5 parameters):")
                sample_corr = np.corrcoef(transformed_samples.T)[:min(5, len(self.index_mapping)), :min(5, len(self.index_mapping))]
                for row in sample_corr:
                    print(" ".join([f"{x:.2f}" for x in row]))
            
            # Save to CSV with parameter names as headers
            header = ",".join(param_names[:min(20, len(self.index_mapping))])  # First 20 params for readability
            np.savetxt(f'transformed_samples_URR_BW.csv', 
                      transformed_samples[:, :min(20, len(self.index_mapping))], 
                      delimiter=',', header=header, comments="")
            print(f"\nTransformed samples saved to transformed_samples_URR_BW.csv")
            print("=" * 50)

    def extract_samples(self):
        """
        Extracts samples from urre_data using index_mapping and returns them as a dictionary.

        Returns:
            samples_dict: A dictionary where keys are (l_idx, j_idx, e_idx, param_name)
                        and values are lists of samples, including the original value.
        """
        samples_dict = {}

        # Build a mapping from indices to RP objects
        rp_mapping = {}
        for l_idx, l_value in enumerate(self.urre_data.Llist):
            for j_idx, j_value in enumerate(l_value.Jlist):
                for e_idx, rp in enumerate(j_value.RP):
                    rp_mapping[(l_idx, j_idx, e_idx)] = rp

        # Loop over index_mapping to extract samples
        for item in self.index_mapping:
            l_idx = item['l_idx']
            j_idx = item['j_idx']
            e_idx = item['e_idx']
            param_name = item['param_name'].decode('utf-8')

            key = (l_idx, j_idx, e_idx, param_name)

            rp = rp_mapping[(l_idx, j_idx, e_idx)]
            param_list = getattr(rp, param_name)
            # Include the original value (first element)
            samples = param_list  # This includes the original value at param_list[0]

            # Initialize the list if the key is not present
            if key not in samples_dict:
                samples_dict[key] = []

            samples_dict[key].extend(samples)

        return samples_dict

    def get_param_names(self):
        """
        Returns the list of parameter names based on MPAR and LFW.
        """
        if self.MPAR == 1:
            return ['D']
        elif self.MPAR == 2:
            return ['D', 'GN']
        elif self.MPAR == 3:
            return ['D', 'GN', 'GG']
        elif self.MPAR == 4:
            if self.LFW == 0:
                return ['D', 'GN', 'GG', 'GX']
            elif self.LFW == 1:
                return ['D', 'GN', 'GG', 'GF']
        elif self.MPAR == 5:
            return ['D', 'GN', 'GG', 'GF', 'GX']
        else:
            raise ValueError(f"Unsupported MPAR value: {self.MPAR}")
        
    def update_tape(self, tape, sample_index=1, sample_name=""):
        """
        Updates the tape with the sampled parameters.

        Parameters:
        - tape: The ENDF tape object to update.
        - sample_index: Index of the sample to use.
        """
        updated_params = self.urre_data.update_resonance_parameters(sample_index)
        self._update_resonance_range(tape, updated_params)

    def write_to_hdf5(self, hdf5_group):
        """
        Writes the L_matrix and urre_data to the given HDF5 group.
        """
        # Write L_matrix
        hdf5_group.create_dataset('L_matrix', data=self.L_matrix)
        
        # Write index_mapping
        hdf5_group.create_dataset('index_mapping', data=self.index_mapping)

        # Write mean_vector
        if hasattr(self, 'mean_vector') and self.mean_vector is not None:
            hdf5_group.create_dataset('mean_vector', data=self.mean_vector)

        # Write attributes
        hdf5_group.attrs['MPAR'] = self.MPAR
        hdf5_group.attrs['LFW'] = self.LFW
        hdf5_group.attrs['NER'] = self.NER
        hdf5_group.attrs['is_cholesky'] = self.is_cholesky

        # Write urre_data
        urre_data_group = hdf5_group.create_group('urre_data')
        self.urre_data.write_to_hdf5(urre_data_group)
            
    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads the L_matrix and urre_data from the given HDF5 group and returns an instance.
        """
        # Read L_matrix
        L_matrix = hdf5_group['L_matrix'][()]
        
        # Read index_mapping
        index_mapping_array = hdf5_group['index_mapping'][()]
        
        # Read mean_vector if available
        mean_vector = hdf5_group['mean_vector'][()] if 'mean_vector' in hdf5_group else None

        # Read attributes
        MPAR = hdf5_group.attrs['MPAR']
        LFW = hdf5_group.attrs['LFW']
        NER = hdf5_group.attrs['NER']
        is_cholesky = hdf5_group.attrs.get('is_cholesky', False)

        # Read urre_data
        urre_data_group = hdf5_group['urre_data']
        urre_data = URREnergyDependent.read_from_hdf5(urre_data_group)

        # Create an instance without calling __init__
        instance = cls.__new__(cls)
        
        # Set attributes that belong to the parent CovarianceBase class
        super(cls, instance).__setattr__('L_matrix', L_matrix)
        super(cls, instance).__setattr__('is_cholesky', is_cholesky)
        
        if mean_vector is not None:
            super(cls, instance).__setattr__('mean_vector', mean_vector)
        
        # Set attributes specific to this class
        instance.MPAR = MPAR
        instance.LFW = LFW
        instance.NER = NER
        instance.urre_data = urre_data
        instance.index_mapping = index_mapping_array
        
        return instance