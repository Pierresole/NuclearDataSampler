import numpy as np
import time
from collections import defaultdict
from ..ResonanceRangeCovariance import ResonanceRangeCovariance
from .Parameters_URR import URREnergyDependent
from ENDFtk import tree
from ENDFtk.MF2.MT151 import ResonanceRange, Isotope, Section
from scipy.linalg import block_diag  # Import block_diag function


class Uncertainty_BW_URR(ResonanceRangeCovariance):
    """
    Class for handling uncertainties in the Unresolved Resonance Region (URR) Breit-Wigner parameters.
    NER : int, energy range index
    MPAR : int, number of parameters in the covariance matrix
    LFW : int, fission width flag
    urre_data : instance of URREnergyDependent, URR parameters
    """

    def __init__(self, mf2_range, mf32_resonance_range, NER):
        self.NER = NER
        super().__init__(mf2_range, NER)
        self.MPAR = mf32_resonance_range.parameters.covariance_matrix.MPAR
        self.LFW = mf32_resonance_range.parameters.LFW

        # Extract parameters and covariance matrices
        self.urre_data = URREnergyDependent.from_endftk(mf2_range, mf32_resonance_range)
        
        # Fill a dictionary to expand the cov matrix for energies
        l_j_ne_dict = {}
        for l_index, l_value in enumerate(mf2_range.parameters.l_values.to_list()):
            l_j_ne_dict[l_index] = {}
            for j_index, j_value in enumerate(l_value.j_values.to_list()):
                l_j_ne_dict[l_index][j_index] = j_value.NE
        
        start_time = time.time()
        self.extract_covariance_matrix(mf32_resonance_range, l_j_ne_dict)
        print(f"Time for extracting covariance matrix: {time.time() - start_time:.4f} seconds")
        
        start_time = time.time()
        self.compute_L_matrix()
        print(f"Time for compute_L_matrix: {time.time() - start_time:.4f} seconds")
        
    
    def get_covariance_type(self):
        """
        Override to return the specific covariance type.
        """
        return "URR_BreitWigner"
    

    def extract_covariance_matrix(self, mf32_resonance_range, LJ_structure):
         # Get the base spin-level covariance matrix
        number_parameters = mf32_resonance_range.parameters.covariance_matrix.MPAR  # Number of parameters per (L,J)
        NPAR = mf32_resonance_range.parameters.covariance_matrix.NPAR  # Total number of parameters
        
        # Extract the relative covariance matrix (upper triangular form)
        relative_cov_matrix_upper = mf32_resonance_range.parameters.covariance_matrix.covariance_matrix
        
        # Convert to full symmetric matrix
        relative_cov_matrix_spin = np.zeros((NPAR, NPAR))
        triu_indices = np.triu_indices(NPAR)
        relative_cov_matrix_spin[triu_indices] = relative_cov_matrix_upper
        relative_cov_matrix_spin = relative_cov_matrix_spin + relative_cov_matrix_spin.T - np.diag(np.diag(relative_cov_matrix_spin))
        
        expanded_rel_cov_matrix = self.expand_relative_covariance(LJ_structure, 
                                                                  relative_cov_matrix_spin, 
                                                                  MPAR= number_parameters,
                                                                  fully_correlated_energies=False)
        
        rel_std_dev_vector = np.array(self.urre_data.get_relative_uncertainty())
        covariance_matrix = expanded_rel_cov_matrix * np.outer(rel_std_dev_vector, rel_std_dev_vector)

        super().__setattr__('covariance_matrix', covariance_matrix)
    
    
    def expand_relative_covariance(self, LJ_structure, relative_cov_matrix, MPAR=2, fully_correlated_energies=True):
        """
        Expand relative covariance matrix from (L,J,param) to (L,J,energy,param).

        Parameters:
        -----------
        LJ_structure: dict
            {L: {J: NEnergies}}.

        relative_cov_matrix: ndarray
            Relative covariance matrix shape (NPAR, NPAR), with NPAR=MPAR*N(L,J).

        MPAR: int
            Number of parameters per energy.

        fully_correlated_energies: bool
            If True, parameters fully correlated across energies.
            If False, parameters correlated only within the same energy.

        Returns:
        --------
        expanded_rel_cov_matrix: ndarray
            Expanded covariance matrix shape (N_total_params, N_total_params).
        """

        LJ_list = [(L, J) for L in sorted(LJ_structure) for J in sorted(LJ_structure[L])]
        NJS = len(LJ_list)
        expanded_dim = MPAR * sum(LJ_structure[L][J] for L, J in LJ_list)

        expanded_rel_cov_matrix = np.zeros((expanded_dim, expanded_dim))

        orig_i = 0
        exp_i = 0
        for L1, J1 in LJ_list:
            NE1 = LJ_structure[L1][J1]
            for e1 in range(NE1):
                orig_j = 0
                exp_j = 0
                for L2, J2 in LJ_list:
                    NE2 = LJ_structure[L2][J2]
                    cov_block = relative_cov_matrix[
                        orig_i:orig_i + MPAR,
                        orig_j:orig_j + MPAR
                    ]

                    for e2 in range(NE2):
                        if fully_correlated_energies or ((L1, J1, e1) == (L2, J2, e2)):
                            expanded_rel_cov_matrix[
                                exp_i:exp_i + MPAR,
                                exp_j:exp_j + MPAR
                            ] = cov_block
                        else:
                            expanded_rel_cov_matrix[
                                exp_i:exp_i + MPAR,
                                exp_j:exp_j + MPAR
                            ] = np.zeros((MPAR, MPAR))
                        exp_j += MPAR
                    orig_j += MPAR
                exp_i += MPAR
            orig_i += MPAR

        return expanded_rel_cov_matrix
    
        
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
            How to apply samples (ignored - operation mode is determined by batch_size)
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
            nominal_values = []
            uncertainties = []
            param_idx_to_info = {}  # Map parameter indices to info for debug output
        
        # Process each batch of samples
        for sample_batch_idx, current_samples in enumerate(sample_list):
            # Counter to track which sample we're using within the current batch
            sample_index = 0
            
            # Loop through all L-values
            for l_idx, l_value in enumerate(self.urre_data.Llist):
                # Loop through all J-values in this L-group
                for j_idx, j_value in enumerate(l_value.Jlist):
                    # Loop through all energy points in this J-group
                    for e_idx, rp in enumerate(j_value.RP):
                        # Process each parameter type
                        param_info = [
                            ('D', rp.D, rp.DD, 'positive'),
                            ('GN', rp.GN, rp.DGN, 'positive'),
                            ('GG', rp.GG, rp.DGG, 'positive'),
                            ('GF', rp.GF, rp.DGF, 'positive'),
                            ('GX', rp.GX, rp.DGX, 'positive')
                        ]
                        
                        for param_name, param_list, uncertainty, constraint_type in param_info:
                            # Skip parameters without uncertainty or empty parameters
                            if uncertainty is None or uncertainty <= 0 or param_list is None or len(param_list) == 0:
                                continue
                            
                            # Get the nominal value
                            nominal_value = param_list[0]
                            
                            # Apply the sample if we have enough samples
                            if sample_index < len(current_samples):
                                # For copula-based samples, transform uniform values to appropriate distribution
                                if use_copula:
                                    # Get the uniform value
                                    u_value = current_samples[sample_index]
                                    
                                    # Ensure the uniform value is in a safe range for ppf transformation
                                    u_value = np.clip(u_value, 0.001, 0.999)
                                    
                                    if constraint_type == 'positive' and nominal_value > 0:
                                        # For positive parameters, use truncated normal
                                        # Calculate lower bound in standard units to prevent negative values
                                        a = -nominal_value / uncertainty
                                        
                                        # Adjust for significant truncation
                                        if a > -10:
                                            # Calculate adjusted mean and scale for truncnorm
                                            loc = self.calculate_adjusted_mean(0.0, a)
                                            scale = self.calculate_adjusted_sigma(1.0, a, 10.0, loc)
                                            z_value = truncnorm.ppf(u_value, a - loc, 10.0 - loc, loc=loc, scale=scale)
                                        else:
                                            # No significant truncation needed
                                            z_value = norm.ppf(u_value)
                                    else:
                                        # For parameters that can be negative, use standard normal
                                        z_value = norm.ppf(u_value)
                                    
                                    # Apply the sample as deviation
                                    sampled_value = nominal_value + z_value * uncertainty
                                else:
                                    # Standard approach - samples are already z-values
                                    sample = current_samples[sample_index]
                                    
                                    # Apply absolute uncertainty to the nominal value
                                    sampled_value = nominal_value + sample * uncertainty
                                
                                # Apply constraints for positive parameters
                                if constraint_type == 'positive' and nominal_value > 0:
                                    # Ensure positive values with a minimum threshold
                                    min_value = max(1e-10, nominal_value * 0.001)
                                    sampled_value = max(sampled_value, min_value)
                                
                                # Store transformed sample for debug output
                                if debug:
                                    transformed_samples[sample_batch_idx, sample_index] = sampled_value
                                    if sample_batch_idx == 0:  # Collect metadata only once
                                        param_name_str = f"{param_name}_L{l_idx}_J{j_idx}_E{e_idx}"
                                        param_names.append(param_name_str)
                                        nominal_values.append(nominal_value)
                                        uncertainties.append(uncertainty)
                                        param_idx_to_info[sample_index] = (param_name_str, nominal_value, uncertainty)
                                
                                # Determine the effective sample index
                                effective_sample_idx = sample_batch_idx + 1  # +1 because index 0 is the original value
                                
                                # Apply the value based on the operation mode
                                if operation_mode == 'stack':
                                    if effective_sample_idx < len(param_list):
                                        # Replace existing sample
                                        param_list[effective_sample_idx] = sampled_value
                                    else:
                                        # Add new sample
                                        param_list.append(sampled_value)
                                elif operation_mode == 'replace':
                                    # Clear all existing samples except the original if this is first parameter
                                    if sample_batch_idx == 0 and sample_index == 0 and len(param_list) > 1:
                                        param_list.clear()
                                        param_list.append(nominal_value)
                                    
                                    # Add the new sampled value
                                    if effective_sample_idx < len(param_list):
                                        param_list[effective_sample_idx] = sampled_value
                                    else:
                                        param_list.append(sampled_value)
                                
                                # Move to next sample
                                sample_index += 1
            
            # Verify all samples in this batch were used
            if sample_index != len(current_samples):
                print(f"Warning: Not all samples in batch {sample_batch_idx+1} were used. Used {sample_index} out of {len(current_samples)}")

        # Debug output of transformed samples
        if debug:
            print(f"\n=== Debug Output for URR Breit-Wigner (Transformed Samples) ===")
            print(f"Number of parameters: {len(param_names)}")
            print(f"Number of samples: {batch_size}")
            print(f"Sampling method: {sampling_method}")
            
            # Print sample matrix
            print("\nTransformed sample matrix (first 5 samples, first 10 parameters):")
            display_samples = transformed_samples[:min(5, batch_size), :min(10, len(param_names))]
            for i, sample in enumerate(display_samples):
                print(f"Sample {i+1}: {sample}")
            
            # Print parameter verification
            print("\nParameter verification (first 10 parameters):")
            print(f"{'Parameter':<20} {'Nominal':<12} {'Uncertainty':<12} {'Ratio':<12}")
            for i in range(min(10, len(param_names))):
                if i in param_idx_to_info:
                    param_key, nominal, uncert = param_idx_to_info[i]
                    ratio = uncert/nominal if nominal != 0 else 0
                    print(f"{param_key:<20} {nominal:<12.6g} {uncert:<12.6g} {ratio:<12.6g}")

            # Calculate sample statistics if we have multiple samples
            if batch_size > 1:
                sample_means = np.mean(transformed_samples, axis=0)
                sample_stds = np.std(transformed_samples, axis=0)
                
                # Calculate percentage differences
                mean_pct_diff = np.zeros(len(param_names))
                std_pct_diff = np.zeros(len(param_names))
                
                for i in range(len(param_names)):
                    if i in param_idx_to_info:
                        _, nominal, uncert = param_idx_to_info[i]
                        # Avoid division by zero
                        if abs(nominal) > 1e-10:
                            mean_pct_diff[i] = 100.0 * (sample_means[i] - nominal) / nominal
                        if uncert > 1e-10:
                            std_pct_diff[i] = 100.0 * (sample_stds[i] - uncert) / uncert
                
                # Print parameters with most divergent means
                print("\nTop 5 parameters with largest mean percentage difference:")
                param_diff_info = []
                for i in range(len(param_names)):
                    if i in param_idx_to_info:
                        param_name, nominal, uncert = param_idx_to_info[i]
                        param_diff_info.append({
                            'param_name': param_name,
                            'nominal': nominal,
                            'uncertainty': uncert,
                            'mean': sample_means[i],
                            'std': sample_stds[i],
                            'mean_pct_diff': mean_pct_diff[i],
                            'std_pct_diff': std_pct_diff[i]
                        })
                
                # Sort and print statistics
                sorted_by_mean_diff = sorted(param_diff_info, key=lambda x: abs(x['mean_pct_diff']), reverse=True)
                print(f"{'Parameter':<20} {'Nominal':<12} {'Mean':<12} {'Diff%':<12} {'Uncertainty':<12}")
                for i, info in enumerate(sorted_by_mean_diff[:5]):
                    print(f"{info['param_name']:<20} {info['nominal']:<12.6g} {info['mean']:<12.6g} {info['mean_pct_diff']:<12.2f} {info['uncertainty']:<12.6g}")
                
                # Calculate and print sample correlations
                print("\nTransformed sample correlation matrix (first 8 parameters):")
                sample_corr = np.corrcoef(transformed_samples.T)[:min(8, len(param_names)), :min(8, len(param_names))]
                for row in sample_corr:
                    print(" ".join([f"{x:.2f}" for x in row]))
                
                # Compare with original correlation matrix if available
                if hasattr(self, 'covariance_matrix') and self.covariance_matrix is not None:
                    # Convert covariance to correlation
                    std_devs = np.sqrt(np.diag(self.covariance_matrix))
                    std_dev_matrix = np.outer(std_devs, std_devs)
                    orig_corr = self.covariance_matrix / std_dev_matrix
                    
                    print("\nOriginal correlation matrix (first 8 parameters):")
                    orig_corr_display = orig_corr[:min(8, len(param_names)), :min(8, len(param_names))]
                    for row in orig_corr_display:
                        print(" ".join([f"{x:.2f}" for x in row]))
            
                # Save to CSV with parameter names as headers
                header = ",".join(param_names[:min(20, len(param_names))])  # First 20 params for readability
                
                # Create a new array with statistics rows added
                csv_data = np.vstack([
                    nominal_values[:min(20, len(param_names))],
                    uncertainties[:min(20, len(param_names))],
                    transformed_samples[:, :min(20, len(param_names))]
                ])
                
                # Save with row labels
                csv_filename = 'transformed_samples_URR_BW.csv'
                with open(csv_filename, 'w') as f:
                    f.write("# Row,"+header+"\n")
                    f.write(f"Nominal,{','.join([f'{x:.8g}' for x in nominal_values[:min(20, len(param_names))]])}\n")
                    f.write(f"Uncertainty,{','.join([f'{x:.8g}' for x in uncertainties[:min(20, len(param_names))]])}\n")
                    
                    if batch_size > 1:
                        f.write(f"Mean,{','.join([f'{x:.8g}' for x in sample_means[:min(20, len(param_names))]])}\n")
                        f.write(f"StdDev,{','.join([f'{x:.8g}' for x in sample_stds[:min(20, len(param_names))]])}\n")
                        f.write(f"MeanPctDiff,{','.join([f'{x:.8g}' for x in mean_pct_diff[:min(20, len(param_names))]])}\n")
                        f.write(f"StdPctDiff,{','.join([f'{x:.8g}' for x in std_pct_diff[:min(20, len(param_names))]])}\n")
                    
                    # Add the actual samples
                    for i in range(min(batch_size, transformed_samples.shape[0])):
                        f.write(f"Sample{i+1},{','.join([f'{x:.8g}' for x in transformed_samples[i, :min(20, len(param_names))]])}\n")
                
                print(f"\nTransformed samples saved to {csv_filename}")
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

        # Write mean_vector
        # if hasattr(self, 'mean_vector') and self.mean_vector is not None:
        #     hdf5_group.create_dataset('mean_vector', data=self.mean_vector)

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
        
        return instance