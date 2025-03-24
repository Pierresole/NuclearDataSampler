import numpy as np
from collections import defaultdict
from .Parameters_RM_RRR import ReichMooreData
from ..ResonanceRangeCovariance import ResonanceRangeCovariance
from ENDFtk import tree
import time

class Uncertainty_RM_RRR(ResonanceRangeCovariance):
    def __init__(self, mf2_resonance_ranges, mf32_resonance_range, NER):
        # Initialize base attributes
        self.NER = NER
        
        # Extract covariance matrix only once at initialization
        diag_uncertainties = None
        cov_matrix = None
        
        if mf32_resonance_range.parameters.LCOMP == 1:
            start_time = time.time()
            # Extract the covariance block
            if mf32_resonance_range.parameters.NSRS > 0:
                covariance_data = mf32_resonance_range.parameters.short_range_blocks[0]
                covariance_order = covariance_data.NPARB
                MPAR = covariance_data.MPAR  # Number of parameters per resonance
                
                # Get the flattened upper triangular matrix data
                cm = np.array(covariance_data.covariance_matrix)
                
                # Create empty covariance matrix
                cov_matrix = np.zeros((covariance_order, covariance_order))
                
                # Use numpy indexing for direct assignment to upper triangle
                triu_indices = np.triu_indices(covariance_order)
                cov_matrix[triu_indices] = cm
                
                # Make it symmetric
                cov_matrix = cov_matrix + cov_matrix.T - np.diag(np.diag(cov_matrix))
                # Extract diagonal (standard deviations)
                diag_uncertainties = {
                    'diag': np.sqrt(np.diag(cov_matrix)),
                    'MPAR': MPAR,
                    'ER': covariance_data.ER
                }
                print(f"Time for extracting covariance matrix: {time.time() - start_time:.4f} seconds")
        else:
            raise ValueError(f"Unsupported LCOMP value: {mf32_resonance_range.parameters.LCOMP}")
        
        # Set the covariance matrix as an attribute of CovarianceBase
        super().__setattr__('covariance_matrix', cov_matrix)
        
        # Initialize data model with uncertainty information
        start_time = time.time()
        self.rm_data = ReichMooreData.from_endftk(mf2_resonance_ranges, mf32_resonance_range, diag_uncertainties)
        print(f"Time for ReichMooreData.from_endftk: {time.time() - start_time:.4f} seconds")
        
        # Remove zero variance parameters from the covariance matrix
        start_time = time.time()
        self._filter_covariance_matrix()
        print(f"Time for _filter_covariance_matrix: {time.time() - start_time:.4f} seconds")

        # Compute L matrix for sampling
        start_time = time.time()
        self.compute_L_matrix()
        print(f"Time for compute_L_matrix: {time.time() - start_time:.4f} seconds")
        
    def get_covariance_type(self):
        """
        Override to return the specific covariance type.
        """
        return "Uncertainty_RM_RRR"
        
    def _filter_covariance_matrix(self):
        """
        Removes zero variance parameters from the covariance matrix and reports on removed parameters.
        No explicit mapping is needed as filtering is handled when sampling.
        """
        if self.covariance_matrix is None:
            return
            
        # Get the diagonal elements of the covariance matrix
        variances = np.diag(self.covariance_matrix)
        
        # Find indices where variance is not zero (use small threshold for numerical precision)
        non_zero_indices = np.where(variances > 1e-15)[0]
        zero_variance_indices = np.where(variances <= 1e-15)[0]
        
        if len(zero_variance_indices) == 0:
            # No parameters to remove
            print("No zero variance parameters found, using full covariance matrix")
            return
        
        # Count parameters by type for reporting
        original_size = self.covariance_matrix.shape[0]
        filtered_size = len(non_zero_indices)
        
        # Print diagnostics
        print(f"Removed {original_size - filtered_size} parameters with zero variance")
        print(f"Original covariance matrix size: {original_size}x{original_size}")
        
        # Update the covariance matrix by removing zero variance rows and columns
        self.covariance_matrix = self.covariance_matrix[np.ix_(non_zero_indices, non_zero_indices)]
        
        print(f"Filtered covariance matrix size: {filtered_size}x{filtered_size}")

    def _perturb_radius_parameters(self, sample_list, operation_mode, use_copula, sampling_method, same_perturbation=False):
        """
        Perturbs the radius parameters (AP and APL) based on the given samples.
        
        Parameters:
        -----------
        sample_list : list
            List of samples to use (each sample is a numpy array)
        operation_mode : str
            Operation mode ('replace' or 'stack')
        use_copula : bool
            Whether copula transformation was used
        sampling_method : str
            The sampling method used
        same_perturbation : bool
            If True, applies identical perturbations to AP and all APL values
            assuming they represent the same physical radius parameter. ENDF manual not clear on that.
        """
        from scipy.stats import norm, qmc, truncnorm
        
        # Number of samples to generate
        num_samples = len(sample_list)
        
        # If using same perturbation, generate common perturbation values for all radius parameters
        common_z_values = None
        if same_perturbation:
            if sampling_method == "Simple":
                common_z_values = np.random.normal(size=num_samples)
            elif sampling_method == "LHS":
                from pyDOE3 import lhs
                common_u_values = lhs(1, samples=num_samples).flatten()
                common_z_values = norm.ppf(common_u_values)
            elif sampling_method == "Sobol":
                sampler = qmc.Sobol(d=1, scramble=True)
                common_u_values = sampler.random(num_samples).flatten()
                common_z_values = norm.ppf(common_u_values)
        
        # Perturb AP if it has uncertainty
        if self.rm_data.DAP is not None and self.rm_data.DAP > 0:
            # Use either common perturbation values or generate specific ones for AP
            if same_perturbation:
                z_ap_values = common_z_values
            else:
                if sampling_method == "Simple":
                    z_ap_values = np.random.normal(size=num_samples)
                elif sampling_method == "LHS":
                    from pyDOE3 import lhs
                    u_ap_values = lhs(1, samples=num_samples).flatten()
                    z_ap_values = norm.ppf(u_ap_values)
                elif sampling_method == "Sobol":
                    sampler = qmc.Sobol(d=1, scramble=True)
                    u_ap_values = sampler.random(num_samples).flatten()
                    z_ap_values = norm.ppf(u_ap_values)
            
            # Apply samples
            nominal_ap = self.rm_data.AP[0]
            
            # Store perturbed values if we need to apply identical perturbations to APL
            perturbed_ap_values = []
            
            for sample_batch_idx, z_ap in enumerate(z_ap_values):
                # Calculate sampled value for AP
                # For positive parameters, use truncated normal distribution
                # Define lower bound for truncated normal (prevent negative or zero values)
                a = -1.0 / self.rm_data.DAP if self.rm_data.DAP > 0 else -np.inf
                
                # For truncated normal, we use the percentage point function (ppf)
                # with uniform random values between 0 and 1
                if use_copula:
                    # If using copula, we already have uniform values from norm.cdf(correlated_z)
                    u_ap = sample_list[sample_batch_idx][0]  # Use the first uniform value from samples
                    # Map from (0,1) to truncated normal using the inverse CDF (ppf)
                    z_ap = truncnorm.ppf(u_ap, a, np.inf, loc=0, scale=1)
                
                sampled_ap = nominal_ap * (1.0 + z_ap * self.rm_data.DAP)
                
                # Ensure positive value
                sampled_ap = max(sampled_ap, 1e-10)
                
                # Store for potential reuse with APL
                perturbed_ap_values.append((z_ap, sampled_ap))
                
                # Determine the effective sample index
                effective_sample_idx = sample_batch_idx + 1  # +1 because index 0 is the nominal value
                
                # Update the AP list according to the operation mode
                if operation_mode == 'stack':
                    if effective_sample_idx < len(self.rm_data.AP):
                        self.rm_data.AP[effective_sample_idx] = sampled_ap
                    else:
                        self.rm_data.AP.append(sampled_ap)
                elif operation_mode == 'replace':
                    # Clear all existing samples except the nominal
                    if len(self.rm_data.AP) > 1:
                        self.rm_data.AP = [self.rm_data.AP[0]]
                    # Add the new sample
                    self.rm_data.AP.append(sampled_ap)
        
        # Perturb APL for each L-group independently
        for l_group_idx, l_group in enumerate(self.rm_data.LGroups):
            if l_group.DAPL is not None and l_group.DAPL > 0:
                # If using same perturbation as AP, use stored AP values
                if same_perturbation and self.rm_data.DAP is not None and self.rm_data.DAP > 0:
                    # Use the same z-values as for AP but adjust for the specific APL uncertainty
                    nominal_apl = l_group.APL[0]
                    
                    for sample_batch_idx, (z_ap, _) in enumerate(perturbed_ap_values):
                        # Scale the relative perturbation by the ratio of uncertainties
                        # This ensures consistent relative perturbations
                        relative_uncertainty_ratio = l_group.DAPL / self.rm_data.DAP
                        z_apl = z_ap
                        
                        # Adjust the sampling bounds for truncated normal if needed
                        a = -1.0 / l_group.DAPL if l_group.DAPL > 0 else -np.inf
                        
                        # For truncated normal, we may need to re-sample if the z-value is outside valid range
                        if z_apl < a:
                            # Resample within valid range if common z-value would make APL negative
                            z_apl = truncnorm.rvs(a, np.inf, loc=0, scale=1)
                        
                        # Calculate sampled value for APL using same relative perturbation as AP
                        sampled_apl = nominal_apl * (1.0 + z_apl * l_group.DAPL)
                        
                        # Ensure positive value
                        sampled_apl = max(sampled_apl, 1e-10)
                        
                        # Update APL using the same pattern as for AP
                        effective_sample_idx = sample_batch_idx + 1
                        
                        if operation_mode == 'stack':
                            if effective_sample_idx < len(l_group.APL):
                                l_group.APL[effective_sample_idx] = sampled_apl
                            else:
                                l_group.APL.append(sampled_apl)
                        elif operation_mode == 'replace':
                            if len(l_group.APL) > 1:
                                l_group.APL = [l_group.APL[0]]
                            l_group.APL.append(sampled_apl)
                else:
                    # Independent perturbation for each L-group's APL
                    # Use either common perturbation values or generate specific ones for this APL
                    if same_perturbation:
                        z_apl_values = common_z_values
                    else:
                        if sampling_method == "Simple":
                            z_apl_values = np.random.normal(size=num_samples)
                        elif sampling_method == "LHS":
                            from pyDOE3 import lhs
                            u_apl_values = lhs(1, samples=num_samples).flatten()
                            z_apl_values = norm.ppf(u_apl_values)
                        elif sampling_method == "Sobol":
                            sampler = qmc.Sobol(d=1, scramble=True)
                            u_apl_values = sampler.random(num_samples).flatten()
                            z_apl_values = norm.ppf(u_apl_values)
                    
                    # Apply samples
                    nominal_apl = l_group.APL[0]
                    
                    for sample_batch_idx, z_apl in enumerate(z_apl_values):
                        # Define lower bound for truncated normal (prevent negative or zero values)
                        a = -1.0 / l_group.DAPL if l_group.DAPL > 0 else -np.inf
                        
                        # For truncated normal, use the percentage point function (ppf)
                        if use_copula:
                            # If using copula, we already have uniform values from norm.cdf(correlated_z)
                            u_apl = sample_list[sample_batch_idx][l_group_idx + 1]  # Use the next uniform value
                            # Map from (0,1) to truncated normal using the inverse CDF (ppf)
                            z_apl = truncnorm.ppf(u_apl, a, np.inf, loc=0, scale=1)
                        
                        # Calculate sampled value for APL
                        sampled_apl = nominal_apl * (1.0 + z_apl * l_group.DAPL)
                        
                        # Ensure positive value
                        sampled_apl = max(sampled_apl, 1e-10)
                        
                        # Determine the effective sample index
                        effective_sample_idx = sample_batch_idx + 1  # +1 because index 0 is the nominal value
                        
                        # Update the APL list according to the operation mode
                        if operation_mode == 'stack':
                            if effective_sample_idx < len(l_group.APL):
                                l_group.APL[effective_sample_idx] = sampled_apl
                            else:
                                l_group.APL.append(sampled_apl)
                        elif operation_mode == 'replace':
                            # Clear all existing samples except the nominal
                            if len(l_group.APL) > 1:
                                l_group.APL = [l_group.APL[0]]
                            # Add the new sample
                            l_group.APL.append(sampled_apl)

    def _apply_samples(self, samples, mode="stack", use_copula=False, batch_size=1, sampling_method="Simple"):
        """
        Apply generated samples to the resonance parameters with uncertainties.
        
        Parameters:
        -----------
        samples : numpy.ndarray
            The samples generated by sample_parameters. Can be a single sample or a batch of samples.
            If use_copula=True, these are uniform values that need to be transformed.
        mode : str
            Ignored parameter - operation mode is determined by batch_size
        use_copula : bool
            Whether copula transformation was used. If True, samples contains uniform values
            that need to be transformed to the appropriate distribution.
        batch_size : int
            Number of samples in the batch (1 for Simple method, >1 for LHS/Sobol)
        sampling_method : str
            The sampling method used ('Simple', 'LHS', or 'Sobol')
        """
        # Handle single sample vs batch sample format
        if batch_size == 1:
            # Single sample (1D array)
            sample_list = [samples]  # Convert to list with one element for consistent processing
            operation_mode = 'replace'
        else:
            # Batch of samples (2D array - samples[sample_index][parameter_index])
            sample_list = samples
            operation_mode = 'stack'
        
        from scipy.stats import norm, truncnorm
        
        # Perturb AP and APL radius parameters
        # Set same_perturbation=True if you want identical perturbations for AP and APL
        self._perturb_radius_parameters(sample_list, operation_mode, use_copula, sampling_method, same_perturbation=True)

        # Now process the covariance matrix samples
        for sample_batch_idx, current_samples in enumerate(sample_list):
            # Counter to track which sample we're using within the current batch
            sample_index = 0
            
            # Loop through all LGroups and resonances
            for l_group_idx, l_group in enumerate(self.rm_data.LGroups):                
                for resonance_idx, resonance in enumerate(l_group.resonances):
                    
                    # Process each parameter that has uncertainty (non-None value)
                    # Parameter order: ER, GN, GG, GFA, GFB
                    param_info = [
                        (resonance.ER, resonance.DER, 'ER', 'positive'),
                        (resonance.GN, resonance.DGN, 'GN', 'positive'),
                        (resonance.GG, resonance.DGG, 'GG', 'positive'),
                        (resonance.GFA, resonance.DGFA, 'GFA', 'signed'),
                        (resonance.GFB, resonance.DGFB, 'GFB', 'signed')
                    ]
                                        
                    for param_list, uncertainty, param_name, constraint_type in param_info:
                        # Skip parameters without uncertainty or empty parameters
                        if uncertainty is None or param_list is None or len(param_list) == 0:
                            continue
                        
                        # Get the nominal value
                        nominal_value = param_list[0]
                        
                        # Apply the sample if we have enough samples
                        if sample_index < len(current_samples):
                            # For copula-based samples, transform uniform values to appropriate distribution
                            if use_copula:
                                u_value = current_samples[sample_index]
                                
                                if constraint_type == 'positive' and nominal_value > 0:
                                    # For positive parameters, use truncated normal
                                    a = -1.0 / uncertainty if uncertainty > 0 else -np.inf
                                    z_value = truncnorm.ppf(u_value, a, np.inf, loc=0, scale=1)
                                else:
                                    # For signed parameters, use standard normal
                                    z_value = norm.ppf(u_value)
                                
                                # Apply the transformed z-value
                                sampled_value = nominal_value * (1.0 + z_value * uncertainty)
                            else:
                                # Standard approach - samples are already z-values
                                sample = current_samples[sample_index]
                                sampled_value = nominal_value * (1.0 + sample)
                            
                            # Apply constraints if needed
                            if constraint_type == 'positive' and nominal_value > 0:
                                sampled_value = max(sampled_value, 1e-10)  # Ensure positive with a small threshold
                            
                            # Determine the effective sample index
                            effective_sample_idx = sample_batch_idx + 1  # +1 because index 0 is the nominal value
                            
                            # Update the parameter list according to the operation mode
                            if operation_mode == 'stack':
                                if effective_sample_idx < len(param_list):
                                    # Already has a value at this position, replace it
                                    param_list[effective_sample_idx] = sampled_value
                                else:
                                    # Append the new sampled value
                                    param_list.append(sampled_value)
                            elif operation_mode == 'replace':
                                # Clear all existing samples except the nominal if this is the first parameter
                                if sample_batch_idx == 0 and sample_index == 0 and len(param_list) > 1:
                                    param_list = [param_list[0]]
                                    
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

    def update_tape(self, tape, sample_index=1, sample_name=""):
        """
        Updates the tape with the sampled parameters.

        Parameters:
        - tape: The ENDF tape object to update.
        - sample_index: Index of the sample to use.
        """
        self._update_resonance_range(tape, updated_parameters=self.rm_data.reconstruct(sample_index))

    @classmethod
    def from_covariance_data(cls, tape, NER, covariance_data):
        """
        Initializes the AveragedBreitWigner object using covariance data.

        Parameters:
        - tape: The ENDF tape object.
        - NER: The resonance range index.
        - covariance_data: The covariance data from the HDF5 file.

        Returns:
        - An instance of AveragedBreitWigner.
        """
        instance = cls(tape, NER)
        instance.set_covariance_data(covariance_data)
        return instance

    def set_covariance_data(self, covariance_data):
        """
        Sets covariance data from the HDF5 file into the object's attributes.

        Parameters:
        - covariance_data: The covariance data dictionary from the HDF5 file.
        """
        self.MPAR = covariance_data['MPAR']
        self.LFW = covariance_data['LFW']
        self.param_names = covariance_data['param_names']
        self.L_values = [{'L': L, 'J': J} for L, J in zip(covariance_data['L_values'], covariance_data['J_values'])]

        # Reconstruct parameters from covariance_data
        self.parameters = []
        groups = covariance_data['groups']
        for idx, group_key in enumerate(groups):
            group_data = groups[group_key]
            param_list = []
            for param_name in self.param_names:
                param_values = group_data[param_name]
                param_list.append(param_values)
            self.parameters.append({
                'L': self.L_values[idx]['L'],
                'J': self.L_values[idx]['J'],
                'parameters': param_list
            })

        # Set covariance matrix
        self.covariance_matrix = covariance_data['relative_covariance_matrix']
        self.num_parameters = self.covariance_matrix.shape[0]

    # def sample_parameters(self):
    #     """
    #     Samples new parameters based on the covariance matrix.
    #     """
    #     if self.L_matrix is None:
    #         # Compute L_matrix if not already computed
    #         self.compute_L_matrix()

    #     # Generate standard normal random variables
    #     N = np.random.normal(size=self.L_matrix.shape[0])

    #     # Compute sampled relative deviations
    #     Y = self.L_matrix @ N  # Y has size (num_parameters,)

    #     # Apply deviations to parameters
    #     idx_in_Y = 0
    #     sampled_parameters = []

    #     for group in self.parameters:
    #         sampled_group = {'L': group['L'], 'J': group['J'], 'sampled_parameters': {}}
    #         for param_name, mean_values in zip(self.param_names, group['parameters']):
    #             relative_deviation = Y[idx_in_Y]
    #             idx_in_Y += 1
    #             # Apply the deviation uniformly to the parameter array
    #             sampled_values = mean_values * (1 + relative_deviation)
    #             sampled_group['sampled_parameters'][param_name] = sampled_values
    #         sampled_parameters.append(sampled_group)

    #     return sampled_parameters
    
    def extract_covariance_matrix(self, mf32_range):
        """
        Extracts the covariance matrix using the method from the base class.
        """
        if mf32_range.parameters.LCOMP == 1:
            self.extract_covariance_matrix_LCOMP1(mf32_range)
        else:
            raise ValueError(f"Unsupported LCOMP value: {mf32_range.parameters.LCOMP}")
        
    def extract_covariance_matrix_LCOMP1(self, mf32_range):
        """
        Extracts the relative covariance matrix and constructs it efficiently from flattened data.
        """
        if mf32_range.parameters.NLRS > 0 or mf32_range.parameters.NSRS > 1:
            raise ValueError(f"Number of short-range covariance ({mf32_range.parameters.NSRS} > 1) or long-range ({mf32_range.parameters.NLRS} > 0) not supported.")
        
        covariance_data = mf32_range.parameters.short_range_blocks[0]
        covariance_order = covariance_data.NPARB  # Order of the matrix

        # Get the flattened upper triangular matrix data
        cm = np.array(covariance_data.covariance_matrix)
        
        # Create an empty matrix
        cov_matrix = np.zeros((covariance_order, covariance_order))
        
        # Use numpy indexing for direct assignment to upper triangle
        # Calculate indices for the upper triangular part
        triu_indices = np.triu_indices(covariance_order)
        
        # Assign values to upper triangle
        cov_matrix[triu_indices] = cm
        
        # Make symmetric by adding the transpose (excluding diagonal)
        # Since we already filled the diagonal in the upper triangle assignment
        cov_matrix = cov_matrix + cov_matrix.T - np.diag(np.diag(cov_matrix))
    
        # Set the covariance matrix as an attribute of CovarianceBase
        super().__setattr__('covariance_matrix', cov_matrix)

    def construct_mean_vector(self):
        """
        Constructs the mean vector from the mean parameters.
        """
        mean_values = []
        for group in self.parameters:
            for param_values in group['parameters']:
                # The mean of relative deviations is zero, but we store zeros for consistency
                mean_values.append(0.0)
        self.mean_vector = np.array(mean_values)

    def remove_zero_variance_parameters(self):
        """
        Removes parameters with zero variance from the covariance matrix and updates the index mapping accordingly.
        """
        if self.covariance_matrix is None:
            return
            
        # Get the diagonal elements of the covariance matrix
        variances = np.diag(self.covariance_matrix)
        
        # Find indices where variance is not zero (use small threshold to account for numerical precision)
        non_zero_indices = np.where(variances > 1e-15)[0]
        
        if len(non_zero_indices) == self.covariance_matrix.shape[0]:
            # No parameters to remove
            self.filtered_index_mapping = self.index_mapping
            self.original_to_filtered_index = {i: i for i in range(len(self.index_mapping))}
            return
        
        # Create mappings between original and filtered indices
        self.original_to_filtered_index = {old_idx: new_idx for new_idx, old_idx in enumerate(non_zero_indices)}
        
        # Create a filtered index_mapping
        self.filtered_index_mapping = [self.index_mapping[i] for i in non_zero_indices]
        
        # Print diagnostics
        print(f"Removed {len(self.index_mapping) - len(self.filtered_index_mapping)} parameters with zero variance")
        print(f"Original covariance matrix size: {self.covariance_matrix.shape}")
        
        # Update the covariance matrix by removing zero variance rows and columns
        self.covariance_matrix = self.covariance_matrix[np.ix_(non_zero_indices, non_zero_indices)]
        
        print(f"Filtered covariance matrix size: {self.covariance_matrix.shape}")

    def _build_index_mapping(self):
        """
        Builds a mapping from covariance matrix indices to resonance parameters.
        Each entry in index_mapping is a tuple (l_group_idx, resonance_idx, param_type)
        where param_type is:
        0 = ER (resonance energy)
        1 = GN (neutron width)
        2 = GG (gamma width)
        3 = GFA (first fission width)
        4 = GFB (second fission width)
        """
        self.index_mapping = []
        
        # Loop through L-groups
        for l_group_idx, l_group in enumerate(self.rm_data.LGroups):
            # Loop through resonances
            for resonance_idx, resonance in enumerate(l_group.resonances):
                # Skip resonances without uncertainties
                if resonance.index is None:
                    continue
                    
                # Check which parameters have uncertainties (non-None values)
                param_types = [(0, 'ER', resonance.DER), 
                              (1, 'GN', resonance.DGN), 
                              (2, 'GG', resonance.DGG), 
                              (3, 'GFA', resonance.DGFA), 
                              (4, 'GFB', resonance.DGFB)]
                
                for param_type, param_name, uncertainty in param_types:
                    # Only add parameters with non-zero uncertainty
                    if uncertainty is not None and uncertainty > 0:
                        self.index_mapping.append((l_group_idx, resonance_idx, param_type))
                        
                        # Set constraints based on parameter type
                        if param_name in resonance.constraints:
                            self.parameter_constraints[len(self.index_mapping) - 1] = resonance.constraints[param_name]
        
        # Diagnostic information
        print(f"Built index_mapping with {len(self.index_mapping)} parameters")
        print(f"Parameter constraints: {len(self.parameter_constraints)} constraints defined")

    def get_nominal_parameters(self):
        """
        Extracts the nominal parameters for parameters that have non-zero variance.
        Returns a numpy array of parameter values corresponding to filtered_index_mapping.
        """
        mapping_to_use = self.filtered_index_mapping if self.filtered_index_mapping is not None else self.index_mapping
        
        nominal_values = np.zeros(len(mapping_to_use))
        
        for i, (l_group_idx, resonance_idx, param_idx) in enumerate(mapping_to_use):
            l_group = self.rm_data.LGroups[l_group_idx]
            resonance = l_group.resonances[resonance_idx]
            
            # Extract the value based on param_idx
            if param_idx == 0 and resonance.ER and len(resonance.ER) > 0:
                nominal_values[i] = resonance.ER[0]
            elif param_idx == 1 and resonance.GN and len(resonance.GN) > 0:
                nominal_values[i] = resonance.GN[0]
            elif param_idx == 2 and resonance.GG and len(resonance.GG) > 0:
                nominal_values[i] = resonance.GG[0]
            elif param_idx == 3 and resonance.GFA and len(resonance.GFA) > 0:
                nominal_values[i] = resonance.GFA[0]
            elif param_idx == 4 and resonance.GFB and len(resonance.GFB) > 0:
                nominal_values[i] = resonance.GFB[0]
        
        return nominal_values

    def write_to_hdf5(self, hdf5_group):
        """
        Writes the L_matrix and rml_data to the given HDF5 group.
        """
        hdf5_group.attrs['NER'] = self.NER
        
        # Write L_matrix
        hdf5_group.create_dataset('L_matrix', data=self.L_matrix)

        # Write index_mapping as a compound dataset
        # dt = np.dtype([('l_group', 'i4'), ('resonance', 'i4'), ('parameter', 'i4')])
        # index_data = np.array(self.index_mapping, dtype=dt)
        # hdf5_group.create_dataset('index_mapping', data=index_data)

        # Write rml_data
        rml_data_group = hdf5_group.create_group('Parameters')
        self.rm_data.write_to_hdf5(rml_data_group)
        
    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads the L_matrix and rml_data from the given HDF5 group and returns an instance.
        """
        NER = hdf5_group.attrs['NER']
        
        # Read L_matrix
        L_matrix = hdf5_group['L_matrix'][()]

        # Read index_mapping
        # index_data = hdf5_group['index_mapping'][()]
        # index_mapping = [(int(idx['l_group']), int(idx['resonance']), int(idx['parameter'])) 
        #                 for idx in index_data]

        # Read rml_data
        rm_data_group = hdf5_group['Parameters']
                
        rm_data = ReichMooreData.read_from_hdf5(rm_data_group)
        
        # Create an instance and set attributes
        instance = cls.__new__(cls)
        instance.NER = NER
        
        # Set L_matrix on the parent CovarianceBase class
        super(cls, instance).__setattr__('L_matrix', L_matrix)
        # Set is_cholesky to False as default, since we don't know if it was a Cholesky decomposition
        # super(cls, instance).__setattr__('is_cholesky', hdf5_group.attrs.get('is_cholesky', False))
        
        # Set attributes specific to this class
        instance.rm_data = rm_data
        # instance._build_index_mapping()
        # instance.index_mapping = index_mapping

        return instance