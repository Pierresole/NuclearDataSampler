import numpy as np
from collections import defaultdict
from .Parameters_RM_RRR import ReichMooreData
from ..ResonanceRangeCovariance import ResonanceRangeCovariance
from ENDFtk import tree
import time
from scipy.optimize import root_scalar

class Uncertainty_RM_RRR(ResonanceRangeCovariance):
    """
    Class to handle the uncertainty of Reich-Moore resonance parameters.
    NER : int, energy range index
    rm_data : ReichMooreData, data model with uncertainty information
    """
    
    def __init__(self, mf2_resonance_ranges, mf32_resonance_range, NER):
        # Initialize base attributes
        self.NER = NER
        
        start_time = time.time()
        self.extract_covariance_matrix(mf32_resonance_range) 
        print(f"Time for extracting covariance matrix: {time.time() - start_time:.4f} seconds")

        
        # Initialize data model with uncertainty information
        start_time = time.time()
        self.rm_data = ReichMooreData.from_endftk(mf2_resonance_ranges, mf32_resonance_range)
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
        from scipy.stats import norm, truncnorm, qmc
        
        # Number of samples to generate
        num_samples = len(sample_list)
        
        # If using same perturbation, generate common perturbation values for all radius parameters
        common_z_values = None
        if same_perturbation:
            if sampling_method == "Simple":
                common_z_values = np.random.normal(size=num_samples)
            elif sampling_method == "LHS":
                sampler = qmc.LatinHypercube(d=1, scramble=True)
                common_u_values = sampler.random(num_samples).flatten()
                common_z_values = norm.ppf(common_u_values)
            elif sampling_method == "Sobol":
                sampler = qmc.Sobol(d=1, scramble=True)
                common_u_values = sampler.random(num_samples).flatten()
                common_z_values = norm.ppf(common_u_values)
            elif sampling_method == "Halton":
                sampler = qmc.Halton(d=1, scramble=True)
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
                    sampler = qmc.LatinHypercube(d=1, scramble=True)
                    u_ap_values = sampler.random(num_samples).flatten()
                    z_ap_values = norm.ppf(u_ap_values)
                elif sampling_method == "Sobol":
                    sampler = qmc.Sobol(d=1, scramble=True)
                    u_ap_values = sampler.random(num_samples).flatten()
                    z_ap_values = norm.ppf(u_ap_values)
                elif sampling_method == "Halton":
                    sampler = qmc.Halton(d=1, scramble=True)
                    u_ap_values = sampler.random(num_samples).flatten()
                    z_ap_values = norm.ppf(u_ap_values)
            
            # Apply samples
            nominal_ap = self.rm_data.AP[0]
            
            # Store perturbed values if we need to apply identical perturbations to APL
            perturbed_ap_values = []
            
            for sample_batch_idx, z_ap in enumerate(z_ap_values):
                # If using copula, we already have uniform values from norm.cdf(correlated_z)
                if use_copula:
                    # Get uniform value from samples
                    u_ap = sample_list[sample_batch_idx][0]
                    
                    # Ensure the uniform value is in a safe range for ppf transformation
                    # Use a more conservative range to avoid numerical issues
                    u_ap = np.clip(u_ap, 0.001, 0.999)
                    
                    z_ap = norm.ppf(u_ap)
                
                # Use absolute uncertainty instead of relative
                sampled_ap = nominal_ap + z_ap * self.rm_data.DAP
                
                # Ensure positive value for physical parameters
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
                        z_apl = z_ap
                        
                        # FIXED: Apply absolute uncertainty instead of relative
                        sampled_apl = nominal_apl + z_apl * l_group.DAPL
                        
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
                            sampler = qmc.LatinHypercube(d=1, scramble=True)
                            u_apl_values = sampler.random(num_samples).flatten()
                            z_apl_values = norm.ppf(u_apl_values)
                        elif sampling_method == "Sobol":
                            sampler = qmc.Sobol(d=1, scramble=True)
                            u_apl_values = sampler.random(num_samples).flatten()
                            z_apl_values = norm.ppf(u_apl_values)
                        elif sampling_method == "Halton":
                            sampler = qmc.Halton(d=1, scramble=True)
                            u_apl_values = sampler.random(num_samples).flatten()
                            z_apl_values = norm.ppf(u_apl_values)
                    
                    # Apply samples
                    nominal_apl = l_group.APL[0]
                    
                    for sample_batch_idx, z_apl in enumerate(z_apl_values):
                        # For truncated normal, use the percentage point function (ppf)
                        if use_copula:
                            # If using copula, we already have uniform values from norm.cdf(correlated_z)
                            u_apl = sample_list[sample_batch_idx][l_group_idx + 1]  # Use the next uniform value
                            
                            # Ensure the uniform value is in a safe range for ppf transformation
                            u_apl = np.clip(u_apl, 0.001, 0.999)
                            
                            z_apl = norm.ppf(u_apl)
                        
                        # FIXED: Apply absolute uncertainty instead of relative
                        sampled_apl = nominal_apl + z_apl * l_group.DAPL
                        
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


    def _apply_samples(self, samples, mode="stack", use_copula=False, batch_size=1, 
                    sampling_method="Simple", debug=False, radius_only=False):
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
        debug : bool
            If True, print and save the transformed parameter samples
        radius_only : bool
            If True, only perturb scattering radius parameters (AP and APL)
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
        
        # For debug mode, we'll collect transformed samples
        if debug:
            n_params = samples.shape[1] if batch_size > 1 else len(samples)
            transformed_samples = np.zeros((batch_size, n_params))
            param_names = []
            param_indices = []
        
        # Perturb AP and APL radius parameters
        # Set same_perturbation=True if you want identical perturbations for AP and APL
        self._perturb_radius_parameters(sample_list, operation_mode, use_copula, sampling_method, same_perturbation=True)
        radius_only = False
        # If radius_only is True, skip the rest of the parameter perturbations
        if radius_only:
            return
        
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
                                # Get the uniform value
                                u_value = current_samples[sample_index]
                                
                                # Ensure the uniform value is in a safe range for ppf transformation
                                # Use a more conservative range to avoid numerical issues
                                u_value = np.clip(u_value, 0.001, 0.999)
                                
                                if constraint_type == 'positive' and nominal_value > 0:
                                    # For positive parameters, use truncated normal
                                    # Calculate the lower bound in standard units to prevent negative values
                                    a = -nominal_value / uncertainty if uncertainty > 0 else -np.inf
                                    
                                    # Adjust mean and standard deviation if needed
                                    if a > -10:  # Only adjust if truncation has significant effect
                                        # Calculate adjusted mean parameter for truncnorm
                                        # loc = self.calculate_adjusted_mean(0.0, a)
                                        
                                        # Calculate adjusted standard deviation for truncnorm
                                        # scale = self.calculate_adjusted_sigma(1.0, a, 10.0, loc)
                                        
                                        # Use truncated normal with adjusted parameters
                                        # z_value = truncnorm.ppf(u_value, a - loc, 10.0 - loc, loc=loc, scale=scale)
                                        z_value = truncnorm.ppf(u_value, a, np.inf, loc=0.0, scale=1.0)

                                        
                                        # Apply adjusted uncertainty
                                        sampled_value = nominal_value + z_value * uncertainty # / scale
                                    else:
                                        # If lower bound is far away, use regular normal
                                        z_value = norm.ppf(u_value)
                                        sampled_value = nominal_value + z_value * uncertainty
                                else:
                                    # For signed parameters, use standard normal
                                    # Use clipped uniform value to prevent numerical issues
                                    z_value = norm.ppf(u_value)
                                    sampled_value = nominal_value + z_value * uncertainty
                            else:
                                # Standard approach - samples are already z-values
                                sample = current_samples[sample_index]
                                
                                # Apply absolute uncertainty to the nominal value
                                sampled_value = nominal_value + sample * uncertainty
                            
                            # Apply constraints if needed
                            if constraint_type == 'positive' and nominal_value > 0:
                                # Ensure positive with a small threshold
                                # For very small values, use a percentage of the nominal
                                min_value = max(1e-10, nominal_value * 0.001)
                                sampled_value = max(sampled_value, min_value)
                            
                            # Store transformed sample for debug output
                            if debug:
                                transformed_samples[sample_batch_idx, sample_index] = sampled_value
                                if sample_batch_idx == 0:  # Collect names only once
                                    param_names.append(f"{param_name}_L{l_group_idx}_R{resonance_idx}")
                                    param_indices.append(sample_index)
                            
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

        # Debug output of transformed samples
        if debug:
            print(f"\n=== Debug Output for {self.__class__.__name__} (Transformed Samples) ===")
            print(f"Number of parameters: {n_params}")
            print(f"Number of samples: {batch_size}")
            print(f"Sampling method: {sampling_method}")
            
            # Collect nominal values and uncertainties for parameters
            nominal_values = np.zeros(n_params)
            uncertainties = np.zeros(n_params)
            
            # Map parameter indices to their nominal values and uncertainties
            param_idx_to_info = {}
            
            for l_group_idx, l_group in enumerate(self.rm_data.LGroups):
                for resonance_idx, resonance in enumerate(l_group.resonances):
                    param_info = [
                        (resonance.ER, resonance.DER, 'ER', 0),
                        (resonance.GN, resonance.DGN, 'GN', 1),
                        (resonance.GG, resonance.DGG, 'GG', 2),
                        (resonance.GFA, resonance.DGFA, 'GFA', 3),
                        (resonance.GFB, resonance.DGFB, 'GFB', 4)
                    ]
                    
                    for param_list, uncertainty, param_name, param_type in param_info:
                        if uncertainty is not None and param_list is not None and len(param_list) > 0:
                            param_key = f"{param_name}_L{l_group_idx}_R{resonance_idx}"
                            # Find the index in param_names (if it exists)
                            if param_key in param_names:
                                idx = param_names.index(param_key)
                                nominal_values[idx] = param_list[0]
                                uncertainties[idx] = uncertainty
                                param_idx_to_info[idx] = (param_key, param_list[0], uncertainty)
            
            # Calculate sample statistics
            sample_means = np.mean(transformed_samples, axis=0) if batch_size > 1 else transformed_samples[0]
            sample_stds = np.std(transformed_samples, axis=0) if batch_size > 1 else np.zeros(n_params)
            
            # Calculate percentage differences for means and standard deviations
            mean_pct_diff = np.zeros(n_params)
            std_pct_diff = np.zeros(n_params)
            
            for i in range(n_params):
                if i in param_idx_to_info:
                    _, nominal, uncert = param_idx_to_info[i]
                    # Avoid division by zero for mean percentage difference
                    if abs(nominal) > 1e-10:
                        mean_pct_diff[i] = 100.0 * (sample_means[i] - nominal) / nominal
                    else:
                        mean_pct_diff[i] = 0.0 if abs(sample_means[i]) < 1e-10 else 100.0  # 100% if nominal near zero but sample isn't
                        
                    # Avoid division by zero for std percentage difference
                    if uncert > 1e-10:
                        std_pct_diff[i] = 100.0 * (sample_stds[i] - uncert) / uncert
                    else:
                        std_pct_diff[i] = 0.0 if sample_stds[i] < 1e-10 else 100.0  # 100% if uncert near zero but sample isn't
            
            # Create parameter info array for sorting
            param_diff_info = []
            for i in range(n_params):
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
                    
            # Sort by mean percentage difference (abs value)
            sorted_by_mean_diff = sorted(param_diff_info, key=lambda x: abs(x['mean_pct_diff']), reverse=True)
            # Sort by std percentage difference (abs value)
            sorted_by_std_diff = sorted(param_diff_info, key=lambda x: abs(x['std_pct_diff']), reverse=True)
            
            # Print sample matrix
            print("\nTransformed sample matrix (first 5 samples, first 10 parameters):")
            display_samples = transformed_samples[:min(5, batch_size), :min(10, n_params)]
            for i, sample in enumerate(display_samples):
                print(f"Sample {i+1}: {sample}")
            
            # Print comparison of nominal, uncertainty, and sample statistics
            print("\nParameter verification (first 10 parameters):")
            print(f"{'Parameter':<20} {'Nominal':<12} {'Uncertainty':<12} {'Mean':<12} {'Std Dev':<12} {'Std/Uncert':<12}")
            for i in range(min(10, n_params)):
                if i in param_idx_to_info:
                    param_key, nominal, uncert = param_idx_to_info[i]
                    ratio = sample_stds[i]/uncert if uncert > 0 else 0
                    print(f"{param_key:<20} {nominal:<12.6g} {uncert:<12.6g} {sample_means[i]:<12.6g} {sample_stds[i]:<12.6g} {ratio:<12.6g}")
            
            # Print parameters with most divergent means
            print("\nTop 5 parameters with largest mean percentage difference:")
            print(f"{'Parameter':<20} {'Nominal':<12} {'Mean':<12} {'Diff%':<12} {'Uncertainty':<12}")
            for i, info in enumerate(sorted_by_mean_diff[:5]):
                print(f"{info['param_name']:<20} {info['nominal']:<12.6g} {info['mean']:<12.6g} {info['mean_pct_diff']:<12.2f} {info['uncertainty']:<12.6g}")
                
            # Print parameters with most divergent standard deviations
            print("\nTop 5 parameters with largest std dev percentage difference:")
            print(f"{'Parameter':<20} {'Uncertainty':<12} {'Std Dev':<12} {'Diff%':<12} {'Nominal':<12}")
            for i, info in enumerate(sorted_by_std_diff[:5]):
                print(f"{info['param_name']:<20} {info['uncertainty']:<12.6g} {info['std']:<12.6g} {info['std_pct_diff']:<12.2f} {info['nominal']:<12.6g}")
            
            # Calculate and print sample correlations
            if batch_size > 1:
                print("\nTransformed sample correlation matrix (first 5 parameters):")
                sample_corr = np.corrcoef(transformed_samples.T)[:min(8, n_params), :min(8, n_params)]
                for row in sample_corr:
                    print(" ".join([f"{x:.2f}" for x in row]))
                
                # Compare with original correlation matrix
                if hasattr(self, 'L_matrix') and self.L_matrix is not None:
                    orig_corr = self.L_matrix @ self.L_matrix.T
                    
                    print("\nOriginal correlation matrix (first 5 parameters):")
                    orig_corr_display = orig_corr[:min(8, n_params), :min(8, n_params)]
                    for row in orig_corr_display:
                        print(" ".join([f"{x:.2f}" for x in row]))
            
            # Save to CSV with parameter names as headers and statistics in first rows
            header = ",".join(param_names[:min(20, n_params)])  # First 20 params for readability
            
            # Create a new array with statistics rows added (including percentage differences)
            csv_data = np.vstack([
                nominal_values[:min(20, n_params)],
                uncertainties[:min(20, n_params)],
                sample_means[:min(20, n_params)],
                sample_stds[:min(20, n_params)],
                mean_pct_diff[:min(20, n_params)],
                std_pct_diff[:min(20, n_params)],
                transformed_samples[:, :min(20, n_params)]
            ])
            
            # Save with row labels
            csv_filename = f'transformed_samples_{self.__class__.__name__}.csv'
            with open(csv_filename, 'w') as f:
                f.write("# Row,"+header+"\n")
                f.write(f"Nominal,{','.join([f'{x:.8g}' for x in nominal_values[:min(20, n_params)]])}\n")
                f.write(f"Uncertainty,{','.join([f'{x:.8g}' for x in uncertainties[:min(20, n_params)]])}\n")
                f.write(f"SampleMean,{','.join([f'{x:.8g}' for x in sample_means[:min(20, n_params)]])}\n")
                f.write(f"SampleStdDev,{','.join([f'{x:.8g}' for x in sample_stds[:min(20, n_params)]])}\n")
                f.write(f"MeanPctDiff,{','.join([f'{x:.8g}' for x in mean_pct_diff[:min(20, n_params)]])}\n")
                f.write(f"StdPctDiff,{','.join([f'{x:.8g}' for x in std_pct_diff[:min(20, n_params)]])}\n")
                
                # Add the actual samples
                for i in range(min(batch_size, transformed_samples.shape[0])):
                    f.write(f"Sample{i+1},{','.join([f'{x:.8g}' for x in transformed_samples[i, :min(20, n_params)]])}\n")
            
            print(f"\nTransformed samples and statistics saved to {csv_filename}")
            print("=" * 50)


    def update_tape(self, tape, sample_index=1, sample_name=""):
        """
        Updates the tape with the sampled parameters.

        Parameters:
        - tape: The ENDF tape object to update.
        - sample_index: Index of the sample to use.
        """
        self._update_resonance_range(tape, updated_parameters=self.rm_data.reconstruct(sample_index))


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

        # Read rml_data
        rm_data_group = hdf5_group['Parameters']
                
        rm_data = ReichMooreData.read_from_hdf5(rm_data_group)
        
        # Create an instance and set attributes
        instance = cls.__new__(cls)
        instance.NER = NER
        
        # Set L_matrix on the parent CovarianceBase class
        super(cls, instance).__setattr__('L_matrix', L_matrix)
        
        # Set attributes specific to this class
        instance.rm_data = rm_data

        return instance