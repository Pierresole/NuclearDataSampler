import numpy as np
from collections import defaultdict
from .Parameters_RML_RRR import RMatrixLimited
from ..ResonanceRangeCovariance import ResonanceRangeCovariance
from ENDFtk import tree
from ENDFtk.MF2.MT151 import ResonanceRange
import time

class Uncertainty_RML_RRR(ResonanceRangeCovariance):
    """
    Class for RMatrixLimited resonance range uncertainty data.
    Attributes:
    - rml_data: RMatrixLimited object.
    - covariance_matrix: The covariance matrix.
    - L_matrix: The Cholesky decomposition of the covariance matrix.
    """
    def __init__(self, mf2_resonance_ranges, mf32_resonance_range, NER, want_reduced: bool = False):
        # Initialize rml_data
        self.NER = NER
        self.covariance_matrix = None
        start_time = time.time()
        self.rml_data = RMatrixLimited.from_endftk(mf2_resonance_ranges, mf32_resonance_range, want_reduced)
        print(f"Time for RMatrixLimited.from_endftk: {time.time() - start_time:.4f} seconds")
        
        self.index_mapping = []

        start_time = time.time()
        self.extract_correlation_matrix(mf32_resonance_range, want_reduced)
        print(f"Time for extract_covariance_matrix: {time.time() - start_time:.4f} seconds")
        
        # Filter out parameters with zero standard deviation before computing L_matrix
        # self.filter_zero_std_dev_parameters()
        
        start_time = time.time()
        self.compute_L_matrix()
        print(f"Time for compute_L_matrix: {time.time() - start_time:.4f} seconds")
       
    @classmethod
    def from_covariance_data(cls, tape, NER, covariance_data):
        """
        Initializes the object using covariance data.

        Parameters:
        - tape: The ENDF tape object.
        - NER: The resonance range index.
        - covariance_data: The covariance data from the HDF5 file.

        Returns:
        - An instance of RML.
        """
        instance = cls(tape, NER)
        instance.set_covariance_data(covariance_data)
        return instance

    def get_covariance_type(self):
        """
        Override to return the specific covariance type.
        """
        return "Uncertainty_RML_RRR"
    
    
    def extract_correlation_matrix(self, mf32_range, to_reduced: bool = True):
        """
        Extracts the correlation matrix using the method from the base class.
        Also sets the initial full index_mapping before filtering.
        """
        if mf32_range.parameters.LCOMP == 2:
            self.extract_correlation_matrix_LCOMP2(mf32_range, to_reduced)
        else:
            raise ValueError(f"Unsupported LCOMP value: {mf32_range.parameters.LCOMP}")
        

    def extract_correlation_matrix_LCOMP2(self, mf32_range, to_reduced: bool = True):
        """
        Reconstructs the correlation matrix from standard deviations and correlation coefficients when LCOMP == 2.
        
        Parameters:
        -----------
        mf32_range : The MF32 resonance range
        to_reduced : bool, optional
            If True, converts widths to reduced form.
        """
        cm = mf32_range.parameters.correlation_matrix
        
        NParams = sum( ((sgroup.NCH + 1 ) * sgroup.NRSA) for sgroup in mf32_range.parameters.uncertainties.spin_groups.to_list())
        if cm.NNN != NParams:
            raise ValueError(f"Mismatch between number of parameters ({NParams}) and size of correlation matrix (NNN={cm.NNN}). Not sure if it will work ... ?")
        
        correlation_matrix = np.identity(cm.NNN)   
             
        I_arr = np.array(cm.I) - 1  # zero-based
        J_arr = np.array(cm.J) - 1
        corr_arr = np.array(cm.correlations)

        # Fill the upper triangle
        correlation_matrix[I_arr, J_arr] = corr_arr
        # Fill the lower triangle (since the matrix is symmetric)
        correlation_matrix[J_arr, I_arr] = corr_arr
        
        # Set the initial, unfiltered index mapping right after extraction
        index_mapping, std_devs = self.rml_data.get_standard_deviations()
        
        non_zero_indices = np.where(np.array(std_devs) > 1e-16)[0]
        correlation_matrix = correlation_matrix[np.ix_(non_zero_indices, non_zero_indices)]
        
        self.index_mapping = [index_mapping[i] for i in non_zero_indices]
               
        # Convert to reduced covariance matrix if requested (multiply by Jacobian)
        if to_reduced:
            # covariance_matrix = self.rml_data.extract_covariance_matrix_LCOMP2(covariance_matrix)
            correlation_matrix = self.rml_data.extract_covariance_matrix_LCOMP2(correlation_matrix)
        
        # Set the covariance matrix and correlation matrix attributes
        # self.correlation_matrix = correlation_matrix
        # Set the covariance matrix as an attribute of CovarianceBase
        # This ensures we're using the one from the parent class
        super().__setattr__('correlation_matrix', correlation_matrix)


    def update_tape(self, tape, sample_index=1, sample_name=""):
        """
        Updates the tape with the sampled parameters.

        Parameters:
        - tape: The ENDF tape object to update.
        - sample_index: Index of the sample to use.
        """
        self._update_resonance_range(tape, updated_parameters = self.rml_data.reconstruct(sample_index))
        # if sample_index == 1:
        #     tape.to_file(f'sampled_tape_{sample_name}.endf')
        # else:
        #     tape.to_file(f'sampled_tape_{sample_index+1}.endf')


    def write_to_hdf5(self, hdf5_group):
        """
        Writes the L_matrix and rml_data to the given HDF5 group.
        """
        hdf5_group.attrs['NER'] = self.NER
        
        # Write L_matrix
        hdf5_group.create_dataset('L_matrix', data=self.L_matrix)

        # Write index_mapping as a compound dataset
        dt = np.dtype([('spin_group', 'i4'), ('resonance', 'i4'), ('parameter', 'i4')])
        index_data = np.array(self.index_mapping, dtype=dt)
        hdf5_group.create_dataset('index_mapping', data=index_data)

        # Write rml_data
        rml_data_group = hdf5_group.create_group('Parameters')
        self.rml_data.write_to_hdf5(rml_data_group)
        
        
    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads the L_matrix and rml_data from the given HDF5 group and returns an instance.
        """
        NER = hdf5_group.attrs['NER']
        
        # Read L_matrix
        L_matrix = hdf5_group['L_matrix'][()]

        # Read index_mapping
        index_data = hdf5_group['index_mapping'][()]
        index_mapping = [(int(idx['spin_group']), int(idx['resonance']), int(idx['parameter'])) 
                        for idx in index_data]

        # Read rml_data
        rml_data_group = hdf5_group['Parameters']
                
        rml_data = RMatrixLimited.read_from_hdf5(rml_data_group)
        
        # Create an instance and set attributes
        instance = cls.__new__(cls)
        instance.NER = NER
        
        # Set L_matrix on the parent CovarianceBase class
        super(cls, instance).__setattr__('L_matrix', L_matrix)
        # Set is_cholesky to False as default, since we don't know if it was a Cholesky decomposition
        # super(cls, instance).__setattr__('is_cholesky', hdf5_group.attrs.get('is_cholesky', False))
        
        # Set attributes specific to this class
        instance.rml_data = rml_data
        instance.index_mapping = index_mapping

        return instance


    def _apply_samples(self, samples, mode="stack", use_copula=False, batch_size=1, sampling_method="Simple", debug=False):
        """
        Apply generated samples to the resonance parameters.
        
        Args:
            samples: The samples generated by sample_parameters. Corresponds ONLY to parameters with non-zero std dev.
                     If use_copula=True, these are correlated uniform values. Otherwise, they are perturbations.
            mode: Either 'stack' (append) or 'replace' (overwrite first sample)
            use_copula: If True, samples are correlated uniforms and will be transformed to normal deviates.
            batch_size: Number of samples in the batch (1 for Simple method, >1 for LHS/Sobol)
            sampling_method: The sampling method used ('Simple', 'LHS', or 'Sobol')
            debug: If True, print and save the transformed parameter samples
        """
        if mode not in ['stack', 'replace']:
            raise ValueError("Mode must be either 'stack' or 'replace'")

        import numpy as np
        from scipy.stats import norm, truncnorm

        # Handle single sample vs batch sample format
        if batch_size == 1:
            # Ensure samples is treated as a 2D array with one row
            sample_list = samples.reshape(1, -1) if isinstance(samples, np.ndarray) else np.array([samples])
            operation_mode = mode
        else:
            sample_list = samples
            operation_mode = 'stack' # Force stack mode for batch processing beyond the first sample

        # For debug mode, collect transformed samples
        n_filtered_params = len(self.index_mapping) # Number of parameters with non-zero std dev
        if debug:
            transformed_samples = np.zeros((batch_size, n_filtered_params))
            param_names = []
            param_indices = []
            # Collect nominal values and uncertainties ONLY for filtered parameters
            nominal_values_filtered = np.zeros(n_filtered_params)
            uncertainties_filtered = np.zeros(n_filtered_params)
            param_idx_to_info = {} # For debug output

        # Pre-calculate nominal values and standard deviations for filtered parameters
        _, nominal_parameters_filtered = self.rml_data.get_nominal_parameters_with_uncertainty()
        _, std_devs_filtered = self.rml_data.get_standard_deviations(non_null_only=True)

        for sample_batch_idx, current_samples_batch in enumerate(sample_list):
            # Process each sampled value
            for filtered_idx, (spin_group_idx, resonance_idx, param_idx) in enumerate(self.index_mapping):
                spin_group = self.rml_data.ListSpinGroup[spin_group_idx]
                resonance = spin_group.ResonanceParameters[resonance_idx]
                
                # Get nominal value and uncertainty for this parameter
                nominal_value = nominal_parameters_filtered[filtered_idx]
                uncertainty = std_devs_filtered[filtered_idx]
                
                # Get the sampled uniform value
                u_value = current_samples_batch[filtered_idx]
                u_value = np.clip(u_value, 0.001, 0.999)  # Ensure safe range
                
                # Determine if this is a fission channel parameter (can be negative)
                is_fission_channel = False
                if param_idx > 0 and resonance.FissionChannels is not None:
                    channel_idx = param_idx - 1  # param_idx 0 is ER, 1+ are GAM channels
                    is_fission_channel = channel_idx in resonance.FissionChannels
                
                # Transform uniform to normal deviate based on parameter type
                if use_copula:
                    if self.rml_data.IFG == 0 and not is_fission_channel and param_idx > 0:
                        # Physical widths (non-fission) - use truncated normal to ensure positivity
                        a = - nominal_value / uncertainty if uncertainty > 0 else -np.inf
                        
                        if a > -10:  # Only adjust if truncation has significant effect
                            # Calculate adjusted mean and sigma for truncnorm
                            loc = self.calculate_adjusted_mean(nominal_value, uncertainty)
                            # scale = self.calculate_adjusted_sigma(1.0, a, 10.0, loc)
                            
                            # Use truncated normal with adjusted parameters
                            z_value = truncnorm.ppf(u_value, a - loc, np.inf, loc=loc, scale=1.0)
                            # z_value = truncnorm.ppf(u_value, a - loc, np.inf, loc=loc, scale=scale)
                        else:
                            # If lower bound is far away, use regular normal
                            z_value = norm.ppf(u_value)
                    else:
                        # For energies, reduced widths, or fission channels - use standard normal
                        z_value = norm.ppf(u_value)
                    
                    # Calculate sampled value
                    sampled_value = nominal_value + z_value * uncertainty
                else:
                    # If not using copula, samples are direct perturbations
                    sampled_value = nominal_value + current_samples_batch[filtered_idx]
                
                # For physical widths that aren't fission channels, ensure positive values
                if self.rml_data.IFG == 0 and not is_fission_channel and param_idx > 0:
                    sampled_value = max(sampled_value, 1e-10)
                
                # Store for debug output if needed
                if debug:
                    transformed_samples[sample_batch_idx, filtered_idx] = sampled_value
                    if sample_batch_idx == 0:
                        if param_idx == 0:
                            param_name = f"ER_SG{spin_group_idx}_R{resonance_idx}"
                        else:
                            ch_type = "FIS" if is_fission_channel else "GAM"
                            param_name = f"{ch_type}{param_idx-1}_SG{spin_group_idx}_R{resonance_idx}"
                        param_names.append(param_name)
                        param_indices.append(filtered_idx)
                        param_idx_to_info[filtered_idx] = (param_name, nominal_value, uncertainty)
                
                # Update the actual parameter value
                if param_idx == 0:  # ER parameter
                    param_list = resonance.ER
                else:  # GAM parameter
                    channel_idx = param_idx - 1
                    if channel_idx < len(resonance.GAM):
                        param_list = resonance.GAM[channel_idx]
                    else:
                        print(f"Warning: channel_idx {channel_idx} out of range for GAM in SG {spin_group_idx}, Res {resonance_idx}")
                        continue
                
                # if (spin_group_idx, resonance_idx, param_idx) == (2, 25, 2):
                #     print(nominal_value, uncertainty, sampled_value)
                    
                # Apply the sampled value according to the operation mode
                sample_idx = sample_batch_idx + 1  # +1 because index 0 is nominal
                if operation_mode == 'stack':
                    if sample_idx < len(param_list):
                        param_list[sample_idx] = sampled_value
                    else:
                        param_list.append(sampled_value)
                elif operation_mode == 'replace':
                    if len(param_list) > 1:
                        param_list[1] = sampled_value
                    else:
                        param_list.append(sampled_value)

        if debug:
            print(f"\n=== Debug Output for {self.__class__.__name__} (Transformed Samples - Filtered Params Only) ===")
            print(f"Number of parameters with uncertainty: {n_filtered_params}")
            print(f"Number of samples: {batch_size}")
            print(f"Sampling method: {sampling_method}")

            # Gather nominal and uncertainty values clearly
            nominal_values = np.array([param_idx_to_info[i][1] for i in range(n_filtered_params)])
            uncertainties = np.array([param_idx_to_info[i][2] for i in range(n_filtered_params)])

            # Compute sample means and std devs
            sample_means = np.mean(transformed_samples, axis=0)
            sample_stds = np.std(transformed_samples, axis=0)

            # Correct calculation for percentage differences
            mean_pct_diff = np.zeros(n_filtered_params)
            std_pct_diff = np.zeros(n_filtered_params)

            for i in range(n_filtered_params):
                nominal = nominal_values[i]
                uncert = uncertainties[i]
                if abs(nominal) > 1e-8:
                    mean_pct_diff[i] = 100.0 * (sample_means[i] - nominal) / nominal
                else:
                    mean_pct_diff[i] = 0.0
                if uncert > 1e-8:
                    std_pct_diff[i] = 100.0 * (sample_stds[i] - uncert) / uncert
                else:
                    std_pct_diff[i] = 0.0

            debug_info = [{
                'Parameter': param_idx_to_info[i][0],
                'Nominal': nominal_values[i],
                'Mean': sample_means[i],
                'Uncertainty': uncertainties[i],
                'Std Dev': sample_stds[i],
                'Mean Diff%': mean_pct_diff[i],
                'Std Diff%': std_pct_diff[i]
            } for i in range(n_filtered_params)]

            # Sorting clearly by Std Diff%
            sorted_mean_diff = sorted(debug_info, key=lambda x: abs(x['Mean Diff%']), reverse=True)

            print("\nTop 5 parameters with largest mean percentage difference:")
            print(f"{'Parameter':<20} {'Nominal':>12} {'Mean':>12} {'Diff%':>10} | {'Uncertainty':>12} {'Std Dev':>12} {'Diff%':>10}")
            print('-'*100)

            # Print top 10 parameters with largest std deviation discrepancy
            for info in sorted_mean_diff[:10]:
                print(f"{info['Parameter']:<20} {info['Nominal']:12.6g} {info['Mean']:12.6g} {info['Mean Diff%']:9.2f}% | "
                    f"{info['Uncertainty']:12.6g} {info['Std Dev']:12.6g} {info['Std Diff%']:9.2f}%")

            print('\nSample Correlation Matrix (first 5 parameters):')
            if batch_size > 1:
                corr_matrix = np.corrcoef(transformed_samples[:, :10].T)
                for row in corr_matrix:
                    print(' '.join(f"{x:7.3f}" for x in row))
                    
            # Compare with original correlation matrix
            orig_corr = self.L_matrix @ self.L_matrix.T
            
            print("\nOriginal correlation matrix (first 5 parameters):")
            orig_corr_display = orig_corr[:min(10, n_filtered_params), :min(10, n_filtered_params)]
            for row in orig_corr_display:
                print(" ".join([f"{x:.3f}" for x in row]))

            print(f"\n{'='*35}\n")