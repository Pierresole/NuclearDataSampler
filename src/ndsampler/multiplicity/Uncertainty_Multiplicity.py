import numpy as np
import time
from collections import defaultdict
from .MultiplicityCovariance import MultiplicityCovariance
from .Parameters_Multiplicity import Multiplicities
from ENDFtk import tree

class Uncertainty_Multiplicity(MultiplicityCovariance):
    """
    Class to handle the uncertainty in neutron multiplicity distributions.
    """
    def __init__(self, mf1mt, mf31mt, mt_number: int):
        """
        Initialize Uncertainty_Multiplicity object.
        
        Parameters:
        - mf1mt: MF1 section for the MT reaction
        - mf31mt: MF31 section for the MT reaction  
        - mt_number: The MT reaction number (455 for delayed, 456 for prompt)
        """
        # Store MT number FIRST before calling super().__init__
        self.MT = mt_number
        
        super().__init__(mf1mt)
        self.mf31mt = mf31mt
        
        print(f"Creating multiplicity uncertainty for MT{mt_number}...")
        
        # Extract parameters and covariance matrices
        start_time = time.time()
        self.parameters = Multiplicities.from_endftk(mf1mt, mf31mt)
        print(f"Time for extracting multiplicity parameters: {time.time() - start_time:.4f} seconds")
        
        # Build the covariance matrix from standard deviations
        start_time = time.time()
        self._build_covariance_from_std_dev()
        print(f"Time for building covariance matrix: {time.time() - start_time:.4f} seconds")
        
        # Compute Cholesky decomposition
        start_time = time.time()
        self.compute_L_matrix()
        print(f"Time for compute_L_matrix (MT{mt_number}): {time.time() - start_time:.4f} seconds")
        
        print(f"✓ Created multiplicity uncertainty for MT{mt_number}")
        
    def _build_covariance_from_std_dev(self):
        """
        Build the covariance matrix from the parameters.
        Maps the RELATIVE covariance matrix from covariance energy grid to multiplicity energy grid.
        """
        n_mult_bins = len(self.parameters.std_dev)
        
        print(f"  Building RELATIVE covariance matrix for {n_mult_bins} multiplicity energy bins")
        
        # Build the relative standard deviation vector for multiplicity grid
        self.std_dev_vector = np.array(self.parameters.std_dev)  # This is relative std dev
        
        # Use the original RELATIVE covariance matrix from parameters if available
        if self.parameters.covariance_matrix is not None:
            print(f"  Mapping RELATIVE covariance matrix from covariance grid to multiplicity grid")
            
            # The original covariance matrix is RELATIVE and on the covariance energy grid
            orig_rel_cov_matrix = self.parameters.covariance_matrix.copy()
            n_cov_bins = orig_rel_cov_matrix.shape[0]
            
            print(f"    Original relative covariance matrix: {n_cov_bins}x{n_cov_bins}")
            print(f"    Target multiplicity matrix: {n_mult_bins}x{n_mult_bins}")
            
            if n_cov_bins == n_mult_bins:
                # Same dimensions - use directly
                self.covariance_matrix = orig_rel_cov_matrix
            else:
                # Different dimensions - create diagonal matrix from mapped relative std_dev
                # This is a simplification - full covariance mapping would be more complex
                print(f"    Using diagonal approximation due to dimension mismatch")
                diagonal_rel_variance = self.std_dev_vector ** 2  # relative variance
                self.covariance_matrix = np.diag(diagonal_rel_variance)
            
            # Check for numerical issues
            eigenvals = np.linalg.eigvals(self.covariance_matrix)
            print(f"  Relative covariance matrix eigenvalue range: [{eigenvals.min():.6e}, {eigenvals.max():.6e}]")
            
            # If we have negative eigenvalues, regularize the matrix
            min_eigenval = eigenvals.min()
            if min_eigenval < 0:
                print(f"  Warning: Negative eigenvalue {min_eigenval:.6e} detected, regularizing matrix")
                # Add small positive value to diagonal to make matrix positive definite
                regularization = abs(min_eigenval) + 1e-12
                self.covariance_matrix += regularization * np.eye(n_mult_bins)
                
        else:
            print(f"  Using diagonal RELATIVE covariance matrix")
            # Fallback to diagonal relative covariance
            diagonal_rel_variance = self.std_dev_vector ** 2  # relative variance
            self.covariance_matrix = np.diag(diagonal_rel_variance)
        
        # Build correlation matrix for compute_L_matrix()
        with np.errstate(divide='ignore', invalid='ignore'):
            # Create correlation matrix from relative covariance matrix
            std_outer = np.outer(self.std_dev_vector, self.std_dev_vector)
            self.correlation_matrix = np.where(std_outer == 0, 0, self.covariance_matrix / std_outer)
            
        print(f"  Relative standard deviation vector range: [{self.std_dev_vector.min():.6f}, {self.std_dev_vector.max():.6f}]")
        print(f"  Relative covariance matrix shape: {self.covariance_matrix.shape}")
        
    def get_covariance_type(self):
        """
        Override to return the covariance type for multiplicity distributions.
        """
        return "MultiplicityDistribution"
        
    def _apply_samples(self, samples, mode="stack", use_copula=False, batch_size=1, 
                      sampling_method="Simple", debug=False):
        """
        Apply generated samples to the multiplicity parameters.
        
        Parameters:
        -----------
        samples : numpy.ndarray
            The samples generated by sample_parameters (shape: [n_samples, n_parameters])
        mode : str
            Operation mode ("stack" or "replace")
        use_copula : bool
            Whether copula transformation was used
        batch_size : int
            Number of samples in the batch
        sampling_method : str
            The sampling method used ('Simple', 'LHS', 'Sobol', etc.)
        debug : bool
            If True, performs detailed statistical analysis and saves debug output
        """
        print(f"Applying {samples.shape[0]} samples to multiplicity parameters...")
        
        # Get nominal multiplicities and energies
        nominal_multiplicities = self.parameters.multiplicities[0]
        n_params = len(nominal_multiplicities)
        n_samples = samples.shape[0]
        
        # For debug mode, collect all transformed samples
        if debug:
            transformed_samples = np.zeros((n_samples, n_params))
            param_names = [f"Mult_E{i+1}" for i in range(n_params)]
            param_energies = self.parameters.energies if hasattr(self.parameters, 'energies') else list(range(n_params))
        
        # Apply samples to generate perturbed multiplicities
        if mode == "replace":
            # Clear existing samples except nominal (index 0)
            self.parameters.multiplicities = [self.parameters.multiplicities[0]]
            self.parameters.factors = [self.parameters.factors[0]]
        
        for i, sample in enumerate(samples):
            # Convert samples to multiplicative factors
            # sample contains perturbations in standard deviation units for the covariance energy grid
            
            # CRITICAL: std_dev from parameters is now RELATIVE standard deviation
            # The parameters.std_dev is correctly mapped to multiplicity grid during creation
            mult_rel_std_dev = np.array(self.parameters.std_dev)  # This is relative std dev
            
            # Map sample perturbations from covariance grid to multiplicity grid if needed
            if len(self.std_dev_vector) == n_params:
                # Direct correspondence - use samples directly with relative std dev
                factors_for_mults = 1.0 + sample * mult_rel_std_dev
            else:
                # Different energy grids - map the sample perturbations
                # Get covariance and multiplicity energy grids
                reaction = self.mf31mt.reactions.to_list()[0]
                cov_data = reaction.explicit_covariances[0]
                cov_energies = np.array(cov_data.energies[:])
                cov_bin_centers = (cov_energies[:-1] + cov_energies[1:]) / 2
                mult_energies = np.array(self.parameters.energies)
                
                # Interpolate sample perturbations to multiplicity energy grid
                mapped_sample = []
                for mult_energy in mult_energies:
                    if mult_energy <= cov_bin_centers[0]:
                        mapped_sample.append(sample[0])
                    elif mult_energy >= cov_bin_centers[-1]:
                        mapped_sample.append(sample[-1])
                    else:
                        # Linear interpolation
                        for j in range(len(cov_bin_centers) - 1):
                            if cov_bin_centers[j] <= mult_energy <= cov_bin_centers[j + 1]:
                                x0, x1 = cov_bin_centers[j], cov_bin_centers[j + 1]
                                y0, y1 = sample[j], sample[j + 1]
                                if x1 - x0 == 0:
                                    mapped_sample.append(y0)
                                else:
                                    y = y0 + (y1 - y0) * (mult_energy - x0) / (x1 - x0)
                                    mapped_sample.append(y)
                                break
                
                mapped_sample = np.array(mapped_sample)
                # Apply relative uncertainties: factors = 1 + sample * relative_std_dev
                factors_for_mults = 1.0 + mapped_sample * mult_rel_std_dev
            
            # Apply factors to nominal multiplicities
            perturbed_multiplicities = [nom * factor for nom, factor in zip(nominal_multiplicities, factors_for_mults)]
            
            # Ensure multiplicities are non-negative
            perturbed_multiplicities = [max(0.0, mult) for mult in perturbed_multiplicities]
            
            # Store debug information
            if debug:
                transformed_samples[i, :] = perturbed_multiplicities
            
            # Store the results
            if mode == "stack":
                self.parameters.multiplicities.append(perturbed_multiplicities)
                self.parameters.factors.append(factors_for_mults.tolist())
            else:  # replace mode
                if len(self.parameters.multiplicities) <= i + 1:
                    self.parameters.multiplicities.append(perturbed_multiplicities)
                    self.parameters.factors.append(factors_for_mults.tolist())
                else:
                    self.parameters.multiplicities[i + 1] = perturbed_multiplicities
                    self.parameters.factors[i + 1] = factors_for_mults.tolist()
        
        # Detailed debug analysis
        if debug:
            print(f"\n{'='*60}")
            print(f"DEBUG ANALYSIS: Multiplicity Uncertainty MT{self.MT}")
            print(f"{'='*60}")
            print(f"Number of energy bins: {n_params}")
            print(f"Number of samples: {n_samples}")
            print(f"Sampling method: {sampling_method}")
            print(f"Energy range: [{param_energies[0]:.2e}, {param_energies[-1]:.2e}] eV")
            print(f"Covariance matrix size: {self.covariance_matrix.shape}")
            print(f"Standard deviation vector size: {len(self.std_dev_vector)}")
            
            # Calculate sample statistics
            nominal_values = np.array(nominal_multiplicities)
            sample_means = np.mean(transformed_samples, axis=0)
            sample_stds = np.std(transformed_samples, axis=0, ddof=1) if n_samples > 1 else np.zeros(n_params)
            
            # Use properly mapped standard deviations for comparison
            if len(self.std_dev_vector) == n_params:
                # Direct correspondence
                expected_rel_stds = self.std_dev_vector
                print(f"  Using direct std_dev mapping (same grid)")
            else:
                # Use the std_dev from parameters which is now correctly mapped to multiplicity grid
                expected_rel_stds = np.array(self.parameters.std_dev)
                print(f"  Using parameters.std_dev (correctly mapped to multiplicity grid: {len(expected_rel_stds)})")
            
            expected_abs_stds = expected_rel_stds * nominal_values
            relative_uncertainties = 100 * expected_rel_stds
            
            # Calculate relative uncertainties from samples
            sample_rel_stds = sample_stds / nominal_values if n_samples > 1 else np.zeros(n_params)
            
            # Calculate percentage differences - CORRECTED TO COMPARE RELATIVE UNCERTAINTIES
            mean_pct_diff = 100 * (sample_means - nominal_values) / nominal_values
            # Compare relative uncertainties (dimensionless) to relative uncertainties (dimensionless)
            rel_std_pct_diff = 100 * (sample_rel_stds - expected_rel_stds) / expected_rel_stds if n_samples > 1 else np.zeros(n_params)
            
            # Print sample matrix preview
            print(f"\nSample matrix preview (first 5 samples, first 10 energy bins):")
            print(f"{'Sample':<8}", end="")
            for j in range(min(10, n_params)):
                print(f"{'E' + str(j+1):<12}", end="")
            print()
            
            for i in range(min(5, n_samples)):
                print(f"{'S' + str(i+1):<8}", end="")
                for j in range(min(10, n_params)):
                    print(f"{transformed_samples[i, j]:<12.6f}", end="")
                print()
            
            # Parameter verification table
            print(f"\nParameter verification (first 10 energy bins):")
            print(f"{'Energy Bin':<10} {'Energy(eV)':<12} {'Nominal':<12} {'RelUnc(%)':<12} {'Mean':<12} {'StdDev':<12} {'ExpRelStd':<12}")
            print("-" * 88)
            for i in range(min(10, n_params)):
                print(f"{'E' + str(i+1):<10} {param_energies[i]:<12.4e} {nominal_values[i]:<12.6f} "
                      f"{relative_uncertainties[i]:<12.4f} {sample_means[i]:<12.6f} "
                      f"{sample_rel_stds[i]:<12.6f} {expected_rel_stds[i]:<12.6f}")
            
            # Find parameters with largest deviations
            param_info = []
            for i in range(n_params):
                param_info.append({
                    'index': i,
                    'name': param_names[i],
                    'energy': param_energies[i],
                    'nominal': nominal_values[i],
                    'rel_unc': relative_uncertainties[i],
                    'mean': sample_means[i],
                    'sample_rel_std': sample_rel_stds[i],
                    'expected_rel_std': expected_rel_stds[i],
                    'mean_pct_diff': mean_pct_diff[i],
                    'rel_std_pct_diff': rel_std_pct_diff[i] if n_samples > 1 else 0.0
                })
            
            # Sort by different criteria
            sorted_by_mean_diff = sorted(param_info, key=lambda x: abs(x['mean_pct_diff']), reverse=True)
            sorted_by_rel_std_diff = sorted(param_info, key=lambda x: abs(x['rel_std_pct_diff']), reverse=True) if n_samples > 1 else param_info
            sorted_by_rel_unc = sorted(param_info, key=lambda x: x['rel_unc'], reverse=True)
            
            # Top parameters with largest mean deviations
            print(f"\nTop 10 energy bins with largest mean percentage deviation:")
            print(f"{'Bin':<8} {'Energy(eV)':<12} {'Nominal':<12} {'Mean':<12} {'Diff(%)':<12} {'RelUnc(%)':<12}")
            print("-" * 76)
            for i, info in enumerate(sorted_by_mean_diff[:10]):
                print(f"{info['name']:<8} {info['energy']:<12.4e} {info['nominal']:<12.6f} "
                      f"{info['mean']:<12.6f} {info['mean_pct_diff']:<12.4f} {info['rel_unc']:<12.4f}")
            
            if n_samples > 1:
                # Top parameters with largest relative std dev deviations
                print(f"\nTop 10 energy bins with largest relative std dev percentage deviation:")
                print(f"{'Bin':<8} {'Energy(eV)':<12} {'ExpRelStd':<12} {'ActRelStd':<12} {'Diff(%)':<12} {'RelUnc(%)':<12}")
                print("-" * 84)
                for i, info in enumerate(sorted_by_rel_std_diff[:10]):
                    print(f"{info['name']:<8} {info['energy']:<12.4e} {info['expected_rel_std']:<12.6f} "
                          f"{info['sample_rel_std']:<12.6f} {info['rel_std_pct_diff']:<12.4f} {info['rel_unc']:<12.4f}")
            
            # Top parameters with highest relative uncertainties
            print(f"\nTop 10 energy bins with highest relative uncertainties:")
            print(f"{'Bin':<8} {'Energy(eV)':<12} {'Nominal':<12} {'RelUnc(%)':<12} {'AbsUnc':<12}")
            print("-" * 64)
            for i, info in enumerate(sorted_by_rel_unc[:10]):
                abs_unc = info['nominal'] * info['rel_unc'] / 100
                print(f"{info['name']:<8} {info['energy']:<12.4e} {info['nominal']:<12.6f} "
                      f"{info['rel_unc']:<12.4f} {abs_unc:<12.6f}")
            
            # Correlation analysis if we have multiple samples
            if n_samples > 1 and hasattr(self, 'correlation_matrix') and self.correlation_matrix is not None:
                print(f"\nCorrelation analysis:")
                sample_corr_matrix = np.corrcoef(transformed_samples.T)
                
                # Handle size mismatch between sample correlations and expected correlations
                expected_corr = self.correlation_matrix
                if sample_corr_matrix.shape[0] != expected_corr.shape[0]:
                    print(f"  Note: Sample correlation matrix size ({sample_corr_matrix.shape}) != Expected size ({expected_corr.shape})")
                    # Use the smaller dimension for comparison
                    min_size = min(sample_corr_matrix.shape[0], expected_corr.shape[0])
                    sample_corr_subset = sample_corr_matrix[:min_size, :min_size]
                    expected_corr_subset = expected_corr[:min_size, :min_size]
                else:
                    sample_corr_subset = sample_corr_matrix
                    expected_corr_subset = expected_corr
                
                # Compare sample correlations with expected correlations
                corr_diff = np.abs(sample_corr_subset - expected_corr_subset)
                max_corr_diff = np.max(corr_diff)
                mean_corr_diff = np.mean(corr_diff)
                
                print(f"Expected correlation matrix shape: {expected_corr.shape}")
                print(f"Sample correlation matrix shape: {sample_corr_matrix.shape}")
                print(f"Comparison matrix size: {sample_corr_subset.shape}")
                print(f"Maximum correlation difference: {max_corr_diff:.6f}")
                print(f"Mean correlation difference: {mean_corr_diff:.6f}")
                
                # Find pairs with largest correlation differences
                n_corr = sample_corr_subset.shape[0]
                corr_pairs = []
                for i in range(n_corr):
                    for j in range(i+1, n_corr):
                        if abs(expected_corr_subset[i,j]) > 0.1:  # Only consider significant correlations
                            diff = abs(sample_corr_subset[i,j] - expected_corr_subset[i,j])
                            corr_pairs.append({
                                'i': i, 'j': j,
                                'expected': expected_corr_subset[i,j],
                                'sample': sample_corr_subset[i,j],
                                'diff': diff
                            })
                
                corr_pairs.sort(key=lambda x: x['diff'], reverse=True)
                
                if corr_pairs:
                    print(f"\nTop 5 correlation pairs with largest deviations:")
                    print(f"{'Pair':<12} {'Expected':<12} {'Sample':<12} {'Difference':<12}")
                    print("-" * 48)
                    for pair in corr_pairs[:5]:
                        pair_name = f"E{pair['i']+1}-E{pair['j']+1}"
                        print(f"{pair_name:<12} {pair['expected']:<12.6f} {pair['sample']:<12.6f} {pair['diff']:<12.6f}")
            
            # Save detailed results to CSV
            try:
                import pandas as pd
                
                # Create comprehensive dataframe
                results_data = {
                    'Energy_Bin': [f"E{i+1}" for i in range(n_params)],
                    'Energy_eV': param_energies,
                    'Nominal_Multiplicity': nominal_values,
                    'Relative_Uncertainty_Percent': relative_uncertainties,
                    'Expected_Rel_StdDev': expected_rel_stds,
                    'Expected_Abs_StdDev': expected_abs_stds,
                    'Sample_Mean': sample_means,
                    'Sample_Rel_StdDev': sample_rel_stds if n_samples > 1 else [0]*n_params,
                    'Mean_Percent_Diff': mean_pct_diff,
                    'RelStdDev_Percent_Diff': rel_std_pct_diff if n_samples > 1 else [0]*n_params
                }
                
                # Add individual sample columns (first 20 samples for readability)
                for i in range(min(20, n_samples)):
                    results_data[f'Sample_{i+1}'] = transformed_samples[i, :]
                
                df = pd.DataFrame(results_data)
                
                # Save to CSV with MT-specific filename
                csv_filename = f"multiplicity_debug_MT{self.MT}_{sampling_method}_{n_samples}samples.csv"
                df.to_csv(csv_filename, index=False)
                print(f"\nDetailed results saved to: {csv_filename}")
                
            except ImportError:
                print("\nNote: pandas not available, skipping CSV export")
            
            # Summary statistics
            print(f"\n{'='*60}")
            print("SUMMARY STATISTICS")
            print(f"{'='*60}")
            print(f"Mean deviation range: [{mean_pct_diff.min():.4f}%, {mean_pct_diff.max():.4f}%]")
            if n_samples > 1:
                print(f"Rel StdDev deviation range: [{rel_std_pct_diff.min():.4f}%, {rel_std_pct_diff.max():.4f}%]")
            print(f"Relative uncertainty range: [{relative_uncertainties.min():.4f}%, {relative_uncertainties.max():.4f}%]")
            print(f"Multiplicity range: [{nominal_values.min():.6f}, {nominal_values.max():.6f}]")
            
            # Quality metrics
            mean_abs_mean_diff = np.mean(np.abs(mean_pct_diff))
            if n_samples > 1:
                mean_abs_rel_std_diff = np.mean(np.abs(rel_std_pct_diff))
                print(f"\nQuality Metrics:")
                print(f"Mean absolute mean deviation: {mean_abs_mean_diff:.4f}%")
                print(f"Mean absolute rel std dev deviation: {mean_abs_rel_std_diff:.4f}%")
                
                if mean_abs_mean_diff < 5.0:
                    print("✓ Mean sampling quality: GOOD")
                elif mean_abs_mean_diff < 10.0:
                    print("⚠ Mean sampling quality: FAIR")
                else:
                    print("✗ Mean sampling quality: POOR")
                    
                if mean_abs_rel_std_diff < 10.0:
                    print("✓ Rel StdDev sampling quality: GOOD")
                elif mean_abs_rel_std_diff < 20.0:
                    print("⚠ Rel StdDev sampling quality: FAIR")
                else:
                    print("✗ Rel StdDev sampling quality: POOR")
            
            print(f"{'='*60}")
        
        print(f"✓ Applied samples to multiplicity parameters")

    def write_to_hdf5(self, hdf5_group):
        """
        Override base class method to write to MT-specific subgroup.
        """
        # Create MT-specific subgroup first
        mt_group = hdf5_group.require_group(f'MT{self.MT}')
        
        # Write common covariance data to the MT-specific subgroup
        if self.L_matrix is not None:
            mt_group.create_dataset('L_matrix', data=self.L_matrix)
            
        if hasattr(self, 'mean_vector') and self.mean_vector is not None:
            mt_group.create_dataset('mean_vector', data=self.mean_vector)
            
        if hasattr(self, 'std_dev_vector') and self.std_dev_vector is not None:
            mt_group.create_dataset('std_dev_vector', data=self.std_dev_vector)
            
        mt_group.attrs['is_cholesky'] = self.is_cholesky
        
        # Write additional multiplicity-specific data
        self.write_additional_data_to_hdf5(mt_group)

    def write_additional_data_to_hdf5(self, hdf5_group):
        """
        Write additional data specific to multiplicity distributions.
        Note: hdf5_group is already the MT-specific subgroup.
        """
        if self.parameters is not None:
            param_group = hdf5_group.require_group('Parameters')
            self.parameters.write_to_hdf5(param_group)

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Read multiplicity uncertainty from HDF5.
        """
        # This would need to be implemented based on the HDF5 structure
        # For now, just return None
        return None

    def update_tape(self, tape, sample_index=1, sample_name=""):
        """
        Updates the ENDF tape with sampled multiplicity values for the given sample_index.
        Also updates the total neutron multiplicity (MT=452) by summing prompt and delayed.
        """
        from ENDFtk.MF1 import TabulatedMultiplicity
        from ENDFtk.MF1.MT456 import Section as SectionPrompt
        from ENDFtk.MF1.MT455 import Section as SectionDelayed
        from ENDFtk.MF1.MT452 import Section as SectionTotalNubar
        
        # Get sampled multiplicities for this sample
        sampled_multiplicities = self.parameters.reconstruct(sample_index)
        
        # Parse the section to update (use dynamic MT number)
        mf1mt = tape.MAT(tape.material_numbers[0]).MF(1).MT(self.MT).parse()
        
                # Create new section with perturbed multiplicities
        if self.MT == 456:  # Prompt neutrons
            new_section = SectionPrompt(
                zaid=mf1mt.ZA,
                awr=mf1mt.AWR,
                multiplicity=TabulatedMultiplicity(
                    [len(self.parameters.energies)],
                    [mf1mt.LNU],
                    self.parameters.energies,
                    sampled_multiplicities
                )
            )
        elif self.MT == 455:  # Delayed neutrons
            new_section = SectionDelayed(
                zaid=mf1mt.ZA,
                awr=mf1mt.AWR,
                constants=mf1mt.delayed_groups,  # Use the original delayed groups constants
                multiplicity=TabulatedMultiplicity(
                    [len(self.parameters.energies)],
                    [mf1mt.LNU],
                    self.parameters.energies,
                    sampled_multiplicities
                )
            )
        else:
            raise ValueError(f"Unsupported MT number for multiplicity: {self.MT}")
        
        # Now update the total multiplicity section (MT=452) by summing prompt and delayed
        mat_num = tape.material_numbers[0]
        mat = tape.MAT(mat_num)
        mf1 = mat.MF(1)
        
        # Get prompt and delayed sections
        prompt_section = None
        delayed_section = None
        
        if mf1.has_MT(456):
            if self.MT == 456:
                # Use the new section we just created
                prompt_section = new_section
            else:
                # Parse existing prompt section
                prompt_section = mf1.MT(456).parse()
        
        if mf1.has_MT(455):
            if self.MT == 455:
                # Use the new section we just created
                delayed_section = new_section
            else:
                # Parse existing delayed section
                delayed_section = mf1.MT(455).parse()
        
        # Create union energy grid and sum multiplicities
        if prompt_section is not None and delayed_section is not None:
            # Both sections exist - need to create union energy grid
            prompt_energies = prompt_section.nubar.energies.to_list()
            prompt_multiplicities = prompt_section.nubar.multiplicities.to_list()
            delayed_energies = delayed_section.nubar.energies.to_list()
            delayed_multiplicities = delayed_section.nubar.multiplicities.to_list()
            
            # Create union energy grid
            union_energies = self._create_union_energy_grid(prompt_energies, delayed_energies)
            
            # Interpolate both prompt and delayed onto union grid
            prompt_interp = self._interpolate_multiplicities(union_energies, prompt_energies, prompt_multiplicities)
            delayed_interp = self._interpolate_multiplicities(union_energies, delayed_energies, delayed_multiplicities)
            
            # Sum to get total
            summed_multiplicities = [p + d for p, d in zip(prompt_interp, delayed_interp)]
            
        elif prompt_section is not None:
            # Only prompt exists
            union_energies = prompt_section.nubar.energies.to_list()
            summed_multiplicities = prompt_section.nubar.multiplicities.to_list()
            
        elif delayed_section is not None:
            # Only delayed exists
            union_energies = delayed_section.nubar.energies.to_list()
            summed_multiplicities = delayed_section.nubar.multiplicities.to_list()
            
        else:
            raise ValueError("Neither prompt nor delayed neutron sections found")
        
        # Create the total neutron section (MT=452)
        if mf1.has_MT(452):
            mf1mt452 = mf1.MT(452).parse()
            new_total = SectionTotalNubar(
                zaid=mf1mt452.ZA,
                awr=mf1mt452.AWR,
                multiplicity=TabulatedMultiplicity(
                    [len(union_energies)],
                    [mf1mt452.nubar.LNU],
                    union_energies,
                    summed_multiplicities
                )
            )
        else:
            # Create new total section
            new_total = SectionTotalNubar(
                zaid=mf1mt.ZA,
                awr=mf1mt.AWR,
                multiplicity=TabulatedMultiplicity(
                    [len(union_energies)],
                    [1],  # Default LNU value
                    union_energies,
                    summed_multiplicities
                )
            )
        
        # Insert/replace sections in the tape
        mf1.insert_or_replace(new_section)
        mf1.insert_or_replace(new_total)

    def _create_union_energy_grid(self, energies1, energies2, eps=1e-8):
        """
        Create a union energy grid from two energy grids.
        """
        # Combine and sort all unique energies
        all_energies = set(energies1 + energies2)
        union_energies = sorted(all_energies)
        
        # Remove duplicates within tolerance
        filtered_energies = []
        for energy in union_energies:
            if not filtered_energies or abs(energy - filtered_energies[-1]) > eps:
                filtered_energies.append(energy)
        
        return filtered_energies

    def _map_std_dev_to_multiplicity_grid(self):
        """
        Map standard deviations from covariance energy grid to multiplicity energy grid.
        This ensures that perturbations are applied at the correct energy points.
        
        Returns:
        --------
        mapped_std_dev : np.ndarray
            Standard deviations mapped to multiplicity energy grid
        """
        # Get multiplicity energy points
        mult_energies = np.array(self.parameters.energies)
        
        # Get covariance energy grid from MF31 data
        reaction = self.mf31mt.reactions.to_list()[0]
        cov_data = reaction.explicit_covariances[0]
        cov_energies = np.array(cov_data.energies[:])  # Energy bin edges
        
        # Create bin centers for covariance energy grid
        cov_bin_centers = (cov_energies[:-1] + cov_energies[1:]) / 2
        
        # Ensure we have the right number of std_dev values for covariance bins
        if len(self.std_dev_vector) != len(cov_bin_centers):
            print(f"Warning: std_dev_vector length ({len(self.std_dev_vector)}) != covariance bins ({len(cov_bin_centers)})")
            # Use available data, truncate if necessary
            n_bins = min(len(self.std_dev_vector), len(cov_bin_centers))
            cov_bin_centers = cov_bin_centers[:n_bins]
            std_dev_for_mapping = self.std_dev_vector[:n_bins]
        else:
            std_dev_for_mapping = self.std_dev_vector
        
        # Interpolate standard deviations to multiplicity energy grid
        mapped_std_dev = []
        for mult_energy in mult_energies:
            # Find the appropriate std_dev value for this multiplicity energy
            if mult_energy <= cov_bin_centers[0]:
                # Use first covariance bin
                mapped_std_dev.append(std_dev_for_mapping[0])
            elif mult_energy >= cov_bin_centers[-1]:
                # Use last covariance bin
                mapped_std_dev.append(std_dev_for_mapping[-1])
            else:
                # Linear interpolation between covariance bins
                for i in range(len(cov_bin_centers) - 1):
                    if cov_bin_centers[i] <= mult_energy <= cov_bin_centers[i + 1]:
                        # Linear interpolation
                        x0, x1 = cov_bin_centers[i], cov_bin_centers[i + 1]
                        y0, y1 = std_dev_for_mapping[i], std_dev_for_mapping[i + 1]
                        
                        if x1 - x0 == 0:
                            mapped_std_dev.append(y0)
                        else:
                            y = y0 + (y1 - y0) * (mult_energy - x0) / (x1 - x0)
                            mapped_std_dev.append(y)
                        break
        
        return np.array(mapped_std_dev)
    
    def _interpolate_multiplicities(self, target_energies, source_energies, source_multiplicities):
        """
        Interpolate multiplicities onto a new energy grid using linear interpolation.
        """
        interpolated = []
        
        for target_energy in target_energies:
            if target_energy <= source_energies[0]:
                # Extrapolate using first value
                interpolated.append(source_multiplicities[0])
            elif target_energy >= source_energies[-1]:
                # Extrapolate using last value
                interpolated.append(source_multiplicities[-1])
            else:
                # Linear interpolation
                for i in range(len(source_energies) - 1):
                    if source_energies[i] <= target_energy <= source_energies[i + 1]:
                        # Linear interpolation formula
                        x0, x1 = source_energies[i], source_energies[i + 1]
                        y0, y1 = source_multiplicities[i], source_multiplicities[i + 1]
                        
                        if x1 - x0 == 0:
                            interpolated.append(y0)
                        else:
                            y = y0 + (y1 - y0) * (target_energy - x0) / (x1 - x0)
                            interpolated.append(y)
                        break
        
        return interpolated