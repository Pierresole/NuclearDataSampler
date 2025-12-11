from collections import defaultdict
from .EnergyDistributionCovariance import EnergyDistributionCovariance
from .Parameters_Energydist import EnergyDistributionData
from ENDFtk import tree
import numpy as np
import time

class Uncertainty_Energydist(EnergyDistributionCovariance):
    """
    Class for handling uncertainties in energy distributions (e.g., PFNS from MF5/MF35).
    
    Similar structure to Uncertainty_Angular but for secondary energy distributions.
    """

    def __init__(self, mf5mt, mf35mt, mt_number, incident_energy_indices=None):
        """
        Initialize Uncertainty_Energydist object.
        
        Parameters:
        - mf5mt: MF5 section for the MT reaction
        - mf35mt: MF35 section for the MT reaction
        - mt_number: The MT reaction number (e.g., 18 for fission)
        - incident_energy_indices: List of incident energy indices to process
                                   If None, all incident energies with covariance will be processed
        """
        # Store MT number FIRST
        self.MT = mt_number
        self.requested_incident_energy_indices = incident_energy_indices
        
        super().__init__(mf5mt)

        print(f"Creating energy distribution uncertainty for MT{mt_number}...")
        
        # Initialize covariance type tracking (maps incident_energy_index -> LB flag)
        self.covariance_type_map = {}  # Will store LB flags per incident energy
        
        # Extract parameters and covariance matrices
        start_time = time.time()
        self.energy_data = EnergyDistributionData.from_endftk(mf5mt, mf35mt)
        print(f"Time for extracting distributions: {time.time() - start_time:.4f} seconds")
        
        start_time = time.time()
        self.extract_relcov_matrix(mf35mt)
        print(f"Time for extracting covariance matrix: {time.time() - start_time:.4f} seconds")
        
        # Store covariance energy mesh
        if not hasattr(self, 'covariance_energy_mesh'):
            self.covariance_energy_mesh = self._extract_covariance_mesh(mf35mt)
        
        # Remove null-variance parameters and build index mapping
        start_time = time.time()
        rel_cov_full = self.relative_covariance_matrix
        
        print(f"Full relative covariance matrix shape: {rel_cov_full.shape}")
        
        # Remove null-variance parameters
        reduced_cov, index_map = self._remove_null_variance_and_track(rel_cov_full)
        
        # Store results
        self.covariance_index_map = index_map
        self.active_parameter_indices = list(range(len(index_map)))
        
        print(f"✓ Stored covariance_index_map with {len(self.covariance_index_map)} entries")
        
        # Compute Cholesky decomposition on REDUCED matrix
        eigenvalues = np.linalg.eigvalsh(reduced_cov)
        if np.any(eigenvalues < -1e-10):
            print(f"  ⚠️  Negative eigenvalues detected (min: {eigenvalues.min():.6e})")
            print(f"  Applying trace-preserving positive definite correction...")
            reduced_cov = self._make_positive_definite_preserve_trace(reduced_cov, epsilon=1e-10)
            eigenvalues = np.linalg.eigvalsh(reduced_cov)
            print(f"  ✓ Corrected (min eigenvalue: {eigenvalues.min():.6e})")
        
        # Cholesky decomposition
        try:
            self.L_matrix = np.linalg.cholesky(reduced_cov)
            self.is_cholesky = True
            print(f"  ✓ Cholesky decomposition successful")
        except np.linalg.LinAlgError:
            print(f"  ⚠️  Cholesky failed, using eigenvalue decomposition fallback")
            eigenvalues, eigenvectors = np.linalg.eigh(reduced_cov)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            self.L_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            self.is_cholesky = False
        
        # Verify reconstruction
        reconstructed_cov = self.L_matrix @ self.L_matrix.T
        print(f"  Reduced covariance[:4,:4]:")
        print(f"  {reduced_cov[:4,:4]}")
        print(f"  Reconstructed from L[:4,:4]:")
        print(f"  {reconstructed_cov[:4,:4]}")
        
        # Set relative_covariance_matrix for compatibility
        self.relative_covariance_matrix_full = reduced_cov
        super().__setattr__('relative_covariance_matrix', self.relative_covariance_matrix_full)
        
        print(f"Time for null-variance removal & Cholesky: {time.time() - start_time:.4f} seconds")
        print(f"✓ Created energy distribution uncertainty for MT{mt_number}")
    
    def get_covariance_type(self):
        """
        Override to return the specific covariance type.
        """
        return "EnergyDistribution"
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _extract_covariance_mesh(self, mf35mt):
        """Extract the union of all outgoing energy meshes from MF35 covariance blocks."""
        all_energies = set()
        
        for block_idx in range(mf35mt.number_energy_blocks):
            block = mf35mt.energy_blocks[block_idx]
            energies = block.energies[:]
            all_energies.update(energies)
        
        return sorted(list(all_energies))
    
    def _remove_null_variance_and_track(self, full_cov_matrix, tol=1e-12):
        """
        Remove rows/columns with null variance from the covariance matrix while maintaining
        a mapping structure to track which (incident_idx, outgoing_bin) each index corresponds to.
        """
        # Find non-zero variance indices
        diag = np.diag(full_cov_matrix)
        non_zero_mask = np.abs(diag) >= tol
        non_zero_indices = np.where(non_zero_mask)[0]
        
        # Build index mapping
        # For energy distributions, we need to map to (incident_energy_idx, outgoing_bin_idx)
        index_map = []
        
        # Reconstruct structure from distributions
        current_idx = 0
        for dist in self.energy_data.distributions:
            n_bins = len(dist.outgoing_energies) - 1
            for bin_idx in range(n_bins):
                if current_idx in non_zero_indices:
                    index_map.append((
                        dist.incident_energy_index,
                        bin_idx,
                        dist.outgoing_energies[bin_idx],
                        dist.outgoing_energies[bin_idx + 1]
                    ))
                current_idx += 1
        
        # Extract reduced covariance matrix
        reduced_cov = full_cov_matrix[np.ix_(non_zero_indices, non_zero_indices)]
        
        print(f"Original matrix size: {full_cov_matrix.shape[0]} x {full_cov_matrix.shape[1]}")
        print(f"Reduced matrix size: {reduced_cov.shape[0]} x {reduced_cov.shape[1]}")
        print(f"Removed {full_cov_matrix.shape[0] - reduced_cov.shape[0]} null variance elements")
        
        return reduced_cov, index_map
    
    def _reconstruct_full_perturbation(self, perturbation_vector, index_map):
        """Reconstruct full perturbation with zeros for pruned parameters.
        
        Args:
            perturbation_vector: Perturbations for active parameters only
            index_map: Mapping from reduced indices to (incident_idx, bin_idx, E_low, E_high)
            
        Returns:
            Dictionary mapping incident_energy_index to perturbation array
        """
        full_perturbation = {}
        
        for i, (incident_idx, bin_idx, _, _) in enumerate(index_map):
            if incident_idx not in full_perturbation:
                # Find how many bins this incident energy has
                n_bins = len([d for d in self.energy_data.distributions if d.incident_energy_index == incident_idx][0].outgoing_energies) - 1
                full_perturbation[incident_idx] = np.zeros(n_bins)
            
            full_perturbation[incident_idx][bin_idx] = perturbation_vector[i]
        
        return full_perturbation
    
    def _make_positive_definite_preserve_trace(self, cov_matrix, epsilon=1e-8):
        """Make a covariance matrix positive definite while preserving its trace."""
        # Ensure symmetry
        cov_sym = (cov_matrix + cov_matrix.T) / 2
        
        # Original trace
        trace_original = np.trace(cov_sym)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_sym)
        
        # Threshold eigenvalues
        eigenvalues_corrected = np.maximum(eigenvalues, epsilon)
        
        # Rescale to preserve trace
        trace_corrected = np.sum(eigenvalues_corrected)
        scale_factor = trace_original / trace_corrected
        eigenvalues_rescaled = eigenvalues_corrected * scale_factor
        
        # Reconstruct matrix
        cov_corrected = eigenvectors @ np.diag(eigenvalues_rescaled) @ eigenvectors.T
        
        return cov_corrected
    
    def extract_relcov_matrix(self, mf35mt):
        """Extract the covariance matrix from MF35.
        
        NEW BEHAVIOR: Does NOT convert absolute to relative.
        - LB=5: Stores relative covariance (as-is)
        - LB=7: Stores absolute covariance (as-is)
        
        Tracks covariance type per incident energy for proper perturbation application.
        """
        # Count total number of parameters
        total_params = 0
        for dist in self.energy_data.distributions:
            n_bins = len(dist.outgoing_energies) - 1
            total_params += n_bins
        
        # Initialize full covariance matrix
        full_cov = np.zeros((total_params, total_params))
        
        # Fill in covariance blocks
        current_row = 0
        for block_idx in range(mf35mt.number_energy_blocks):
            block = mf35mt.energy_blocks[block_idx]
            
            # Get covariance matrix for this block
            NE = block.NE - 1  # Number of outgoing energy bins
            values = block.values[:]
            LB = block.LB  # 5 = relative, 7 = absolute
            
            # Build symmetric matrix
            cov_matrix = np.zeros((NE, NE))
            triu_indices = np.triu_indices(NE)
            cov_matrix[triu_indices] = values
            cov_matrix[(triu_indices[1], triu_indices[0])] = values
            
            # Find which incident energies this block covers
            E1 = block.E1
            E2 = block.E2
            
            covered_dists = [d for d in self.energy_data.distributions 
                           if E1 <= d.incident_energy <= E2]
            
            # Store covariance type for each covered incident energy
            for dist in covered_dists:
                self.covariance_type_map[dist.incident_energy_index] = LB
            
            # NO CONVERSION - store covariance as-is
            if LB == 5:
                print(f"  Block {block_idx}: LB=5 (relative covariance) - stored as-is")
            elif LB == 7:
                print(f"  Block {block_idx}: LB=7 (absolute covariance) - stored as-is")
                print(f"    → Will use ADDITIVE perturbations: P_sample = P_nominal + δ")
            else:
                print(f"  ⚠️  Block {block_idx}: Unknown LB flag = {LB}")
            
            # Place this covariance block in the full matrix
            if len(covered_dists) > 0:
                full_cov[current_row:current_row+NE, current_row:current_row+NE] = cov_matrix
                current_row += NE
        
        super().__setattr__('relative_covariance_matrix', full_cov)
        self.covariance_energy_mesh = self._extract_covariance_mesh(mf35mt)
        
        print(f"\n✓ Covariance type map created:")
        for inc_idx, lb_flag in sorted(self.covariance_type_map.items()):
            cov_type = "relative" if lb_flag == 5 else "absolute" if lb_flag == 7 else f"unknown({lb_flag})"
            print(f"    Incident energy index {inc_idx}: LB={lb_flag} ({cov_type})")
    
    # =========================================================================
    # SAMPLING METHODS
    # =========================================================================
    
    def _apply_samples(self, samples, mode="stack", use_copula=False, batch_size=1, 
                      sampling_method="Simple", debug=False):
        """Apply samples to energy distributions."""
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        n_samples, n_reduced_params = samples.shape
        
        # Validate dimensions
        if not hasattr(self, 'covariance_index_map'):
            raise ValueError("covariance_index_map not initialized")
        
        expected_reduced_params = len(self.covariance_index_map)
        if n_reduced_params != expected_reduced_params:
            raise ValueError(f"Sample dimension mismatch: got {n_reduced_params}, expected {expected_reduced_params}")
        
        # Clear if replace mode
        if mode == 'replace':
            for dist in self.energy_data.distributions:
                dist.rel_deviation = []
        
        # Process each sample
        for sample_idx in range(n_samples):
            perturbation_vector = samples[sample_idx, :]
            
            # Reconstruct full perturbation
            full_perturbation = self._reconstruct_full_perturbation(
                perturbation_vector,
                self.covariance_index_map
            )
            
            # Store deviations for each distribution
            # Type depends on covariance type (LB flag):
            # - LB=5: relative deviations (multiplicative)
            # - LB=7: absolute deviations (additive)
            for dist in self.energy_data.distributions:
                incident_idx = dist.incident_energy_index
                
                if incident_idx in full_perturbation:
                    delta = full_perturbation[incident_idx].tolist()
                else:
                    # No perturbation for this incident energy
                    n_bins = len(dist.outgoing_energies) - 1
                    delta = [0.0] * n_bins
                
                # Ensure rel_deviation list is long enough
                while len(dist.rel_deviation) <= sample_idx + 1:
                    dist.rel_deviation.append(None)
                
                dist.rel_deviation[sample_idx + 1] = delta
                
                # Store covariance type in distribution for proper reconstruction
                if not hasattr(dist, 'covariance_type'):
                    dist.covariance_type = self.covariance_type_map.get(incident_idx, 5)  # Default to relative
        
        print(f"Applied {n_samples} samples to energy distributions")
    
    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """Read Uncertainty_Energydist from HDF5 group."""
        # Read L_matrix
        L_matrix = hdf5_group['L_matrix'][()]
        
        # Read mean_vector if it exists
        mean_vector = hdf5_group['mean_vector'][()] if 'mean_vector' in hdf5_group else None
        
        # Read std_dev_vector if it exists
        std_dev_vector = hdf5_group['std_dev_vector'][()] if 'std_dev_vector' in hdf5_group else None
        
        # Read is_cholesky flag
        is_cholesky = hdf5_group.attrs.get('is_cholesky', False)
        
        # Read MT number
        mt_number = hdf5_group.attrs.get('MT', 18)
        
        # Read energy_data
        energy_data_group = hdf5_group['Parameters']
        energy_data = EnergyDistributionData.read_from_hdf5(energy_data_group)
        
        # Create instance
        instance = cls.__new__(cls)
        
        # Initialize parent class attributes
        super(cls, instance).__init__(None)
        
        # Set attributes from HDF5
        instance.L_matrix = L_matrix
        instance.mean_vector = mean_vector
        instance.std_dev_vector = std_dev_vector
        instance.is_cholesky = is_cholesky
        instance.energy_data = energy_data
        instance.MT = mt_number
        
        # Restore covariance index map
        if 'covariance_index_map' in hdf5_group:
            index_map_array = hdf5_group['covariance_index_map'][()]
            instance.covariance_index_map = [
                (int(row['incident_idx']), int(row['bin_index']), float(row['E_low']), float(row['E_high']))
                for row in index_map_array
            ]
        
        # Restore active parameter indices
        if 'active_parameter_indices' in hdf5_group:
            instance.active_parameter_indices = hdf5_group['active_parameter_indices'][()].tolist()
        else:
            instance.active_parameter_indices = list(range(L_matrix.shape[0]))
        
        return instance
    
    def write_additional_data_to_hdf5(self, hdf5_group):
        """Write energy distribution specific data to HDF5."""
        if self.energy_data is not None:
            energy_group = hdf5_group.require_group('Parameters')
            self.energy_data.write_to_hdf5(energy_group)
        
        # Save the MT number as an attribute
        hdf5_group.attrs['MT'] = self.MT
        
        # Save covariance index map
        if hasattr(self, 'covariance_index_map'):
            index_map_array = np.array(self.covariance_index_map, dtype=[
                ('incident_idx', 'i4'),
                ('bin_index', 'i4'),
                ('E_low', 'f8'),
                ('E_high', 'f8')
            ])
            hdf5_group.create_dataset('covariance_index_map', data=index_map_array)
        
        # Save active parameter indices
        if hasattr(self, 'active_parameter_indices'):
            hdf5_group.create_dataset('active_parameter_indices', data=self.active_parameter_indices)
    
    # =========================================================================
    # TAPE UPDATE METHODS
    # =========================================================================
    
    def update_tape(self, tape, sample_index=1, sample_name=""):
        """
        Update the ENDF tape with sampled energy distributions.
        
        For now, this is a placeholder. Full implementation would require:
        1. Reading the MF5 section
        2. Applying perturbations to probability distributions
        3. Reconstructing the ENDF format
        4. Updating the tape
        """
        print(f"⚠️  update_tape for energy distributions not yet fully implemented")
        print(f"   Sampled data is stored in energy_data.distributions[i].rel_deviation[{sample_index}]")
        # TODO: Implement full MF5 reconstruction similar to MF4 for angular
        pass
