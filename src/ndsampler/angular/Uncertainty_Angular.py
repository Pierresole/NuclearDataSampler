import numpy as np
import time
from collections import defaultdict
from .AngularDistributionCovariance import AngularDistributionCovariance
from .Parameters_Angular import AngularDistributionData
from ENDFtk import tree
from scipy.linalg import block_diag  # Import block_diag function

class Uncertainty_Angular(AngularDistributionCovariance):
    """
    Class for handling uncertainties in angular distributions.
    """

    def __init__(self, mf4mt, mf34mt, mt_number, legendre_orders=None):
        """
        Initialize Uncertainty_Angular object.
        
        Parameters:
        - mf4mt: MF4 section for the MT reaction
        - mf34mt: MF34 section for the MT reaction
        - mt_number: The MT reaction number
        - legendre_orders: List of Legendre orders to process (from covariance dict)
                          If None, all orders found in the data will be processed
        """
        # Store MT number FIRST before calling super().__init__
        self.MT = mt_number
        self.requested_legendre_orders = legendre_orders  # Store requested orders
        
        super().__init__(mf4mt)

        # Final check that MT is still set after base class initialization
        if not hasattr(self, 'MT') or self.MT != mt_number:
            self.MT = mt_number

        print(f"Creating angular distribution uncertainty for MT{mt_number}...")
        
        # Extract parameters and covariance matrices (following notebook methodology)
        start_time = time.time()
        # Get the first reaction mf34mt.reactions from MF34 (usually MT=2 elastic scattering)
        self.legendre_data = AngularDistributionData.from_endftk(mf4mt, mf34mt.reactions.to_list()[0])
        print(f"Time for extracting coefficients: {time.time() - start_time:.4f} seconds")
        
        start_time = time.time()
        self.extract_relcov_matrix(mf34mt.reactions.to_list()[0])
        print(f"Time for extracting covariance matrix: {time.time() - start_time:.4f} seconds")
        
        # Store covariance energy mesh
        if not hasattr(self, 'covariance_energy_mesh'):
            raise AttributeError("covariance_energy_mesh not set by extract_relcov_matrix")
        
        # Following notebook: Remove null-variance parameters and build index mapping
        start_time = time.time()
        NL = len(self.legendre_data.coefficients)
        rel_cov_full = self.relative_covariance_matrix
        
        print(f"Full relative covariance matrix shape: {rel_cov_full.shape}")
        
        # Apply trace-preserving PD correction if needed
        # rel_cov_corrected = self._make_positive_definite_preserve_trace(rel_cov_full, epsilon=1e-8)
        
        # Remove null-variance parameters
        reduced_cov, index_map = self._remove_null_variance_and_track(rel_cov_full, self.covariance_energy_mesh, NL)
        
        # Store results
        # self.reduced_relative_covariance_matrix = reduced_cov
        self.covariance_index_map = index_map  # Always created, even if no pruning
        self.active_parameter_indices = list(range(len(index_map)))
        
        print(f"✓ Stored covariance_index_map with {len(self.covariance_index_map)} entries")
        
        # Compute Cholesky decomposition DIRECTLY on REDUCED matrix
        # DO NOT use base class compute_L_matrix() - it applies another eigenvalue correction
        # that doesn't preserve trace, causing under-dispersion!
        # self.relative_covariance_matrix_full = rel_cov_corrected

        # Check if matrix is positive semi-definite and apply trace-preserving correction
        eigenvalues = np.linalg.eigvalsh(reduced_cov)
        if np.any(eigenvalues < -1e-10):
            print(f"Warning: Covariance matrix has {np.sum(eigenvalues < 0)} negative eigenvalues")
            print(f"Min eigenvalue: {eigenvalues.min()}")
            print(f"Applying trace-preserving positive-definite correction...")
            
            # Use trace-preserving correction instead of nearestPD
            reduced_cov_corrected = self._make_positive_definite_preserve_trace(reduced_cov, epsilon=1e-8)
            reduced_cov = reduced_cov_corrected
        # Cholesky decomposition for sampling
        try:
            self.L_matrix = np.linalg.cholesky(reduced_cov)
            self.is_cholesky = True
        except np.linalg.LinAlgError:
            print("Cholesky decomposition failed, using eigenvalue decomposition with trace preservation")
            # Fallback: eigenvalue thresholding with trace preservation
            eigenvalues, eigenvectors = np.linalg.eigh(reduced_cov)
            trace_orig = np.sum(eigenvalues)
            eigenvalues_corrected = np.maximum(eigenvalues, 1e-8)
            # Rescale to preserve trace
            eigenvalues_corrected = eigenvalues_corrected * (trace_orig / np.sum(eigenvalues_corrected))
            self.L_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues_corrected))
            self.is_cholesky = True
                    
        # Verify
        reconstructed_cov = self.L_matrix @ self.L_matrix.T
        print(f"  Reduced covariance[:4,:4]:")
        print(f"  {reduced_cov[:4,:4]}")
        print(f"  Reconstructed from L[:4,:4]:")
        print(f"  {reconstructed_cov[:4,:4]}")
        
        # Set relative_covariance_matrix for compatibility
        self.relative_covariance_matrix_full = reduced_cov
        super().__setattr__('relative_covariance_matrix', self.relative_covariance_matrix_full)
        
        print(f"Time for null-variance removal & Cholesky: {time.time() - start_time:.4f} seconds")
        
        print(f"✓ Created angular distribution uncertainty for MT{mt_number}")
    
    def get_covariance_type(self):
        """
        Override to return the specific covariance type.
        """
        return "AngularDistribution"

    # =========================================================================
    # Functions for handling covariance matrix pruning, mapping, and
    # extracting full relative covariance matrix
    # =========================================================================
    
    def _remove_null_variance_and_track(self, full_cov_matrix, all_mesh, NL, tol=1e-12):
        """
        Remove rows/columns with null variance from the covariance matrix while maintaining
        a mapping structure to track which (Legendre order, energy bin) each index corresponds to.
        
        Parameters:
        -----------
        full_cov_matrix : np.ndarray
            Full covariance matrix of shape (NL*N, NL*N) where N is number of energy bins
        all_mesh : list
            Energy mesh boundaries (length N+1)
        NL : int
            Number of Legendre orders
        tol : float
            Tolerance for considering variance as zero
            
        Returns:
        --------
        reduced_cov : np.ndarray
            Covariance matrix with null variance rows/cols removed
        index_map : list of tuples
            List of (legendre_order, energy_bin_index, energy_low, energy_high) for each 
            row/column in the reduced matrix
        """
        N = len(all_mesh) - 1  # Number of energy bins
        
        # Find non-zero variance indices
        diag = np.diag(full_cov_matrix)
        non_zero_mask = np.abs(diag) >= tol
        non_zero_indices = np.where(non_zero_mask)[0]
        
        # Build index mapping: for each kept index, store (legendre_order, bin_index, E_low, E_high)
        index_map = []
        for idx in non_zero_indices:
            legendre_order = (idx // N) + 1  # 1-based Legendre order
            bin_index = idx % N
            energy_low = all_mesh[bin_index]
            energy_high = all_mesh[bin_index + 1]
            index_map.append((legendre_order, bin_index, energy_low, energy_high))
        
        # Extract reduced covariance matrix
        reduced_cov = full_cov_matrix[np.ix_(non_zero_indices, non_zero_indices)]
        
        print(f"Original matrix size: {full_cov_matrix.shape[0]} x {full_cov_matrix.shape[1]}")
        print(f"Reduced matrix size: {reduced_cov.shape[0]} x {reduced_cov.shape[1]}")
        print(f"Removed {full_cov_matrix.shape[0] - reduced_cov.shape[0]} null variance elements")
        
        return reduced_cov, index_map
    
    def _reconstruct_full_perturbation(self, perturbation_vector, index_map, all_mesh, NL):
        """Reconstruct full perturbation vector with zeros for pruned parameters.
        
        Args:
            perturbation_vector: Perturbations for active parameters only
            index_map: Mapping from reduced indices to (L, bin, E_low, E_high)
            all_mesh: Energy mesh boundaries
            NL: Number of Legendre orders
            
        Returns:
            Full perturbation array of shape (NL, N) with zeros for null-variance bins
        """
        N = len(all_mesh) - 1
        full_perturbation = np.zeros((NL, N))
        
        for i, (legendre_order, bin_index, _, _) in enumerate(index_map):
            full_perturbation[legendre_order - 1, bin_index] = perturbation_vector[i]
        
        return full_perturbation

    def _make_positive_definite_preserve_trace(self, cov_matrix, epsilon=1e-8):
        """
        Make a covariance matrix positive definite while preserving its trace.
        
        This method thresholds negative eigenvalues and then rescales to maintain
        the original trace (sum of variances), preventing under-dispersion.
        
        Parameters:
        -----------
        cov_matrix : np.ndarray
            Input covariance matrix (possibly with negative eigenvalues)
        epsilon : float
            Minimum eigenvalue threshold
            
        Returns:
        --------
        cov_corrected : np.ndarray
            Positive definite covariance matrix with preserved trace
        """
        # Ensure symmetry
        cov_sym = (cov_matrix + cov_matrix.T) / 2
        
        # Original trace
        trace_original = np.trace(cov_sym)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_sym)
        
        # Count negative eigenvalues
        n_negative = np.sum(eigenvalues < 0)
        min_eigenvalue = eigenvalues.min()
        
        print(f"  Original matrix:")
        print(f"    Trace: {trace_original:.6f}")
        print(f"    Min eigenvalue: {min_eigenvalue:.6e}")
        print(f"    Negative eigenvalues: {n_negative}")
        
        # Threshold eigenvalues
        eigenvalues_corrected = np.maximum(eigenvalues, epsilon)
        
        # Rescale to preserve trace
        trace_corrected = np.sum(eigenvalues_corrected)
        scale_factor = trace_original / trace_corrected
        eigenvalues_rescaled = eigenvalues_corrected * scale_factor
        
        print(f"  Corrected matrix:")
        print(f"    Min eigenvalue: {eigenvalues_rescaled.min():.6e}")
        print(f"    Trace: {np.sum(eigenvalues_rescaled):.6f}")
        print(f"    Trace preservation: {np.sum(eigenvalues_rescaled) / trace_original:.8f}")
        print(f"    Scale factor applied: {scale_factor:.8f}")
        
        # Reconstruct matrix
        cov_corrected = eigenvectors @ np.diag(eigenvalues_rescaled) @ eigenvectors.T
        
        return cov_corrected

    @staticmethod           
    def mesh_union(mesh1, mesh2, eps=1e-8):
        union = np.unique(np.concatenate((mesh1, mesh2)))
        diff = np.diff(union)
        mask = diff < eps
        if np.any(mask):
            keep = np.ones_like(union, dtype=bool)
            keep[1:][mask] = False
            union = union[keep]
        return union

    @staticmethod
    def expand_matrix_fast(original_matrix, original_row_mesh, original_col_mesh, union_row_mesh, union_col_mesh):
        original_row_mesh = np.array(sorted(original_row_mesh))
        original_col_mesh = np.array(sorted(original_col_mesh))
        union_row_mesh = np.array(sorted(union_row_mesh))
        union_col_mesh = np.array(sorted(union_col_mesh))

        original_row_size = len(original_row_mesh) - 1
        original_col_size = len(original_col_mesh) - 1

        row_indices = np.searchsorted(original_row_mesh, union_row_mesh[:-1], side='right') - 1
        col_indices = np.searchsorted(original_col_mesh, union_col_mesh[:-1], side='right') - 1

        row_indices = np.clip(row_indices, 0, original_row_size-1)
        col_indices = np.clip(col_indices, 0, original_col_size-1)

        expanded_matrix = original_matrix[np.ix_(row_indices, col_indices)]
        return expanded_matrix

    @staticmethod
    def add_matrices_with_mesh(matrixA, rowMeshA, colMeshA, matrixB, rowMeshB, colMeshB, epsilon=1e-8):
        if matrixA.size == 0:
            return matrixB.copy(), sorted(rowMeshB), sorted(colMeshB)
        if matrixB.size == 0:
            return matrixA.copy(), sorted(rowMeshA), sorted(colMeshA)

        rowMeshA = np.array(sorted(rowMeshA))
        colMeshA = np.array(sorted(colMeshA))
        rowMeshB = np.array(sorted(rowMeshB))
        colMeshB = np.array(sorted(colMeshB))

        union_row_mesh = Uncertainty_Angular.mesh_union(rowMeshA, rowMeshB, epsilon)
        union_col_mesh = Uncertainty_Angular.mesh_union(colMeshA, colMeshB, epsilon)

        expandedA = Uncertainty_Angular.expand_matrix_fast(matrixA, rowMeshA, colMeshA, union_row_mesh, union_col_mesh)
        expandedB = Uncertainty_Angular.expand_matrix_fast(matrixB, rowMeshB, colMeshB, union_row_mesh, union_col_mesh)

        result = expandedA + expandedB
        return result, union_row_mesh.tolist(), union_col_mesh.tolist()

    @staticmethod
    def subblock_to_matrix(subblock):
        # LB==5: symmetric, upper triangle stored
        if hasattr(subblock, "LB") and subblock.LB == 5:
            N = subblock.NE - 1
            mesh = subblock.energies.to_list()
            mat = np.zeros((N, N))
            triu_indices = np.triu_indices(N)
            mat[triu_indices] = subblock.values.to_list()
            mat = mat + mat.T - np.diag(np.diag(mat))
            return mat, mesh, mesh
        # LB==1: diagonal
        elif hasattr(subblock, "LB") and subblock.LB == 1:
            mesh = subblock.first_array_energies.to_list()
            vals = subblock.first_array_fvalues.to_list()
            mat = np.diag(vals)
            return mat, mesh, mesh
        # CovariancePairs (LB==1)
        elif hasattr(subblock, "number_pairs"):
            mesh = subblock.first_array_energies.to_list()
            vals = subblock.first_array_fvalues.to_list()
            mat = np.diag(vals)
            return mat, mesh, mesh
        else:
            raise NotImplementedError(f"Unknown subblock type: {type(subblock)}, LB={getattr(subblock, 'LB', 'N/A')}")

    @staticmethod
    def block_to_matrix(block):
        # block is ENDFtk.SquareMatrix or ENDFtk.LegendreBlock
        # block.data.to_list() gives subblocks
        subblocks = block.data.to_list() if hasattr(block, "data") else [block]
        matrix = np.zeros((0,0))
        row_mesh = []
        col_mesh = []
        for sub in subblocks:
            submat, subrow, subcol = Uncertainty_Angular.subblock_to_matrix(sub)
            if matrix.size == 0:
                matrix = submat
                row_mesh = subrow
                col_mesh = subcol
            else:
                matrix, row_mesh, col_mesh = Uncertainty_Angular.add_matrices_with_mesh(
                    matrix, row_mesh, col_mesh, submat, subrow, subcol
                )
        return matrix, row_mesh, col_mesh

    def extract_relcov_matrix(self, mt2):
        NL = mt2.NL
        NL1 = mt2.NL1
        nblocks = mt2.number_legendre_blocks
        blocks = mt2.legendre_blocks.to_list()
        # First, collect all unique energy mesh points for all blocks
        all_mesh = set()
        for block in blocks:
            for sub in block.data.to_list():
                if hasattr(sub, "LB") and sub.LB == 5 and hasattr(sub, "energies"):
                    all_mesh.update(sub.energies.to_list())
                elif hasattr(sub, "LB") and sub.LB == 1 and hasattr(sub, "first_array_energies"):
                    all_mesh.update(sub.first_array_energies.to_list())
                elif hasattr(sub, "number_pairs"):
                    all_mesh.update(sub.first_array_energies.to_list())
        all_mesh = sorted(all_mesh)
        N = len(all_mesh) - 1

        # Prepare the full relative covariance matrix
        full_rel_cov = np.zeros((NL*N, NL1*N))
        # For each block (l, l1), fill the corresponding submatrix
        for idx, block in enumerate(blocks):
            l = block.L
            l1 = block.L1
            mat, row_mesh, col_mesh = Uncertainty_Angular.block_to_matrix(block)
            # Expand to the global mesh
            mat_expanded = Uncertainty_Angular.expand_matrix_fast(mat, row_mesh, col_mesh, all_mesh, all_mesh)
            # Place in the full matrix
            full_rel_cov[(l-1)*N:l*N, (l1-1)*N:l1*N] = mat_expanded
            if l != l1:
                # Fill symmetric block
                full_rel_cov[(l1-1)*N:l1*N, (l-1)*N:l*N] = mat_expanded.T
        super().__setattr__('relative_covariance_matrix', full_rel_cov)  # Store for compute_L_matrix()
        
        # Store the fine energy mesh used by covariance matrix
        self.covariance_energy_mesh = sorted(list(all_mesh))

    def _build_covariance_to_legendre_mapping(self):
        """
        Build mapping from covariance matrix parameters (fine grid) to legendre_data bins (coarse grid).
        
        The covariance matrix describes correlations on a fine unified energy mesh (e.g., 42 bins per order),
        while legendre_data stores nominal coefficients on a coarser mesh (e.g., 10-12 bins per order).
        
        This method creates a mapping that tells us: "for legendre_data bin i of order l,
        which covariance matrix parameters should be aggregated?"
        """
        if not hasattr(self, 'covariance_energy_mesh'):
            print("⚠️  Covariance energy mesh not stored - cannot build mapping")
            return
        
        cov_mesh = np.array(self.covariance_energy_mesh)
        n_cov_bins_per_order = len(cov_mesh) - 1
        
        # Build mapping: legendre_bin_to_cov_bins[(order, leg_bin_idx)] = [cov_param_indices]
        self.legendre_bin_to_cov_bins = {}
        
        for coeff in self.legendre_data.coefficients:
            order = coeff.order
            leg_energies = np.array(coeff.energies)
            n_leg_bins = len(leg_energies) - 1
            
            # For each legendre bin, find which covariance bins overlap with it
            for leg_bin_idx in range(n_leg_bins):
                leg_e_low = leg_energies[leg_bin_idx]
                leg_e_high = leg_energies[leg_bin_idx + 1]
                
                # Find covariance bins that overlap with [leg_e_low, leg_e_high]
                overlapping_cov_bins = []
                for cov_bin_idx in range(n_cov_bins_per_order):
                    cov_e_low = cov_mesh[cov_bin_idx]
                    cov_e_high = cov_mesh[cov_bin_idx + 1]
                    
                    # Check for overlap
                    if cov_e_high > leg_e_low and cov_e_low < leg_e_high:
                        # Convert to global covariance parameter index
                        global_cov_param = (order - 1) * n_cov_bins_per_order + cov_bin_idx
                        overlapping_cov_bins.append(global_cov_param)
                
                self.legendre_bin_to_cov_bins[(order, leg_bin_idx)] = overlapping_cov_bins
        
        print(f"Built mapping: {len(self.legendre_bin_to_cov_bins)} legendre bins mapped to covariance parameters")


    # =========================================================================
    # Functions for applying samples to Legendre coefficients
    # =========================================================================

    def _apply_samples(self, samples, mode="stack", use_copula=False, batch_size=1, 
                      sampling_method="Simple", debug=False):
        """Apply samples to Legendre coefficients.
        
        Following mf34_test_clever.ipynb methodology:
        1. Samples are in reduced space (active parameters only)
        2. Reconstruct full grid with zeros for null-variance bins
        3. Store as relative deviations δ where a_sample = a_nominal × (1 + δ)
        
        Args:
            samples: Shape (n_samples, n_active_params) - perturbations for active parameters
            mode: "stack" or "replace"
            use_copula: Transform from uniform to normal
            debug: Enable diagnostic output
        """
        # Store for debug
        if debug:
            self.stored_samples = samples.copy()
        
        # Ensure proper shape
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        n_samples, n_reduced_params = samples.shape
        
        # Validate dimensions
        if not hasattr(self, 'covariance_index_map'):
            raise AttributeError("covariance_index_map not found - initialization failed")
        
        expected_reduced_params = len(self.covariance_index_map)
        if n_reduced_params != expected_reduced_params:
            raise ValueError(f"Sample dimension mismatch: expected {expected_reduced_params}, got {n_reduced_params}")
        
        # Clear if replace mode
        if mode == 'replace':
            for coeff in self.legendre_data.coefficients:
                coeff.rel_deviation.clear()  # Clear existing deviations
        
        # Process each sample
        NL = len(self.legendre_data.coefficients)
        N = len(self.covariance_energy_mesh) - 1
        
        for sample_idx in range(n_samples):
            reduced_sample = samples[sample_idx, :]
            
            # Transform copula if needed
            if use_copula:
                from scipy.stats import norm
                reduced_sample = norm.ppf(reduced_sample)
            
            # Reconstruct full perturbation grid (NL × N) with zeros for null-variance
            full_perturbation = self._reconstruct_full_perturbation(
                reduced_sample, 
                self.covariance_index_map, 
                self.covariance_energy_mesh, 
                NL
            )
            if sample_idx == 1:
                print(f"Full perturbation shape: {full_perturbation.shape}")
            
            # Store relative deviations in each LegendreCoefficient object
            # CRITICAL: Store the FULL perturbation array (N bins from global covariance mesh)
            # not just the first n_bins from the per-order mesh
            effective_sample_idx = sample_idx + 1
            for coeff in self.legendre_data.coefficients:
                order_idx = coeff.order - 1  # Convert to 0-based
                
                # Extend rel_deviation list if needed
                while len(coeff.rel_deviation) <= effective_sample_idx:
                    coeff.rel_deviation.append(None)
                
                # Store ALL N bins (global covariance mesh), not just per-order bins
                coeff.rel_deviation[effective_sample_idx] = full_perturbation[order_idx, :].tolist()
        
        print(f"Applied {n_samples} samples to Legendre coefficients")
        
        if debug:
            self._debug_covariance_comparison(samples, debug=True)
            reduced_sample = samples[sample_idx, :]
            
            # Transform copula if needed
            if use_copula:
                from scipy.stats import norm
                reduced_sample = norm.ppf(reduced_sample)
            
            # Reconstruct full perturbation grid (NL × N) with zeros for null-variance
            full_perturbation = self._reconstruct_full_perturbation(
                reduced_sample, 
                self.covariance_index_map, 
                self.covariance_energy_mesh, 
                NL
            )
            
            # Store relative deviations in each LegendreCoefficient object
            effective_sample_idx = sample_idx + 1
            for coeff in self.legendre_data.coefficients:
                l_idx = coeff.order - 1  # 0-based
                delta_arr = full_perturbation[l_idx, :]  # All N bins for this order
                
                # Ensure list is long enough
                while len(coeff.rel_deviation) <= effective_sample_idx:
                    coeff.rel_deviation.append(None)
                
                coeff.rel_deviation[effective_sample_idx] = delta_arr.tolist()
        

    # def _verify_coefficient_sampling_statistics(self, debug=True):
    #     """
    #     Verify that the sampling statistics match the expected standard deviations.
    #     """
    #     if debug:
    #         # Aggregate relative deviations and compare to identity (since L_matrix encodes covariance)
    #         all_deltas = []
    #         for coeff in self.legendre_data.coefficients:
    #             # Skip nominal (index 0)
    #             for sidx in range(1, len(coeff.rel_deviation)):
    #                 if sidx < len(coeff.rel_deviation) and coeff.rel_deviation[sidx] is not None:
    #                     all_deltas.append(coeff.rel_deviation[sidx])
    #         if not all_deltas:
    #             print("   No relative deviations stored yet.")
    #             return
    #         delta_matrix = np.vstack(all_deltas)
    #         sample_std = delta_matrix.std(axis=0, ddof=1)
    #         print(f"   Relative deviation sample std (first 10): {sample_std[:10]}")

    # def _debug_show_sample_matrix(self, samples, debug=True):
    #     """
    #     Show the sample matrix for debugging purposes.
    #     """
    #     if debug:
    #         print(f"\n📋 SAMPLE MATRIX (active parameters only):")
    #         print(f"   Shape: {samples.shape}")
    #         print(f"   Sample matrix (first 5 rows, first 10 cols):")
    #         max_rows = min(5, samples.shape[0])
    #         max_cols = min(10, samples.shape[1])
    #         for i in range(max_rows):
    #             row_str = "   " + " ".join(f"{samples[i,j]:8.4f}" for j in range(max_cols))
    #             if samples.shape[1] > max_cols:
    #                 row_str += " ..."
    #             print(row_str)
    #         if samples.shape[0] > max_rows:
    #             print("   ...")

    # def _debug_covariance_comparison(self, samples, debug=True):
    #     """
    #     Compare the empirical covariance matrix from samples with the original relative covariance matrix.
    #     Identifies coefficients that differ by more than 10%.
    #     """
    #     print(f"\n🔍 COVARIANCE MATRIX COMPARISON:")
    #     print(f"   Debug flag: {debug}, samples shape: {samples.shape}")
        
    #     if not debug:
    #         print("   Debug mode not enabled")
    #         return
            
    #     if samples.shape[0] < 10:
    #         print(f"   Not enough samples for comparison ({samples.shape[0]} < 10)")
    #         return
        
    #     # Debug: Check what covariance matrices are available
    #     available_matrices = []
    #     if hasattr(self, 'reduced_relative_covariance_matrix'):
    #         available_matrices.append(f"reduced_relative_covariance_matrix: {self.reduced_relative_covariance_matrix.shape}")
    #     if hasattr(self, 'relative_covariance_matrix_full'):
    #         available_matrices.append(f"relative_covariance_matrix_full: {self.relative_covariance_matrix_full.shape}")
    #     if hasattr(self, 'relative_covariance_matrix'):
    #         available_matrices.append(f"relative_covariance_matrix: {self.relative_covariance_matrix.shape}")
    #     print(f"   Available matrices: {available_matrices}")
        
    #     # Compute empirical covariance from samples
    #     empirical_cov = np.cov(samples, rowvar=False)
        
    #     # Get the reduced (active parameters only) relative covariance matrix
    #     if hasattr(self, 'reduced_relative_covariance_matrix'):
    #         expected_cov = self.reduced_relative_covariance_matrix
    #     elif hasattr(self, 'relative_covariance_matrix_full'):
    #         # Use the full matrix that was saved during initialization
    #         active_indices = getattr(self, 'active_parameter_indices', list(range(samples.shape[1])))
    #         expected_cov = self.relative_covariance_matrix_full[np.ix_(active_indices, active_indices)]
    #     elif hasattr(self, 'relative_covariance_matrix'):
    #         # If no reduced matrix available, extract from full matrix using active indices
    #         active_indices = getattr(self, 'active_parameter_indices', list(range(samples.shape[1])))
    #         expected_cov = self.relative_covariance_matrix[np.ix_(active_indices, active_indices)]
    #     else:
    #         print("   ⚠️  No reference covariance matrix available for comparison")
    #         return
        
    #     if empirical_cov.shape != expected_cov.shape:
    #         print(f"   ⚠️  Shape mismatch: empirical {empirical_cov.shape} vs expected {expected_cov.shape}")
    #         return
        
    #     # Compare diagonal elements (variances)
    #     empirical_var = np.diag(empirical_cov)
    #     expected_var = np.diag(expected_cov)
        
    #     # Calculate relative differences
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         rel_diff = np.abs(empirical_var - expected_var) / np.maximum(expected_var, 1e-10)
        
    #     # Find parameters with >10% difference
    #     threshold = 0.10  # 10%
    #     large_diff_indices = np.where(rel_diff > threshold)[0]
        
    #     print(f"   Total active parameters: {len(empirical_var)}")
    #     print(f"   Parameters with >10% variance difference: {len(large_diff_indices)}")
        
    #     if len(large_diff_indices) > 0:
    #         print(f"\n   🚨 LARGE DIFFERENCES (>{threshold*100:.0f}%):")
    #         print(f"   {'Param':<6} {'Order':<6} {'Bin':<4} {'Expected':<12} {'Empirical':<12} {'Rel.Diff':<10}")
    #         print(f"   {'-'*60}")
            
    #         # Map parameter indices back to Legendre orders and bins
    #         active_indices = getattr(self, 'active_parameter_indices', list(range(len(self.parameter_index_map))))
            
    #         for local_idx in large_diff_indices[:20]:  # Show first 20 problematic parameters
    #             global_idx = active_indices[local_idx] if local_idx < len(active_indices) else local_idx
                
    #             if global_idx < len(self.parameter_index_map):
    #                 coeff, bin_idx = self.parameter_index_map[global_idx]
    #                 order = coeff.order
    #             else:
    #                 order = "?"
    #                 bin_idx = "?"
                
    #             expected_val = expected_var[local_idx]
    #             empirical_val = empirical_var[local_idx]
    #             rel_diff_val = rel_diff[local_idx]
                
    #             print(f"   {local_idx:<6} {order:<6} {bin_idx:<4} {expected_val:<12.6f} {empirical_val:<12.6f} {rel_diff_val:<10.2%}")
            
    #         if len(large_diff_indices) > 20:
    #             print(f"   ... and {len(large_diff_indices) - 20} more")
    #     else:
    #         print(f"   ✅ All parameters within {threshold*100:.0f}% tolerance")
        
    #     # Summary statistics
    #     print(f"\n   📊 VARIANCE COMPARISON STATISTICS:")
    #     print(f"   Mean relative difference: {np.mean(rel_diff):.2%}")
    #     print(f"   Max relative difference: {np.max(rel_diff):.2%}")
    #     print(f"   RMS relative difference: {np.sqrt(np.mean(rel_diff**2)):.2%}")
        
    #     # Frobenius norm comparison for full matrices
    #     frobenius_diff = np.linalg.norm(empirical_cov - expected_cov, 'fro')
    #     frobenius_expected = np.linalg.norm(expected_cov, 'fro')
    #     frobenius_rel = frobenius_diff / frobenius_expected if frobenius_expected > 0 else np.inf
        
    #     print(f"   Matrix Frobenius norm difference: {frobenius_rel:.2%}")
        
    #     return {
    #         'empirical_cov': empirical_cov,
    #         'expected_cov': expected_cov,
    #         'large_diff_indices': large_diff_indices,
    #         'rel_diff': rel_diff,
    #         'frobenius_rel_diff': frobenius_rel
    #     }

    
    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads the L_matrix and legendre_data from the given HDF5 group and returns an instance.
        """        
        # Read L_matrix
        L_matrix = hdf5_group['L_matrix'][()]
        
        # Read mean_vector if it exists
        mean_vector = hdf5_group['mean_vector'][()] if 'mean_vector' in hdf5_group else None
        
        # Read std_dev_vector if it exists
        std_dev_vector = hdf5_group['std_dev_vector'][()] if 'std_dev_vector' in hdf5_group else None
        
        # Read is_cholesky flag
        is_cholesky = hdf5_group.attrs.get('is_cholesky', False)
        
        # Read MT number
        mt_number = hdf5_group.attrs.get('MT', 2)  # Default to 2 if not found

        # Read legendre_data
        leg_data_group = hdf5_group['Parameters']
        leg_data = AngularDistributionData.read_from_hdf5(leg_data_group)
        
        # Create an instance and set attributes
        instance = cls.__new__(cls)
        
        # Initialize parent class attributes
        super(cls, instance).__init__(None)  # Pass None since we're reading from HDF5
        
        # Set attributes from HDF5
        instance.L_matrix = L_matrix
        instance.mean_vector = mean_vector
        instance.std_dev_vector = std_dev_vector
        instance.is_cholesky = is_cholesky
        instance.legendre_data = leg_data
        instance.MT = mt_number  # Restore MT number
        
        # Restore covariance energy mesh
        if 'covariance_energy_mesh' in hdf5_group:
            instance.covariance_energy_mesh = hdf5_group['covariance_energy_mesh'][()].tolist()
        
        # Restore covariance index map
        if 'covariance_index_map' in hdf5_group:
            index_map_array = hdf5_group['covariance_index_map'][()]
            instance.covariance_index_map = [
                (int(row['legendre_order']), int(row['bin_index']), float(row['E_low']), float(row['E_high']))
                for row in index_map_array
            ]
        
        # Restore active parameter indices
        if 'active_parameter_indices' in hdf5_group:
            instance.active_parameter_indices = hdf5_group['active_parameter_indices'][()].tolist()
        else:
            # Fallback: assume all L_matrix parameters are active
            instance.active_parameter_indices = list(range(L_matrix.shape[0]))
        
        return instance

    def write_additional_data_to_hdf5(self, hdf5_group):
        if self.legendre_data is not None:
            leg_group = hdf5_group.require_group('Parameters')
            self.legendre_data.write_to_hdf5(leg_group)
        
        # Save the MT number as an attribute
        hdf5_group.attrs['MT'] = self.MT
        
        # Save covariance energy mesh
        if hasattr(self, 'covariance_energy_mesh'):
            hdf5_group.create_dataset('covariance_energy_mesh', data=self.covariance_energy_mesh)
        
        # Save covariance index map (needed for reconstruction)
        if hasattr(self, 'covariance_index_map'):
            # Store as structured array: (legendre_order, bin_index, E_low, E_high)
            import numpy as np
            index_map_array = np.array(self.covariance_index_map, dtype=[
                ('legendre_order', 'i4'),
                ('bin_index', 'i4'),
                ('E_low', 'f8'),
                ('E_high', 'f8')
            ])
            hdf5_group.create_dataset('covariance_index_map', data=index_map_array)
        
        # Save active parameter indices
        if hasattr(self, 'active_parameter_indices'):
            hdf5_group.create_dataset('active_parameter_indices', data=self.active_parameter_indices)


    # =========================================================================
    # Functions for updating ENDF tape with sampled coefficients
    # =========================================================================

    def update_tape(self, tape, sample_index=1, sample_name=""):
        """
        Updates the ENDF tape with sampled Legendre coefficients for the given sample_index.
        """
        from ENDFtk.MF4 import Section, LegendreDistributions, LegendreCoefficients, MixedDistributions
        # Parse the section to update (use dynamic MT number)
        mf4mt = tape.MAT(tape.material_numbers[0]).MF(4).MT(self.MT).parse()

        # Build factor dictionary directly from stored relative deviations 'd' to avoid
        # reconstructing absolute coefficients then re-dividing.
        # factors = 1 + d per covariance bin.
        # CRITICAL: Use global covariance mesh size (N bins), not per-order mesh size
        n_cov_bins = len(self.covariance_energy_mesh) - 1
        factors_dict = {}
        for coeff_data in self.legendre_data.coefficients:
            if sample_index < len(coeff_data.rel_deviation) and coeff_data.rel_deviation[sample_index] is not None:
                delta = coeff_data.rel_deviation[sample_index]
                # delta should have n_cov_bins elements from global mesh
                if len(delta) != n_cov_bins:
                    # Pad or trim if mismatch (shouldn't happen with fix above)
                    adj = (delta + [0.0]*(n_cov_bins - len(delta)))[:n_cov_bins]
                    delta = adj
                factors_dict[coeff_data.order] = [1.0 + d for d in delta]
            else:
                # Nominal (or missing) => unity factors
                factors_dict[coeff_data.order] = [1.0]*n_cov_bins

        perturbed_legendre_dist = self._create_perturbed_legendre_distributions_from_factors(mf4mt, factors_dict)
        
        # Handle mixed distributions (both Legendre and tabulated)
        if mf4mt.LTT == 3:
            # Mixed case: both Legendre and tabulated
            perturbed_distributions = MixedDistributions(
                legendre=perturbed_legendre_dist,
                tabulated=mf4mt.distributions.tabulated  # Keep original tabulated part
            )
        elif mf4mt.LTT == 1: # Pure Legendre case
            # Pure Legendre case
            perturbed_distributions = perturbed_legendre_dist
        
        # Create new Section and replace in tape
        new_section = Section(
            mt=mf4mt.MT,
            zaid=mf4mt.ZA,
            awr=mf4mt.AWR,
            lct=mf4mt.LCT,
            distributions=perturbed_distributions
        )
        
        mat_num = tape.material_numbers[0]
        tape.MAT(mat_num).MF(4).insert_or_replace(new_section)
    
    def _create_perturbed_legendre_distributions_from_factors(self, mf4mt, multiplicative_factors):
        """Create perturbed Legendre distributions directly from bin-wise multiplicative factors.

        Parameters
        ----------
        mf4mt : ENDFtk MF4 section (parsed)
        multiplicative_factors : dict[int, list[float]]
            Mapping Legendre order -> factor per covariance bin (length = n_bins for that order).
        """
        from ENDFtk.MF4 import LegendreDistributions, LegendreCoefficients

        # Access original distributions (pure or mixed)
        if mf4mt.LTT == 1:
            original_dist = mf4mt.distributions
        elif mf4mt.LTT == 3:
            original_dist = mf4mt.distributions.legendre
        original_distributions = original_dist.angular_distributions.to_list()
        original_energies = [dist.incident_energy for dist in original_distributions]

        # CRITICAL: Use global covariance mesh boundaries, not per-order boundaries
        # The multiplicative_factors are indexed by global covariance bins
        covariance_boundaries = self.covariance_energy_mesh
        
        # Union mesh of original + covariance boundaries
        union_energies = self.mesh_union(original_energies, covariance_boundaries)

        enhanced_energies = []
        enhanced_coeffs_data = []

        # Helper: fetch nominal coefficients at any energy
        get_nominal = self.legendre_data.get_coefficients_at_energy

        for energy in union_energies:
            base_coeffs = get_nominal(energy)  # nominal L>=1 list
            is_boundary = energy in covariance_boundaries
            if is_boundary:
                b_idx = covariance_boundaries.index(energy)
                first = (b_idx == 0)
                last = (b_idx == len(covariance_boundaries)-1)

                def apply_bin(bin_idx):
                    return self._apply_multiplicative_factors_to_coefficients(base_coeffs, multiplicative_factors, bin_index=bin_idx)

                if first:
                    # E0 nominal
                    enhanced_energies.append(energy); enhanced_coeffs_data.append(base_coeffs.copy())
                    # E0' first bin factors
                    enhanced_energies.append(energy); enhanced_coeffs_data.append(apply_bin(0))
                elif last:
                    prev_bin = b_idx - 1
                    pert = apply_bin(prev_bin)
                    enhanced_energies.append(energy); enhanced_coeffs_data.append(pert)
                    enhanced_energies.append(energy); enhanced_coeffs_data.append(pert.copy())
                else:
                    prev_bin = b_idx - 1
                    next_bin = b_idx
                    prev_coeffs = apply_bin(prev_bin)
                    enhanced_energies.append(energy); enhanced_coeffs_data.append(prev_coeffs)
                    enhanced_energies.append(energy); enhanced_coeffs_data.append(prev_coeffs.copy())
                    next_coeffs = apply_bin(next_bin)
                    enhanced_energies.append(energy); enhanced_coeffs_data.append(next_coeffs)
            else:
                bin_idx = self._find_bin_index_for_energy(energy, covariance_boundaries)
                pert = self._apply_multiplicative_factors_to_coefficients(
                    base_coeffs, multiplicative_factors, bin_index=bin_idx)
                enhanced_energies.append(energy); enhanced_coeffs_data.append(pert)

        enhanced_n_points = len(enhanced_energies)
        new_boundaries = [enhanced_n_points]
        new_interpolants = [2]
        new_legendre_coeffs = [LegendreCoefficients(E, coeffs) for E, coeffs in zip(enhanced_energies, enhanced_coeffs_data)]
        return LegendreDistributions(new_boundaries, new_interpolants, new_legendre_coeffs)

    def _apply_multiplicative_factors_to_coefficients(self, base_coeffs, multiplicative_factors, bin_index):
        """
        Apply bin-wise multiplicative factors to coefficients.
        
        Parameters:
        - base_coeffs: Original coefficients [a1, a2, a3, ...] for this energy point
        - multiplicative_factors: {order: [factor_per_bin]} dictionary
        - bin_index: Which covariance bin this energy point belongs to
        """
        perturbed_coeffs = []
        
        for l_order in range(1, len(base_coeffs) + 1):
            if l_order <= len(base_coeffs):
                original_val = base_coeffs[l_order - 1]  # base_coeffs[0] = a1, etc.
            else:
                original_val = 0.0
            
            # Apply multiplicative factor if available for this order
            if l_order in multiplicative_factors:
                factors = multiplicative_factors[l_order]
                if 0 <= bin_index < len(factors):
                    factor = factors[bin_index]
                    final_val = original_val * factor
                    # if l_order == 1:
                    #     print(f"    Applying factor {factor} (original={original_val}, final={final_val})")
                else:
                    final_val = original_val  # No factor for this bin
            else:
                final_val = original_val  # No factors for this order
            
            perturbed_coeffs.append(final_val)
        
        return perturbed_coeffs
    
    def _find_bin_index_for_energy(self, energy, covariance_boundaries):
        """
        Find which covariance bin this energy falls into.
        Match the notebook implementation exactly.
        """
        if energy <= covariance_boundaries[0]:
            return 0
        if energy >= covariance_boundaries[-1]:
            return len(covariance_boundaries) - 2
        for i in range(len(covariance_boundaries) - 1):
            if covariance_boundaries[i] <= energy < covariance_boundaries[i+1]:
                return i
        return len(covariance_boundaries) - 2

    def _interpolate_and_perturb_coefficients(self, energy, original_coeffs, sample_coefficients_dict, boundary_type="regular", bin_index=None):
        """
        Interpolate Legendre coefficients at given energy and apply perturbations using the additive approach.
        
        Parameters:
        - energy: Energy point where coefficients are needed
        - original_coeffs: Original coefficients from MF4 (L=1, L=2, ...)
        - sample_coefficients_dict: {order: [actual_coefficients]} from sampled data (NOT factors)
        - boundary_type: "regular" or ignored if bin_index is provided
        - bin_index: Specific bin index to use for perturbation (overrides boundary_type logic)
        """
        # Process L≥1 coefficients only (L=0 is implicit = 1.0, never written to ENDF)
        perturbed_coeffs = []
        
        # Process L≥1 coefficients
        for l_order in range(1, len(original_coeffs) + 1):
            if l_order <= len(original_coeffs):
                original_val = original_coeffs[l_order - 1]  # original_coeffs[0] = a_1, etc.
            else:
                original_val = 0.0  # Higher orders not present
            
            # Use perturbed coefficient if this Legendre order has covariance data
            final_val = original_val  # Default to original value
            if l_order in sample_coefficients_dict:
                coeff_data = next((c for c in self.legendre_data.coefficients if c.order == l_order), None)
                if coeff_data is not None:
                    if bin_index is not None:
                        # Use specific bin index - directly use the perturbed coefficient
                        if 0 <= bin_index < len(sample_coefficients_dict[l_order]):
                            final_val = sample_coefficients_dict[l_order][bin_index]
                    else:
                        # Use boundary type logic to determine which bin to use
                        perturbed_val = self._get_perturbed_coefficient_at_energy(
                            energy, coeff_data, sample_coefficients_dict[l_order], boundary_type
                        )
                        final_val = perturbed_val
            
            perturbed_coeffs.append(final_val)
        
        return perturbed_coeffs
    
    def _get_perturbed_coefficient_at_energy(self, energy, coeff_data, perturbed_coeffs, boundary_type):
        """
        Get perturbed coefficient for given energy, handling boundary logic correctly.
        
        Parameters:
        - energy: Energy point
        - coeff_data: LegendreCoefficient object containing energy bins
        - perturbed_coeffs: List of actual perturbed coefficients for this Legendre order
        - boundary_type: "regular", "upper", or "lower"
        
        Returns:
        - The perturbed coefficient value for this energy
        """
        bin_boundaries = coeff_data.energies  # Energy boundaries defining bins
        
        if boundary_type == "regular":
            # Regular energy point - find which bin it belongs to
            bin_idx = np.searchsorted(bin_boundaries[:-1], energy, side='right') - 1
            bin_idx = np.clip(bin_idx, 0, len(perturbed_coeffs) - 1)
            return perturbed_coeffs[bin_idx]
        
        elif boundary_type == "upper":
            # Upper edge of previous bin
            bin_idx = np.searchsorted(bin_boundaries, energy, side='left') - 1
            bin_idx = max(0, bin_idx)  # Previous bin, or 0 if this is first boundary
            if bin_idx < len(perturbed_coeffs):
                return perturbed_coeffs[bin_idx]
            else:
                # Fallback to nominal if bin doesn't exist
                nominal_coeffs = coeff_data.legcoeff[0] if coeff_data.legcoeff else []
                return nominal_coeffs[bin_idx] if bin_idx < len(nominal_coeffs) else 0.0
                
        elif boundary_type == "lower":
            # Lower edge of next bin
            bin_idx = np.searchsorted(bin_boundaries, energy, side='right')
            if bin_idx < len(perturbed_coeffs):
                return perturbed_coeffs[bin_idx]
            else:
                # Fallback to nominal if bin doesn't exist
                nominal_coeffs = coeff_data.legcoeff[0] if coeff_data.legcoeff else []
                return nominal_coeffs[bin_idx] if bin_idx < len(nominal_coeffs) else 0.0
        
        # Fallback to nominal coefficient
        nominal_coeffs = coeff_data.legcoeff[0] if coeff_data.legcoeff else []
        return nominal_coeffs[0] if nominal_coeffs else 0.0
    
    def _get_perturbation_factor_at_energy(self, energy, coeff_data, factors, boundary_type):
        """
        Get perturbation factor for given energy, handling boundary logic correctly.
        
        Parameters:
        - energy: Energy point
        - coeff_data: LegendreCoefficient object containing energy bins
        - factors: List of perturbation factors for this Legendre order
        - boundary_type: "regular", "upper", or "lower"
        """
        bin_boundaries = coeff_data.energies  # Energy boundaries defining bins
        
        if boundary_type == "regular":
            # Regular energy point - find which bin it belongs to
            bin_idx = np.searchsorted(bin_boundaries[:-1], energy, side='right') - 1
            bin_idx = np.clip(bin_idx, 0, len(factors) - 1)
            return factors[bin_idx]
        
        elif boundary_type == "upper":
            # Upper edge of previous bin
            bin_idx = np.searchsorted(bin_boundaries, energy, side='left') - 1
            bin_idx = max(0, bin_idx)  # Previous bin, or 0 if this is first boundary
            if bin_idx < len(factors):
                return factors[bin_idx]
            else:
                return 1.0  # No perturbation if bin doesn't exist
                
        elif boundary_type == "lower":
            # Lower edge of next bin
            bin_idx = np.searchsorted(bin_boundaries, energy, side='right')
            if bin_idx < len(factors):
                return factors[bin_idx]
            else:
                return 1.0  # No perturbation if bin doesn't exist
        
        return 1.0
        
    def _verify_sampling_statistics(self, debug=True):
        """
        Private method to verify that the sampled parameters match the expected covariance structure.
        Called internally by _apply_samples when debug=True.
        
        Args:
            debug (bool): If True, print detailed verification results
            
        Returns:
            dict: Dictionary containing verification metrics
        """
        if not hasattr(self.legendre_data, 'coefficients') or not self.legendre_data.coefficients:
            return {"error": "No Legendre coefficient data available"}
        
        # Check if samples exist
        sample_count = 0
        for coeff in self.legendre_data.coefficients:
            if len(coeff.factor) > 0:
                sample_count = len(coeff.factor)
                break
        
        if sample_count == 0:
            return {"error": "No samples found - run sample_parameters() first"}
        
        if debug:
            print(f"🔍 VERIFYING SAMPLING STATISTICS FOR MT{self.MT}")
            print(f"{'='*60}")
            print(f"Number of samples: {sample_count}")
            print(f"Theoretical parameters: {len(self.std_dev_vector)}")
            print(f"Legendre orders: {[c.order for c in self.legendre_data.coefficients]}")
        
        # Collect all sampled factors into a matrix
        all_factors = []
        param_mapping = []  # Track which parameter each column represents
        
        for coeff in self.legendre_data.coefficients:
            for sample_idx in range(sample_count):
                if sample_idx < len(coeff.factor):
                    factors = coeff.factor[sample_idx]
                    if sample_idx == 0:  # First sample, establish structure
                        start_idx = len(all_factors[0]) if all_factors else 0
                        for bin_idx in range(len(factors)):
                            param_mapping.append({
                                'order': coeff.order,
                                'bin': bin_idx,
                                'energy_range': f"[{self.energy_mesh[bin_idx]:.2e}, {self.energy_mesh[bin_idx+1]:.2e}]" if hasattr(self, 'energy_mesh') else f"bin_{bin_idx}",
                                'theoretical_std': self.std_dev_vector[start_idx + bin_idx] if start_idx + bin_idx < len(self.std_dev_vector) else 0
                            })
                    
                    if len(all_factors) <= sample_idx:
                        all_factors.append([])
                    all_factors[sample_idx].extend(factors)
        
        # Convert to numpy array
        factor_matrix = np.array(all_factors)  # Shape: (n_samples, n_parameters)
        
        if debug:
            print(f"Factor matrix shape: {factor_matrix.shape}")
        
        # Convert factors to log-space (since factors are multiplicative, log(factor) should be normally distributed)
        log_factors = np.log(factor_matrix)
        
        # Compute sample statistics
        sample_mean = np.mean(log_factors, axis=0)
        sample_std = np.std(log_factors, axis=0, ddof=1)  # Unbiased estimator
        sample_corr = np.corrcoef(log_factors.T)
        
        # Compare with theoretical values
        theoretical_std = self.std_dev_vector[:len(sample_std)]
        # Reconstruct correlation matrix from L_matrix (same approach as RML_RRR)
        theoretical_corr_full = self.L_matrix @ self.L_matrix.T
        theoretical_corr = theoretical_corr_full[:len(sample_std), :len(sample_std)]
        
        # Compute verification metrics
        std_error = np.abs(sample_std - theoretical_std)
        std_rel_error = std_error / (theoretical_std + 1e-10)  # Avoid division by zero
        
        # For correlation, compute element-wise differences
        corr_error = np.abs(sample_corr - theoretical_corr)
        
        results = {
            "sample_count": sample_count,
            "parameter_count": len(sample_std),
            "mean_convergence": np.max(np.abs(sample_mean)),  # Should be close to 0
            "std_max_error": np.max(std_error),
            "std_max_rel_error": np.max(std_rel_error),
            "std_rms_error": np.sqrt(np.mean(std_error**2)),
            "corr_max_error": np.max(corr_error),
            "corr_rms_error": np.sqrt(np.mean(corr_error**2)),
            "sample_std": sample_std,
            "theoretical_std": theoretical_std,
            "sample_corr": sample_corr,
            "theoretical_corr": theoretical_corr
        }
        
        if debug:
            print(f"\n📊 VERIFICATION RESULTS:")
            print(f"   Mean convergence (should be ≈0): {results['mean_convergence']:.6f}")
            print(f"   Std deviation:")
            print(f"     Max absolute error: {results['std_max_error']:.6f}")
            print(f"     Max relative error: {results['std_max_rel_error']:.1%}")
            print(f"     RMS error: {results['std_rms_error']:.6f}")
            print(f"   Correlation:")
            print(f"     Max absolute error: {results['corr_max_error']:.6f}")
            print(f"     RMS error: {results['corr_rms_error']:.6f}")
        
        # Show detailed comparison for first few parameters
        if debug and len(param_mapping) > 0:
            print(f"\n📋 DETAILED PARAMETER COMPARISON (first 10):")
            print(f"{'Order':<5} {'Bin':<3} {'Energy Range':<20} {'Theoretical σ':<12} {'Sample σ':<12} {'Rel Error':<10}")
            print("-" * 75)
            for i in range(min(10, len(param_mapping))):
                p = param_mapping[i]
                theo_std = theoretical_std[i] if i < len(theoretical_std) else 0
                samp_std = sample_std[i] if i < len(sample_std) else 0
                rel_err = abs(samp_std - theo_std) / (theo_std + 1e-10)
                print(f"L={p['order']:<3} {p['bin']:<3} {p['energy_range']:<20} {theo_std:<12.6f} {samp_std:<12.6f} {rel_err:<10.1%}")
        
        return results
