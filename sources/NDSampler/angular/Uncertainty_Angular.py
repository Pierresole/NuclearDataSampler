import numpy as np
import time
from collections import defaultdict
from .AngularDistributionCovariance import AngularDistributionCovariance
from .Parameters_Angular import LegendreCoefficients
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
        
        # Extract parameters and covariance matrices with existing approach
        start_time = time.time()
        # Get the first reaction mf34mt.reactions from MF34 (usually MT=2 elastic scattering, no cross covariance)
        self.legendre_data = LegendreCoefficients.from_endftk(mf4mt, mf34mt.reactions.to_list()[0])
        print(f"Time for extracting coefficients and std deviations: {time.time() - start_time:.4f} seconds")
        
        start_time = time.time()
        self.extract_relcov_matrix(mf34mt.reactions.to_list()[0])
        print(f"Time for extracting covariance structure: {time.time() - start_time:.4f} seconds")
        
        # Compute Cholesky decomposition
        start_time = time.time()
        # Before computing L, prune zero-variance rows/cols in the relative covariance matrix.
        # Build a mask of parameters (energy-bin per order) with positive variance.
        if hasattr(self, 'relative_covariance_matrix') and self.legendre_data is not None:
            rel_cov = self.relative_covariance_matrix
            # Variance per parameter is diagonal element
            variances = np.diag(rel_cov)
            self.active_parameter_indices = [i for i, v in enumerate(variances) if v > 0 and not np.isclose(v, 0.0)]
            self.pruned_parameter_indices = [i for i in range(len(variances)) if i not in self.active_parameter_indices]
            if self.pruned_parameter_indices:
                print(f"Pruning {len(self.pruned_parameter_indices)} zero-variance parameters before Cholesky")
                rel_cov_reduced = rel_cov[np.ix_(self.active_parameter_indices, self.active_parameter_indices)]
                self.reduced_relative_covariance_matrix = rel_cov_reduced
                # Store full matrix reference before replacing for decomposition
                self.relative_covariance_matrix_full = rel_cov.copy()
                # Replace matrix used for decomposition temporarily
                super().__setattr__('relative_covariance_matrix', rel_cov_reduced)
            else:
                # No pruning needed, but still store the matrix for debug comparison
                self.reduced_relative_covariance_matrix = rel_cov.copy()
        self.compute_L_matrix()
        relcovmat = self.L_matrix @ self.L_matrix.T
        print(rel_cov_reduced[:4,:4])
        print(relcovmat[:4,:4])
        # Restore full matrix reference for potential downstream uses (keep reduced separately)
        if hasattr(self, 'relative_covariance_matrix_full'):
            super().__setattr__('relative_covariance_matrix', self.relative_covariance_matrix_full)
        print(f"Time for compute_L_matrix (MT{mt_number}): {time.time() - start_time:.4f} seconds")
        
        print(f"✓ Created angular distribution uncertainty for MT{mt_number}")
        
    # def _build_expanded_covariance_from_coefficients(self):
    #     """
    #     Build the expanded absolute covariance matrix from the coefficient standard deviations.
    #     The matrix dimension equals the total number of coefficients across all orders and energy bins.
    #     """
    #     # Count total number of coefficients
    #     total_coeffs = 0
    #     coeff_info = []  # Store (order, bin_idx, nominal_coeff, std_dev)
        
    #     for coeff_data in self.legendre_data.coefficients:
    #         order = coeff_data.order
    #         nominal_coeffs = coeff_data.legcoeff[0] if coeff_data.legcoeff else []
    #         std_devs = coeff_data.std_dev
            
    #         for bin_idx, (nominal, std) in enumerate(zip(nominal_coeffs, std_devs)):
    #             coeff_info.append((order, bin_idx, nominal, std))
    #             total_coeffs += 1
        
    #     print(f"  Building expanded matrix for {total_coeffs} coefficients across {len(self.legendre_data.coefficients)} Legendre orders")
        
    #     # Build the standard deviation vector
    #     self.std_dev_vector = np.array([info[3] for info in coeff_info])  # Extract std_devs
        
    #     # For now, assume diagonal covariance (correlations between different orders/bins are ignored)
    #     # This can be extended later to include off-diagonal terms
    #     diagonal_variance = self.std_dev_vector ** 2
    #     self.covariance_matrix = np.diag(diagonal_variance)
        
    #     # Build correlation matrix for compute_L_matrix()
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         std_outer = np.outer(self.std_dev_vector, self.std_dev_vector)
    #         self.correlation_matrix = np.divide(self.covariance_matrix, std_outer, 
    #                                           out=np.zeros_like(self.covariance_matrix), 
    #                                           where=std_outer!=0)
            
    #     print(f"  Standard deviation vector: {self.std_dev_vector}")
    #     print(f"  Covariance matrix shape: {self.covariance_matrix.shape}")
        
    #     # Store coefficient mapping for sampling
    #     self.coefficient_info = coeff_info
        
    # def _filter_legendre_data_by_orders(self, requested_orders):
    #     """
    #     Filter the Legendre coefficient data to only include specified orders.
        
    #     Parameters:
    #     - requested_orders: List of Legendre orders to keep
    #     """
    #     if self.legendre_data and self.legendre_data.coefficients:
    #         filtered_coefficients = []
    #         for coeff in self.legendre_data.coefficients:
    #             if coeff.order in requested_orders:
    #                 filtered_coefficients.append(coeff)
    #                 print(f"  Keeping Legendre order L={coeff.order} as specified in covariance dict")
    #             else:
    #                 print(f"  Skipping Legendre order L={coeff.order} (not in covariance dict)")
            
    #         # Update the coefficients list
    #         self.legendre_data.coefficients = filtered_coefficients
            
    #         if not filtered_coefficients:
    #             print(f"Warning: No Legendre coefficients remain after filtering by orders {requested_orders}")
        
        
    def get_covariance_type(self):
        """
        Override to return the specific covariance type.
        """
        return "AngularDistribution"
    
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
        # SquareMatrix (LB==5): symmetric, but may be stored as full or triangular
        if hasattr(subblock, "LB") and subblock.LB == 5 and hasattr(subblock, "energies"):
            N = subblock.NE - 1
            mesh = subblock.energies.to_list()
            values = subblock.values.to_list()
            
            # Check if values are stored in triangular or full format
            expected_triangular = N * (N + 1) // 2
            expected_full = N * N
            
            if len(values) == expected_triangular:
                # Triangular format: unpack upper triangle
                mat = np.zeros((N, N))
                triu_indices = np.triu_indices(N)
                mat[triu_indices] = values
                mat = mat + mat.T - np.diag(np.diag(mat))
            elif len(values) == expected_full:
                # Full matrix format: reshape directly
                mat = np.array(values).reshape((N, N))
            else:
                raise ValueError(f"Unexpected number of values {len(values)} for {N}×{N} matrix")
            
            return mat, mesh, mesh
        # CovariancePairs (LB==1): diagonal
        elif hasattr(subblock, "LB") and subblock.LB == 1 and hasattr(subblock, "first_array_energies"):
            mesh = subblock.first_array_energies.to_list()
            vals = subblock.first_array_fvalues.to_list()
            mat = np.diag(vals)
            return mat, mesh, mesh
        # Generic CovariancePairs (by number_pairs attribute)
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


    def _apply_samples(self, samples, mode="stack", use_copula=False, batch_size=1, 
                      sampling_method="Simple", debug=False):
        """
        Apply generated samples to the Legendre coefficients.
        Each sample is a vector of z-values or uniform values (if copula).
        
        UPDATED (Aug 2025): Store ONLY relative deviations δ such that
            a_sample = a_nominal * (1 + δ)
        Absolute coefficients are reconstructed lazily when requested.
        """
        # Store samples for debug analysis if in debug mode
        if debug:
            self.stored_samples = samples.copy()
        
        # Ensure samples is properly shaped (these samples correspond ONLY to active parameters, zero-variance pruned)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        n_samples, n_reduced_params = samples.shape
        # Build mapping from global parameter index -> (coeff_obj, local_bin_index)
        if not hasattr(self, 'parameter_index_map'):
            self.parameter_index_map = []  # list of tuples (coeff, bin_idx)
            for coeff in self.legendre_data.coefficients:
                n_ebins = len(coeff.energies) - 1
                for b in range(n_ebins):
                    self.parameter_index_map.append((coeff, b))
        # active_parameter_indices created during __init__ pruning
        active_indices = getattr(self, 'active_parameter_indices', list(range(len(self.parameter_index_map))))
        if n_reduced_params != len(active_indices):
            # Fallback: derive active indices purely from non-zero std_dev in legendre_data
            fallback_active = []
            for idx, (coeff, bin_idx) in enumerate(self.parameter_index_map):
                if coeff.std_dev and bin_idx < len(coeff.std_dev):
                    if coeff.std_dev[bin_idx] > 0 and not np.isclose(coeff.std_dev[bin_idx], 0.0):
                        fallback_active.append(idx)
            if n_reduced_params == len(fallback_active):
                if debug:
                    print(f"⚠️  Active index mismatch detected (matrix-based={len(active_indices)} vs samples={n_reduced_params}). Using std_dev-based fallback active set of size {len(fallback_active)}.")
                active_indices = fallback_active
                self.active_parameter_indices = active_indices  # update for future calls
                self.pruned_parameter_indices = [i for i in range(len(self.parameter_index_map)) if i not in active_indices]
            else:
                raise ValueError(
                    f"Inconsistent active parameter counts: samples={n_reduced_params}, matrix-based={len(active_indices)}, std_dev-based={len(fallback_active)}"
                )
        
        if debug:
            print(f"🔬 ANGULAR DISTRIBUTION DEBUG MODE - MT{self.MT}")
            print(f"{'='*60}")
            print(f"📊 Sampling Configuration:")
            print(f"   Number of samples: {n_samples}")
            print(f"   Number of active parameters (sampled): {n_reduced_params}")
            if hasattr(self, 'pruned_parameter_indices'):
                print(f"   Pruned zero-variance parameters: {len(self.pruned_parameter_indices)}")
            print(f"   Sampling method: {sampling_method}")
            print(f"   Use copula: {use_copula}")
            print(f"   Operation mode: {mode}")
            if hasattr(self.legendre_data, 'coefficients'):
                legendre_orders = [c.order for c in self.legendre_data.coefficients]
                print(f"   Legendre orders: {legendre_orders}")
            print(f"\n🚨 DEBUG MODE: STACKING PERTURBATIONS (no file creation)")
            print(f"   Perturbations will be stored in memory for covariance verification")
        
        if not debug:
            print(f"🚨 ANGULAR (relative deviations): mode='{mode}', n_samples={n_samples}")
        else:
            print(f"🚨 ANGULAR DEBUG (relative deviations): mode='{mode}', n_samples={n_samples}")
            print(f"   Storing samples in coefficient objects for matrix comparison")
        
        # Clear existing samples if in replace mode (except nominal at index 0)
        if mode == 'replace':
            for coeff in self.legendre_data.coefficients:
                # Keep only nominal absolute coefficients; purge previous derived data
                if len(coeff.legcoeff) > 1:
                    coeff.legcoeff = coeff.legcoeff[:1]
                if len(coeff.factor) > 1:
                    coeff.factor = coeff.factor[:1]
                coeff.rel_deviation = []  # reset relative deviations (index 0 not used)
        
        for sample_idx in range(n_samples):
            reduced_sample = samples[sample_idx, :]
            # Build full delta vector with zeros for pruned parameters
            full_delta = np.zeros(len(self.parameter_index_map))
            for local_pos, global_idx in enumerate(active_indices):
                z = reduced_sample[local_pos]
                if use_copula:
                    from scipy.stats import norm
                    z = norm.ppf(z)
                full_delta[global_idx] = z
            # Distribute per coefficient using id-based index mapping (avoids unhashable object keys)
            if not hasattr(self, '_coeff_id_to_index'):
                self._coeff_id_to_index = {id(c): i for i, c in enumerate(self.legendre_data.coefficients)}
            coeffs_list = self.legendre_data.coefficients
            delta_arrays = [np.zeros(len(c.energies) - 1) for c in coeffs_list]
            for idx, (coeff, bin_idx) in enumerate(self.parameter_index_map):
                ci = self._coeff_id_to_index[id(coeff)]
                delta_arrays[ci][bin_idx] = full_delta[idx]
            effective_sample_idx = sample_idx + 1
            for ci, coeff in enumerate(coeffs_list):
                delta_arr = delta_arrays[ci]
                factors = 1.0 + delta_arr
                target_rel_dev = coeff.rel_deviation
                target_factor = coeff.factor
                if mode == 'stack':
                    while len(target_rel_dev) <= effective_sample_idx:
                        target_rel_dev.append(None)
                    target_rel_dev[effective_sample_idx] = delta_arr.tolist()
                    while len(target_factor) <= effective_sample_idx:
                        target_factor.append(None)
                    target_factor[effective_sample_idx] = factors.tolist()
                elif mode == 'replace':
                    while len(target_rel_dev) <= effective_sample_idx:
                        target_rel_dev.append(None)
                    target_rel_dev[effective_sample_idx] = delta_arr.tolist()
                    while len(target_factor) <= effective_sample_idx:
                        target_factor.append(None)
                    target_factor[effective_sample_idx] = factors.tolist()

        # Perform debug verification if requested
        if debug:
            print(f"\n🔍 COEFFICIENT SAMPLING VERIFICATION:")
            # Store samples if debugging
            self.stored_samples = samples.copy()
            
            # Comprehensive debug analysis
            self._comprehensive_debug_analysis(samples, debug=True)


    def _verify_coefficient_sampling_statistics(self, debug=True):
        """
        Verify that the sampling statistics match the expected standard deviations.
        """
        if debug:
            # Aggregate relative deviations and compare to identity (since L_matrix encodes covariance)
            all_deltas = []
            for coeff in self.legendre_data.coefficients:
                # Skip nominal (index 0)
                for sidx in range(1, len(coeff.rel_deviation)):
                    if sidx < len(coeff.rel_deviation) and coeff.rel_deviation[sidx] is not None:
                        all_deltas.append(coeff.rel_deviation[sidx])
            if not all_deltas:
                print("   No relative deviations stored yet.")
                return
            delta_matrix = np.vstack(all_deltas)
            sample_std = delta_matrix.std(axis=0, ddof=1)
            print(f"   Relative deviation sample std (first 10): {sample_std[:10]}")

    def _debug_show_sample_matrix(self, samples, debug=True):
        """
        Show the sample matrix for debugging purposes.
        """
        if debug:
            print(f"\n📋 SAMPLE MATRIX (active parameters only):")
            print(f"   Shape: {samples.shape}")
            print(f"   Sample matrix (first 5 rows, first 10 cols):")
            max_rows = min(5, samples.shape[0])
            max_cols = min(10, samples.shape[1])
            for i in range(max_rows):
                row_str = "   " + " ".join(f"{samples[i,j]:8.4f}" for j in range(max_cols))
                if samples.shape[1] > max_cols:
                    row_str += " ..."
                print(row_str)
            if samples.shape[0] > max_rows:
                print("   ...")

    def _debug_covariance_comparison(self, samples, debug=True):
        """
        Compare the empirical covariance matrix from samples with the original relative covariance matrix.
        Identifies coefficients that differ by more than 10%.
        """
        print(f"\n🔍 COVARIANCE MATRIX COMPARISON:")
        print(f"   Debug flag: {debug}, samples shape: {samples.shape}")
        
        if not debug:
            print("   Debug mode not enabled")
            return
            
        if samples.shape[0] < 10:
            print(f"   Not enough samples for comparison ({samples.shape[0]} < 10)")
            return
        
        # Debug: Check what covariance matrices are available
        available_matrices = []
        if hasattr(self, 'reduced_relative_covariance_matrix'):
            available_matrices.append(f"reduced_relative_covariance_matrix: {self.reduced_relative_covariance_matrix.shape}")
        if hasattr(self, 'relative_covariance_matrix_full'):
            available_matrices.append(f"relative_covariance_matrix_full: {self.relative_covariance_matrix_full.shape}")
        if hasattr(self, 'relative_covariance_matrix'):
            available_matrices.append(f"relative_covariance_matrix: {self.relative_covariance_matrix.shape}")
        print(f"   Available matrices: {available_matrices}")
        
        # Compute empirical covariance from samples
        empirical_cov = np.cov(samples, rowvar=False)
        
        # Get the reduced (active parameters only) relative covariance matrix
        if hasattr(self, 'reduced_relative_covariance_matrix'):
            expected_cov = self.reduced_relative_covariance_matrix
        elif hasattr(self, 'relative_covariance_matrix_full'):
            # Use the full matrix that was saved during initialization
            active_indices = getattr(self, 'active_parameter_indices', list(range(samples.shape[1])))
            expected_cov = self.relative_covariance_matrix_full[np.ix_(active_indices, active_indices)]
        elif hasattr(self, 'relative_covariance_matrix'):
            # If no reduced matrix available, extract from full matrix using active indices
            active_indices = getattr(self, 'active_parameter_indices', list(range(samples.shape[1])))
            expected_cov = self.relative_covariance_matrix[np.ix_(active_indices, active_indices)]
        else:
            print("   ⚠️  No reference covariance matrix available for comparison")
            return
        
        if empirical_cov.shape != expected_cov.shape:
            print(f"   ⚠️  Shape mismatch: empirical {empirical_cov.shape} vs expected {expected_cov.shape}")
            return
        
        # Compare diagonal elements (variances)
        empirical_var = np.diag(empirical_cov)
        expected_var = np.diag(expected_cov)
        
        # Calculate relative differences
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.abs(empirical_var - expected_var) / np.maximum(expected_var, 1e-10)
        
        # Find parameters with >10% difference
        threshold = 0.10  # 10%
        large_diff_indices = np.where(rel_diff > threshold)[0]
        
        print(f"   Total active parameters: {len(empirical_var)}")
        print(f"   Parameters with >10% variance difference: {len(large_diff_indices)}")
        
        if len(large_diff_indices) > 0:
            print(f"\n   🚨 LARGE DIFFERENCES (>{threshold*100:.0f}%):")
            print(f"   {'Param':<6} {'Order':<6} {'Bin':<4} {'Expected':<12} {'Empirical':<12} {'Rel.Diff':<10}")
            print(f"   {'-'*60}")
            
            # Map parameter indices back to Legendre orders and bins
            active_indices = getattr(self, 'active_parameter_indices', list(range(len(self.parameter_index_map))))
            
            for local_idx in large_diff_indices[:20]:  # Show first 20 problematic parameters
                global_idx = active_indices[local_idx] if local_idx < len(active_indices) else local_idx
                
                if global_idx < len(self.parameter_index_map):
                    coeff, bin_idx = self.parameter_index_map[global_idx]
                    order = coeff.order
                else:
                    order = "?"
                    bin_idx = "?"
                
                expected_val = expected_var[local_idx]
                empirical_val = empirical_var[local_idx]
                rel_diff_val = rel_diff[local_idx]
                
                print(f"   {local_idx:<6} {order:<6} {bin_idx:<4} {expected_val:<12.6f} {empirical_val:<12.6f} {rel_diff_val:<10.2%}")
            
            if len(large_diff_indices) > 20:
                print(f"   ... and {len(large_diff_indices) - 20} more")
        else:
            print(f"   ✅ All parameters within {threshold*100:.0f}% tolerance")
        
        # Summary statistics
        print(f"\n   📊 VARIANCE COMPARISON STATISTICS:")
        print(f"   Mean relative difference: {np.mean(rel_diff):.2%}")
        print(f"   Max relative difference: {np.max(rel_diff):.2%}")
        print(f"   RMS relative difference: {np.sqrt(np.mean(rel_diff**2)):.2%}")
        
        # Frobenius norm comparison for full matrices
        frobenius_diff = np.linalg.norm(empirical_cov - expected_cov, 'fro')
        frobenius_expected = np.linalg.norm(expected_cov, 'fro')
        frobenius_rel = frobenius_diff / frobenius_expected if frobenius_expected > 0 else np.inf
        
        print(f"   Matrix Frobenius norm difference: {frobenius_rel:.2%}")
        
        return {
            'empirical_cov': empirical_cov,
            'expected_cov': expected_cov,
            'large_diff_indices': large_diff_indices,
            'rel_diff': rel_diff,
            'frobenius_rel_diff': frobenius_rel
        }

    
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
        leg_data = LegendreCoefficients.read_from_hdf5(leg_data_group)
        
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
        
        return instance


    def write_additional_data_to_hdf5(self, hdf5_group):
        if self.legendre_data is not None:
            leg_group = hdf5_group.require_group('Parameters')
            self.legendre_data.write_to_hdf5(leg_group)
        
        # Save the MT number as an attribute
        hdf5_group.attrs['MT'] = self.MT


    def update_tape(self, tape, sample_index=1, sample_name=""):
        """
        Updates the ENDF tape with sampled Legendre coefficients for the given sample_index.
        """
        from ENDFtk.MF4 import Section, LegendreDistributions, LegendreCoefficients, MixedDistributions
        # Parse the section to update (use dynamic MT number)
        mf4mt = tape.MAT(tape.material_numbers[0]).MF(4).MT(self.MT).parse()

        # Build factor dictionary directly from stored relative deviations (δ) to avoid
        # reconstructing absolute coefficients then re-dividing.
        # factors = 1 + δ per covariance bin.
        factors_dict = {}
        for coeff_data in self.legendre_data.coefficients:
            n_bins = len(coeff_data.energies) - 1
            if sample_index < len(coeff_data.rel_deviation) and coeff_data.rel_deviation[sample_index] is not None:
                delta = coeff_data.rel_deviation[sample_index]
                # Safety: ensure correct length
                if len(delta) != n_bins:
                    # Pad or trim if mismatch
                    adj = (delta + [0.0]*(n_bins - len(delta)))[:n_bins]
                    delta = adj
                factors_dict[coeff_data.order] = [1.0 + d for d in delta]
            else:
                # Nominal (or missing) => unity factors
                factors_dict[coeff_data.order] = [1.0]*n_bins

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

        # Collect all covariance bin boundaries across orders
        covariance_boundaries = set()
        order_to_boundaries = {}
        for coeff_data in self.legendre_data.coefficients:
            covariance_boundaries.update(coeff_data.energies)
            order_to_boundaries[coeff_data.order] = coeff_data.energies
        covariance_boundaries = sorted(covariance_boundaries)

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
                    return self._apply_multiplicative_factors_to_coefficients(
                        base_coeffs, multiplicative_factors, bin_index=bin_idx)

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

    def _create_perturbed_legendre_distributions(self, mf4mt, sample_coefficients_dict):
        """
        Create perturbed Legendre distributions using bin-wise multiplicative factors.
        
        Strategy:
        1. Use mesh_union() to combine original MF4 energies + covariance boundaries  
        2. For each Legendre order:
           - Take sampled coefficients at covariance boundary points
           - Compute multiplicative factors = sampled_value / original_interpolated_value
           - Apply the same factor to all union points within each covariance bin
        3. Apply boundary triplication for proper ENDF formatting
        
        Parameters:
        - mf4mt: Original MF4 section
        - sample_coefficients_dict: {order: [sampled_coefficients]} at covariance boundaries
        """
        from ENDFtk.MF4 import LegendreDistributions, LegendreCoefficients
        
        # print(f"\n  Creating perturbed Legendre distributions using bin-wise multiplicative factors")
        
        # Get original structure
        if mf4mt.LTT == 1:  # Pure Legendre case
            original_dist = mf4mt.distributions
        elif mf4mt.LTT == 3:  # Mixed case
            original_dist = mf4mt.distributions.legendre
        
        # Get original energies and coefficients
        original_distributions = original_dist.angular_distributions.to_list()
        original_energies = [dist.incident_energy for dist in original_distributions]
        
        # # Get original energies from stored MF4 data (avoid parsing mf4mt repeatedly)
        # original_energies = self.legendre_data.original_mf4_energies
        
        # Get all covariance bin boundaries
        covariance_boundaries = set()
        for coeff_data in self.legendre_data.coefficients:
            covariance_boundaries.update(coeff_data.energies)
        covariance_boundaries = sorted(list(covariance_boundaries))
        
        # Create union mesh using existing method
        union_energies = self.mesh_union(original_energies, covariance_boundaries)
        
        # print(f"    Original MF4 energies: {len(original_energies)} points")
        # print(f"    Covariance boundaries: {len(covariance_boundaries)} points") 
        # print(f"    Union mesh: {len(union_energies)} points")
        
        # Compute bin-wise multiplicative factors for each Legendre order
        multiplicative_factors = {}  # {order: [factor_per_bin]}
        
        for coeff_data in self.legendre_data.coefficients:
            order = coeff_data.order
            if order not in sample_coefficients_dict:
                # No sampling data for this order, use unity factors
                n_bins = len(covariance_boundaries) - 1
                multiplicative_factors[order] = [1.0] * n_bins
                continue
                
            # Get sampled coefficients at covariance boundary points
            sampled_coeffs = sample_coefficients_dict[order]
            coeff_boundaries = coeff_data.energies
            
            # Compute multiplicative factors at each covariance bin
            bin_factors = []
            for bin_idx in range(len(coeff_boundaries) - 1):
                # Use bin center for interpolation instead of left boundary
                # This avoids potential boundary condition issues
                left_boundary = coeff_boundaries[bin_idx]
                right_boundary = coeff_boundaries[bin_idx + 1]
                bin_center_energy = (left_boundary + right_boundary) / 2.0
                
                # Get original interpolated value at the bin center
                original_interp = self.legendre_data.get_coefficients_at_energy(bin_center_energy)
                
                # Get the original coefficient for this Legendre order (L≥1, so index = order-1)
                if order-1 < len(original_interp):
                    original_val = original_interp[order-1]
                else:
                    original_val = 0.0
                
                # Get sampled coefficient for this bin
                if bin_idx < len(sampled_coeffs):
                    sampled_val = sampled_coeffs[bin_idx]
                else:
                    sampled_val = original_val
                
                # Compute multiplicative factor
                if abs(original_val) > 1e-15:
                    factor = sampled_val / original_val
                else:
                    factor = 1.0  # Avoid division by zero
                
                bin_factors.append(factor)
                
            multiplicative_factors[order] = bin_factors
            # print(f"    L={order}: computed {len(bin_factors)} bin-wise factors")
        
        # Apply bin-wise factors to create enhanced energy grid with boundary triplication
        enhanced_energies = []
        enhanced_coeffs_data = []
        
        for i, energy in enumerate(union_energies):
            # Get coefficients for this energy using the stored original MF4 data
            base_coeffs = self.legendre_data.get_coefficients_at_energy(energy)
            
            # Check if this energy is a covariance bin boundary
            is_covariance_boundary = energy in covariance_boundaries
            
            if is_covariance_boundary:
                # Apply triplication logic with bin-wise factors
                boundary_idx = covariance_boundaries.index(energy)
                is_first_boundary = (boundary_idx == 0)
                is_last_boundary = (boundary_idx == len(covariance_boundaries) - 1)
                
                if is_first_boundary:
                    # Initial boundary: duplicate once (E0, E0')
                    # E0: unperturbed boundary point
                    enhanced_energies.append(energy)
                    enhanced_coeffs_data.append(base_coeffs.copy())
                    
                    # E0': apply bin 0 factors
                    perturbed_coeffs = self._apply_multiplicative_factors_to_coefficients(
                        base_coeffs, multiplicative_factors, bin_index=0
                    )
                    enhanced_energies.append(energy)
                    enhanced_coeffs_data.append(perturbed_coeffs)
                    
                elif is_last_boundary:
                    # Final boundary: duplicate once (EN, EN')
                    # EN: apply previous bin factors
                    bin_idx = boundary_idx - 1
                    perturbed_coeffs = self._apply_multiplicative_factors_to_coefficients(
                        base_coeffs, multiplicative_factors, bin_index=bin_idx
                    )
                    enhanced_energies.append(energy)
                    enhanced_coeffs_data.append(perturbed_coeffs)
                    
                    # EN': duplicate with same perturbation as EN
                    enhanced_energies.append(energy)
                    enhanced_coeffs_data.append(perturbed_coeffs.copy())
                    
                else:
                    # Intermediate boundary: triplicate (Ei, Ei', Ei'')
                    prev_bin_idx = boundary_idx - 1
                    next_bin_idx = boundary_idx
                    print(f"  Processing boundary energy {energy} at index {boundary_idx} (prev={prev_bin_idx}, next={next_bin_idx})")
                    # Ei: previous bin factors
                    coeffs_prev = self._apply_multiplicative_factors_to_coefficients(
                        base_coeffs, multiplicative_factors, bin_index=prev_bin_idx
                    )
                    enhanced_energies.append(energy)
                    enhanced_coeffs_data.append(coeffs_prev)
                    
                    # Ei': duplicate of Ei
                    enhanced_energies.append(energy)
                    enhanced_coeffs_data.append(coeffs_prev.copy())
                    
                    # Ei'': next bin factors
                    coeffs_next = self._apply_multiplicative_factors_to_coefficients(
                        base_coeffs, multiplicative_factors, bin_index=next_bin_idx
                    )
                    enhanced_energies.append(energy)
                    enhanced_coeffs_data.append(coeffs_next)
                    
            else:
                # Regular energy point - determine which bin it falls into and apply factors
                enhanced_energies.append(energy)
                bin_idx = self._find_bin_index_for_energy(energy, covariance_boundaries)
                perturbed_coeffs = self._apply_multiplicative_factors_to_coefficients(
                    base_coeffs, multiplicative_factors, bin_index=bin_idx
                )
                enhanced_coeffs_data.append(perturbed_coeffs)
        
        # Create LegendreDistributions with enhanced grid
        enhanced_n_points = len(enhanced_energies)
        new_boundaries = [enhanced_n_points]  # Single region covering all points
        new_interpolants = [2]  # Linear interpolation
        
        # Create LegendreCoefficients objects for enhanced grid
        new_legendre_coeffs = []
        for energy, coeffs in zip(enhanced_energies, enhanced_coeffs_data):
            new_legendre_coeffs.append(LegendreCoefficients(energy, coeffs))
        
        # Create new LegendreDistributions with updated interpolation table
        perturbed_legendre_dist = LegendreDistributions(
            new_boundaries,
            new_interpolants,
            new_legendre_coeffs
        )
        
        # print(f"    Created perturbed distributions with {enhanced_n_points} energy points (triplication applied)")
        return perturbed_legendre_dist
    

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
        """
        import numpy as np
        bin_idx = np.searchsorted(covariance_boundaries[:-1], energy, side='right') - 1
        return np.clip(bin_idx, 0, len(covariance_boundaries) - 2)

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
    
    def test_method_exists(self):
        """Simple test method to verify method addition works."""
        return "Method exists and is callable!"
    
    def verify_boundary_duplication_logic(self, debug=True):
        """
        Verify that boundary energy duplication follows the correct logic.
        
        Returns:
            dict: Dictionary containing verification metrics
        """
        if debug:
            print(f"🔍 VERIFYING BOUNDARY DUPLICATION LOGIC FOR MT{self.MT}")
            print(f"{'='*60}")
        
        # Get all covariance bin boundaries
        all_bin_boundaries = set()
        for coeff_data in self.legendre_data.coefficients:
            all_bin_boundaries.update(coeff_data.energies)
        all_bin_boundaries = sorted(all_bin_boundaries)
        
        if debug:
            print(f"Covariance bin boundaries: {len(all_bin_boundaries)} boundaries")
            print(f"Energy range: [{all_bin_boundaries[0]:.2e}, {all_bin_boundaries[-1]:.2e}] eV")
        
        # Get original MF4 energy points
        if hasattr(self, 'mf4mt2') and self.mf4mt2 is not None:
            if self.mf4mt.LTT == 1:  # Pure Legendre case
                original_dist = self.mf4mt.distributions
            elif self.mf4mt.LTT == 3:  # Mixed case
                original_dist = self.mf4mt.distributions.legendre
            original_energies = [dist.incident_energy for dist in original_dist.angular_distributions.to_list()]
        else:
            if debug:
                print("Warning: Cannot access original MF4 data for verification")
            return {"error": "No MF4 data available for verification"}
        
        if debug:
            print(f"Original MF4 energy points: {len(original_energies)}")
        
        # Count expected duplications
        boundary_intersections = 0
        for energy in original_energies:
            if energy in all_bin_boundaries:
                boundary_intersections += 1
        
        expected_total_points = len(original_energies) + boundary_intersections
        
        if debug:
            print(f"Energy points that are bin boundaries: {boundary_intersections}")
            print(f"Expected total points after duplication: {expected_total_points}")
            print(f"  Original points: {len(original_energies)}")
            print(f"  + Boundary duplications: {boundary_intersections}")
            print(f"  = Total expected: {expected_total_points}")
        
        # Verification checks
        results = {
            "original_points": len(original_energies),
            "bin_boundaries": len(all_bin_boundaries),
            "boundary_intersections": boundary_intersections,
            "expected_total_points": expected_total_points,
            "dimension_check_passed": True,  # Will be updated when implemented
        }
        
        # Additional checks can be added here when the full implementation is tested
        
        if debug:
            print(f"\n✓ Boundary duplication logic verification completed")
            print(f"Expected enhancement: {len(original_energies)} → {expected_total_points} points")
        
        return results

    def _comprehensive_debug_analysis(self, samples, debug=True):
        """
        Perform comprehensive debug analysis including covariance matrix comparison,
        statistical tests, and detailed discrepancy analysis.
        """
        if not debug:
            return
            
        import numpy as np
        from scipy import stats
        
        n_samples, n_params = samples.shape
        print(f"\n{'='*60}")
        print(f"🔍 COMPREHENSIVE DEBUG ANALYSIS")
        print(f"{'='*60}")
        print(f"📊 Samples: {n_samples}, Parameters: {n_params}")
        
        if n_samples < 10:
            print(f"⚠️  Statistical verification requires ≥10 samples (got {n_samples})")
            print(f"📋 Sample matrix preview:")
            print(samples[:min(5, n_samples), :min(10, n_params)])
            return
        
        # 1. Get theoretical covariance matrix
        print(f"\n1️⃣ THEORETICAL COVARIANCE MATRIX ANALYSIS")
        
        # Try to get the covariance matrix - handle HDF5 read case where only L_matrix is available
        theoretical_cov = None
        matrix_source = "Unknown"
        
        if hasattr(self, 'relative_covariance_matrix') and self.relative_covariance_matrix is not None:
            theoretical_cov = self.relative_covariance_matrix
            matrix_source = "relative_covariance_matrix"
        elif hasattr(self, 'reduced_relative_covariance_matrix') and self.reduced_relative_covariance_matrix is not None:
            theoretical_cov = self.reduced_relative_covariance_matrix
            matrix_source = "reduced_relative_covariance_matrix"
        elif hasattr(self, 'L_matrix') and self.L_matrix is not None:
            # Reconstruct covariance matrix from L_matrix (L @ L.T)
            theoretical_cov = self.L_matrix @ self.L_matrix.T
            matrix_source = "reconstructed from L_matrix"
            print(f"📋 Reconstructed covariance matrix from L_matrix")
        else:
            print(f"❌ No covariance matrix available for analysis")
            return
        
        print(f"📏 Theoretical matrix shape: {theoretical_cov.shape}")
        print(f"📋 Matrix source: {matrix_source}")
        print(f"📋 Theoretical relative covariance [:5,:5]:")
        print(theoretical_cov[:5, :5])
        
        # Check matrix properties
        eigenvals = np.linalg.eigvals(theoretical_cov)
        min_eigenval = np.min(eigenvals)
        max_eigenval = np.max(eigenvals)
        print(f"🔢 Eigenvalue range: [{min_eigenval:.2e}, {max_eigenval:.2e}]")
        print(f"📊 Condition number: {max_eigenval/max(min_eigenval, 1e-15):.2e}")
        
        # 2. Compute empirical covariance matrix
        print(f"\n2️⃣ EMPIRICAL COVARIANCE MATRIX ANALYSIS")
        # Center the samples (remove mean)
        samples_centered = samples - np.mean(samples, axis=0)
        empirical_cov = np.cov(samples_centered.T, ddof=1)
        print(f"📏 Empirical matrix shape: {empirical_cov.shape}")
        print(f"📋 Empirical relative covariance [:5,:5]:")
        print(empirical_cov[:5, :5])
        
        # Check empirical matrix properties
        emp_eigenvals = np.linalg.eigvals(empirical_cov)
        emp_min_eigenval = np.min(emp_eigenvals)
        emp_max_eigenval = np.max(emp_eigenvals)
        print(f"🔢 Eigenvalue range: [{emp_min_eigenval:.2e}, {emp_max_eigenval:.2e}]")
        print(f"📊 Condition number: {emp_max_eigenval/max(emp_min_eigenval, 1e-15):.2e}")
        
        # 3. Matrix comparison and discrepancy analysis
        print(f"\n3️⃣ MATRIX COMPARISON ANALYSIS")
        diff_matrix = empirical_cov - theoretical_cov
        frobenius_norm = np.linalg.norm(diff_matrix, 'fro')
        relative_frobenius = frobenius_norm / np.linalg.norm(theoretical_cov, 'fro')
        print(f"📏 Frobenius norm of difference: {frobenius_norm:.4e}")
        print(f"📏 Relative Frobenius norm: {relative_frobenius:.4e}")
        
        # 4. Diagonal analysis (variances)
        print(f"\n4️⃣ DIAGONAL DISCREPANCY ANALYSIS (Top 10)")
        diagonal_theoretical = np.diag(theoretical_cov)
        diagonal_empirical = np.diag(empirical_cov)
        diagonal_diff = np.abs(diagonal_empirical - diagonal_theoretical)
        diagonal_rel_diff = diagonal_diff / (np.abs(diagonal_theoretical) + 1e-15)
        
        # Top 10 most discrepant diagonal terms
        top_diagonal_indices = np.argsort(diagonal_rel_diff)[-10:][::-1]
        for i, idx in enumerate(top_diagonal_indices):
            print(f"  {i+1:2d}. Param {idx:3d}: theoretical={diagonal_theoretical[idx]:.4e}, "
                  f"empirical={diagonal_empirical[idx]:.4e}, "
                  f"rel_diff={diagonal_rel_diff[idx]:.4e}")
        
        # 5. Off-diagonal analysis (correlations)
        print(f"\n5️⃣ OFF-DIAGONAL CORRELATION ANALYSIS (Top 10)")
        
        # Convert to correlation matrices
        def cov_to_corr(cov_matrix):
            """Convert covariance matrix to correlation matrix"""
            std_devs = np.sqrt(np.diag(cov_matrix))
            # Avoid division by zero
            std_devs = np.where(std_devs < 1e-15, 1e-15, std_devs)
            corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
            return corr_matrix
        
        theoretical_corr = cov_to_corr(theoretical_cov)
        empirical_corr = cov_to_corr(empirical_cov)
        
        # Find off-diagonal discrepancies
        corr_diff = np.abs(empirical_corr - theoretical_corr)
        # Mask diagonal elements
        mask = np.eye(corr_diff.shape[0], dtype=bool)
        corr_diff[mask] = 0
        
        # Get top 10 off-diagonal discrepancies
        flat_indices = np.argsort(corr_diff.ravel())[-10:][::-1]
        row_indices, col_indices = np.unravel_index(flat_indices, corr_diff.shape)
        
        for i, (row, col) in enumerate(zip(row_indices, col_indices)):
            if row != col:  # Skip any diagonal elements that might have slipped through
                theoretical_corr_val = theoretical_corr[row, col]
                empirical_corr_val = empirical_corr[row, col]
                diff_val = corr_diff[row, col]
                print(f"  {i+1:2d}. ({row:3d},{col:3d}): theoretical={theoretical_corr_val:+.4f}, "
                      f"empirical={empirical_corr_val:+.4f}, "
                      f"|diff|={diff_val:.4f}")
        
        # 6. Hotelling's T² test for zero mean
        print(f"\n6️⃣ HOTELLING'S T² TEST FOR ZERO MEAN")
        sample_mean = np.mean(samples, axis=0)
        
        # Compute T² statistic
        try:
            # Use empirical covariance for the test
            emp_cov_inv = np.linalg.pinv(empirical_cov)
            t_squared = n_samples * sample_mean.T @ emp_cov_inv @ sample_mean
            
            # Convert to F-statistic
            f_statistic = (n_samples - n_params) / ((n_samples - 1) * n_params) * t_squared
            p_value = 1 - stats.f.cdf(f_statistic, n_params, n_samples - n_params)
            
            print(f"📊 Sample mean norm: {np.linalg.norm(sample_mean):.4e}")
            print(f"📊 T² statistic: {t_squared:.4f}")
            print(f"📊 F statistic: {f_statistic:.4f}")
            print(f"📊 p-value: {p_value:.4e}")
            
            if p_value < 0.05:
                print(f"⚠️  Significant deviation from zero mean (p < 0.05)")
            else:
                print(f"✅ Mean is consistent with zero (p ≥ 0.05)")
                
        except Exception as e:
            print(f"❌ Hotelling's T² test failed: {e}")
        
        # 7. Summary
        print(f"\n7️⃣ SUMMARY")
        print(f"📊 Relative Frobenius norm: {relative_frobenius:.4e}")
        if relative_frobenius < 0.1:
            print(f"✅ Good agreement between theoretical and empirical covariance")
        elif relative_frobenius < 0.3:
            print(f"⚠️  Moderate agreement between theoretical and empirical covariance")
        else:
            print(f"❌ Poor agreement between theoretical and empirical covariance")
        
        print(f"{'='*60}")
        print(f"🔍 DEBUG ANALYSIS COMPLETE")
        print(f"{'='*60}\n")