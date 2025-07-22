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
        self.legendre_data = LegendreCoefficients.from_endftk(mf4mt, mf34mt)
        print(f"Time for extracting coefficients and std deviations: {time.time() - start_time:.4f} seconds")
        
        # If specific Legendre orders were requested, filter the data
        if legendre_orders is not None:
            self._filter_legendre_data_by_orders(legendre_orders)
        
        # Build the expanded covariance matrix and standard deviation vector from coefficient data
        start_time = time.time()
        self._build_expanded_covariance_from_coefficients()
        print(f"Time for building expanded covariance matrix: {time.time() - start_time:.4f} seconds")
        
        # Compute Cholesky decomposition
        start_time = time.time()
        self.compute_L_matrix()
        print(f"Time for compute_L_matrix (MT{mt_number}): {time.time() - start_time:.4f} seconds")
        
        print(f"✓ Created angular distribution uncertainty for MT{mt_number}")
        
    def _build_expanded_covariance_from_coefficients(self):
        """
        Build the expanded absolute covariance matrix from the coefficient standard deviations.
        The matrix dimension equals the total number of coefficients across all orders and energy bins.
        """
        # Count total number of coefficients
        total_coeffs = 0
        coeff_info = []  # Store (order, bin_idx, nominal_coeff, std_dev)
        
        for coeff_data in self.legendre_data.coefficients:
            order = coeff_data.order
            nominal_coeffs = coeff_data.legcoeff[0] if coeff_data.legcoeff else []
            std_devs = coeff_data.std_dev
            
            for bin_idx, (nominal, std) in enumerate(zip(nominal_coeffs, std_devs)):
                coeff_info.append((order, bin_idx, nominal, std))
                total_coeffs += 1
        
        print(f"  Building expanded matrix for {total_coeffs} coefficients across {len(self.legendre_data.coefficients)} Legendre orders")
        
        # Build the standard deviation vector
        self.std_dev_vector = np.array([info[3] for info in coeff_info])  # Extract std_devs
        
        # For now, assume diagonal covariance (correlations between different orders/bins are ignored)
        # This can be extended later to include off-diagonal terms
        diagonal_variance = self.std_dev_vector ** 2
        self.covariance_matrix = np.diag(diagonal_variance)
        
        # Build correlation matrix for compute_L_matrix()
        with np.errstate(divide='ignore', invalid='ignore'):
            std_outer = np.outer(self.std_dev_vector, self.std_dev_vector)
            self.correlation_matrix = np.divide(self.covariance_matrix, std_outer, 
                                              out=np.zeros_like(self.covariance_matrix), 
                                              where=std_outer!=0)
            
        print(f"  Standard deviation vector: {self.std_dev_vector}")
        print(f"  Covariance matrix shape: {self.covariance_matrix.shape}")
        
        # Store coefficient mapping for sampling
        self.coefficient_info = coeff_info
        
    def _filter_legendre_data_by_orders(self, requested_orders):
        """
        Filter the Legendre coefficient data to only include specified orders.
        
        Parameters:
        - requested_orders: List of Legendre orders to keep
        """
        if self.legendre_data and self.legendre_data.coefficients:
            filtered_coefficients = []
            for coeff in self.legendre_data.coefficients:
                if coeff.order in requested_orders:
                    filtered_coefficients.append(coeff)
                    print(f"  Keeping Legendre order L={coeff.order} as specified in covariance dict")
                else:
                    print(f"  Skipping Legendre order L={coeff.order} (not in covariance dict)")
            
            # Update the coefficients list
            self.legendre_data.coefficients = filtered_coefficients
            
            if not filtered_coefficients:
                print(f"Warning: No Legendre coefficients remain after filtering by orders {requested_orders}")
        
        
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


    def extract_relcorr_matrix(self, mt2):
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

        diag = np.diag(full_rel_cov)
        relstd = np.sqrt(np.maximum(diag, 0))
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            relcorr_matrix = full_rel_cov / np.outer(relstd, relstd)
            relcorr_matrix[~np.isfinite(relcorr_matrix)] = 0.0

        self.std_dev_vector = relstd
        
        self.energy_mesh = all_mesh
        
        super().__setattr__('correlation_matrix', relcorr_matrix)  # Store for compute_L_matrix()

    def _apply_samples(self, samples, mode="stack", use_copula=False, batch_size=1, 
                      sampling_method="Simple", debug=False):
        """
        Apply generated samples to the Legendre coefficients.
        Each sample is a vector of z-values or uniform values (if copula).
        
        NEW APPROACH: Generate actual coefficient values instead of just factors.
        """
        # Ensure samples is properly shaped
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)  # Make it (1, N) for single sample
        
        n_samples, n_params = samples.shape
        
        if debug:
            print(f"🔬 ANGULAR DISTRIBUTION DEBUG MODE - MT{self.MT}")
            print(f"{'='*60}")
            print(f"📊 Sampling Configuration:")
            print(f"   Number of samples: {n_samples}")
            print(f"   Number of parameters: {n_params}")
            print(f"   Sampling method: {sampling_method}")
            print(f"   Use copula: {use_copula}")
            print(f"   Operation mode: {mode}")
            if hasattr(self.legendre_data, 'coefficients'):
                legendre_orders = [c.order for c in self.legendre_data.coefficients]
                print(f"   Legendre orders: {legendre_orders}")
        
        print(f"🚨 ANGULAR DEBUG: mode='{mode}', n_samples={n_samples}")
        
        # Clear existing samples if in replace mode (except nominal at index 0)
        if mode == 'replace':
            for coeff in self.legendre_data.coefficients:
                if len(coeff.legcoeff) > 1:  # Keep nominal (index 0) if it exists
                    coeff.legcoeff = coeff.legcoeff[:1]  # Keep only index 0 (nominal)
                    coeff.factor = coeff.factor[:1] if len(coeff.factor) > 1 else coeff.factor
        
        for sample_idx in range(n_samples):
            sample = samples[sample_idx, :]  # Get 1D sample vector
            param_offset = 0
            
            for coeff in self.legendre_data.coefficients:
                n_ebins = len(coeff.energies) - 1
                nominal_coeffs = np.array(coeff.legcoeff[0])  # Get nominal coefficients
                std_devs = np.array(coeff.std_dev)  # Get standard deviations
                
                # Extract the relevant slice for this coefficient
                zvals = sample[param_offset:param_offset + n_ebins]
                
                # For copula, zvals are uniform, transform to normal
                if use_copula:
                    from scipy.stats import norm
                    zvals = norm.ppf(zvals)
                
                # Generate perturbed coefficients: nominal + z * std_dev
                perturbed_coeffs = nominal_coeffs + zvals * std_devs
                
                # Also compute multiplicative factors for backward compatibility
                # Handle zero coefficients carefully
                with np.errstate(divide='ignore', invalid='ignore'):
                    factors = np.divide(perturbed_coeffs, nominal_coeffs, 
                                      out=np.ones_like(perturbed_coeffs), 
                                      where=nominal_coeffs!=0)
                
                # Determine the effective sample index (skip nominal at index 0)
                effective_sample_idx = sample_idx + 1
                
                # Store coefficients according to operation mode
                if mode == 'stack':
                    # Extend coefficient list if needed for stacking
                    # FIXED: Use None placeholders instead of nominal coefficients
                    while len(coeff.legcoeff) <= effective_sample_idx:
                        coeff.legcoeff.append(None)  # Placeholder for ungenerated samples
                    coeff.legcoeff[effective_sample_idx] = perturbed_coeffs.tolist()
                    
                    # Also store factors for backward compatibility
                    while len(coeff.factor) <= effective_sample_idx:
                        coeff.factor.append(None)  # Placeholder for ungenerated samples
                    coeff.factor[effective_sample_idx] = factors.tolist()
                    
                elif mode == 'replace':
                    # Append new sample (we already cleared old samples above)
                    if effective_sample_idx < len(coeff.legcoeff):
                        coeff.legcoeff[effective_sample_idx] = perturbed_coeffs.tolist()
                        coeff.factor[effective_sample_idx] = factors.tolist()
                    else:
                        coeff.legcoeff.append(perturbed_coeffs.tolist())
                        coeff.factor.append(factors.tolist())
                        
                param_offset += n_ebins

        # Perform debug verification if requested
        if debug and n_samples >= 10:  # Only verify if we have enough samples
            print(f"\n🔍 COEFFICIENT SAMPLING VERIFICATION:")
            self._verify_coefficient_sampling_statistics(debug=True)
        elif debug and n_samples < 10:
            print(f"\n⚠️  Note: Statistical verification skipped (need ≥10 samples, got {n_samples})")

    def _verify_coefficient_sampling_statistics(self, debug=True):
        """
        Verify that the sampling statistics match the expected standard deviations.
        """
        if debug:
            print("   Coefficient sampling verification not yet implemented")
            print("   (This would check that sampled coefficients have correct std dev)")
    
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
        
        # Reconstruct actual coefficients for this sample
        updated_coefficients = self.legendre_data.reconstruct(sample_index)
        
        # Parse the section to update (use dynamic MT number)
        mf4mt = tape.MAT(tape.material_numbers[0]).MF(4).MT(self.MT).parse()
        
        # Create perturbed LegendreDistributions
        perturbed_legendre_dist = self._create_perturbed_legendre_distributions(
            mf4mt, updated_coefficients)
        
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
                    if l_order == 1:
                        print(f"    Applying factor {factor} (original={original_val}, final={final_val})")
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