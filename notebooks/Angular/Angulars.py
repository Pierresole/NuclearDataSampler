import ENDFtk
import numpy as np
import matplotlib.pyplot as plt

def plot_legendre_coeffs(angulard, orders):
    """
    Plot Legendre coefficients for given order(s) as a function of incident energy.

    Parameters:
    angulard: parsed angular distribution object
    orders: int or list of ints, Legendre order(s) to plot
    """
    if isinstance(orders, int):
        orders = [orders]

    energies = [dist.incident_energy for dist in angulard.angular_distributions.to_list()]
    max_order = max(len(dist.coefficients[:]) for dist in angulard.distributions.legendre.angular_distributions.to_list())
    coeff_array = np.zeros((len(energies), max_order))

    for i, dist in enumerate(angulard.distributions.legendre.angular_distributions.to_list()):
        coeffs = dist.coefficients
        coeff_array[i, :len(coeffs)] = coeffs

    for l in orders:
        if l < max_order:
            plt.plot(energies, coeff_array[:, l], label=r'$a_{l=%d}$' % l)
        else:
            print(f"Order {l} exceeds available maximum order {max_order-1}")

    plt.xscale('log')
    plt.xlim(1000,5e7)
    plt.xlabel('Energy [eV]')
    plt.ylabel('Legendre Coefficient aₗ')
    plt.title(r'$^{26}Al$ Elastic Angular Distributions')
    plt.legend()
    plt.show()

def mesh_union(mesh1, mesh2, eps=1e-8):
    union = np.unique(np.concatenate((mesh1, mesh2)))
    diff = np.diff(union)
    mask = diff < eps
    if np.any(mask):
        keep = np.ones_like(union, dtype=bool)
        keep[1:][mask] = False
        union = union[keep]
    return union

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

def add_matrices_with_mesh(matrixA, rowMeshA, colMeshA, matrixB, rowMeshB, colMeshB, epsilon=1e-8):
    if matrixA.size == 0:
        return matrixB.copy(), sorted(rowMeshB), sorted(colMeshB)
    if matrixB.size == 0:
        return matrixA.copy(), sorted(rowMeshA), sorted(colMeshA)

    rowMeshA = np.array(sorted(rowMeshA))
    colMeshA = np.array(sorted(colMeshA))
    rowMeshB = np.array(sorted(rowMeshB))
    colMeshB = np.array(sorted(colMeshB))

    union_row_mesh = mesh_union(rowMeshA, rowMeshB, epsilon)
    union_col_mesh = mesh_union(colMeshA, colMeshB, epsilon)

    expandedA = expand_matrix_fast(matrixA, rowMeshA, colMeshA, union_row_mesh, union_col_mesh)
    expandedB = expand_matrix_fast(matrixB, rowMeshB, colMeshB, union_row_mesh, union_col_mesh)

    result = expandedA + expandedB
    return result, union_row_mesh.tolist(), union_col_mesh.tolist()

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
        raise NotImplementedError("Unknown subblock type")

def block_to_matrix(block):
    # block is ENDFtk.SquareMatrix or ENDFtk.LegendreBlock
    # block.data.to_list() gives subblocks
    subblocks = block.data.to_list() if hasattr(block, "data") else [block]
    matrix = np.zeros((0,0))
    row_mesh = []
    col_mesh = []
    for sub in subblocks:
        submat, subrow, subcol = subblock_to_matrix(sub)
        if matrix.size == 0:
            matrix = submat
            row_mesh = subrow
            col_mesh = subcol
        else:
            matrix, row_mesh, col_mesh = add_matrices_with_mesh(
                matrix, row_mesh, col_mesh, submat, subrow, subcol
            )
    return matrix, row_mesh, col_mesh

def covariance_to_correlation_and_relstd(rel_cov):
    diag = np.diag(rel_cov)
    std = np.sqrt(np.maximum(diag, 0))
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = rel_cov / np.outer(std, std)
        corr[~np.isfinite(corr)] = 0.0
        relstd = std.copy()
    return corr, relstd

def retrieve_full_covariance_matrix(mt2):
    NL = mt2.NL
    NL1 = mt2.NL1
    nblocks = mt2.number_legendre_blocks
    blocks = mt2.legendre_blocks.to_list()
    # First, collect all unique energy mesh points for all blocks
    all_mesh = set()
    for block in blocks:
        for sub in block.data.to_list():
            if hasattr(sub, "LB") and sub.LB == 5:
                all_mesh.update(sub.energies.to_list())
            elif hasattr(sub, "LB") and sub.LB == 1:
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
        mat, row_mesh, col_mesh = block_to_matrix(block)
        # Expand to the global mesh
        mat_expanded = expand_matrix_fast(mat, row_mesh, col_mesh, all_mesh, all_mesh)
        # Place in the full matrix
        full_rel_cov[(l-1)*N:l*N, (l1-1)*N:l1*N] = mat_expanded
        if l != l1:
            # Fill symmetric block
            full_rel_cov[(l1-1)*N:l1*N, (l-1)*N:l*N] = mat_expanded.T

    correlation_matrix, relative_std_vector = covariance_to_correlation_and_relstd(full_rel_cov)
    return correlation_matrix, relative_std_vector, all_mesh, full_rel_cov

def get_legendre_cov_block(legendre_blocks, LegOrder1, LegOrder2):
    """
    Retrieve the Legendre covariance block for the specified orders.

    Parameters:
    legendre_blocks : list
        List of LegendreBlock objects (e.g., angularu.reactions[0].legendre_blocks)
    LegOrder1 : int
        First Legendre order (1-based)
    LegOrder2 : int
        Second Legendre order (1-based)

    Returns:
    block : ENDFtk.MF34.LegendreBlock
        The block corresponding to (LegOrder1, LegOrder2), or None if not found.
    """
    for block in legendre_blocks:
        if (block.first_legendre_order == LegOrder1 and block.second_legendre_order == LegOrder2) or \
           (block.first_legendre_order == LegOrder2 and block.second_legendre_order == LegOrder1):
            return block
    return None

def build_legendre_cov_matrix(legendre_blocks, order1=1, order2=1):
    """
    Build the covariance matrix for the specified Legendre orders from MF34 legendre_blocks.

    Parameters:
    legendre_blocks : list
        List of LegendreBlock objects (e.g., angularu.reactions[0].legendre_blocks)
    order1 : int
        First Legendre order (default 1)
    order2 : int
        Second Legendre order (default 1)

    Returns:
    cov_matrix : np.ndarray
        Covariance matrix (nBins x nBins)
    bin_boundaries : list
        List of bin boundary energies (length nBins+1)
    """
    block = get_legendre_cov_block(legendre_blocks, order1, order2)
    if block is None:
        raise ValueError(f"No block found for Legendre orders ({order1},{order2})")
    mat, row_mesh, col_mesh = block_to_matrix(block)
    # nBins = block.data[0].NE - 1
    # bin_boundaries = block.data[0].energies[:]
    # cov_matrix = np.zeros((nBins, nBins))
    # triu_indices = np.triu_indices(nBins)
    # cov_matrix[triu_indices] = block.data[0].values[:]
    # cov_matrix[(triu_indices[1], triu_indices[0])] = block.data[0].values[:]
    # return cov_matrix, bin_boundaries
    return mat, row_mesh

def remove_null_variance_and_track(full_cov_matrix, all_mesh, NL, tol=1e-12):
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


def reconstruct_full_perturbation(perturbation_vector, index_map, all_mesh, NL):
    """
    Reconstruct a full perturbation vector on the complete energy grid,
    filling zeros for bins that had null variance.
    
    Parameters:
    -----------
    perturbation_vector : np.ndarray
        Perturbation vector of length equal to reduced matrix size
    index_map : list of tuples
        Mapping from reduced matrix indices to (legendre_order, bin_index, E_low, E_high)
    all_mesh : list
        Energy mesh boundaries
    NL : int
        Number of Legendre orders
        
    Returns:
    --------
    full_perturbation : np.ndarray
        Full perturbation vector of shape (NL, N) where N is number of energy bins
    """
    N = len(all_mesh) - 1
    full_perturbation = np.zeros((NL, N))
    
    for i, (legendre_order, bin_index, _, _) in enumerate(index_map):
        full_perturbation[legendre_order - 1, bin_index] = perturbation_vector[i]
    
    return full_perturbation

def extract_nominal_legendre_coeffs(angulard, all_mesh, NL=6):
    """
    Extract nominal Legendre coefficients on the specified energy mesh.
    
    Parameters:
    -----------
    angulard : ENDFtk MF4 Section
        Angular distribution section from ENDF file
    all_mesh : list
        Energy mesh boundaries (length N+1)
    NL : int
        Number of Legendre orders to extract (default 6)
    
    Returns:
    --------
    nominal_coeffs : np.ndarray
        Array of shape (NL, N) with nominal Legendre coefficient values
    """
    # Get distributions from MF4
    distributions = angulard.distributions.legendre.angular_distributions.to_list()
    energies = [dist.incident_energy for dist in distributions]
    
    N = len(all_mesh) - 1
    nominal_coeffs = np.zeros((NL, N))
    
    # For each energy bin in all_mesh, interpolate nominal values
    for bin_idx in range(N):
        E_center = (all_mesh[bin_idx] + all_mesh[bin_idx + 1]) / 2
        
        # Find nearest energies in the distribution data
        idx = np.searchsorted(energies, E_center)
        if idx == 0:
            idx = 1
        elif idx >= len(energies):
            idx = len(energies) - 1
        
        # Linear interpolation between two nearest points
        E1, E2 = energies[idx-1], energies[idx]
        dist1, dist2 = distributions[idx-1], distributions[idx]
        
        if E2 != E1:
            w = (E_center - E1) / (E2 - E1)
        else:
            w = 0.5
        
        # Interpolate each coefficient
        coeffs1 = dist1.coefficients[:]
        coeffs2 = dist2.coefficients[:]
        
        for l in range(NL):
            c1 = coeffs1[l] if l < len(coeffs1) else 0.0
            c2 = coeffs2[l] if l < len(coeffs2) else 0.0
            nominal_coeffs[l, bin_idx] = (1 - w) * c1 + w * c2
    
    return nominal_coeffs

def apply_perturbations_to_coefficients(nominal_coeffs, perturbation_samples, index_map, all_mesh, NL=6):
    """
    Apply perturbation samples to nominal Legendre coefficients.
    
    Parameters:
    -----------
    nominal_coeffs : np.ndarray
        Nominal coefficients of shape (NL, N)
    perturbation_samples : np.ndarray
        Perturbation samples of shape (n_samples, n_active_params)
    index_map : list of tuples
        Mapping from reduced indices to (legendre_order, bin_index, E_low, E_high)
    all_mesh : list
        Energy mesh boundaries
    NL : int
        Number of Legendre orders
        
    Returns:
    --------
    perturbed_coeffs_list : list of np.ndarray
        List of perturbed coefficient arrays, each of shape (NL, N)
    """
    n_samples = perturbation_samples.shape[0]
    N = len(all_mesh) - 1
    perturbed_coeffs_list = []
    
    for sample_idx in range(n_samples):
        # Reconstruct full perturbation for this sample
        full_perturbation = reconstruct_full_perturbation(
            perturbation_samples[sample_idx], 
            index_map, 
            all_mesh, 
            NL
        )
        
        # Apply multiplicative perturbation: coeff * (1 + delta)
        perturbed_coeffs = nominal_coeffs * (1 + full_perturbation)
        perturbed_coeffs_list.append(perturbed_coeffs)
    
    return perturbed_coeffs_list

def create_enhanced_legendre_distributions(angulard, perturbation_samples, index_map, 
                                          all_mesh, NL=6):
    """
    Create enhanced Legendre distributions with boundary duplication, mimicking
    _create_perturbed_legendre_distributions_from_factors.
    
    This function replicates the exact logic of update_tape:
    1. Get union of original MF4 energies and covariance boundaries
    2. For each energy, interpolate nominal coefficients from original MF4 data
    3. Apply multiplicative factors based on which covariance bin the energy falls into
    4. At covariance boundaries, create discontinuities by duplicating points with different factors
    
    Parameters:
    -----------
    angulard : ENDFtk MF4 Section
        Angular distribution section from ENDF file
    perturbation_samples : np.ndarray
        Perturbation samples of shape (n_samples, n_active_params)
    index_map : list of tuples
        Mapping from reduced indices to (legendre_order, bin_index, E_low, E_high)
    all_mesh : list
        Energy mesh boundaries (covariance boundaries)
    NL : int
        Number of Legendre orders
        
    Returns:
    --------
    enhanced_data_list : list of dict
        List of dictionaries, each containing:
        - 'energies': array of energy points (with boundary duplications)
        - 'coefficients': array of shape (NL, n_enhanced_points)
    """
    # Get original MF4 data for interpolation
    distributions = angulard.distributions.legendre.angular_distributions.to_list()
    original_energies = [dist.incident_energy for dist in distributions]
    
    # Build interpolation functions for nominal coefficients from original MF4 data
    # This preserves the continuous shape of the original data
    nominal_interp_funcs = []
    for l_order in range(NL):
        energies_list = []
        coeffs_list = []
        for dist in distributions:
            coeffs = dist.coefficients[:]
            energies_list.append(dist.incident_energy)
            if l_order < len(coeffs):
                coeffs_list.append(coeffs[l_order])
            else:
                coeffs_list.append(0.0)
        
        # Create interpolation function (log-linear interpolation for energy)
        from scipy.interpolate import interp1d
        interp_func = interp1d(np.log(energies_list), coeffs_list, 
                               kind='linear', bounds_error=False, 
                               fill_value=(coeffs_list[0], coeffs_list[-1]))
        nominal_interp_funcs.append(interp_func)
    
    # Helper to get nominal coefficient at any energy via interpolation
    def get_nominal_at_energy(energy):
        """Returns list of nominal coefficients for all L orders at given energy."""
        log_e = np.log(energy)
        return [interp_func(log_e) for interp_func in nominal_interp_funcs]
    
    # Create union of original energies and covariance boundaries
    covariance_boundaries = all_mesh
    union_energies = sorted(set(original_energies) | set(covariance_boundaries))
    
    n_samples = perturbation_samples.shape[0]
    enhanced_data_list = []
    
    for sample_idx in range(n_samples):
        # Reconstruct full perturbation for this sample
        full_perturbation = reconstruct_full_perturbation(
            perturbation_samples[sample_idx], 
            index_map, 
            all_mesh, 
            NL
        )
        
        # Build multiplicative factors dictionary: factors[l_order] = 1 + delta
        # Keys are 1-based (L=1, L=2, ...) to match _apply_multiplicative_factors_to_coefficients
        multiplicative_factors = {}
        for l in range(NL):
            multiplicative_factors[l + 1] = [1.0 + d for d in full_perturbation[l, :]]
        
        enhanced_energies = []
        enhanced_coeffs = []
        
        # Helper to apply multiplicative factors to base coefficients
        def apply_factors(base_coeffs, bin_idx):
            """Apply bin-specific multiplicative factors to coefficients."""
            perturbed = []
            for l_order_1based in range(1, NL + 1):
                base_val = base_coeffs[l_order_1based - 1]
                if l_order_1based in multiplicative_factors:
                    factors = multiplicative_factors[l_order_1based]
                    if bin_idx < len(factors):
                        factor = factors[bin_idx]
                    else:
                        factor = 1.0
                else:
                    factor = 1.0
                perturbed.append(base_val * factor)
            return perturbed
        
        # Process each energy in union mesh
        for energy in union_energies:
            # Get nominal coefficients at this energy via interpolation
            base_coeffs = get_nominal_at_energy(energy)
            
            is_boundary = energy in covariance_boundaries
            
            if is_boundary:
                b_idx = covariance_boundaries.index(energy)
                first = (b_idx == 0)
                last = (b_idx == len(covariance_boundaries) - 1)
                
                if first:
                    # First boundary: nominal, then perturbed with first bin factors
                    enhanced_energies.append(energy)
                    enhanced_coeffs.append(base_coeffs.copy())
                    
                    enhanced_energies.append(energy)
                    enhanced_coeffs.append(apply_factors(base_coeffs, 0))
                    
                elif last:
                    # Last boundary: perturbed with previous bin factors (duplicated)
                    prev_bin = b_idx - 1
                    pert = apply_factors(base_coeffs, prev_bin)
                    
                    enhanced_energies.append(energy)
                    enhanced_coeffs.append(pert.copy())
                    enhanced_energies.append(energy)
                    enhanced_coeffs.append(pert.copy())
                    
                else:
                    # Interior boundary: prev_bin, duplicate, next_bin
                    prev_bin = b_idx - 1
                    next_bin = b_idx
                    
                    prev_coeffs = apply_factors(base_coeffs, prev_bin)
                    enhanced_energies.append(energy)
                    enhanced_coeffs.append(prev_coeffs)
                    
                    enhanced_energies.append(energy)
                    enhanced_coeffs.append(prev_coeffs.copy())
                    
                    next_coeffs = apply_factors(base_coeffs, next_bin)
                    enhanced_energies.append(energy)
                    enhanced_coeffs.append(next_coeffs)
            else:
                # Not a boundary: find which bin and apply factors
                bin_idx = None
                for i in range(len(covariance_boundaries) - 1):
                    if covariance_boundaries[i] <= energy < covariance_boundaries[i+1]:
                        bin_idx = i
                        break
                
                if bin_idx is None:
                    if energy < covariance_boundaries[0]:
                        bin_idx = 0
                    else:
                        bin_idx = len(covariance_boundaries) - 2
                
                pert_coeffs = apply_factors(base_coeffs, bin_idx)
                enhanced_energies.append(energy)
                enhanced_coeffs.append(pert_coeffs)
        
        # Convert to arrays
        enhanced_energies_arr = np.array(enhanced_energies)
        enhanced_coeffs_arr = np.array(enhanced_coeffs).T  # Shape: (NL, n_points)
        
        enhanced_data_list.append({
            'energies': enhanced_energies_arr,
            'coefficients': enhanced_coeffs_arr
        })
    
    return enhanced_data_list

def plot_perturbed_legendre_coeffs(angulard, perturbation_samples, index_map, all_mesh,
                                   legendre_order=0, max_samples=None, NL=6,
                                   xlim=(1e3, 5e7), figsize=(12, 6)):
    """
    Plot nominal and perturbed Legendre coefficients for a specific order using
    enhanced energy mesh with boundary duplications.
    
    Parameters:
    -----------
    angulard : ENDFtk MF4 Section
        Angular distribution section from ENDF file
    perturbation_samples : np.ndarray
        Perturbation samples of shape (n_samples, n_active_params)
    index_map : list of tuples
        Mapping from reduced indices to (legendre_order, bin_index, E_low, E_high)
    all_mesh : list
        Energy mesh boundaries
    legendre_order : int
        Legendre order to plot (0-based, so 0 = L=1, 1 = L=2, etc.)
    max_samples : int, optional
        Maximum number of samples to plot
    NL : int
        Number of Legendre orders
    xlim : tuple
        X-axis limits
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis
    """
    # Get original nominal data
    distributions = angulard.distributions.legendre.angular_distributions.to_list()
    nominal_energies = [dist.incident_energy for dist in distributions]
    nominal_coeffs = []
    for dist in distributions:
        coeffs = dist.coefficients[:]
        if legendre_order < len(coeffs):
            nominal_coeffs.append(coeffs[legendre_order])
        else:
            nominal_coeffs.append(0.0)
    
    # Create enhanced distributions for all samples
    enhanced_data_list = create_enhanced_legendre_distributions(
        angulard, perturbation_samples, index_map, all_mesh, NL=NL
    )
    
    # Limit number of samples if requested
    n_samples = len(enhanced_data_list)
    if max_samples is not None and n_samples > max_samples:
        n_samples = max_samples
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot perturbed samples first (so nominal is on top)
    for i in range(n_samples):
        data = enhanced_data_list[i]
        ax.plot(data['energies'], data['coefficients'][legendre_order, :], 
               alpha=0.5, linewidth=1, color='red', 
               label='Perturbed samples' if i == 0 else '')
    
    # Plot nominal on top
    ax.plot(nominal_energies, nominal_coeffs, 
           'b-', linewidth=2, label='Nominal', zorder=100)
    
    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_xlabel('Energy [eV]', fontsize=12)
    ax.set_ylabel(f'$a_{{{legendre_order}}}$ (L={legendre_order+1})', fontsize=12)
    ax.set_title(f'Legendre Coefficient L={legendre_order+1}: Nominal vs {n_samples} Perturbed Samples', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

