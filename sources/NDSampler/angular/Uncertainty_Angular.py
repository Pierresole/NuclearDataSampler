import numpy as np
import time
from collections import defaultdict
from .AngularDistributionCovariance import AngularDistributionCovariance
from .Parameters_Angular import LegendreCoefficients
from ENDFtk import tree
from ENDFtk.MF4 import ResonanceRange, Isotope, Section
from scipy.linalg import block_diag  # Import block_diag function

class Uncertainty_Angular(AngularDistributionCovariance):
    """
    Class for handling uncertainties in angular distributions.
    """

    def __init__(self, mf4mt2, mf34mt2):
        super().__init__(mf4mt2)

        # Extract parameters and covariance matrices
        self.legendre_data = LegendreCoefficients.from_endftk(mf4mt2, mf34mt2)
        
        start_time = time.time()
        mf34mt2mt2 = mf34mt2.reactions.to_list()[0]
        full_corr_matrix, relative_std_vector, energy_mesh = self.retrieve_full_covariance_matrix(mf34mt2mt2)
        super().__setattr__('covariance_matrix', full_corr_matrix)
        self.mean_vector = np.zeros(full_corr_matrix.shape[0])
        self.std_dev_vector = relative_std_vector
        print(f"Time for extracting covariance matrix: {time.time() - start_time:.4f} seconds")
        
        start_time = time.time()
        self.compute_L_matrix()
        print(f"Time for compute_L_matrix: {time.time() - start_time:.4f} seconds")
        
        
    def get_covariance_type(self):
        """
        Override to return the specific covariance type.
        """
        return "AngularDistribution"
    
               
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
        return correlation_matrix, relative_std_vector, all_mesh

    def _apply_samples(self, samples, mode="stack", use_copula=False, batch_size=1, 
                      sampling_method="Simple", debug=False):
        """
        Apply generated samples to the Legendre coefficient factors.
        Each sample is a vector of z-values or uniform values (if copula).
        """
        # For each sample, update the factors for each LegendreCoefficient
        # Assume self.legendre_data.coefficients is a list of LegendreCoefficient
        n_params = samples.shape[1] if batch_size > 1 else len(samples)
        # Reshape for consistency
        sample_list = samples if batch_size > 1 else [samples]
        for sample_idx, sample in enumerate(sample_list):
            param_offset = 0
            for coeff in self.legendre_data.coefficients:
                n_ebins = len(coeff.energies) - 1
                # Extract the relevant slice for this coefficient
                zvals = sample[param_offset:param_offset + n_ebins]
                # For copula, zvals are uniform, transform to normal
                if use_copula:
                    from scipy.stats import norm
                    zvals = norm.ppf(zvals)
                # Compute multiplicative factors: exp(z * rel_std)
                rel_std = self.std_dev_vector[param_offset:param_offset + n_ebins]
                factors = np.exp(zvals * rel_std)
                # Store factors for this sample
                if len(coeff.factor) <= sample_idx:
                    # Extend factor list if needed
                    while len(coeff.factor) <= sample_idx:
                        coeff.factor.append([1.0] * n_ebins)
                coeff.factor[sample_idx] = factors.tolist()
                param_offset += n_ebins

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads the L_matrix and rml_data from the given HDF5 group and returns an instance.
        """        
        # Read L_matrix
        L_matrix = hdf5_group['L_matrix'][()]

        # Read rml_data
        leg_data_group = hdf5_group['Parameters']
                
        leg_data = LegendreCoefficients.read_from_hdf5(leg_data_group)
        
        # Create an instance and set attributes
        instance = cls.__new__(cls)
        
        # Set L_matrix on the parent CovarianceBase class
        super(cls, instance).__setattr__('L_matrix', L_matrix)
        
        # Set attributes specific to this class
        instance.legendre_data = leg_data

        return instance

    def write_additional_data_to_hdf5(self, hdf5_group):
        if self.legendre_data is not None:
            leg_group = hdf5_group.require_group('Parameters')
            self.legendre_data.write_to_hdf5(leg_group)

    def update_tape(self, tape, sample_index=1, sample_name=""):
        """
        Updates the ENDF tape with sampled Legendre coefficients for the given sample_index.
        """
        from ENDFtk.MF4 import Section
        # Reconstruct factors for this sample
        updated_factors = self.legendre_data.reconstruct(sample_index)
        # Parse the section to update
        mf4mt2 = tape.MAT(tape.material_numbers[0]).MF(4).MT(2).parse()
        # Update the distributions using the new factors
        # This is a stub; actual implementation depends on ENDFtk API
        # For each Legendre order, multiply the coefficients by the sampled factor
        # ...apply updated_factors to mf4mt2.distributions...
        # Create new Section and replace in tape
        new_section = Section(
            mt=2,
            lct=mf4mt2.reference_frame,
            zaid=mf4mt2.target_identifier,
            awr=mf4mt2.atomic_weight_ratio,
            distributions=mf4mt2.distributions  # Should be updated with new factors
        )
        mat_num = tape.material_numbers[0]
        tape.MAT(mat_num).MF(4).insert_or_replace(new_section)