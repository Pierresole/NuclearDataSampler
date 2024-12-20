import h5py
import numpy as np
from ENDFtk import tree

class CovarianceExtractor:
    def __init__(self, tape_list, hdf5_filename):
        """
        Initialize the CovarianceExtractor.

        Parameters:
            tape_list (list): List of tape file paths or ENDFtk tape objects.
            hdf5_filename (str): The name of the HDF5 file to store the covariance data.
        """
        self.tape_list = tape_list
        self.hdf5_filename = hdf5_filename
        self.tapes = {}  # Dictionary of tapes keyed by MAT number
        self.missing_mat_mt = set()  # Set to store missing (MAT, MT) pairs
        self.load_tapes()
        self.hdf5_file = h5py.File(self.hdf5_filename, 'w')
        self.parse_tapes()
        self.hdf5_file.close()

    def load_tapes(self):
        """
        Load tapes from the provided list.
        """
        for tape_input in self.tape_list:
            if isinstance(tape_input, str):
                # Assume it's a file path
                tape = tree.Tape.from_file(tape_input)
                tape_name = tape_input
            elif isinstance(tape_input, tree.Tape):
                tape = tape_input
                tape_name = f"tape_{len(self.tapes)}"
            else:
                raise ValueError("Invalid tape input.")

            for mat in tape.material_numbers:
                if mat in self.tapes:
                    print(f"Warning: Duplicate MAT {mat} found in tapes.")
                self.tapes[mat] = tape.MAT(mat)

    def parse_tapes(self):
        """
        Parse all tapes and extract covariance data.
        """
        for mat, mat_tree in self.tapes.items():
            print(f"Parsing tape for MAT {mat}")
            # self.parse_mf31(mat_tree, mat)
            self.parse_mf32(mat_tree, mat)
            # self.parse_mf33(mat_tree, mat)
            # self.parse_mf34(mat_tree, mat)

    def parse_mf31(self, mat_tree, mat):
        """
        Parse MF31 covariances (nubar) in the given tape.

        Parameters:
            mat_tree: The MAT tree from ENDFtk.
            mat: Material number of the tape.
        """
        if mat_tree.has_MF(31):
            mf31 = mat_tree.MF(31).parse()
            for section in mf31.sections:
                MT = section.MT
                MAT1 = mat
                MF1 = 31
                MT1 = MT  # Assuming self-correlation
                covariance_id = {
                    'MAT': mat,
                    'MF': 31,
                    'MT': MT,
                    'MAT1': MAT1,
                    'MF1': MF1,
                    'MT1': MT1
                }
                isSYM = True  # Assuming symmetry
                # Get mean vectors and energy grids
                mean_vector, energy_grid = self.get_nubar_mean_vector_and_energy_grid(mat, MT)
                # Extract covariance matrix
                cov_matrix = self.get_covariance_matrix(section.covariance)
                # Store data into HDF5
                self.store_covariance_data(covariance_id, cov_matrix, mean_vector, None,
                                           energy_grid, None, isSYM)

    def parse_mf32(self, mat_tree, mat):
        """
        Parse MF32 covariances (model parameters) in the given tape.

        Parameters:
            mat_tree: The MAT tree from ENDFtk.
            mat: Material number of the tape.
        """
        if mat_tree.has_MF(32):
            mf32 = mat_tree.MF(32).MT(151).parse()
            iNER = 0
            for res_range in mf32.isotopes[0].resonance_ranges.to_list():
                covariance_id = {
                    'MAT': mat,
                    'MF': 32,
                    'MT': 151,
                    'NER': iNER,
                    'MAT1': mat,
                    'MF1': 32,
                    'MT1': 151,
                    'NER1': iNER
                }
                isSYM = True  # Assuming symmetry
                # Get mean vectors
                mean_vector = res_range.parameters.parameter_values
                # Covariance matrix
                cov_vector = res_range.parameters.covariance_matrix.covariance_matrix
                NPAR = res_range.parameters.covariance_matrix.NPAR
                cov_matrix = self.upper_triangular_to_full(cov_vector, NPAR)
                # Store data into HDF5
                self.store_covariance_data(covariance_id, cov_matrix, mean_vector, None, None, None, isSYM)
                iNER += 1

    def parse_mf33(self, mat_tree, mat):
        """
        Parse MF33 covariances (cross sections) in the given tape.

        Parameters:
            mat_tree: The MAT tree from ENDFtk.
            mat: Material number of the tape.
        """
        if mat_tree.has_MF(33):
            mf33 = mat_tree.MF(33).parse()
            for section in mf33.sections:
                MT = section.MT
                for sub in section.reactions:
                    MAT1 = sub.MAT1
                    MF1 = 33
                    MT1 = sub.MT1
                    covariance_id = {
                        'MAT': mat,
                        'MF': 33,
                        'MT': MT,
                        'MAT1': MAT1,
                        'MF1': MF1,
                        'MT1': MT1
                    }
                    isSYM = (mat == MAT1 and MT == MT1)
                    # Check if MAT1 is in provided tapes
                    if MAT1 not in self.tapes:
                        self.missing_mat_mt.add((MAT1, MT1))
                        continue  # Skip processing if MAT1 is missing
                    # Get mean vectors and energy grids
                    mean_vector_row, energy_grid_row = self.get_mean_vector_and_energy_grid(mat, MT)
                    mean_vector_col, energy_grid_col = self.get_mean_vector_and_energy_grid(MAT1, MT1)
                    # Covariance matrix
                    cov_matrix = self.combine_explicit_covariances(sub.explicit_covariances)
                    # Store data into HDF5
                    self.store_covariance_data(covariance_id, cov_matrix,
                                               mean_vector_row, mean_vector_col,
                                               energy_grid_row, energy_grid_col, isSYM)

    def parse_mf34(self, mat_tree, mat):
        """
        Parse MF34 covariances (angular distributions) in the given tape.

        Parameters:
            mat_tree: The MAT tree from ENDFtk.
            mat: Material number of the tape.
        """
        if 34 in mat_tree.mf_list:
            mf34 = mat_tree.MF(34).parse()
            for section in mf34.sections:
                MT = section.MT
                LEG = section.L
                for sub in section.reactions:
                    MAT1 = sub.MAT1
                    MF1 = 34
                    MT1 = sub.MT1
                    LEG1 = sub.L1
                    covariance_id = {
                        'MAT': mat,
                        'MF': 34,
                        'MT': MT,
                        'LEG': LEG,
                        'MAT1': MAT1,
                        'MF1': MF1,
                        'MT1': MT1,
                        'LEG1': LEG1
                    }
                    isSYM = (mat == MAT1 and MT == MT1 and LEG == LEG1)
                    # Check if MAT1 is in provided tapes
                    if MAT1 not in self.tapes:
                        self.missing_mat_mt.add((MAT1, MT1))
                        continue  # Skip processing if MAT1 is missing
                    # Get mean vectors and energy grids
                    mean_vector_row, energy_grid_row = self.get_legendre_mean_vector_and_energy_grid(mat, MT, LEG)
                    mean_vector_col, energy_grid_col = self.get_legendre_mean_vector_and_energy_grid(MAT1, MT1, LEG1)
                    # Covariance matrix
                    cov_matrix = self.combine_explicit_covariances(sub.explicit_covariances)
                    # Store data into HDF5
                    self.store_covariance_data(covariance_id, cov_matrix,
                                               mean_vector_row, mean_vector_col,
                                               energy_grid_row, energy_grid_col, isSYM)

    def get_nubar_mean_vector_and_energy_grid(self, mat, mt):
        """
        Get the mean vector and energy grid for nubar (MF1, MT452).

        Parameters:
            mat (int): Material number.
            mt (int): Reaction number.

        Returns:
            mean_vector (np.ndarray): The mean vector for nubar.
            energy_grid (np.ndarray): The energy grid.
        """
        if mat in self.tapes:
            mat_tree = self.tapes[mat]
            if 1 in mat_tree.mf_list:
                mf1 = mat_tree.MF(1).parse()
                for section in mf1.nubar:
                    if section.MT == mt:
                        mean_vector = section.NU  # Hypothetical attribute for mean nubar values
                        energy_grid = section.E  # Hypothetical attribute for energy grid
                        return mean_vector, energy_grid
        return None, None

    def get_mean_vector_and_energy_grid(self, mat, mt):
        """
        Get the mean vector and energy grid for the given MAT and MT.

        Parameters:
            mat (int): Material number.
            mt (int): Reaction number.

        Returns:
            mean_vector (np.ndarray): The mean vector.
            energy_grid (np.ndarray): The energy grid.
        """
        if mat in self.tapes:
            mat_tree = self.tapes[mat]
            if 3 in mat_tree.mf_list:
                mf3 = mat_tree.MF(3).parse()
                section = mf3.section(mt)
                if section:
                    mean_vector = section.cross_sections.Y
                    energy_grid = section.cross_sections.x
                    return mean_vector, energy_grid
        return None, None

    def get_legendre_mean_vector_and_energy_grid(self, mat, mt, leg_order):
        """
        Get the Legendre coefficients mean vector and energy grid for the given MAT, MT, and Legendre order.

        Parameters:
            mat (int): Material number.
            mt (int): Reaction number.
            leg_order (int): Legendre order.

        Returns:
            mean_vector (np.ndarray): The mean vector of Legendre coefficients.
            energy_grid (np.ndarray): The energy grid.
        """
        if mat in self.tapes:
            mat_tree = self.tapes[mat]
            if 4 in mat_tree.mf_list:
                mf4 = mat_tree.MF(4).parse()
                section = mf4.section(mt)
                if section:
                    mean_vector = []
                    energy_grid = []
                    for angular_dist in section.angular_distributions:
                        if angular_dist.L == leg_order:
                            mean_vector.append(angular_dist.coefficients)
                            energy_grid.append(angular_dist.E)
                    if mean_vector:
                        mean_vector = np.concatenate(mean_vector)
                        energy_grid = np.concatenate(energy_grid)
                        return mean_vector, energy_grid
        return None, None

    def combine_explicit_covariances(self, explicit_covariances):
        """
        Combine explicit covariances to construct the covariance matrix.

        Parameters:
            explicit_covariances (list): List of explicit covariance data.

        Returns:
            cov_matrix (np.ndarray): The combined covariance matrix.
        """
        # Placeholder implementation; actual implementation depends on LB flags and data structure
        cov_matrices = []
        for explicit in explicit_covariances:
            LB = explicit.LB
            # Handle different LB types accordingly
            # For simplicity, let's assume we can extract the covariance matrix directly
            cov_matrix = explicit.covariance_matrix()
            cov_matrices.append(cov_matrix)
        # Combine the matrices (this may involve summing them)
        if cov_matrices:
            cov_matrix = sum(cov_matrices)
        else:
            cov_matrix = None
        return cov_matrix

    def get_covariance_matrix(self, covariance_section):
        """
        Extract the covariance matrix from a covariance section.

        Parameters:
            covariance_section: The covariance section.

        Returns:
            cov_matrix (np.ndarray): The covariance matrix.
        """
        # Placeholder implementation; actual extraction depends on data structure
        # Assuming covariance_section has an attribute 'matrix' with the covariance data
        cov_matrix = covariance_section.matrix
        return cov_matrix

    def store_covariance_data(self, covariance_id, cov_matrix, mean_vector_row, mean_vector_col,
                              energy_grid_row, energy_grid_col, isSYM):
        """
        Store the covariance data into the HDF5 file.

        Parameters:
            covariance_id (dict): Dictionary of covariance identifiers.
            cov_matrix (np.ndarray): Covariance matrix data.
            mean_vector_row (np.ndarray): Mean vector for rows.
            mean_vector_col (np.ndarray): Mean vector for columns.
            energy_grid_row (np.ndarray): Energy grid for rows.
            energy_grid_col (np.ndarray): Energy grid for columns.
            isSYM (bool): Symmetry flag.
        """
        group_name = self.create_group_name(covariance_id)
        group = self.hdf5_file.create_group(group_name)
        # Store attributes
        for key, value in covariance_id.items():
            group.attrs[key] = value
        group.attrs['isSYM'] = isSYM
        # Store datasets
        if cov_matrix is not None:
            group.create_dataset('cov_matrix', data=cov_matrix)
        if isSYM:
            if mean_vector_row is not None:
                group.create_dataset('mean_vector', data=mean_vector_row)
            if energy_grid_row is not None:
                group.create_dataset('energy_grid', data=energy_grid_row)
        else:
            if mean_vector_row is not None:
                group.create_dataset('mean_vector_row', data=mean_vector_row)
            if mean_vector_col is not None:
                group.create_dataset('mean_vector_col', data=mean_vector_col)
            if energy_grid_row is not None:
                group.create_dataset('energy_grid_row', data=energy_grid_row)
            if energy_grid_col is not None:
                group.create_dataset('energy_grid_col', data=energy_grid_col)

    def create_group_name(self, covariance_id):
        """
        Create a unique group name based on the covariance identifiers.

        Parameters:
            covariance_id (dict): Dictionary of covariance identifiers.

        Returns:
            group_name (str): The unique group name.
        """
        # Create a group name by concatenating the identifiers
        parts = [f"{key}{value}" for key, value in covariance_id.items()]
        group_name = "_".join(parts)
        return group_name

    def upper_triangular_to_full(self, cov_vector, NPAR):
        """
        Convert upper triangular covariance vector to full covariance matrix.

        Parameters:
            cov_vector (np.ndarray): Upper triangular elements of the covariance matrix.
            NPAR (int): Number of parameters.

        Returns:
            cov_matrix (np.ndarray): Full covariance matrix.
        """
        cov_matrix = np.zeros((NPAR, NPAR))
        idx = 0
        for i in range(NPAR):
            for j in range(i, NPAR):
                cov_matrix[i, j] = cov_vector[idx]
                cov_matrix[j, i] = cov_vector[idx]
                idx += 1
        return cov_matrix
