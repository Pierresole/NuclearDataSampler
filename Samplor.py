import h5py
import numpy as np
from collections import defaultdict
from scipy.linalg import cholesky

class NuclearDataSampler:
    def __init__(self, covariance_hdf5_filename, samples_hdf5_filename, n_samples):
        """
        Initialize the NuclearDataSampler.

        Parameters:
            covariance_hdf5_filename (str): The HDF5 file containing covariance data.
            samples_hdf5_filename (str): The HDF5 file to store the generated samples.
            n_samples (int): Number of samples to generate.
        """
        self.covariance_hdf5_filename = covariance_hdf5_filename
        self.samples_hdf5_filename = samples_hdf5_filename
        self.n_samples = n_samples
        self.covariance_data = {}
        self.load_covariance_data()

    def load_covariance_data(self):
        """
        Load covariance data from the HDF5 file and organize it by MF type.
        """
        with h5py.File(self.covariance_hdf5_filename, 'r') as hdf5_file:
            for group_name in hdf5_file:
                group = hdf5_file[group_name]
                covariance_id = dict(group.attrs)
                mf_type = covariance_id['MF']
                if mf_type not in self.covariance_data:
                    self.covariance_data[mf_type] = []
                # Prepare data dictionary
                data = {
                    'covariance_id': covariance_id,
                    'isSYM': group.attrs['isSYM'],
                    'cov_matrix': group['cov_matrix'][:]
                }
                # Retrieve mean vectors and energy grids
                if data['isSYM']:
                    data['mean_vector'] = group['mean_vector'][:]
                    data['energy_grid'] = group.get('energy_grid', None)
                else:
                    data['mean_vector_row'] = group['mean_vector_row'][:]
                    data['mean_vector_col'] = group['mean_vector_col'][:]
                    data['energy_grid_row'] = group.get('energy_grid_row', None)
                    data['energy_grid_col'] = group.get('energy_grid_col', None)
                self.covariance_data[mf_type].append(data)

    def assemble_and_sample(self):
        """
        Process each MF type, assemble covariance matrices, perform Cholesky decomposition,
        generate samples, and store them in an HDF5 file.
        """
        with h5py.File(self.samples_hdf5_filename, 'w') as samples_hdf5:
            for mf_type in self.covariance_data:
                print(f"Processing MF{mf_type}")
                covariance_matrices = self.covariance_data[mf_type]
                # Identify groups of correlated covariance matrices
                groups = self.identify_groups(covariance_matrices, mf_type)
                for group_key, group_cov_matrices in groups.items():
                    # Assemble covariance matrix and mean vector
                    C, Mu, indices_info = self.assemble_covariance_matrix_and_mean(group_cov_matrices)
                    # Perform Cholesky decomposition
                    try:
                        L = cholesky(C, lower=True)
                    except np.linalg.LinAlgError:
                        print(f"Warning: Covariance matrix for MF{mf_type}, group {group_key} is not positive definite.")
                        continue
                    # Generate samples
                    Y_samples = self.generate_samples(L, Mu)
                    # Store samples
                    self.store_samples(samples_hdf5, mf_type, group_key, Y_samples, indices_info)

    def identify_groups(self, covariance_matrices, mf_type):
        """
        Identify groups of covariance matrices that need to be assembled based on their correlations.

        Parameters:
            covariance_matrices (list): List of covariance matrices for the MF type.
            mf_type (int): MF type being processed.

        Returns:
            groups (dict): Dictionary where keys are group identifiers, and values are lists of covariance matrices.
        """
        groups = defaultdict(list)
        for cov_data in covariance_matrices:
            covariance_id = cov_data['covariance_id']
            # For MF33, group by correlated reactions (MAT, MF, MT, MAT1, MF1, MT1)
            if mf_type == 33:
                group_key = frozenset([
                    (covariance_id['MAT'], covariance_id['MT']),
                    (covariance_id['MAT1'], covariance_id['MT1'])
                ])
            else:
                # For other MF types, group by MAT
                group_key = covariance_id['MAT']
            groups[group_key].append(cov_data)
        return groups

    def assemble_covariance_matrix_and_mean(self, group_cov_matrices):
        """
        Assemble the covariance matrix and mean vector from the group of covariance matrices.

        Parameters:
            group_cov_matrices (list): List of covariance matrices in the group.

        Returns:
            C (np.ndarray): Assembled covariance matrix.
            Mu (np.ndarray): Assembled mean vector.
            indices_info (list): List of dictionaries with index information for each covariance matrix.
        """
        # Determine total size and index mapping
        total_size = 0
        indices_info = []
        parameter_positions = {}
        for cov_data in group_cov_matrices:
            covariance_id = cov_data['covariance_id']
            key_row = (covariance_id['MAT'], covariance_id.get('MT', covariance_id.get('NER', covariance_id.get('LEG'))))
            key_col = (covariance_id['MAT1'], covariance_id.get('MT1', covariance_id.get('NER1', covariance_id.get('LEG1'))))
            size_row = cov_data['cov_matrix'].shape[0]
            size_col = cov_data['cov_matrix'].shape[1]
            # Update parameter positions
            if key_row not in parameter_positions:
                parameter_positions[key_row] = (total_size, total_size + size_row)
                total_size += size_row
            if key_col not in parameter_positions:
                parameter_positions[key_col] = (total_size, total_size + size_col)
                total_size += size_col
            # Store indices info
            indices_info.append({
                'covariance_id': covariance_id,
                'key_row': key_row,
                'key_col': key_col,
                'size_row': size_row,
                'size_col': size_col,
                'cov_matrix': cov_data['cov_matrix']
            })
        # Initialize assembled covariance matrix and mean vector
        C = np.zeros((total_size, total_size))
        Mu = np.zeros(total_size)
        # Assemble covariance matrix
        for info in indices_info:
            start_row, end_row = parameter_positions[info['key_row']]
            start_col, end_col = parameter_positions[info['key_col']]
            C[start_row:end_row, start_col:end_col] = info['cov_matrix']
            if not info['covariance_id']['isSYM']:
                C[start_col:end_col, start_row:end_row] = info['cov_matrix'].T
        # Assemble mean vector
        for key, (start_idx, end_idx) in parameter_positions.items():
            # Retrieve mean vector for the parameter
            for cov_data in group_cov_matrices:
                if cov_data['covariance_id']['MAT'] == key[0] and \
                   cov_data['covariance_id'].get('MT', cov_data['covariance_id'].get('NER', cov_data['covariance_id'].get('LEG'))) == key[1]:
                    if cov_data['covariance_id']['isSYM']:
                        Mu[start_idx:end_idx] = cov_data['mean_vector']
                    else:
                        if key == cov_data['key_row']:
                            Mu[start_idx:end_idx] = cov_data['mean_vector_row']
                        else:
                            Mu[start_idx:end_idx] = cov_data['mean_vector_col']
                    break
        return C, Mu, parameter_positions

    def generate_samples(self, L, Mu):
        """
        Generate samples using the Cholesky decomposition.

        Parameters:
            L (np.ndarray): Lower-triangular matrix from Cholesky decomposition.
            Mu (np.ndarray): Mean vector.

        Returns:
            Y_samples (np.ndarray): Generated samples.
        """
        n_params = Mu.shape[0]
        N = np.random.randn(self.n_samples, n_params)
        Y_samples = N @ L.T + Mu
        return Y_samples

    def store_samples(self, samples_hdf5, mf_type, group_key, Y_samples, parameter_positions):
        """
        Store the samples in the HDF5 file.

        Parameters:
            samples_hdf5 (h5py.File): HDF5 file to store samples.
            mf_type (int): MF type being processed.
            group_key: Identifier for the group of covariance matrices.
            Y_samples (np.ndarray): Generated samples.
            parameter_positions (dict): Mapping of parameters to their indices in the assembled matrix.
        """
        # Create a group for the MF type
        mf_group = samples_hdf5.require_group(f'MF{mf_type}')
        # For each parameter, extract samples and store
        for key, (start_idx, end_idx) in parameter_positions.items():
            samples = Y_samples[:, start_idx:end_idx]
            mat, identifier = key
            dataset_name = f'MAT{mat}_ID{identifier}'
            dataset = mf_group.create_dataset(dataset_name, data=samples)
            dataset.attrs['MAT'] = mat
            dataset.attrs['Identifier'] = identifier
            dataset.attrs['MF'] = mf_type
            # Additional attributes can be stored as needed
