import numpy as np
from collections import defaultdict
from .ResonanceRangeCovariance import ResonanceRangeCovariance
from .URR_BreitWigner_Parameters import URREnergyDependent
from ..mathmatrix import CovarianceMatrixHandler
from ENDFtk import tree
from ENDFtk.MF2.MT151 import ResonanceRange, Isotope, Section
from scipy.linalg import block_diag  # Import block_diag function


class URRBreitWignerUncertainty(ResonanceRangeCovariance):
    """
    Class for handling uncertainties in the Unresolved Resonance Region (URR) Breit-Wigner parameters.
    """

    def __init__(self, mf2_resonance_range, mf32_resonance_range, NER):
        self.NER = NER
        super().__init__(mf2_resonance_range, NER)
        self.MPAR = mf32_resonance_range.parameters.covariance_matrix.MPAR
        self.LFW = mf32_resonance_range.parameters.LFW

        # Initialize URREnergyDependent data
        self.urre_data = URREnergyDependent(
            SPI=mf2_resonance_range.parameters.SPI,
            AP=mf2_resonance_range.parameters.AP,
            LSSF=mf2_resonance_range.parameters.LSSF
        )

        # Extract parameters and covariance matrices
        self.urre_data.extract_parameters(mf2_resonance_range)
        self.extract_covariance_matrix(mf32_resonance_range)
        self.remove_zero_variance_parameters()
        
        # Process the covariance matrix
        handler = CovarianceMatrixHandler(self.covariance_matrix)

        # Get the decomposition
        self.L_matrix, self.is_cholesky = handler.get_decomposition()

    def get_covariance_matrix(self):
        """
        Override to return the specific covariance type.
        """
        return np.dot(self.L_matrix, self.L_matrix.T)
    
    def get_covariance_type(self):
        """
        Override to return the specific covariance type.
        """
        return "URR_BreitWigner"
    
    def extract_covariance_matrix(self, mf32_resonance_range):
        covariance_data = mf32_resonance_range.parameters.covariance_matrix
        NPAR_spin = covariance_data.NPAR  # Number of parameters at the spin group level

        # Get the spin-level covariance matrix
        relative_cov_matrix_upper = covariance_data.covariance_matrix
        relative_cov_matrix_spin = np.zeros((NPAR_spin, NPAR_spin))
        idx = 0
        for i in range(NPAR_spin):
            for j in range(i, NPAR_spin):
                relative_cov_matrix_spin[i, j] = relative_cov_matrix_upper[idx]
                relative_cov_matrix_spin[j, i] = relative_cov_matrix_upper[idx]
                idx += 1

        # Build the expanded covariance matrix as a block-diagonal matrix
        param_names = self.get_param_names()
        num_params_per_spin = len(param_names)

        block_matrices = []
        index_list = []
        for l_idx, l_value in enumerate(self.urre_data.Llist):
            for j_idx, j_value in enumerate(l_value.Jlist):
                num_energies = len(j_value.RP)
                for e_idx in range(num_energies):
                    # Add to index mapping
                    for p_idx, param_name in enumerate(param_names):
                        index_list.append(
                            (l_idx, j_idx, e_idx, param_name.encode('utf-8'))
                        )
                    # Append the spin-level covariance matrix for this energy
                    block_matrices.append(relative_cov_matrix_spin)
        
        # Build block-diagonal covariance matrix
        self.covariance_matrix = block_diag(*block_matrices)

        # Convert index_list to structured array
        dtype = np.dtype([
            ('l_idx', 'i4'),
            ('j_idx', 'i4'),
            ('e_idx', 'i4'),
            ('param_name', 'S20')  # String of max length 20
        ])
        self.index_mapping = np.array(index_list, dtype=dtype)

    def remove_zero_variance_parameters(self):
        """
        Removes parameters with zero variance and updates the covariance matrix accordingly.
        """
        # Identify zero variance parameters
        zero_variance_indices = np.where(np.diag(self.covariance_matrix) == 0.0)[0]
        if zero_variance_indices.size == 0:
            return

        # Remove parameters with zero variance from index_mapping
        self.index_mapping = [
            self.index_mapping[i]
            for i in range(len(self.index_mapping))
            if i not in zero_variance_indices
        ]

        # Remove entries from covariance matrix
        self.covariance_matrix = np.delete(self.covariance_matrix, zero_variance_indices, axis=0)
        self.covariance_matrix = np.delete(self.covariance_matrix, zero_variance_indices, axis=1)

    def sample_parameters(self, mode='stack'):
        if mode not in ['stack', 'replace']:
            raise ValueError("Mode must be either 'stack' or 'replace'.")

        # Generate standard normal random variables
        N = np.random.normal(size=self.L_matrix.shape[0])
        # Compute sampled relative deviations
        Y = self.L_matrix @ N
        
        # Apply deviations to parameters
        for idx_in_Y, item in enumerate(self.index_mapping):
            l_idx = item['l_idx']
            j_idx = item['j_idx']
            e_idx = item['e_idx']
            param_name = item['param_name'].decode('utf-8')  # Decode bytes to string

            relative_deviation = Y[idx_in_Y]
            l_value = self.urre_data.Llist[l_idx]
            j_value = l_value.Jlist[j_idx]
            rp = j_value.RP[e_idx]

            # Get the original value
            original_value = getattr(rp, param_name)[0]

            # Compute the sampled value
            sampled_value = original_value * (1 + relative_deviation)

            # Apply the value based on the mode
            param_list = getattr(rp, param_name)
            if mode == 'stack' or len(param_list) == 1:
                param_list.append(sampled_value)
            elif mode == 'replace':
                param_list[1] = sampled_value

    def extract_samples(self):
        """
        Extracts samples from urre_data using index_mapping and returns them as a dictionary.

        Returns:
            samples_dict: A dictionary where keys are (l_idx, j_idx, e_idx, param_name)
                        and values are lists of samples, including the original value.
        """
        samples_dict = {}

        # Build a mapping from indices to RP objects
        rp_mapping = {}
        for l_idx, l_value in enumerate(self.urre_data.Llist):
            for j_idx, j_value in enumerate(l_value.Jlist):
                for e_idx, rp in enumerate(j_value.RP):
                    rp_mapping[(l_idx, j_idx, e_idx)] = rp

        # Loop over index_mapping to extract samples
        for idx, item in enumerate(self.index_mapping):
            l_idx = item['l_idx']
            j_idx = item['j_idx']
            e_idx = item['e_idx']
            param_name = item['param_name'].decode('utf-8')

            key = (l_idx, j_idx, e_idx, param_name)

            rp = rp_mapping[(l_idx, j_idx, e_idx)]
            param_list = getattr(rp, param_name)
            # Include the original value (first element)
            samples = param_list  # This includes the original value at param_list[0]

            # Initialize the list if the key is not present
            if key not in samples_dict:
                samples_dict[key] = []

            samples_dict[key].extend(samples)

        return samples_dict

    def get_param_names(self):
        """
        Returns the list of parameter names based on MPAR and LFW.
        """
        if self.MPAR == 1:
            return ['D']
        elif self.MPAR == 2:
            return ['D', 'GN']
        elif self.MPAR == 3:
            return ['D', 'GN', 'GG']
        elif self.MPAR == 4:
            if self.LFW == 0:
                return ['D', 'GN', 'GG', 'GX']
            elif self.LFW == 1:
                return ['D', 'GN', 'GG', 'GF']
        elif self.MPAR == 5:
            return ['D', 'GN', 'GG', 'GF', 'GX']
        else:
            raise ValueError(f"Unsupported MPAR value: {self.MPAR}")
        
    def update_tape(self, tape, sample_index=1, sample_name=""):
        """
        Updates the tape with the sampled parameters.

        Parameters:
        - tape: The ENDF tape object to update.
        - sample_index: Index of the sample to use.
        """
        updated_params = self.urre_data.update_resonance_parameters(sample_index)
        self._update_resonance_range(tape, updated_params)
        # if sample_index == 1:
        #     tape.to_file(f'sampled_tape_{sample_name}.endf')
        # else:
        #     tape.to_file(f'sampled_tape_{sample_index+1}.endf')

    def write_to_hdf5(self, hdf5_group):
        """
        Writes the L_matrix and urre_data to the given HDF5 group.
        """
        # Write L_matrix
        hdf5_group.create_dataset('L_matrix', data=self.L_matrix)
        
        # Write index_mapping
        hdf5_group.create_dataset('index_mapping', data=self.index_mapping)


        # Write attributes
        hdf5_group.attrs['MPAR'] = self.MPAR
        hdf5_group.attrs['LFW'] = self.LFW
        hdf5_group.attrs['NER'] = self.NER

        # Write urre_data
        urre_data_group = hdf5_group.create_group('urre_data')
        self.urre_data.write_to_hdf5(urre_data_group)

    def reconstruct_index_mapping(self, index_mapping_array):
        """
        Reconstructs the index_mapping from the stored array and urre_data.
        """
        self.index_mapping = []
        for item in index_mapping_array:
            l_idx = item['l_idx']
            j_idx = item['j_idx']
            param_name = item['param_name'].decode('utf-8')  # Decode bytes to string

            l_value = self.urre_data.Llist[l_idx]
            j_value = l_value.Jlist[j_idx]

            self.index_mapping.append((l_value, j_value, param_name))
            
    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads the L_matrix and urre_data from the given HDF5 group and returns an instance.
        """
        # Read L_matrix
        L_matrix = hdf5_group['L_matrix'][()]
        
        index_mapping_array = hdf5_group['index_mapping'][()]

        # Read attributes
        MPAR = hdf5_group.attrs['MPAR']
        LFW = hdf5_group.attrs['LFW']
        NER = hdf5_group.attrs['NER']

        # Read urre_data
        urre_data_group = hdf5_group['urre_data']
        urre_data = URREnergyDependent.read_from_hdf5(urre_data_group)

        # Create an instance without calling __init__
        instance = cls.__new__(cls)
        instance.L_matrix = L_matrix
        instance.MPAR = MPAR
        instance.LFW = LFW
        instance.NER = NER
        instance.urre_data = urre_data

        # If needed, set other attributes or perform additional initialization
        instance.index_mapping = index_mapping_array
        # instance.reconstruct_index_mapping(index_mapping_array)

        return instance