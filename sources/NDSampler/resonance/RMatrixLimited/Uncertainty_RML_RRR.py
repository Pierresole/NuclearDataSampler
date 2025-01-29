import numpy as np
from collections import defaultdict
from .Parameters_RML_RRR import RMatrixLimited
from ..ResonanceRangeCovariance import ResonanceRangeCovariance
from ENDFtk import tree
from ENDFtk.MF2.MT151 import ResonanceRange
import time

class Uncertainty_RML_RRR(ResonanceRangeCovariance):
    """
    Class for RMatrixLimited resonance range uncertainty data.
    Attributes:
    - rml_data: RMatrixLimited object.
    - covariance_matrix: The covariance matrix.
    - L_matrix: The Cholesky decomposition of the covariance matrix.
    """
    def __init__(self, mf2_resonance_ranges, mf32_resonance_range, NER):
        # Initialize urre_data
        self.NER = NER
        self.covariance_matrix = None
        start_time = time.time()
        self.rml_data = RMatrixLimited.from_endftk(mf2_resonance_ranges, mf32_resonance_range)
        print(f"Time for RMatrixLimited.from_endftk: {time.time() - start_time:.4f} seconds")
        
        self.index_mapping = []

        start_time = time.time()
        self.extract_covariance_matrix(mf32_resonance_range)
        print(f"Time for extract_covariance_matrix: {time.time() - start_time:.4f} seconds")
        
        start_time = time.time()
        self.compute_L_matrix()
        print(f"Time for compute_L_matrix: {time.time() - start_time:.4f} seconds")
        
    @classmethod
    def from_covariance_data(cls, tape, NER, covariance_data):
        """
        Initializes the AveragedBreitWigner object using covariance data.

        Parameters:
        - tape: The ENDF tape object.
        - NER: The resonance range index.
        - covariance_data: The covariance data from the HDF5 file.

        Returns:
        - An instance of AveragedBreitWigner.
        """
        instance = cls(tape, NER)
        instance.set_covariance_data(covariance_data)
        return instance

    def extract_resonance_parameters(self):
        """
        Extracts resonance parameters from the tape.
        """
        # Parse MF2 MT151
        mf2mt151 = self.tape.MAT(self.tape.material_numbers[0]).MF(2).MT(151).parse()
        mf32mt151 = self.tape.MAT(self.tape.material_numbers[0]).MF(32).MT(151).parse()
        resonance_ranges = mf2mt151.isotopes[0].resonance_ranges.to_list()
        self.resonance_range = resonance_ranges[self.NER]
        self.resonance_range32 = resonance_ranges = mf32mt151.isotopes[0].resonance_ranges[self.NER]

        # Extract MPAR and LFW from resonance parameters
        self.MPAR = self.resonance_range.parameters.MPAR
        self.LFW = self.resonance_range.parameters.LFW

        # Get parameter names
        self.param_names = self.get_param_names()

        # Extract parameters
        self.extract_parameters()

    def extract_covariance_data(self):
        """
        Extracts covariance data from the tape (MF32).
        """
        # Implement extraction of covariance data from MF32
        # This method would be used during the extraction context
        pass  # Replace with actual implementation

    def get_covariance_type(self):
        """
        Override to return the specific covariance type.
        """
        return "Uncertainty_RML_RRR"
    
    def set_covariance_data(self, covariance_data):
        """
        Sets covariance data from the HDF5 file into the object's attributes.

        Parameters:
        - covariance_data: The covariance data dictionary from the HDF5 file.
        """
        self.MPAR = covariance_data['MPAR']
        self.LFW = covariance_data['LFW']
        self.param_names = covariance_data['param_names']
        self.L_values = [{'L': L, 'J': J} for L, J in zip(covariance_data['L_values'], covariance_data['J_values'])]

        # Reconstruct parameters from covariance_data
        self.parameters = []
        groups = covariance_data['groups']
        for idx, group_key in enumerate(groups):
            group_data = groups[group_key]
            param_list = []
            for param_name in self.param_names:
                param_values = group_data[param_name]
                param_list.append(param_values)
            self.parameters.append({
                'L': self.L_values[idx]['L'],
                'J': self.L_values[idx]['J'],
                'parameters': param_list
            })

        # Set covariance matrix
        self.covariance_matrix = covariance_data['relative_covariance_matrix']
        self.num_parameters = self.covariance_matrix.shape[0]

    def sample_parameters(self, mode='stack'):
        """
        Sample resonance parameters using the computed L matrix.
        
        Args:
            mode: Either 'stack' (append) or 'replace' (overwrite first sample)
        """
        if mode not in ['stack', 'replace']:
            raise ValueError("Mode must be either 'stack' or 'replace'")
    
        # Generate standard normal random variables
        random_vector = np.random.normal(size=self.L_matrix.shape[0])
        
        # Calculate correlated samples: mean + L @ random
        
        nominal_parameters = self.rml_data.get_nominal_parameters()

        sampled_values = nominal_parameters + self.L_matrix @ random_vector
        # Map the sampled values back to the data structure using index_mapping
        for sample_idx, (spin_group_idx, resonance_idx, param_idx) in enumerate(self.index_mapping):
            spin_group = self.rml_data.ListSpinGroup[spin_group_idx]
            resonance = spin_group.ResonanceParameters[resonance_idx]
            
            # Determine which parameter list to modify (ER or GAM)
            if param_idx == 0:  # ER parameter
                param_list = resonance.ER
                sampled_value = sampled_values[sample_idx]
            else:  # GAM parameter
                param_list = resonance.GAM[param_idx - 1]
                sampled_value = sampled_values[sample_idx]

            # Update the parameter list according to the specified mode
            if mode == 'stack' or len(param_list) == 1:
                param_list.append(sampled_value)
            elif mode == 'replace':
                param_list[1] = sampled_value

    def extract_covariance_matrix(self, mf32_range):
        """
        Extracts the covariance matrix using the method from the base class.
        """
        if mf32_range.parameters.LCOMP == 2:
            self.extract_covariance_matrix_LCOMP2(mf32_range, True)
        else:
            raise ValueError(f"Unsupported LCOMP value: {mf32_range.parameters.LCOMP}")
        
        self.index_mapping = [
            (J_idx, R_idx, P_idx)
            for J_idx, spingroup in enumerate(self.rml_data.ListSpinGroup)
            for R_idx, R_value in enumerate(spingroup.ResonanceParameters)
            for P_idx in range(len(R_value.GAM) + 1)  # Number of channels plus the energy
        ]
    
    def extract_covariance_matrix_LCOMP2(self, mf32_range, to_reduced: bool = True):
        """
        Reconstructs the covariance matrix from standard deviations and correlation coefficients when LCOMP == 2.
        """
        cm = mf32_range.parameters.correlation_matrix
        NNN = cm.NNN  # Order of the correlation matrix
        correlations = cm.correlations  # List of correlation coefficients
        I = cm.I  # List of row indices (one-based)
        J = cm.J  # List of column indices (one-based)
        
        NParams = sum( ((sgroup.NCH + 1 ) * sgroup.NRSA) for sgroup in mf32_range.parameters.uncertainties.spin_groups.to_list())
        if NNN != NParams:
            raise ValueError(f"Mismatch between number of parameters ({NParams}) and size of correlation matrix (NNN={NNN}).")
        
        # Initialize the correlation matrix
        correlation_matrix = np.identity(NParams)
        
        # Fill in the off-diagonal elements
        for idx, corr_value in enumerate(correlations):
            i = I[idx] - 1  # Convert to zero-based index
            j = J[idx] - 1  # Convert to zero-based index
            correlation_matrix[i, j] = corr_value
            correlation_matrix[j, i] = corr_value  # Symmetric matrix
        
        # Now, compute the covariance matrix
        std_devs = self.rml_data.get_non_none_std_devs()
        # for spingroup in mf32_range.parameters.uncertainties.spin_groups.to_list():
        #     for iER, DER in enumerate(spingroup.parameters.DER[:]):
        #         std_devs.append(DER)
        #         for iCH in range(spingroup.NCH):
        #             std_devs.append(spingroup.parameters.DGAM[iER][iCH])

        covariance_matrix = np.outer(std_devs, std_devs) * correlation_matrix
        
        # Convert to reduced covariance matrix
        if to_reduced:
            covariance_matrix = self.rml_data.extract_covariance_matrix_LCOMP2(covariance_matrix)
        
        self.covariance_matrix = covariance_matrix

    def construct_mean_vector(self):
        """
        Constructs the mean vector from the mean parameters.
        """
        mean_values = []
        for group in self.parameters:
            for param_values in group['parameters']:
                # The mean of relative deviations is zero, but we store zeros for consistency
                mean_values.append(0.0)
        self.mean_vector = np.array(mean_values)

    def remove_zero_variance_parameters(self):
        """
        Removes parameters with zero variance and updates the covariance matrix accordingly.
        """
        # Identify zero variance parameters
        zero_variance_indices = np.where(np.diag(self.covariance_matrix) == 0.0)[0]
        if zero_variance_indices.size == 0:
            # No zero variance parameters
            return

        # Map covariance matrix indices to (group_idx, param_idx)
        index_mapping = []
        for group_idx, group in enumerate(self.parameters):
            num_params_in_group = len(group['parameters'])
            for param_idx in range(num_params_in_group):
                index_mapping.append((group_idx, param_idx))

        # Identify parameters to delete
        parameters_to_delete = [index_mapping[idx] for idx in zero_variance_indices]

        # Group parameters to delete by group index
        parameters_to_delete_by_group = defaultdict(list)
        for group_idx, param_idx in parameters_to_delete:
            parameters_to_delete_by_group[group_idx].append(param_idx)

        # Delete parameters from groups
        for group_idx, param_indices in parameters_to_delete_by_group.items():
            group = self.parameters[group_idx]
            # Delete parameters in reverse order
            for param_idx in sorted(param_indices, reverse=True):
                del group['parameters'][param_idx]
            # If no parameters remain, mark for deletion
            if not group['parameters']:
                group['delete'] = True

        # Remove groups marked for deletion
        self.parameters = [group for group in self.parameters if not group.get('delete', False)]

        # Update L_values and num_LJ_groups
        self.L_values = [{'L': group['L'], 'J': group['J']} for group in self.parameters]
        self.num_parameters = len(self.L_values) * len(self.param_names)

        # Delete rows and columns from covariance matrix
        self.covariance_matrix = np.delete(self.covariance_matrix, zero_variance_indices, axis=0)
        self.covariance_matrix = np.delete(self.covariance_matrix, zero_variance_indices, axis=1)

        # Update mean vector
        self.mean_vector = np.delete(self.mean_vector, zero_variance_indices)

    def compute_L_matrix(self):
        """
        Computes the Cholesky decomposition (L matrix) of the covariance matrix.
        """
        try:
            self.L_matrix = np.linalg.cholesky(self.covariance_matrix)
        except np.linalg.LinAlgError:
            # Handle non-positive definite covariance matrix
            eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix)
            eigenvalues[eigenvalues < 0] = 0
            self.L_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    def update_tape(self, tape, sample_index=1, sample_name=""):
        """
        Updates the tape with the sampled parameters.

        Parameters:
        - tape: The ENDF tape object to update.
        - sample_index: Index of the sample to use.
        """
        self._update_resonance_range(tape, updated_parameters = self.rml_data.reconstruct(sample_index))
        # if sample_index == 1:
        #     tape.to_file(f'sampled_tape_{sample_name}.endf')
        # else:
        #     tape.to_file(f'sampled_tape_{sample_index+1}.endf')

    def write_to_hdf5(self, hdf5_group):
        """
        Writes the L_matrix and rml_data to the given HDF5 group.
        """
        hdf5_group.attrs['NER'] = self.NER
        
        # Write L_matrix
        hdf5_group.create_dataset('L_matrix', data=self.L_matrix)

        # Write index_mapping as a compound dataset
        dt = np.dtype([('spin_group', 'i4'), ('resonance', 'i4'), ('parameter', 'i4')])
        index_data = np.array(self.index_mapping, dtype=dt)
        hdf5_group.create_dataset('index_mapping', data=index_data)

        # Write rml_data
        rml_data_group = hdf5_group.create_group('Parameters')
        self.rml_data.write_to_hdf5(rml_data_group)
        
    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads the L_matrix and rml_data from the given HDF5 group and returns an instance.
        """
        NER = hdf5_group.attrs['NER']
        
        # Read L_matrix
        L_matrix = hdf5_group['L_matrix'][()]

        # Read index_mapping
        index_data = hdf5_group['index_mapping'][()]
        index_mapping = [(int(idx['spin_group']), int(idx['resonance']), int(idx['parameter'])) 
                        for idx in index_data]

        # Read rml_data
        rml_data_group = hdf5_group['Parameters']
                
        rml_data = RMatrixLimited.read_from_hdf5(rml_data_group)
        
        # Create an instance and set attributes
        instance = cls.__new__(cls)
        instance.NER = NER
        instance.L_matrix = L_matrix
        instance.rml_data = rml_data
        instance.index_mapping = index_mapping

        return instance