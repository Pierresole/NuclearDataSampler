import numpy as np
from collections import defaultdict
from .Parameters import ResonanceData
from ENDFtk import tree  

class RRRReichMooreUncertainty:
    def __init__(self, mf2_resonance_ranges, mf32_resonance_range, NER):
        self.MPAR = mf32_resonance_range.parameters.covariance_matrix.MPAR  # Number of parameters per (L,J) group
        self.LFW = mf32_resonance_range.parameters.LFW  # Indicates if fission widths are present
        self.mf2_range  = mf2_resonance_ranges
        self.mf32_range = mf32_resonance_range

        # Initialize urre_data
        self.urre_data = ResonanceData(
            SPI=self.resonance_parameters.SPI,
            AP=self.resonance_parameters.AP,
            LSSF=self.resonance_parameters.LSSF
        )

        # Extract parameters and covariance matrices
        self.extract_parameters(mf2_resonance_ranges)
        self.extract_covariance_matrix()
        self.remove_zero_variance_parametersURR()
        self.compute_L_matrix()
        
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

    def sample_parameters(self):
        """
        Samples new parameters based on the covariance matrix.
        """
        if self.L_matrix is None:
            # Compute L_matrix if not already computed
            self.compute_L_matrix()

        # Generate standard normal random variables
        N = np.random.normal(size=self.L_matrix.shape[0])

        # Compute sampled relative deviations
        Y = self.L_matrix @ N  # Y has size (num_parameters,)

        # Apply deviations to parameters
        idx_in_Y = 0
        sampled_parameters = []

        for group in self.parameters:
            sampled_group = {'L': group['L'], 'J': group['J'], 'sampled_parameters': {}}
            for param_name, mean_values in zip(self.param_names, group['parameters']):
                relative_deviation = Y[idx_in_Y]
                idx_in_Y += 1
                # Apply the deviation uniformly to the parameter array
                sampled_values = mean_values * (1 + relative_deviation)
                sampled_group['sampled_parameters'][param_name] = sampled_values
            sampled_parameters.append(sampled_group)

        return sampled_parameters

    def extract_data(self):
        """
        Extracts resonance parameters and covariance data from the tape.
        """
        # Parse MF2 MT151
        mf2mt151 = self.tape.MAT(self.tape.material_numbers[0]).MF(2).MT(151).parse()
        isotope = mf2mt151.isotopes[0]
        resonance_ranges = isotope.resonance_ranges.to_list()
        self.resonance_range = resonance_ranges[self.NER]

        # Extract MPAR and LFW
        self.MPAR = self.resonance_range.parameters.covariance_matrix.MPAR
        self.LFW = self.resonance_range.parameters.LFW

        # Get parameter names
        self.param_names = self.get_param_names()

        # Extract parameters
        self.extract_parameters()

        # Extract covariance matrix
        self.extract_covariance_matrix()

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

    def extract_parameters(self):
        """
        Extracts the mean parameters from the resonance range and constructs the parameter list.
        """
        l_values = self.resonance_range.parameters.l_values.to_list()

        parameters = []

        for l_value in l_values:
            L = l_value.L
            j_values = l_value.j_values.to_list()
            for j_value in j_values:
                J = j_value.AJ
                # Store (L, J) group information
                self.L_values.append({'L': L, 'J': J, 'num_energies': len(j_value.ES)})

                # For each parameter type in MPAR, store the mean values
                param_list = []
                for param_name in self.param_names:
                    # Access the attribute directly
                    mean_values = np.array(getattr(j_value, param_name))
                    param_list.append(mean_values)
                    # Update total number of parameters
                    self.num_parameters += 1

                parameters.append({
                    'L': L,
                    'J': J,
                    'parameters': param_list
                })

        self.parameters = parameters

    def extract_covariance_matrix(self):
        """
        Extracts the relative covariance matrix and constructs it according to the parameter ordering.
        """
        covariance_data = self.resonance_range.parameters.covariance_matrix
        covariance_order = covariance_data.NPAR  # Order of the matrix

        # Build the covariance matrix from the upper triangular data
        relative_cov_matrix_upper = covariance_data.covariance_matrix  # 1D array
        relative_cov_matrix = np.zeros((covariance_order, covariance_order))
        idx = 0
        for i in range(covariance_order):
            for j in range(i, covariance_order):
                relative_cov_matrix[i, j] = relative_cov_matrix_upper[idx]
                if i != j:
                    relative_cov_matrix[j, i] = relative_cov_matrix_upper[idx]
                idx += 1

        self.covariance_matrix = relative_cov_matrix

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

    def update_tape(self, sampled_parameters):
        """
        Updates the ENDF tape with the sampled parameters.

        Parameters:
        - sampled_parameters: List of dictionaries containing sampled parameters per group.
        """
        # Create a copy of the resonance ranges
        mf2mt151 = self.tape.MAT(self.tape.material_numbers[0]).MF(2).MT(151).parse()
        isotope = mf2mt151.isotopes[0]
        resonance_ranges = isotope.resonance_ranges.copy()

        # Update the resonance range with matching NER
        resonance_ranges[self.NER] = self._update_resonance_range(self.resonance_range, sampled_parameters)

        # Create new isotope with updated resonance ranges
        new_isotope = tree.Isotope(
            ZAI=isotope.ZAI,
            ABN=isotope.ABN,
            LFW=isotope.LFW,
            resonance_ranges=resonance_ranges
        )

        # Create new section with the updated isotope
        new_section = tree.MF2(
            ZA=mf2mt151.ZA,
            AWR=mf2mt151.AWR,
            isotopes=[new_isotope]
        )

        # Replace the existing section in the tape
        mat_num = self.tape.material_numbers[0]
        self.tape.MAT(mat_num).MF(2).MT(151).data = new_section

    def _update_resonance_range(self, resonance_range, sampled_parameters):
        """
        Updates a resonance range with the sampled parameters.

        Parameters:
        - resonance_range: The original resonance range.
        - sampled_parameters: List of dictionaries containing sampled parameters per group.

        Returns:
        - new_resonance_range: The updated resonance range.
        """
        # Build new l_values with updated parameters
        l_values = resonance_range.parameters.l_values.to_list()
        new_l_values = []

        # Map (L, J) to sampled parameters
        sampled_params_dict = {}
        for group in sampled_parameters:
            L = group['L']
            J = group['J']
            sampled_params = group['sampled_parameters']
            sampled_params_dict[(L, J)] = sampled_params

        for l_value in l_values:
            L = l_value.L
            j_values = l_value.j_values.to_list()
            new_j_values = []

            for j_value in j_values:
                J = j_value.AJ

                # Get the sampled parameters for this (L, J)
                key = (L, J)
                if key in sampled_params_dict:
                    sampled_params = sampled_params_dict[key]

                    # Create new j_value with updated parameters
                    new_j_value = tree.UnresolvedEnergyDependent.JValue(
                        AJ=J,
                        AMUN=j_value.AMUN,
                        AMUG=j_value.AMUG,
                        AMUF=j_value.AMUF,
                        AMUX=j_value.AMUX,
                        INT=j_value.INT,
                        ES=j_value.ES[:],  # Energies remain the same
                        D=sampled_params.get('D', j_value.D).tolist(),
                        GN=sampled_params.get('GN', j_value.GN).tolist(),
                        GG=sampled_params.get('GG', j_value.GG).tolist(),
                        GF=sampled_params.get('GF', j_value.GF).tolist(),
                        GX=sampled_params.get('GX', j_value.GX).tolist()
                    )
                else:
                    # No sampled parameters for this (L, J), keep original
                    new_j_value = j_value

                new_j_values.append(new_j_value)

            # Create new l_value with updated j_values
            new_l_value = tree.UnresolvedEnergyDependent.LValue(
                AWRI=l_value.AWRI,
                L=L,
                j_values=new_j_values
            )

            new_l_values.append(new_l_value)

        # Create new parameters object
        new_parameters = tree.UnresolvedEnergyDependent(
            SPI=resonance_range.parameters.SPI,
            AP=resonance_range.parameters.AP,
            LSSF=resonance_range.parameters.LSSF,
            L_values=new_l_values
        )

        # Create new resonance range
        new_resonance_range = tree.ResonanceRange(
            EL=resonance_range.EL,
            EH=resonance_range.EH,
            NAPS=resonance_range.NAPS,
            parameters=new_parameters
        )

        return new_resonance_range
