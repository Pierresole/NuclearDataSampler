import bisect
import numpy as np
import ENDFtk

#-----------------
# Base class   
#-----------------

class ResonanceCovariance:
    def __init__(self, resonance_range, NER):
        """
        Base class for resonance covariance data.

        Parameters:
        - resonance_range: The resonance range object from MF32.
        - NER: Energy range index (integer).
        """
        self.resonance_range = resonance_range
        self.NER = NER  # Energy range identifier
        self.LRF = resonance_range.LRF  # Resonance formalism flag
        self.LRU = resonance_range.LRU  # Resonance type (resolved or unresolved)
        self.resonance_parameters = resonance_range.parameters
        self.covariance_matrix = None
        self.parameters = None
        self.AP = None  # Scattering radius
        self.DAP = None  # Scattering radius uncertainty

    @classmethod
    def from_resonance_range(cls, resonance_range, mf2_resonance_ranges, NER):
        LRU = resonance_range.LRU
        LRF = resonance_range.LRF
        if LRU == 1 and LRF == 2:
            return MultiLevelBreitWignerCovariance(resonance_range, mf2_resonance_ranges, NER)
        if LRU == 1 and LRF == 3:
            return ReichMooreCovariance(resonance_range, mf2_resonance_ranges, NER)
        if LRU == 1 and LRF == 7:
            return RMatrixLimitedCovariance(resonance_range, mf2_resonance_ranges, NER)
        elif LRU == 2 and LRF == 1:
            return AveragedBreitWignerCovariance(resonance_range, mf2_resonance_ranges, NER)
        else:
            raise NotImplementedError("Resonance covariance format not supported")
     
    #-----------------
    # Matrix operator
    #-----------------
    
    def extract_covariance_matrix_LCOMP2(self):
        """
        Reconstructs the covariance matrix from standard deviations and correlation coefficients when LCOMP == 2.
        """
        cm = self.resonance_parameters.correlation_matrix
        NNN = cm.NNN  # Order of the correlation matrix
        correlations = cm.correlations  # List of correlation coefficients
        I = cm.I  # List of row indices (one-based)
        J = cm.J  # List of column indices (one-based)
        
        # Initialize the correlation matrix
        correlation_matrix = np.identity(NNN)
        
        # Fill in the off-diagonal elements
        for idx, corr_value in enumerate(correlations):
            i = I[idx] - 1  # Convert to zero-based index
            j = J[idx] - 1  # Convert to zero-based index
            correlation_matrix[i, j] = corr_value
            correlation_matrix[j, i] = corr_value  # Symmetric matrix
        
        # Now, compute the covariance matrix
        self.covariance_matrix = np.outer(self.std_dev_vector, self.std_dev_vector) * correlation_matrix
           
    def delete_parameters(self, indices_to_delete):
        """
        Deletes parameters by indices and updates the covariance matrix and parameters list.

        Parameters:
        - indices_to_delete: List of indices of parameters to delete.
        """
        # Ensure indices are sorted in descending order to avoid index shifting issues
        indices_to_delete = sorted(indices_to_delete, reverse=True)

        # Delete rows and columns from the covariance matrix
        self.covariance_matrix = np.delete(self.covariance_matrix, indices_to_delete, axis=0)
        self.covariance_matrix = np.delete(self.covariance_matrix, indices_to_delete, axis=1)
        
        # Delete parameters from the list
        for idx in indices_to_delete:
            del self.parameters[idx]
        
        # Update indices in parameters
        for idx, param in enumerate(self.parameters):
            param['index'] = idx
        
        # Update mean vector and standard deviation vector
        self.mean_vector = np.delete(self.mean_vector, indices_to_delete)
        if hasattr(self, 'std_dev_vector') and self.std_dev_vector is not None:
            self.std_dev_vector = np.delete(self.std_dev_vector, indices_to_delete)
        
        # Update NPAR
        self.NPAR = self.covariance_matrix.shape[0]

    def remove_zero_variance_parameters(self):
        """
        Removes parameters with zero variance and updates the covariance matrix accordingly.
        """
        # Identify parameters with non-zero standard deviation
        if hasattr(self, 'std_dev_vector'):
            non_zero_indices = np.where(self.std_dev_vector != 0.0)[0]
        else:
            non_zero_indices = np.where(np.diag(self.covariance_matrix) != 0.0)[0]

        # Update parameters and vectors
        self.parameters = [self.parameters[i] for i in non_zero_indices]
        self.mean_vector = self.mean_vector[non_zero_indices]
        if hasattr(self, 'std_dev_vector'):
            self.std_dev_vector = self.std_dev_vector[non_zero_indices]
        # Update the covariance matrix
        self.covariance_matrix = self.covariance_matrix[np.ix_(non_zero_indices, non_zero_indices)]

    def compute_L_matrix(self):
        """
        Computes the decomposition of the covariance matrix and stores it as L_matrix.
        """
        try:
            # Attempt Cholesky decomposition
            self.L_matrix = np.linalg.cholesky(self.covariance_matrix)
            self.is_cholesky = True  # Indicate that L_matrix is a Cholesky decomposition
        except np.linalg.LinAlgError:
            # Handle non-positive definite covariance matrix
            # Use eigenvalue decomposition as a fallback
            eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix)
            # Ensure all eigenvalues are non-negative
            eigenvalues[eigenvalues < 0] = 0
            # Reconstruct L_matrix
            self.L_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            self.is_cholesky = False  # Indicate that L_matrix is not a Cholesky decomposition

    # def remove_zero_variance_parameters(self, tolerance=1e-12):
    #     """
    #     Identifies and removes parameters with zero variance.

    #     Parameters:
    #     - tolerance: The threshold below which a variance is considered zero.
    #     """
    #     zero_variance_indices = self.identify_zero_variance_parameters(tolerance=tolerance)
    #     if zero_variance_indices:
    #         self.delete_parameters(zero_variance_indices)

    # def identify_zero_variance_parameters(self, tolerance=1e-12):
    #     """
    #     Identifies the indices of parameters with variance less than or equal to the tolerance.

    #     Parameters:
    #     - tolerance: The threshold below which a variance is considered zero.

    #     Returns:
    #     - A list of indices of parameters to delete.
    #     """
    #     variances = np.diag(self.covariance_matrix)
    #     zero_variance_indices = [i for i, var in enumerate(variances) if abs(var) <= tolerance]
    #     return zero_variance_indices

    #-----------------
    # Helper functions
    #-----------------

    def _find_nearest_energy(self, energy_list, target_energy, tolerance=1e-5):
        """
        Finds the index of the energy in energy_list that matches target_energy within a tolerance.

        Parameters:
        - energy_list: List of sorted energies.
        - target_energy: The energy value to match.
        - tolerance: The acceptable difference between energies.

        Returns:
        - The index of the matching energy in energy_list, or None if not found.
        """
        idx = bisect.bisect_left(energy_list, target_energy)
        # Check the left neighbor
        if idx > 0 and abs(energy_list[idx - 1] - target_energy) <= tolerance:
            return idx - 1
        # Check the right neighbor
        if idx < len(energy_list) and abs(energy_list[idx] - target_energy) <= tolerance:
            return idx
        return None
        
    #-----------------
    # Communication  
    #-----------------

    def write_to_hdf5(self, hdf5_group):
        """
        Writes the covariance data to an HDF5 group.
        """
        # Write L_matrix
        hdf5_group.create_dataset('L_matrix', data=self.L_matrix)
        # Write mean_vector
        if hasattr(self, 'mean_vector'):
            hdf5_group.create_dataset('mean_vector', data=self.mean_vector)
        # Write standard deviations if available
        if hasattr(self, 'std_dev_vector'):
            hdf5_group.create_dataset('std_dev_vector', data=self.std_dev_vector)
        # Indicate if L_matrix is a Cholesky decomposition
        hdf5_group.attrs['is_cholesky'] = self.is_cholesky
        # Call the derived class method to write format-specific data
        self.write_additional_data_to_hdf5(hdf5_group)


    def print_parameters(self):
        """
        Prints the parameters. This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

#-----------------
# Formalism   
#-----------------

class MultiLevelBreitWignerCovariance(ResonanceCovariance):
    def __init__(self, resonance_range, mf2_resonance_ranges, NER):
        super().__init__(resonance_range, NER)
        self.LCOMP = self.resonance_parameters.LCOMP
        self.LFW = getattr(self.resonance_range, 'LFW', None)
        self.extract_parameters(mf2_resonance_ranges)
        self.extract_covariance_matrix()

    #-----------------
    # Parameter
    #-----------------

    def extract_parameters(self, mf2_resonance_ranges):
        """
        Extracts parameters based on LCOMP value.
        """
        if self.LCOMP == 0:
            raise NotImplementedError("Multiple Short Range not implemented in LCOMP=0")
            # self.extract_parameters_LCOMP0(mf2_resonance_ranges)
        elif self.LCOMP == 1:
            self.extract_parameters_LCOMP1(mf2_resonance_ranges)
        elif self.LCOMP == 2:
            self.extract_parameters_LCOMP2(mf2_resonance_ranges)
        else:
            raise ValueError(f"Unsupported LCOMP value: {self.LCOMP}")

    def extract_parameters_LCOMP0(self, mf2_resonance_ranges):
        """
        Extracts parameters when LCOMP == 0.
        Only uncertainties (standard deviations) are given; parameters are uncorrelated.
        """
        uncertainties = self.resonance_parameters.uncertainties

        # Standard deviations
        DER = uncertainties.DER
        DGN = uncertainties.DGN
        DGG = uncertainties.DGG
        DGF = uncertainties.DGF if self.LFW != 0 else []

        # Mean values
        ER = uncertainties.ER
        GN = uncertainties.GN
        GG = uncertainties.GG
        GF = uncertainties.GF if self.LFW != 0 else []

        # Map resonances from MF2 to MF32
        self._map_resonances(mf2_resonance_ranges, ER)

        # Build parameters list
        self._build_parameters_list(ER, GN, GG, GF, DER, DGN, DGG, DGF)

    def extract_parameters_LCOMP1(self, mf2_resonance_ranges):
        """
        Extracts parameters when LCOMP == 1.
        The full covariance matrix is provided.
        """
        uncertainties = self.resonance_parameters.uncertainties

        # Mean values
        ER = uncertainties.ER
        GN = uncertainties.GN
        GG = uncertainties.GG
        GF = uncertainties.GF if self.LFW != 0 else []

        # No standard deviations are given explicitly; they are derived from the covariance matrix

        # Map resonances from MF2 to MF32
        self._map_resonances(mf2_resonance_ranges, ER)

        # Build parameters list (std_dev will be calculated later)
        self._build_parameters_list(ER, GN, GG, GF, None, None, None, None)

    def extract_parameters_LCOMP2(self, mf2_resonance_ranges):
        """
        Extracts parameters when LCOMP == 2.
        Standard deviations and correlation matrix are provided.
        """
        uncertainties = self.resonance_parameters.uncertainties

        # Standard deviations
        DER = uncertainties.DER
        DGN = uncertainties.DGN
        DGG = uncertainties.DGG
        DGF = uncertainties.DGF if self.LFW != 0 else []

        # Mean values
        ER = uncertainties.ER
        GN = uncertainties.GN
        GG = uncertainties.GG
        GF = uncertainties.GF if self.LFW != 0 else []

        # Map resonances from MF2 to MF32
        self._map_resonances(mf2_resonance_ranges, ER)

        # Build parameters list
        self._build_parameters_list(ER, GN, GG, GF, DER, DGN, DGG, DGF)

    def _map_resonances(self, mf2_resonance_ranges, ER):
        """
        Maps resonances from MF2 to MF32.
        """
        mf2_resonance_range = mf2_resonance_ranges[self.NER]

        mf2_resonances = []
        for l_value in mf2_resonance_range.parameters.l_values.to_list():
            L = l_value.L
            AJ = l_value.AJ
            ER_mf2 = l_value.ER
            for i, ER_value in enumerate(ER_mf2):
                J = AJ[i]
                mf2_resonances.append({'ER': ER_value, 'L': L, 'J': J})

        if len(mf2_resonances) != len(ER):
            raise ValueError("Mismatch in number of resonances between MF2 and MF32")

        self.mf2_resonances = mf2_resonances

    def _build_parameters_list(self, ER, GN, GG, GF, DER, DGN, DGG, DGF):
        """
        Builds the parameters list.
        """
        parameters = []
        index = 0
        for i, resonance in enumerate(self.mf2_resonances):
            L = resonance['L']
            J = resonance['J']
            iE2 = i

            param_list = []
            param_list.append((1, 'ER', ER[i], DER[i] if DER else None))
            param_list.append((2, 'GN', GN[i], DGN[i] if DGN else None))
            param_list.append((3, 'GG', GG[i], DGG[i] if DGG else None))
            if self.LFW != 0 and GF:
                param_list.append((4, 'GF', GF[i], DGF[i] if DGF else None))

            for param_type_index, param_name, mean_value, std_dev in param_list:
                param_dict = {
                    'index': index,
                    'L': L,
                    'J': J,
                    'iE2': iE2,
                    'type': param_type_index,  # 1: ER, 2: GN, 3: GG, 4: GF
                    'name': param_name,
                    'mean': mean_value,
                    'std_dev': std_dev
                }
                parameters.append(param_dict)
                index += 1

        self.parameters = parameters
        self.mean_vector = np.array([param['mean'] for param in self.parameters])
        if parameters[0]['std_dev'] is not None:
            self.std_dev_vector = np.array([param['std_dev'] for param in self.parameters])
        else:
            self.std_dev_vector = None  # Will be calculated from covariance matrix

    #-----------------
    # Covariance
    #-----------------

    def extract_covariance_matrix(self):
        """
        Extracts covariance matrix based on LCOMP value.
        """
        if self.LCOMP == 0:
            self.extract_covariance_matrix_LCOMP0()
        elif self.LCOMP == 1:
            self.extract_covariance_matrix_LCOMP1()
        elif self.LCOMP == 2:
            self.extract_covariance_matrix_LCOMP2()
        else:
            raise ValueError(f"Unsupported LCOMP value: {self.LCOMP}")

    def extract_covariance_matrix_LCOMP0(self):
        """
        Constructs a diagonal covariance matrix when LCOMP == 0.
        """
        if self.std_dev_vector is None:
            raise ValueError("Standard deviations are required for LCOMP=0")
        covariance_matrix = np.diag(self.std_dev_vector ** 2)
        self.covariance_matrix = covariance_matrix

    def extract_covariance_matrix_LCOMP1(self):
        """
        Uses the covariance matrix provided in the data when LCOMP == 1.
        """
        cm = self.resonance_parameters.covariance_matrix
        NPAR = cm.NPAR
        cov_upper = cm.covariance_matrix[:]
        # Reconstruct full covariance matrix
        covariance_matrix = np.zeros((NPAR, NPAR))
        idx = 0
        for i in range(NPAR):
            for j in range(i, NPAR):
                covariance_matrix[i, j] = cov_upper[idx]
                covariance_matrix[j, i] = cov_upper[idx]
                idx += 1
        self.covariance_matrix = covariance_matrix
        # Extract standard deviations
        self.std_dev_vector = np.sqrt(np.diag(covariance_matrix))

    def print_parameters(self):
        """
        Prints the parameters.
        """
        print(f"Parameters for NER={self.NER}, LCOMP={self.LCOMP}:")
        for param in self.parameters:
            std_dev = param['std_dev'] if param['std_dev'] is not None else 'N/A'
            print(f"Index: {param['index']}, L: {param['L']}, J: {param['J']}, "
                  f"Type: {param['type']}, Name: {param['name']}, Mean: {param['mean']}, StdDev: {std_dev}")

        # Print scattering radius and its uncertainty if available
        if self.AP is not None:
            DAP = self.DAP if self.DAP is not None else 'N/A'
            print(f"Scattering radius AP: {self.AP}, DAP: {DAP}")


class ReichMooreCovariance(ResonanceCovariance):
    def __init__(self, resonance_range, mf2_resonance_ranges, NER):
        super().__init__(resonance_range, NER)
        self.LCOMP = self.resonance_parameters.LCOMP
        self.LFW = getattr(self.resonance_range, 'LFW', None)
        
        # Scattering radius and its uncertainties
        self.AP = self.resonance_parameters.AP  # Scattering radius AP
        DAP_obj = self.resonance_parameters.DAP  # DAP object
        self.DAP = DAP_obj.DAP if hasattr(DAP_obj, 'DAP') else None  # Uncertainty DAP
        self.DAPL = DAP_obj.DAPL if hasattr(DAP_obj, 'DAPL') else []  # List of DAPL uncertainties per L group

        self.extract_parameters(mf2_resonance_ranges)
        
        self.extract_covariance_matrix()
        self.remove_zero_variance_parameters()
        
        self.compute_L_matrix()

    #-----------------
    # Parameter
    #-----------------

    def extract_parameters(self, mf2_resonance_ranges):
        """
        Extracts parameters based on LCOMP value.
        """
        if self.LCOMP == 0:
            raise NotImplementedError("Multiple Short Range not implemented in LCOMP=0")
            # self.extract_parameters_LCOMP0(mf2_resonance_ranges)
        elif self.LCOMP == 1:
            self.extract_parameters_LCOMP1(mf2_resonance_ranges)
        elif self.LCOMP == 2:
            self.extract_parameters_LCOMP2(mf2_resonance_ranges)
        else:
            raise ValueError(f"Unsupported LCOMP value: {self.LCOMP}")

    def extract_parameters_LCOMP0(self, mf2_resonance_ranges):
        """
        Extracts parameters when LCOMP == 0.
        Only uncertainties (standard deviations) are given; parameters are uncorrelated.
        """
        uncertainties = self.resonance_parameters.uncertainties

        # Standard deviations
        DER = uncertainties.DER
        DGN = uncertainties.DGN
        DGG = uncertainties.DGG
        DGFA = uncertainties.DGFA
        DGFB = uncertainties.DGFB

        # Mean values
        ER = uncertainties.ER
        GN = uncertainties.GN
        GG = uncertainties.GG
        GFA = uncertainties.GFA
        GFB = uncertainties.GFB

        # Map resonances from MF2 to MF32
        self._map_resonances(mf2_resonance_ranges)

        # Build parameters list
        self._build_parameters_list(ER, GN, GG, GFA, GFB, DER, DGN, DGG, DGFA, DGFB)

    def extract_parameters_LCOMP1(self, mf2_resonance_ranges):
        """
        Extracts parameters when LCOMP == 1.
        The full covariance matrix is provided.
        """
        if self.resonance_parameters.NSRS != 1:
            raise NotImplementedError("Multiple Short Range not implemented in LCOMP=1")
        if self.resonance_parameters.NLRS != 0:
            raise NotImplementedError("Long Range not implemented in LCOMP=1")
        
        uncertainties = self.resonance_parameters.short_range_blocks[0]

        # Mean values
        ER = uncertainties.ER
        GN = uncertainties.GN
        GG = uncertainties.GG
        GFA = uncertainties.GFA
        GFB = uncertainties.GFB

        # Map resonances from MF2 to MF32
        self._map_resonances(mf2_resonance_ranges)

        # Build parameters list (std_dev will be calculated later)
        self._build_parameters_list(ER, GN, GG, GFA, GFB, None, None, None, None, None)

    def extract_parameters_LCOMP2(self, mf2_resonance_ranges):
        """
        Extracts parameters when LCOMP == 2.
        Standard deviations and correlation matrix are provided.
        """
        uncertainties = self.resonance_parameters.uncertainties

        # Standard deviations
        DER = uncertainties.DER
        DGN = uncertainties.DGN
        DGG = uncertainties.DGG
        DGFA = uncertainties.DGFA
        DGFB = uncertainties.DGFB

        # Mean values
        ER = uncertainties.ER
        GN = uncertainties.GN
        GG = uncertainties.GG
        GFA = uncertainties.GFA
        GFB = uncertainties.GFB

        # Map resonances from MF2 to MF32
        self._map_resonances(mf2_resonance_ranges)

        # Build parameters list
        self._build_parameters_list(ER, GN, GG, GFA, GFB, DER, DGN, DGG, DGFA, DGFB)

    def _map_resonances(self, mf2_resonance_ranges):
        """
        Maps resonances from MF2 to MF32 by matching resonance energies.

        Also extracts APL and DAPL per L group.
        """
        mf2_resonance_range = mf2_resonance_ranges[self.NER]
        l_values = mf2_resonance_range.parameters.l_values.to_list()
        self.APLs = []
        self.DAPLs = []

        # Collect APL and DAPL values per L group
        for idx_L, l_value in enumerate(l_values):
            L = l_value.L
            APL = l_value.APL if hasattr(l_value, 'APL') else None
            self.APLs.append({'L': L, 'APL': APL})

            if self.DAPL and idx_L < len(self.DAPL):
                DAPL = self.DAPL[idx_L]
            else:
                DAPL = self.DAP

            self.DAPLs.append({'L': L, 'DAPL': DAPL})

        # Collect resonances from MF32
        uncertainties = self.resonance_parameters.uncertainties
        ER_mf32 = uncertainties.ER  # Energies in MF32

        # Prepare to map resonances
        self.matched_resonances = []
        tolerance = 1e-5  # Adjust tolerance as needed

        import bisect

        # Loop over each ER in MF32
        for idx_mf32, ER_value_mf32 in enumerate(ER_mf32):
            found_match = False
            # Loop over each l_value group in MF2
            for idx_L, l_value in enumerate(l_values):
                L = l_value.L
                ER_mf2 = l_value.ER  # Energies in MF2 for this L group
                NRS = l_value.NRS

                # Since ER_mf2 is sorted, use bisect to find the insertion point
                idx = bisect.bisect_left(ER_mf2, ER_value_mf32)
                # Check the left neighbor
                if idx > 0 and abs(ER_mf2[idx - 1] - ER_value_mf32) <= tolerance:
                    resonance_index = idx - 1
                    found_match = True
                # Check the right neighbor
                elif idx < NRS and abs(ER_mf2[idx] - ER_value_mf32) <= tolerance:
                    resonance_index = idx
                    found_match = True

                if found_match:
                    J = l_value.AJ[resonance_index]
                    self.matched_resonances.append({
                        'mf32_index': idx_mf32,
                        'L': L,
                        'J': J,
                        'L_index': idx_L,
                        'resonance_index': resonance_index
                    })
                    break  # Stop searching after finding a match

            if not found_match:
                raise ValueError(f"Resonance energy {ER_value_mf32} in MF32 not found in MF2 within tolerance.")

    def _build_parameters_list(self, ER, GN, GG, GFA, GFB, DER, DGN, DGG, DGFA, DGFB):
        """
        Builds the parameters list with mapping information and standard deviations.

        Parameters:
        - ER, GN, GG, GFA, GFB: Mean values of the parameters from MF32.
        - DER, DGN, DGG, DGFA, DGFB: Standard deviations of the parameters from MF32.
        """
        parameters = []
        index = 0

        for matched_resonance in self.matched_resonances:
            mf32_idx = matched_resonance['mf32_index']
            L = matched_resonance['L']
            J = matched_resonance['J']
            L_index = matched_resonance['L_index']
            resonance_index = matched_resonance['resonance_index']

            # List of parameter tuples: (param_type_index, param_name, mean_value, std_dev)
            param_list = []
            param_list.append((1, 'ER', ER[mf32_idx], DER[mf32_idx] if DER is not None else None))
            param_list.append((2, 'GN', GN[mf32_idx], DGN[mf32_idx] if DGN is not None else None))
            param_list.append((3, 'GG', GG[mf32_idx], DGG[mf32_idx] if DGG is not None else None))
            # param_list.append((4, 'GFA', GFA[mf32_idx], DGFA[mf32_idx] if DGFA is not None else None))
            # param_list.append((5, 'GFB', GFB[mf32_idx], DGFB[mf32_idx] if DGFB is not None else None))

            for param_type_index, param_name, mean_value, std_dev in param_list:
                # Check if mean_value is not None (some widths may not be provided)
                if mean_value is not None:
                    param_dict = {
                        'index': index,
                        'L': L,
                        'J': J,
                        'L_index': L_index,
                        'resonance_index': resonance_index,
                        'type': param_type_index,  # 1: ER, 2: GN, 3: GG, 4: GFA, 5: GFB
                        'name': param_name,
                        'mean': mean_value,
                        'std_dev': std_dev  # Include standard deviation
                    }
                    parameters.append(param_dict)
                    index += 1

        self.parameters = parameters
        self.mean_vector = np.array([param['mean'] for param in self.parameters])
        self.std_dev_vector = np.array([param['std_dev'] if param['std_dev'] is not None else 0.0 for param in self.parameters])

    #-----------------
    # Covariance
    #-----------------

    def extract_covariance_matrix(self):
        """
        Extracts covariance matrix based on LCOMP value.
        """
        if self.LCOMP == 0:
            self.extract_covariance_matrix_LCOMP0()
        elif self.LCOMP == 1:
            self.extract_covariance_matrix_LCOMP1()
        elif self.LCOMP == 2:
            self.extract_covariance_matrix_LCOMP2()
        else:
            raise ValueError(f"Unsupported LCOMP value: {self.LCOMP}")

    def extract_covariance_matrix_LCOMP0(self):
        """
        Constructs a diagonal covariance matrix when LCOMP == 0.
        """
        if self.std_dev_vector is None:
            raise ValueError("Standard deviations are required for LCOMP=0")
        covariance_matrix = np.diag(self.std_dev_vector ** 2)
        self.covariance_matrix = covariance_matrix

    def extract_covariance_matrix_LCOMP1(self):
        """
        Uses the covariance matrix provided in the data when LCOMP == 1.
        """
        cm = self.resonance_parameters.short_range_blocks[0]
        NPAR = cm.NPARB
        cov_upper = cm.covariance_matrix[:]
        # Reconstruct full covariance matrix
        covariance_matrix = np.zeros((NPAR, NPAR))
        idx = 0
        for i in range(NPAR):
            for j in range(i, NPAR):
                covariance_matrix[i, j] = cov_upper[idx]
                covariance_matrix[j, i] = cov_upper[idx]
                idx += 1
        self.covariance_matrix = covariance_matrix

        # Extract standard deviations from the diagonal of the covariance matrix
        variances = np.diag(covariance_matrix)
        std_dev_vector = np.sqrt(np.abs(variances))  # Use abs in case of negative variances due to numerical issues
        self.std_dev_vector = std_dev_vector

        # Update the std_dev in parameters
        for idx, param in enumerate(self.parameters):
            param['std_dev'] = self.std_dev_vector[idx]

    def write_additional_data_to_hdf5(self, hdf5_group):
        """
        Writes format-specific data to the HDF5 group.
        """
        # Write mapping information
        parameters_L_values = np.array([param['L'] for param in self.parameters], dtype=np.int32)
        resonance_indices = np.array([param['resonance_index'] for param in self.parameters], dtype=np.int32)
        param_types = np.array([param['type'] for param in self.parameters], dtype=np.int32)
        param_names = np.array([param['name'] for param in self.parameters], dtype='S')

        hdf5_group.create_dataset('parameters_L_values', data=parameters_L_values)
        hdf5_group.create_dataset('resonance_indices', data=resonance_indices)
        hdf5_group.create_dataset('param_types', data=param_types)
        hdf5_group.create_dataset('param_names', data=param_names)

        # Write AP and DAP if available
        if self.AP is not None:
            hdf5_group.attrs['AP'] = self.AP
        if self.DAP is not None:
            hdf5_group.attrs['DAP'] = self.DAP

        # Write APL and DAPL if available
        if self.APLs:
            APL_values = np.array([apl['APL'] for apl in self.APLs], dtype=np.float64)
            L_values_for_APL = np.array([apl['L'] for apl in self.APLs], dtype=np.int32)
            hdf5_group.create_dataset('APL_values', data=APL_values)
            hdf5_group.create_dataset('L_values_for_APL', data=L_values_for_APL)

            # Only write DAPL_values if any uncertainties are provided
            DAPL_values = np.array([dapl['DAPL'] for dapl in self.DAPLs], dtype=np.float64)
            if np.any(DAPL_values):
                hdf5_group.create_dataset('DAPL_values', data=DAPL_values)
                # Since L_values_for_DAPL is the same as L_values_for_APL, no need to store it separately

        # Add resonance_format attribute
        hdf5_group.attrs['resonance_format'] = 'ReichMoore'
      
    def print_parameters(self):
        """
        Prints the parameters.
        """
        print(f"Parameters for NER={self.NER}, LCOMP={self.LCOMP}:")
        for param in self.parameters:
            std_dev = param['std_dev'] if param['std_dev'] is not None else 'N/A'
            print(f"Resonance Index: {param['resonance_index']}, L: {param['L']}, J: {param['J']}, "
                f"Type: {param['name']}, Mean: {param['mean']}, StdDev: {std_dev}")

        # Print scattering radius and its uncertainty if available
        if self.AP is not None:
            DAP = self.DAP if self.DAP is not None else 'N/A'
            print(f"Scattering radius AP: {self.AP}, DAP: {DAP}")
        if self.APLs:
            for idx, apl_entry in enumerate(self.APLs):
                L = apl_entry['L']
                APL = apl_entry['APL']
                DAPL = self.DAPLs[idx]['DAPL'] if idx < len(self.DAPLs) else 'N/A'
                print(f"L={L}: APL={APL}, DAPL={DAPL}")



class RMatrixLimitedCovariance(ResonanceCovariance):
    def __init__(self, resonance_range, mf2_resonance_ranges, NER):
        super().__init__(resonance_range, NER)
        self.LCOMP = self.resonance_parameters.LCOMP
        if self.LCOMP != 2:
            raise NotImplementedError("Only LCOMP=2 is implemented for RMatrixLimitedCovariance.")
        
        self.LRF = resonance_range.LRF
        self.LRU = resonance_range.LRU
        
        # Extract the uncertainty data
        self.extract_parameters(mf2_resonance_ranges)
        
        # Extract the covariance matrix
        self.extract_covariance_matrix()
        
        # Remove parameters with zero variance
        self.remove_zero_variance_parameters()
    
    def extract_parameters(self, mf2_resonance_ranges):
        """
        Extracts parameters for LCOMP=2.
        """
        # Map resonances from MF2 to MF32
        self._map_resonances(mf2_resonance_ranges)
        
        # Build parameters list
        self._build_parameters_list()
    
    def _map_resonances(self, mf2_resonance_ranges):
        """
        Maps resonances from MF2 to MF32 based on spin group and resonance energy.
        """
        mf2_resonance_range = mf2_resonance_ranges[self.NER]
        mf2_spin_groups = mf2_resonance_range.parameters.spin_groups
        
        mf32_spin_groups = self.resonance_parameters.uncertainties.spin_groups
        
        # Create a mapping from MF32 resonances to MF2 resonances
        self.matched_resonances = []
        for sg_index, mf32_sg in enumerate(mf32_spin_groups):
            AJ_mf32 = mf32_sg.AJ
            PJ_mf32 = mf32_sg.PJ
            NRSA_mf32 = mf32_sg.NRSA
            ER_mf32 = mf32_sg.parameters.ER
            GAM_mf32 = mf32_sg.parameters.GAM
            DER_mf32 = mf32_sg.parameters.DER
            DGAM_mf32 = mf32_sg.parameters.DGAM
            
            # Find the corresponding spin group in MF2
            mf2_sg = None
            for mf2_sg_candidate in mf2_spin_groups:
                if mf2_sg_candidate.AJ == AJ_mf32 and mf2_sg_candidate.PJ == PJ_mf32:
                    mf2_sg = mf2_sg_candidate
                    break
            if mf2_sg is None:
                raise ValueError(f"No matching spin group in MF2 for AJ={AJ_mf32}, PJ={PJ_mf32}")
            
            # Now match resonances within the spin group based on resonance energy
            mf2_ER = mf2_sg.parameters.ER
            tolerance=1e-5
            
            for res_index, ER_value_mf32 in enumerate(ER_mf32):
                # Find matching resonance in MF2
                idx = self._find_nearest_energy(mf2_ER, ER_value_mf32, tolerance)
                if GAM_mf32[idx][:] != mf2_sg.parameters.GAM[idx][:]:
                    idx+=1
                    if abs(mf2_ER[idx] - ER_value_mf32) > tolerance:
                        raise ValueError(f"No matching resonance in MF32 for AJ={AJ_mf32}, PJ={PJ_mf32}, ER32={ER_value_mf32}")
                    
                if idx is not None:
                    mf2_ER_value = mf2_ER[idx]
                    # Collect the matched resonance data
                    matched_resonance = {
                        'spin_group_index': sg_index,
                        'mf2_sg': mf2_sg,
                        'mf32_sg': mf32_sg,
                        'resonance_index_mf32': res_index,
                        'resonance_index_mf2': idx,
                        'ER_mf32': ER_value_mf32,
                        'ER_mf2': mf2_ER_value,
                        'GAM_mf32': GAM_mf32[res_index],
                        'DER_mf32': DER_mf32[res_index],
                        'DGAM_mf32': DGAM_mf32[res_index]
                    }
                    self.matched_resonances.append(matched_resonance)
                else:
                    raise ValueError(f"Resonance energy {ER_value_mf32} in MF32 not found in MF2 within tolerance.")
    
    def _build_parameters_list(self):
        """
        Builds the parameters list from matched resonances.
        """
        parameters = []
        index = 0
        std_dev_list = []
        
        for matched_resonance in self.matched_resonances:
            sg_index = matched_resonance['spin_group_index']
            mf32_sg = matched_resonance['mf32_sg']
            mf2_sg = matched_resonance['mf2_sg']
            res_index_mf32 = matched_resonance['resonance_index_mf32']
            res_index_mf2 = matched_resonance['resonance_index_mf2']
            ER = matched_resonance['ER_mf32']
            DER = matched_resonance['DER_mf32']
            GAM = matched_resonance['GAM_mf32']
            DGAM = matched_resonance['DGAM_mf32']
            NCH = mf32_sg.NCH
            
            # Add the resonance energy parameter
            param_energy = {
                'index': index,
                'spin_group_index': sg_index,
                'resonance_index_mf2': res_index_mf2,
                'channel_index': None,
                'type': 0,  # 0 for resonance energy
                'name': 'ER',
                'mean': ER,
                'std_dev': DER
            }
            parameters.append(param_energy)
            std_dev_list.append(DER)
            index +=1
            
            # Add the resonance widths for each channel
            for ch_idx in range(NCH):
                param_width = {
                    'index': index,
                    'spin_group_index': sg_index,
                    'resonance_index_mf2': res_index_mf2,
                    'channel_index': ch_idx,
                    'type': ch_idx +1,  # 1-based indexing for channel widths
                    'name': f'GAM_{ch_idx+1}',
                    'mean': GAM[ch_idx],
                    'std_dev': DGAM[ch_idx]
                }
                parameters.append(param_width)
                std_dev_list.append(DGAM[ch_idx])
                index +=1
        
        self.parameters = parameters
        self.mean_vector = np.array([param['mean'] for param in parameters])
        self.std_dev_vector = np.array(std_dev_list)
    
    def extract_covariance_matrix(self):
        """
        Extracts the covariance matrix using the method from the base class.
        """
        self.extract_covariance_matrix_LCOMP2()
    
    def extract_covariance_matrix_LCOMP2(self):
        """
        Reconstructs the covariance matrix from standard deviations and correlation coefficients when LCOMP == 2.
        """
        cm = self.resonance_parameters.correlation_matrix
        NNN = cm.NNN  # Order of the correlation matrix
        correlations = cm.correlations  # List of correlation coefficients
        I = cm.I  # List of row indices (one-based)
        J = cm.J  # List of column indices (one-based)
        
        NPAR = len(self.parameters)
        if NNN != NPAR:
            raise ValueError(f"Mismatch between number of parameters ({NPAR}) and size of correlation matrix (NNN={NNN}).")
        
        # Initialize the correlation matrix
        correlation_matrix = np.identity(NPAR)
        
        # Fill in the off-diagonal elements
        for idx, corr_value in enumerate(correlations):
            i = I[idx] - 1  # Convert to zero-based index
            j = J[idx] - 1  # Convert to zero-based index
            correlation_matrix[i, j] = corr_value
            correlation_matrix[j, i] = corr_value  # Symmetric matrix
        
        # Now, compute the covariance matrix
        std_dev_vector = self.std_dev_vector
        covariance_matrix = np.outer(std_dev_vector, std_dev_vector) * correlation_matrix
        self.covariance_matrix = covariance_matrix

    def print_parameters(self):
        """
        Prints the parameters.
        """
        print(f"Parameters for NER={self.NER}, LCOMP={self.LCOMP}:")
        for param in self.parameters:
            std_dev = param['std_dev'] if param['std_dev'] is not None else 'N/A'
            sg_idx = param['spin_group_index']
            res_idx = param['resonance_index_mf2']
            ch_idx = param['channel_index']
            param_type = 'ER' if param['type']==0 else f'GAM_{param["type"]}'
            print(f"Index: {param['index']}, SpinGroup: {sg_idx}, ResonanceIndex: {res_idx}, "
                  f"ChannelIndex: {ch_idx}, Type: {param_type}, Mean: {param['mean']}, StdDev: {std_dev}")



class AveragedBreitWignerCovariance(ResonanceCovariance):
    def __init__(self, resonance_range, mf2_resonance_ranges, NER):
        super().__init__(resonance_range, NER)
        self.MPAR = self.resonance_parameters.covariance_matrix.MPAR  # Number of parameters per (L,J) group
        self.LFW = self.resonance_parameters.LFW  # Indicates if fission widths are present
        self.L_values = []  # List of (L, J) groups
        self.num_parameters = 0  # Total number of parameters across all (L,J) groups

        # Extract parameters and covariance matrices
        self.extract_parameters(mf2_resonance_ranges)
        
        self.extract_covariance_matrix()
        self.remove_zero_variance_parametersURR()
        
        self.compute_L_matrix()
        
    def remove_zero_variance_parametersURR(self):
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
        from collections import defaultdict
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
        self.num_LJ_groups = len(self.L_values)

        # Delete rows and columns from covariance matrix
        self.covariance_matrix = np.delete(self.covariance_matrix, zero_variance_indices, axis=0)
        self.covariance_matrix = np.delete(self.covariance_matrix, zero_variance_indices, axis=1)

        # Update mean vector
        self.mean_vector = np.delete(self.mean_vector, zero_variance_indices)

        # Update num_parameters
        self.num_parameters = self.covariance_matrix.shape[0]
    
    def extract_parameters(self, mf2_resonance_ranges):
        """
        Extracts the mean parameters from MF2 and constructs the parameter list.
        """
        resonance_range = mf2_resonance_ranges[self.NER]
        l_values = resonance_range.parameters.l_values.to_list()

        parameters = []
        param_names = self.get_param_names()  # Get the parameter names once

        for l_value in l_values:
            L = l_value.L
            j_values = l_value.j_values.to_list()
            for j_value in j_values:
                J = j_value.AJ
                # Store (L, J) group information
                self.L_values.append({'L': L, 'J': J, 'num_energies': len(j_value.ES)})

                # For each parameter type in MPAR, store the mean values
                param_list = []
                for param_name in param_names:
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
        Extracts the relative covariance matrix from MF32 and constructs it according to the parameter ordering.
        """
        covariance_data = self.resonance_parameters.covariance_matrix
        covariance_order = covariance_data.NPAR  # order of the matrix

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

    def get_param_names(self):
        """
        Returns the list of parameter names based on MPAR and LFW.
        """
        param_names = ['D', 'GN', 'GG', 'GF', 'GX']
        # Adjust the list based on MPAR and LFW
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

    def print_parameters(self):
        """
        Prints the parameters.
        """
        print(f"Averaged Breit-Wigner Parameters for NER={self.NER}:")
        for idx, group in enumerate(self.parameters):
            L = group['L']
            J = group['J']
            print(f"(L={L}, J={J}):")
            param_names = self.get_param_names()
            for param_name, values in zip(param_names, group['parameters']):
                print(f"  {param_name}: {values}")

    def write_additional_data_to_hdf5(self, hdf5_group):
        """
        Writes format-specific data to the HDF5 group.
        """
        # Store LFW and MPAR
        hdf5_group.attrs['LFW'] = self.LFW
        hdf5_group.attrs['MPAR'] = self.MPAR
        hdf5_group.attrs['num_LJ_groups'] = len(self.L_values)

        # Store L and J values
        L_values_array = np.array([lj['L'] for lj in self.L_values], dtype=np.int32)
        J_values_array = np.array([lj['J'] for lj in self.L_values], dtype=np.float64)
        hdf5_group.create_dataset('L_values', data=L_values_array)
        hdf5_group.create_dataset('J_values', data=J_values_array)

        # Store the parameter names
        param_names = self.get_param_names()
        param_names_encoded = np.array([name.encode('utf-8') for name in param_names], dtype='S')
        hdf5_group.create_dataset('param_names', data=param_names_encoded)

        # Store the mean parameters
        for idx, group in enumerate(self.parameters):
            group_name = f'group_{idx}'
            group_hdf5 = hdf5_group.create_group(group_name)
            group_hdf5.attrs['L'] = group['L']
            group_hdf5.attrs['J'] = group['J']
            for param_name, values in zip(param_names, group['parameters']):
                group_hdf5.create_dataset(param_name, data=values)


