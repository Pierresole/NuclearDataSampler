import numpy as np
from collections import defaultdict
from ENDFtk import tree  # Assuming ENDFtk is being used

class ReichMoore:
    def __init__(self, tape, NER):
        """
        Initializes the ReichMoore object.

        Parameters:
        - tape: The ENDF tape object.
        - NER: The resonance range index.
        """
        self.tape = tape
        self.NER = NER  # Resonance range index
        self.resonance_range = None  # Will hold the ResonanceRange object
        self.parameters = []
        self.mean_vector = None
        self.std_dev_vector = None
        self.covariance_matrix = None
        self.L_matrix = None
        self.LCOMP = None
        self.AP = None  # Scattering radius
        self.DAP = None  # Uncertainty in AP
        self.APLs = []  # Scattering radius per L
        self.DAPLs = []  # Uncertainties in APLs
        self.param_names = []
        self.param_types = []
        self.param_L_values = []
        self.resonance_indices = []

        # Extract data
        self.extract_data()
        self.remove_zero_variance_parameters()
        self.compute_L_matrix()

    def extract_data(self):
        """
        Extracts resonance parameters and covariance data from the tape.
        """
        # Parse MF2 MT151
        mf2mt151 = self.tape.MAT(self.tape.material_numbers[0]).MF(2).MT(151).parse()
        isotope = mf2mt151.isotopes[0]
        resonance_ranges = isotope.resonance_ranges.to_list()
        self.resonance_range = resonance_ranges[self.NER]

        # Extract LCOMP
        self.LCOMP = self.resonance_range.parameters.LCOMP

        # Extract AP and DAP
        self.AP = self.resonance_range.parameters.AP
        self.DAP = getattr(self.resonance_range.parameters, 'DAP', None)

        # Extract parameters
        self.extract_parameters()

        # Extract covariance matrix
        self.extract_covariance_matrix()

    def extract_parameters(self):
        """
        Extracts parameters based on LCOMP value.
        """
        if self.LCOMP == 0:
            raise NotImplementedError("LCOMP=0 (uncorrelated uncertainties) not implemented.")
        elif self.LCOMP == 1:
            self.extract_parameters_LCOMP1()
        elif self.LCOMP == 2:
            self.extract_parameters_LCOMP2()
        else:
            raise ValueError(f"Unsupported LCOMP value: {self.LCOMP}")

    def extract_parameters_LCOMP1(self):
        """
        Extracts parameters when LCOMP == 1.
        The full covariance matrix is provided.
        """
        uncertainties = self.resonance_range.parameters.short_range_blocks[0]

        # Mean values
        ER = uncertainties.ER
        GN = uncertainties.GN
        GG = uncertainties.GG
        GFA = uncertainties.GFA
        GFB = uncertainties.GFB

        # Map resonances from MF2 to MF32
        self._map_resonances()

        # Build parameters list
        self._build_parameters_list(ER, GN, GG, GFA, GFB)

    def extract_parameters_LCOMP2(self):
        """
        Extracts parameters when LCOMP == 2.
        Standard deviations and correlation matrix are provided.
        """
        uncertainties = self.resonance_range.parameters.uncertainties

        # Mean values
        ER = uncertainties.ER
        GN = uncertainties.GN
        GG = uncertainties.GG
        GFA = uncertainties.GFA
        GFB = uncertainties.GFB

        # Standard deviations
        DER = uncertainties.DER
        DGN = uncertainties.DGN
        DGG = uncertainties.DGG
        DGFA = uncertainties.DGFA
        DGFB = uncertainties.DGFB

        # Map resonances from MF2 to MF32
        self._map_resonances()

        # Build parameters list
        self._build_parameters_list(ER, GN, GG, GFA, GFB, DER, DGN, DGG, DGFA, DGFB)

    def _map_resonances(self):
        """
        Maps resonances from MF2 to MF32 by matching resonance energies.
        """
        # Get resonance energies from MF2
        l_values = self.resonance_range.parameters.l_values.to_list()
        self.mf2_resonances = []
        for l_value in l_values:
            L = l_value.L
            APL = getattr(l_value, 'APL', None)
            self.APLs.append({'L': L, 'APL': APL})
            J_values = l_value.j_values.to_list()
            for j_value in J_values:
                resonances = j_value.resonances
                for res in resonances:
                    self.mf2_resonances.append({
                        'L': L,
                        'J': j_value.AJ,
                        'ER': res.ER,
                        'GN': res.GN,
                        'GG': res.GG,
                        'GFA': res.GFA,
                        'GFB': res.GFB
                    })

        # Get resonance energies from MF32
        if self.LCOMP == 1:
            uncertainties = self.resonance_range.parameters.short_range_blocks[0]
        else:  # LCOMP == 2
            uncertainties = self.resonance_range.parameters.uncertainties
        ER_mf32 = uncertainties.ER

        # Match resonances
        self.matched_resonances = []
        tolerance = 1e-5  # Adjust as needed
        for idx_mf32, ER_value_mf32 in enumerate(ER_mf32):
            found_match = False
            for idx_mf2, res in enumerate(self.mf2_resonances):
                if abs(res['ER'] - ER_value_mf32) <= tolerance:
                    self.matched_resonances.append({
                        'mf32_index': idx_mf32,
                        'mf2_index': idx_mf2,
                        'L': res['L'],
                        'J': res['J'],
                        'ER': res['ER']
                    })
                    found_match = True
                    break
            if not found_match:
                raise ValueError(f"Resonance energy {ER_value_mf32} in MF32 not found in MF2 within tolerance.")

    def _build_parameters_list(self, ER, GN, GG, GFA, GFB, DER=None, DGN=None, DGG=None, DGFA=None, DGFB=None):
        """
        Builds the parameters list with mapping information.

        Parameters:
        - ER, GN, GG, GFA, GFB: Mean values of the parameters from MF32.
        - DER, DGN, DGG, DGFA, DGFB: Standard deviations of the parameters from MF32.
        """
        parameters = []
        param_names = ['ER', 'GN', 'GG', 'GFA', 'GFB']
        param_types = [1, 2, 3, 4, 5]

        for idx, match in enumerate(self.matched_resonances):
            mf32_idx = match['mf32_index']
            L = match['L']
            J = match['J']
            mf2_index = match['mf2_index']

            param_values = [ER[mf32_idx], GN[mf32_idx], GG[mf32_idx], GFA[mf32_idx], GFB[mf32_idx]]
            if DER is not None:
                std_devs = [DER[mf32_idx], DGN[mf32_idx], DGG[mf32_idx], DGFA[mf32_idx], DGFB[mf32_idx]]
            else:
                std_devs = [None] * 5

            for param_type, param_name, mean_value, std_dev in zip(param_types, param_names, param_values, std_devs):
                if mean_value is not None:
                    parameters.append({
                        'L': L,
                        'J': J,
                        'resonance_index': idx,
                        'type': param_type,
                        'name': param_name,
                        'mean': mean_value,
                        'std_dev': std_dev
                    })

        self.parameters = parameters
        self.mean_vector = np.array([param['mean'] for param in self.parameters])
        if any(param['std_dev'] is not None for param in self.parameters):
            self.std_dev_vector = np.array([param['std_dev'] if param['std_dev'] is not None else 0.0 for param in self.parameters])
        else:
            self.std_dev_vector = None

        self.param_names = [param['name'] for param in self.parameters]
        self.param_types = [param['type'] for param in self.parameters]
        self.param_L_values = [param['L'] for param in self.parameters]
        self.resonance_indices = [param['resonance_index'] for param in self.parameters]

    def extract_covariance_matrix(self):
        """
        Extracts covariance matrix based on LCOMP value.
        """
        if self.LCOMP == 1:
            self.extract_covariance_matrix_LCOMP1()
        elif self.LCOMP == 2:
            self.extract_covariance_matrix_LCOMP2()
        else:
            raise NotImplementedError("Covariance extraction for LCOMP=0 not implemented.")

    def extract_covariance_matrix_LCOMP1(self):
        """
        Uses the covariance matrix provided in the data when LCOMP == 1.
        """
        cm = self.resonance_range.parameters.short_range_blocks[0]
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
        std_dev_vector = np.sqrt(np.abs(variances))
        self.std_dev_vector = std_dev_vector

        # Update std_dev in parameters
        for idx, param in enumerate(self.parameters):
            param['std_dev'] = self.std_dev_vector[idx]

    def extract_covariance_matrix_LCOMP2(self):
        """
        Constructs the covariance matrix when LCOMP == 2 using standard deviations and correlation matrix.
        """
        uncertainties = self.resonance_range.parameters.uncertainties
        correlation_matrix = uncertainties.correlation_matrix
        if correlation_matrix is None:
            raise ValueError("Correlation matrix is missing in LCOMP=2.")

        std_dev_vector = self.std_dev_vector
        covariance_matrix = np.outer(std_dev_vector, std_dev_vector) * correlation_matrix
        self.covariance_matrix = covariance_matrix

    def remove_zero_variance_parameters(self):
        """
        Removes parameters with zero variance and updates the covariance matrix accordingly.
        """
        if self.std_dev_vector is None:
            return  # No standard deviations to check

        # Identify zero variance parameters
        zero_variance_indices = np.where(self.std_dev_vector == 0.0)[0]
        if zero_variance_indices.size == 0:
            return

        # Remove parameters with zero variance
        self.parameters = [param for idx, param in enumerate(self.parameters) if idx not in zero_variance_indices]
        self.mean_vector = np.delete(self.mean_vector, zero_variance_indices)
        self.std_dev_vector = np.delete(self.std_dev_vector, zero_variance_indices)
        self.covariance_matrix = np.delete(self.covariance_matrix, zero_variance_indices, axis=0)
        self.covariance_matrix = np.delete(self.covariance_matrix, zero_variance_indices, axis=1)

        # Update mappings
        self.param_names = [param['name'] for param in self.parameters]
        self.param_types = [param['type'] for param in self.parameters]
        self.param_L_values = [param['L'] for param in self.parameters]
        self.resonance_indices = [param['resonance_index'] for param in self.parameters]

    def compute_L_matrix(self):
        """
        Computes the L matrix for sampling.
        """
        try:
            self.L_matrix = np.linalg.cholesky(self.covariance_matrix)
        except np.linalg.LinAlgError:
            # Handle non-positive definite covariance matrix
            eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix)
            eigenvalues[eigenvalues < 0] = 0
            self.L_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    def sample_parameters(self):
        """
        Samples new parameters based on the covariance matrix.

        Returns:
        - sampled_parameters: List of dictionaries containing sampled parameters.
        - sampled_AP: Sampled value of AP (scattering radius), if available.
        - sampled_APLs: List of sampled APL values, if available.
        """
        # Generate standard normal random variables
        N = np.random.normal(size=self.L_matrix.shape[0])

        # Compute sampled deviations
        Y = self.L_matrix @ N + self.mean_vector

        # Sample AP
        if self.AP is not None and self.DAP is not None:
            sampled_AP = np.random.normal(loc=self.AP, scale=self.DAP)
        else:
            sampled_AP = self.AP

        # Sample APLs
        sampled_APLs = []
        for apl_entry, dapl_entry in zip(self.APLs, self.DAPLs):
            APL = apl_entry['APL']
            DAPL = dapl_entry.get('DAPL', None)
            if DAPL is not None:
                sampled_APL = np.random.normal(loc=APL, scale=DAPL)
            else:
                sampled_APL = APL
            sampled_APLs.append(sampled_APL)

        # Reconstruct sampled parameters
        sampled_parameters = []
        for idx, param in enumerate(self.parameters):
            sampled_value = Y[idx]
            sampled_parameters.append({
                'L': param['L'],
                'J': param['J'],
                'resonance_index': param['resonance_index'],
                'type': param['type'],
                'name': param['name'],
                'value': sampled_value
            })

        return sampled_parameters, sampled_AP, sampled_APLs

    def update_tape(self, sampled_parameters, sampled_AP=None, sampled_APLs=[]):
        """
        Updates the ENDF tape with the sampled parameters.

        Parameters:
        - sampled_parameters: List of dictionaries containing sampled parameters.
        - sampled_AP: Sampled value of AP (scattering radius), if available.
        - sampled_APLs: List of sampled APL values, if available.
        """
        # Create a copy of the resonance ranges
        mf2mt151 = self.tape.MAT(self.tape.material_numbers[0]).MF(2).MT(151).parse()
        isotope = mf2mt151.isotopes[0]
        resonance_ranges = isotope.resonance_ranges.copy()

        # Update the resonance range with matching NER
        resonance_ranges[self.NER] = self._update_resonance_range(self.resonance_range, sampled_parameters, sampled_AP, sampled_APLs)

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

    def _update_resonance_range(self, resonance_range, sampled_parameters, sampled_AP, sampled_APLs):
        """
        Updates a resonance range with the sampled parameters.

        Parameters:
        - resonance_range: The original resonance range.
        - sampled_parameters: List of dictionaries containing sampled parameters.
        - sampled_AP: Sampled value of AP (scattering radius), if available.
        - sampled_APLs: List of sampled APL values, if available.

        Returns:
        - new_resonance_range: The updated resonance range.
        """
        # Build new l_values with updated parameters
        l_values = resonance_range.parameters.l_values.to_list()
        new_l_values = []

        # Map parameters
        param_dict = defaultdict(lambda: defaultdict(dict))
        for param in sampled_parameters:
            L = param['L']
            res_idx = param['resonance_index']
            param_name = param['name']
            value = param['value']
            param_dict[L][res_idx][param_name] = value

        # Update l_values
        for idx_L, l_value in enumerate(l_values):
            L = l_value.L
            APL = sampled_APLs[idx_L] if idx_L < len(sampled_APLs) else l_value.APL
            j_values = l_value.j_values.to_list()
            new_j_values = []

            for j_value in j_values:
                J = j_value.AJ
                resonances = j_value.resonances
                new_resonances = []
                for idx_res, res in enumerate(resonances):
                    res_params = param_dict.get(L, {}).get(idx_res, {})
                    ER_new = res_params.get('ER', res.ER)
                    GN_new = res_params.get('GN', res.GN)
                    GG_new = res_params.get('GG', res.GG)
                    GFA_new = res_params.get('GFA', res.GFA)
                    GFB_new = res_params.get('GFB', res.GFB)

                    new_res = tree.Resonance(
                        ER=ER_new,
                        AJ=res.AJ,
                        GN=GN_new,
                        GG=GG_new,
                        GFA=GFA_new,
                        GFB=GFB_new
                    )
                    new_resonances.append(new_res)

                new_j_value = tree.ReichMoore.JValue(
                    AJ=J,
                    resonances=new_resonances
                )
                new_j_values.append(new_j_value)

            new_l_value = tree.ReichMoore.LValue(
                L=L,
                APL=APL,
                j_values=new_j_values
            )
            new_l_values.append(new_l_value)

        # Create new parameters object
        new_parameters = tree.ReichMoore(
            SPI=resonance_range.parameters.SPI,
            AP=sampled_AP if sampled_AP is not None else resonance_range.parameters.AP,
            LAD=resonance_range.parameters.LAD,
            NLSC=resonance_range.parameters.NLSC,
            l_values=new_l_values
        )

        # Create new resonance range
        new_resonance_range = tree.ResonanceRange(
            EL=resonance_range.EL,
            EH=resonance_range.EH,
            NAPS=resonance_range.NAPS,
            parameters=new_parameters
        )

        return new_resonance_range
