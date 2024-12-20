import numpy as np
import h5py
import io
from ENDFtk.tree import Tape
from ENDFtk.MF2.MT151 import Isotope
from ENDFtk.MF2.MT151 import Section

from ResonanceUpdater import RMatrixLimitedUpdater, ReichMooreUpdater, AveragedBreitWignerUpdater
# from AngularUpdater import AngularUpdater
# from CrossSectionUpdater import CrossSectionUpdater

class ENDFSampler:
    """
    This class is designed to handle uncertainty data from ENDF tapes, where each type of covariance data has a specific format. For instance:
        - `ReichMooreCovariance` contains lgroups.
        - `AveragedBreitWignerCovariance` segregates energies within jgroups within lgroups.

    The choice to use specific formats for each type of data is rational because an initial trial of storing parameter positions as integers proved 
    to be very error-prone and subject to developer interpretation. Questions such as :
        "Is the scattering radius the first parameter?", 
        "How are the l-wave scattering radii numbered?", 
    and "What about spins?" led to ambiguities.

    Therefore, the decision was made to store precise formats for each type of data:
        -- Resonance data :
            - For the Reich-Moore formalism, the L number and resonance index are stored.
            - For Unresolved Resonances, the (L, J) couples are stored.
        -- Cross-section data :
        -- Angular data :
        -- Nubar data :
    """
    def __init__(self, hdf5_filename, endf_tape):
        """
        Initializes the ENDFSampler.

        Parameters:
        - hdf5_filename: The name of the HDF5 file containing covariance data.
        - endf_tape: The original ENDF tape object to be sampled.
        """
        self.hdf5_filename = hdf5_filename
        self.endf_tape = endf_tape  # Original tape
        self.covariance_data = []

        # Mapping of covariance types to updater classes
        self.updater_classes = {
            'ReichMooreCovariance': ReichMooreUpdater,
            'RMatrixLimitedCovariance': RMatrixLimitedUpdater,
            'AveragedBreitWignerCovariance': AveragedBreitWignerUpdater,
            # 'AngularCovariance': AngularUpdater,  # Placeholder for future implementation
            # 'CrossSectionCovariance': CrossSectionUpdater,  # Placeholder for future implementation
        }

        self.load_covariance_data()

    def load_covariance_data(self):
        """
        Loads covariance data from the HDF5 file.
        """
        with h5py.File(self.hdf5_filename, 'r') as hdf5_file:
            for group_name in hdf5_file:
                group = hdf5_file[group_name]
                for subgroup_name in group:
                    subgroup = group[subgroup_name]
                    covariance_info = {}
                    covariance_info['group_name'] = group_name
                    covariance_info['subgroup_name'] = subgroup_name
                    
                    if group_name == 'AveragedBreitWignerCovariance':
                        covariance_info.update(self.load_averaged_breit_wigner_covariance(subgroup))
                    elif group_name == 'ReichMooreCovariance':
                        covariance_info.update(self.load_reich_moore_covariance(subgroup))
                    else:
                        # Handle other covariance types
                        pass
                    self.covariance_data.append(covariance_info)

    def load_averaged_breit_wigner_covariance(self, subgroup):
        """
        Loads AveragedBreitWignerCovariance data.
        """
        covariance_info = {}
        covariance_info['resonance_format'] = subgroup.attrs.get('resonance_format', 'AveragedBreitWigner')
        covariance_info['LFW'] = subgroup.attrs['LFW']
        covariance_info['MPAR'] = subgroup.attrs['MPAR']
        covariance_info['num_LJ_groups'] = subgroup.attrs['num_LJ_groups']

        # Load L and J values
        covariance_info['L_values'] = subgroup['L_values'][:]
        covariance_info['J_values'] = subgroup['J_values'][:]

        # Load parameter names
        covariance_info['param_names'] = [
            name.decode('utf-8') if isinstance(name, bytes) else name
            for name in subgroup['param_names'][:]
        ]

        # Load mean parameters
        groups = {}
        for idx in range(covariance_info['num_LJ_groups']):
            group_name_hdf5 = f'group_{idx}'
            group_hdf5 = subgroup[group_name_hdf5]
            group_parameters = {}
            for param_name in covariance_info['param_names']:
                if param_name in group_hdf5:
                    group_parameters[param_name] = group_hdf5[param_name][:]
            groups[group_name_hdf5] = group_parameters
        covariance_info['groups'] = groups

        covariance_info['L_matrix'] = subgroup['L_matrix'][:]
        covariance_info['is_cholesky'] = subgroup.attrs.get('is_cholesky', True)

        return covariance_info

    def load_reich_moore_covariance(self, subgroup):
        """
        Loads ReichMooreCovariance data.
        """
        covariance_info = {}
        covariance_info['mean_vector'] = subgroup['mean_vector'][:]
        covariance_info['L_matrix'] = subgroup['L_matrix'][:]
        covariance_info['is_cholesky'] = subgroup.attrs.get('is_cholesky', True)
        covariance_info['resonance_format'] = subgroup.attrs.get('resonance_format', None)

        # Load mapping information
        covariance_info['param_names'] = [
            name.decode('utf-8') if isinstance(name, bytes) else name
            for name in subgroup['param_names'][:]
        ]
        covariance_info['param_types'] = subgroup['param_types'][:]
        covariance_info['parameters_L_values'] = subgroup['parameters_L_values'][:]
        covariance_info['resonance_indices'] = subgroup['resonance_indices'][:]

        # Load AP and DAP if available
        covariance_info['AP'] = subgroup.attrs.get('AP', None)
        covariance_info['DAP'] = subgroup.attrs.get('DAP', None)

        # Load APL and DAPL if available
        if 'APL_values' in subgroup:
            covariance_info['APL_values'] = subgroup['APL_values'][:]
            covariance_info['L_values_for_APL'] = subgroup['L_values_for_APL'][:]
        if 'DAPL_values' in subgroup:
            covariance_info['DAPL_values'] = subgroup['DAPL_values'][:]

        return covariance_info

    def sample(self, num_samples):
        """
        Generates samples, updates the ENDF tape, and writes each sampled tape to a file.

        Parameters:
        - num_samples: The number of samples to generate.
        """
        original_tape = self.endf_tape  # Keep a reference to the original tape
        for i in range(num_samples):
            print(f"Generating sample {i+1}...")
            # Reset the tape to its original state for each sample
            endf_tape = original_tape

            # Loop over covariance data
            for cov_data in self.covariance_data:
                group_name = cov_data['group_name']

                if group_name == 'AveragedBreitWignerCovariance':
                    self._sample_averaged_breit_wigner(endf_tape, cov_data)
                elif group_name == 'ReichMooreCovariance':
                    self._sample_reich_moore(endf_tape, cov_data)

            # Write the tape to a file
            endf_tape.to_file(f'sampled_tape_random{i+1}.endf')

    def _sample_averaged_breit_wigner(self, endf_tape, cov_data):
        """
        Handles sampling for AveragedBreitWignerCovariance.

        Parameters:
        - endf_tape: The ENDF tape object to update.
        - cov_data: The covariance data dictionary.
        """
        # Extract necessary data
        L_matrix = cov_data['L_matrix']
        param_names = cov_data['param_names']  # ['D', 'GN']
        num_LJ_groups = cov_data['num_LJ_groups']
        groups = cov_data['groups']
        L_values = cov_data['L_values']
        J_values = cov_data['J_values']

        # Generate standard normal random variables
        N = np.random.normal(size=L_matrix.shape[0])

        # Compute sampled relative deviations
        Y = L_matrix @ N

        # Apply deviations to parameters
        index_mapping = []
        for group_idx in range(num_LJ_groups):
            group_key = f'group_{group_idx}'
            for param_name in param_names:
                index_mapping.append((group_key, param_name))

        # Apply the deviations
        for idx_in_Y, (group_key, param_name) in enumerate(index_mapping):
            group = groups[group_key]
            mean_param_values = group[param_name]
            relative_deviation = Y[idx_in_Y]
            sampled_param_values = mean_param_values * (1 + relative_deviation)
            group[param_name] = sampled_param_values

        # Prepare sampled_groups for updating the tape
        sampled_groups = []
        for group_idx in range(num_LJ_groups):
            group_key = f'group_{group_idx}'
            group = groups[group_key]
            L = L_values[group_idx]
            J = J_values[group_idx]
            sampled_parameters = {param_name: group[param_name] for param_name in param_names}
            sampled_groups.append({
                'L': L,
                'J': J,
                'sampled_parameters': sampled_parameters
            })

        # Update the tape with the new parameters
        self.update_tape(endf_tape, cov_data, sampled_groups)

    def _sample_reich_moore(self, endf_tape, cov_data):
        """
        Handles sampling for ReichMooreCovariance.

        Parameters:
        - endf_tape: The ENDF tape object to update.
        - cov_data: The covariance data dictionary.
        """
        mean_vector = cov_data['mean_vector']
        L_matrix = cov_data['L_matrix']
        is_cholesky = cov_data['is_cholesky']

        # Generate a random normal vector
        N = np.random.normal(size=mean_vector.shape)

        # Compute the sample vector
        if is_cholesky:
            Y = L_matrix @ N + mean_vector
        else:
            L_decomp = np.linalg.cholesky(L_matrix)
            Y = L_decomp @ N + mean_vector

        # Sample AP
        if 'DAP' in cov_data and cov_data['DAP'] is not None:
            sampled_AP = np.random.normal(loc=cov_data['AP'], scale=cov_data['DAP'])
        else:
            sampled_AP = cov_data.get('AP', None)

        # Sample APLs
        sampled_APLs = []
        if 'APL_values' in cov_data:
            APL_values = cov_data['APL_values']
            DAPL_values = cov_data.get('DAPL_values', [None]*len(APL_values))
            original_AP = cov_data.get('AP', None)
            for APL_value, DAPL_value in zip(APL_values, DAPL_values):
                if original_AP is not None and APL_value == original_AP:
                    sampled_APL = sampled_AP
                else:
                    if DAPL_value is not None:
                        sampled_APL = np.random.normal(loc=APL_value, scale=DAPL_value)
                    else:
                        sampled_APL = APL_value
                sampled_APLs.append(sampled_APL)

        # Update the tape with the new parameters
        self.update_tape(endf_tape, cov_data, Y, sampled_AP, sampled_APLs)

    def update_tape(self, tape, cov_data, sampled_params, sampled_AP=None, sampled_APLs=[]):
        """
        Updates the tape with the sampled parameters.

        Parameters:
        - tape: The ENDF tape object to update.
        - cov_data: The covariance data dictionary.
        - sampled_params: The sampled parameters vector or list of dictionaries containing sampled parameters per group.
        - sampled_AP: Sampled value of AP (scattering radius), if available.
        - sampled_APLs: List of sampled APL values, if available.
        """
        group_name = cov_data['group_name']
        updater_class = self.updater_classes.get(group_name)
        if updater_class is None:
            raise NotImplementedError(f"Covariance type {group_name} is not yet implemented in the sampler.")

        updater = updater_class()

        if group_name == 'AveragedBreitWignerCovariance':
            self._update_averaged_breit_wigner_tape(tape, cov_data, sampled_params, updater)
        elif group_name == 'ReichMooreCovariance':
            self._update_reich_moore_tape(tape, cov_data, sampled_params, updater, sampled_AP, sampled_APLs)
        else:
            # For other updaters, implement the appropriate method
            updater.update_tape(tape, cov_data, sampled_params)

    def _update_averaged_breit_wigner_tape(self, tape, cov_data, sampled_params, updater):
        """
        Updates the tape for AveragedBreitWignerCovariance.

        Parameters:
        - tape: The ENDF tape object to update.
        - cov_data: The covariance data dictionary.
        - sampled_params: List of dictionaries containing sampled parameters per group.
        - updater: The updater instance for AveragedBreitWignerCovariance.
        """
        subgroup_name = cov_data['subgroup_name']
        NER = int(subgroup_name.split('_')[1])

        mf2mt151 = tape.MAT(tape.material_numbers[0]).MF(2).MT(151).parse()
        isotope = mf2mt151.isotopes[0]
        resonance_ranges = isotope.resonance_ranges.to_list()
    
        # Update the resonance range with matching NER
        updated_ranges = []
        for idx, rr in enumerate(resonance_ranges):
            if idx == NER:
                updated_rr = updater.update_resonance_parameters(tape, rr, sampled_params)
                updated_ranges.append(updated_rr)
            else:
                updated_ranges.append(rr)

        new_isotope = Isotope(
            zai=isotope.ZAI,
            abn=isotope.ABN,
            lfw=isotope.LFW,
            ranges=updated_ranges
        )

        new_section = Section(
            zaid=mf2mt151.ZA,
            awr=mf2mt151.AWR,
            isotopes=[new_isotope]
        )

        # Replace the existing section in the tape
        mat_num = tape.material_numbers[0]
        tape.MAT(mat_num).MF(2).insert_or_replace(new_section)

    def _update_reich_moore_tape(self, tape, cov_data, sampled_params, updater, sampled_AP, sampled_APLs):
        """
        Updates the tape for ReichMooreCovariance.

        Parameters:
        - tape: The ENDF tape object to update.
        - cov_data: The covariance data dictionary.
        - sampled_params: The sampled parameters vector.
        - updater: The updater instance for ReichMooreCovariance.
        - sampled_AP: Sampled value of AP (scattering radius), if available.
        - sampled_APLs: List of sampled APL values, if available.
        """
        subgroup_name = cov_data['subgroup_name']
        NER = int(subgroup_name.split('_')[1])

        mf2mt151 = tape.MAT(tape.material_numbers[0]).MF(2).MT(151).parse()
        isotope = mf2mt151.isotopes[0]
        resonance_ranges = isotope.resonance_ranges.to_list()
    
        # Update the resonance range with matching NER
        updated_ranges = []
        for idx, rr in enumerate(resonance_ranges):
            if idx == NER:
                updated_rr = updater.update_resonance_parameters(tape, rr, sampled_params, cov_data, sampled_AP, sampled_APLs)
                updated_ranges.append(updated_rr)
            else:
                updated_ranges.append(rr)

        new_isotope = Isotope(
            zai=isotope.ZAI,
            abn=isotope.ABN,
            lfw=isotope.LFW,
            ranges=updated_ranges
        )

        new_section = Section(
            zaid=mf2mt151.ZA,
            awr=mf2mt151.AWR,
            isotopes=[new_isotope]
        )

        # Replace the existing section in the tape
        mat_num = tape.material_numbers[0]
        tape.MAT(mat_num).MF(2).insert_or_replace(new_section)

