import h5py
import numpy as np
from ENDFtk.tree import Tape
from .resonance.ResonanceRangeCovariance import ResonanceRangeCovariance
from .angular.AngularDistributionCovariance import AngularDistributionCovariance
import datetime

class NuclearDataSampler:
    def __init__(self, endf_tape, covariance_dict=None, hdf5_filename=None):
        # Set the HDF5 filename
        if hdf5_filename is None:
            # Generate a default filename based on the current timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.hdf5_filename = f'covariance_data_{timestamp}.hdf5'
        else:
            self.hdf5_filename = hdf5_filename

        # Open the HDF5 file
        self.hdf5_file = h5py.File(self.hdf5_filename, 'w')

        self.original_tape = endf_tape
        self.endf_tape = endf_tape

        # If covariance_dict is None, generate it
        if covariance_dict is None:
            self.covariance_dict = self._generate_covariance_dict()
        else:
            self.covariance_dict = covariance_dict

        # Initialize covariance objects based on covariance_dict
        self.covariance_objects = []
        self._initialize_covariance_objects()
        
    def _add_covariance_to_hdf5(self, covariance_objects, covariance_type_name):
        """
        Add multiple covariance objects to the HDF5 file.
        """
        
        for covariance_obj in covariance_objects:
            group_name = covariance_type_name
            group = self.hdf5_file.require_group(group_name)
            subgroup_name = covariance_obj.get_covariance_type()
            subgroup = group.create_group(subgroup_name)
            covariance_obj.write_to_hdf5(subgroup)
    
    def _initialize_covariance_objects(self):
        mat = self.endf_tape.MAT(self.endf_tape.material_numbers[0])

        # Loop over covariance_dict to initialize covariance objects
        for MF, MT_dict in self.covariance_dict.items():
            if mat.has_MF(MF):
                mf_section = mat.MF(MF)
                for MT in MT_dict:
                    if mf_section.has_MT(MT):
                        if MF == 32:
                            covariance_objects = []
                            ResonanceRangeCovariance.fill_from_resonance_range(self.endf_tape, covariance_objects)
                            self.covariance_objects.extend(covariance_objects)
                            self._add_covariance_to_hdf5(covariance_objects, "ResonanceRange")
                        elif MF == 34:
                            covariance_objects = []
                            AngularDistributionCovariance.fill_from_resonance_range(self.endf_tape, covariance_objects)
                            self.covariance_objects.extend(covariance_objects)
                            self._add_covariance_to_hdf5(covariance_objects, "AngularDist")
                            pass
                        # Handle other MFs similarly
                            
    @classmethod
    def get_covariance_dict(cls, endf_tape):
        sampler = cls(endf_tape, covariance_dict={}, hdf5_filename='temp.hdf5')
        # Close the temporary HDF5 file
        sampler.hdf5_file.close()
        return sampler.covariance_dict
       
    def load_and_sample_covariance_objects(self, num_samples):
        """
        Loads covariance data from the HDF5 file and generates samples.
        """
        with h5py.File(self.hdf5_filename, 'r') as hdf5_file:
            covariance_objects = []
            for group_name in hdf5_file:
                print(f"found group {group_name}")
                group = hdf5_file[group_name]
                    
                if group_name == 'ResonanceRange':
                    ResonanceRangeCovariance.read_hdf5_group(group, covariance_objects)
                else:
                    # Handle other covariance types
                    pass

            for i in range(num_samples):
                print(f"Generating sample {i+1}...")
                endf_tape = self.original_tape
                for covariance_obj in covariance_objects:
                    covariance_obj.sample_parameters()
                    covariance_obj.update_tape(endf_tape, i)
                
                # Write the sampled tape to a file
                endf_tape.to_file(f'sampled_tape_{i+1}.endf')
                
    def test_sample_covariance_objects(self, num_samples):
        """
        Loads covariance data from the HDF5 file and generates samples.
        """
        with h5py.File(self.hdf5_filename, 'r') as hdf5_file:
            covariance_objects = []
            for group_name in hdf5_file:
                print(f"found group {group_name}")
                group = hdf5_file[group_name]
                    
                if group_name == 'ResonanceRange':
                    MyResonanceRange.read_hdf5_group(group, covariance_objects)
                else:
                    # Handle other covariance types
                    pass

            for covariance_obj in covariance_objects:
                for _ in range(num_samples):
                    covariance_obj.sample_parameters(mode='stack')
            
                original_variances = {}
                covariance_matrix = np.dot(covariance_obj.L_matrix, covariance_obj.L_matrix.T)

                for idx, item in enumerate(covariance_obj.index_mapping):
                    key = (item['l_idx'], item['j_idx'], item['e_idx'], item['param_name'].decode('utf-8'))
                    variance = covariance_matrix[idx, idx]
                    original_variances[key] = variance
                
                samples_dict = covariance_obj.extract_samples()
                
                variances = {}
                relative_variances = {}
                for key, samples in samples_dict.items():
                    samples_array = np.array(samples)
                    variance = np.var(samples_array, ddof=1)  # Sample variance
                    variances[key] = variance

                    # Original value is the first element
                    original_value = samples_array[0]

                    # Compute relative variance: variance divided by original_value squared
                    relative_variance = variance / original_value**2
                    relative_variances[key] = relative_variance
            
                for key in relative_variances:
                    empirical_relative_variance = relative_variances[key]
                    original_relative_variance = original_variances.get(key, None)
                    if original_relative_variance is not None:
                        print(f"Parameter: {key}")
                        print(f"Empirical Relative Variance: {empirical_relative_variance}")
                        print(f"Original Relative Variance: {original_relative_variance}")
                        print(f"Percent Ratio (Empiric - Origin / Origin): {100 * (empirical_relative_variance - original_relative_variance) / original_relative_variance} %")
                        print()
                    else:
                        print(f"Original relative variance not found for parameter: {key}")


def generate_covariance_dict(endf_tape):
    covariance_dict = {}
    mat = endf_tape.MAT(endf_tape.material_numbers[0])

    # Loop over covariance MF sections
    for MF in [31, 32, 33, 34, 35]:
        if mat.has_MF(MF):
            mf_section = mat.MF(MF)
            covariance_dict[MF] = {}
            # Loop over MT sections within the MF
            for MT in mf_section.section_numbers:
                parsed_section = mf_section.MT(MT).parse()
                if MF == 32:
                    # For MF=32, get the number of resonance ranges
                    num_resonance_ranges = parsed_section.isotopes[0].number_resonance_ranges
                    covariance_dict[MF][MT] = list(range(num_resonance_ranges))
                elif MF == 33:
                    covariance_dict[MF][MT] = []

                    # Loop over reactions in MF33
                    for sub_section in parsed_section.reactions:
                        MT1 = sub_section.MT1
                        covariance_dict[MF][MT].append(MT1)
                elif MF == 34:
                    # Parse the MF34 section
                    covariance_dict[MF][MT] = {}

                    # Loop over reactions in MF34
                    for sub_section in parsed_section.reactions:
                        MT1 = sub_section.MT1
                        covariance_dict[MF][MT][MT1] = {}

                        # Loop over Legendre blocks
                        for legendre_block in sub_section.legendre_blocks:
                            L = legendre_block.L
                            L1 = legendre_block.L1
                            if L not in covariance_dict[MF][MT][MT1]:
                                covariance_dict[MF][MT][MT1][L] = []
                            covariance_dict[MF][MT][MT1][L].append(L1)
                else:
                    # Placeholder for other MFs
                    covariance_dict[MF][MT] = None

    return covariance_dict