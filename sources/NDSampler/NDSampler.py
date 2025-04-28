import h5py
import numpy as np
from ENDFtk.tree import Tape
from .CovarianceBase import CovarianceBase
import datetime
<<<<<<< HEAD
import os
=======
from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class SamplerSettings:
    """Dataclass to store settings for nuclear data sampling."""
    sampling: str = "Simple"  # Options: 'Simple', 'LHS', 'Sobol', 'Halton', etc.
    num_samples: int = 1
    random_seed: Optional[int] = None
    mode: str = "stack" # Keep in memory (Stack) all samples or draw n' replace
    debug: bool = False  # Debug mode for printing sample matrices without updating tapes
    
    # Parameters settings specific to resonances
    widths_to_reduced: bool = False
    
    def __post_init__(self):
        """Validate settings after initialization."""
        valid_sampling_methods = ["Simple", "LHS", "Sobol", "Halton"]
        if self.sampling not in valid_sampling_methods:
            raise ValueError(f"Sampling method '{self.sampling}' not recognized. "
                             f"Valid options are: {', '.join(valid_sampling_methods)}")
>>>>>>> phd_tests

class NDSampler:
    def __init__(self, endf_tape: Tape, covariance_dict: dict = None, settings: Optional[SamplerSettings] = None, hdf5_filename: Optional[str] = None):
        
        # Set the HDF5 filename
        self.hdf5_filename = hdf5_filename or f'covariance_data_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.hdf5'
        self.hdf5_file = h5py.File(self.hdf5_filename, 'w') if hdf5_filename is None else h5py.File(hdf5_filename, 'r')

        self.original_tape = endf_tape
        
        # Convert settings dict to SamplerSettings dataclass
        self.settings = settings or SamplerSettings()

        if hdf5_filename is None:
            self.covariance_dict = covariance_dict or generate_covariance_dict(endf_tape)
            # Initialize covariance objects based on covariance_dict
            self.covariance_objects: List[CovarianceBase] = []
            self._initialize_covariance_objects()
        else:
            self.covariance_dict = covariance_dict or {}
            self.covariance_objects: List[CovarianceBase] = []

    
    def _initialize_covariance_objects(self):   
        mat = self.original_tape.MAT(self.original_tape.material_numbers[0])

        # Loop over covariance_dict to initialize covariance objects
        for MF, MT_dict in self.covariance_dict.items():
            if mat.has_MF(MF):
                mf_section = mat.MF(MF)
                for MT, value_list in MT_dict.items():
                    if mf_section.has_MT(MT):
                        if MF == 31:
                            from .multiplicity.MultiplicityCovariance import MultiplicityCovariance
                            covariance_objects = []
                            MultiplicityCovariance.fill_from_resonance_range(self.original_tape, covariance_objects)
                            self.covariance_objects.extend(covariance_objects)
                            self._add_covariance_to_hdf5(covariance_objects, "Multiplicity")
                        elif MF == 32:
                            from .resonance.ResonanceRangeCovariance import ResonanceRangeCovariance
                            covariance_objects = []
                            
                            # Extract the list of NER values from the covariance dictionary
                            ner_list = value_list  # For MF=32, this is typically a list of NER values [0, 1, ...]
                            
                            # Pass the NER list to fill_from_resonance_range
                            print(f"Processing MF={MF}, MT={MT} with NER list: {ner_list}")
                            ResonanceRangeCovariance.fill_from_resonance_range(
                                self.original_tape, 
                                covariance_objects, 
                                ner_list=ner_list,
                                want_reduced=self.settings.widths_to_reduced
                            )
                            
                            self.covariance_objects.extend(covariance_objects)
                            self._add_covariance_to_hdf5(covariance_objects, "ResonanceRange")
                            
                        elif MF == 34:
                            from .angular.AngularDistributionCovariance import AngularDistributionCovariance
                            covariance_objects = []
                            AngularDistributionCovariance.fill_from_resonance_range(self.original_tape, covariance_objects)
                            self.covariance_objects.extend(covariance_objects)
                            self._add_covariance_to_hdf5(covariance_objects, "AngularDist")
                        else:
                            # Handle other covariance types
                            pass
        
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
                            
    @classmethod
    def get_covariance_dict(cls, endf_tape: Tape):
        sampler = cls(endf_tape, covariance_dict={}, hdf5_filename='temp.hdf5')
        # Close the temporary HDF5 file
        sampler.hdf5_file.close()
        return sampler.covariance_dict
       
<<<<<<< HEAD
    def load_and_sample_covariance_objects(self, num_samples: int = 1, directory: str = 'random'):
        """
        Loads covariance data from the HDF5 file and generates samples.
        """
        # Create the directory if it does not exist, or erase if it exists
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        else:
            os.makedirs(directory)

=======
    def sample(self, num_samples: int = 1):
        # Update the number of samples in settings
        self.settings.num_samples = num_samples
        
>>>>>>> phd_tests
        with h5py.File(self.hdf5_filename, 'r') as hdf5_file:
            covariance_objects: List[CovarianceBase] = []
            for group_name in hdf5_file:
                group = hdf5_file[group_name]
                if group_name == 'ResonanceRange':
                    from .resonance.ResonanceRangeCovariance import ResonanceRangeCovariance
                    ResonanceRangeCovariance.read_hdf5_group(group, covariance_objects)
                elif group_name == 'AngularDist':
                    from .angular.AngularDistributionCovariance import AngularDistributionCovariance
                    AngularDistributionCovariance.read_hdf5_group(group, covariance_objects)
                elif group_name == 'Multiplicity':
                    from .multiplicity.MultiplicityCovariance import MultiplicityCovariance
                    MultiplicityCovariance.read_hdf5_group(group, covariance_objects)
                else:
                    # Handle other covariance types
                    pass

            # Essentially to test the parser/writer
            if num_samples == 0:
                endf_tape: Tape = self.original_tape
                for covariance_obj in covariance_objects:
                    covariance_obj.update_tape(endf_tape, 0)
                endf_tape.to_file(os.path.join(directory, 'sampled_tape_random0.endf'))
                return

            # Configure random seed if specified
            if self.settings.random_seed is not None:
                np.random.seed(self.settings.random_seed)
                
            # Import sampling methods as needed
            try:
                from scipy.stats import qmc
            except ImportError:
                raise ImportError("scipy.stats.qmc is required for advanced sampling methods. "
                                  "Install it using 'pip install scipy'")

            print(f"Generating {num_samples} samples using {self.settings.sampling} method...")
            for covariance_obj in covariance_objects:
                # Generate all samples at once
                covariance_obj.sample_parameters(
                    sampling_method=self.settings.sampling, 
                    mode=self.settings.mode, 
                    use_copula=True, 
                    num_samples=num_samples,
                    debug=self.settings.debug
                )
            
            # Skip tape creation in debug mode
            if self.settings.debug:
                print("Debug mode enabled - skipping tape creation")
                return
                
            # Now create individual tapes for each sample
            for i in range(1, num_samples + 1):
                print(f"Creating tape for sample {i}...")
                endf_tape: Tape = self.original_tape
                for covariance_obj in covariance_objects:
                    covariance_obj.update_tape(endf_tape, i)
                
                # Write the sampled tape to a file
<<<<<<< HEAD
                endf_tape.to_file(os.path.join(directory, f'sampled_tape_random{i}.endf'))
                
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

=======
                endf_tape.to_file(f'sampled_tape_random{i}.endf')
>>>>>>> phd_tests

def generate_covariance_dict(endf_tape):
    """
    Generate a dictionary of covariance data from an ENDF tape.
    """
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