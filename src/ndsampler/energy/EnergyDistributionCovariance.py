import ENDFtk
from ENDFtk.tree import Tape
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from ..CovarianceBase import CovarianceBase

class EnergyDistributionCovariance(CovarianceBase, ABC):
    def __init__(self, mf5mt):
        """
        Base class for energy distribution covariance data.

        Parameters:
        - mf5mt: The energy distribution object from MF5 (e.g., MT=18 for fission).
        """
        self.mf5mt = mf5mt
        self.covariance_matrix = None
        self.parameters = None
        self.energy_data = None  # Will hold EnergyDistributionData

    @staticmethod
    def fill_from_energy_distribution(endf_tape: Tape, covariance_objects: list, mt_covariance_dict: dict = None):
        """
        Factory method to create an energy distribution uncertainty object from an ENDF tape.
        Processes ONE MT reaction at a time.
        
        Parameters:
        - endf_tape: The ENDF tape containing the nuclear data
        - covariance_objects: List to append created uncertainty objects to
        - mt_covariance_dict: Dictionary with structure {MT: incident_energy_indices} for a single MT reaction
                             e.g., {18: [0, 1, 2]} indicating which incident energies have covariance data
        """

        mf35 = endf_tape.MAT(endf_tape.material_numbers[0]).MF(35)
        mf5 = endf_tape.MAT(endf_tape.material_numbers[0]).MF(5)
        
        # Process single MT reaction
        if mt_covariance_dict and len(mt_covariance_dict) > 0:
            # Extract the single MT from the dictionary
            mt_number = list(mt_covariance_dict.keys())[0]
            incident_energy_indices = mt_covariance_dict[mt_number]
            
            print(f"Processing MT{mt_number} energy distribution with incident energy indices {incident_energy_indices}")
        else:
            # Fallback: process all available MT reactions in MF35
            print("Warning: No MT covariance dictionary provided, processing all available MT reactions")
            mt_number = mf35.section_numbers[0] if mf35.section_numbers else None
            incident_energy_indices = None
            
            if mt_number is None:
                print("No MT sections found in MF35")
                return
        
        try:
            # Check if corresponding MF5 section exists
            if not mf5.has_MT(mt_number):
                print(f"Warning: MF5 MT{mt_number} not found, skipping energy distribution for MT{mt_number}")
                return
            
            # Check if MF35 section exists
            if not mf35.has_MT(mt_number):
                print(f"Warning: MF35 MT{mt_number} not found, skipping energy distribution for MT{mt_number}")
                return
            
            print(f"Creating energy distribution uncertainty for MT{mt_number}...")
            mf5mt = mf5.MT(mt_number).parse()
            mf35mt = mf35.MT(mt_number).parse()
            
            from .Uncertainty_Energydist import Uncertainty_Energydist
            energy_uncertainty = Uncertainty_Energydist(mf5mt, mf35mt, mt_number, incident_energy_indices)
            covariance_objects.append(energy_uncertainty)
            print(f"✓ Created energy distribution uncertainty for MT{mt_number}")
            
        except Exception as e:
            print(f"Warning: Could not create energy distribution uncertainty for MT{mt_number}: {e}")
            import traceback
            traceback.print_exc()
            return
        
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

    def write_additional_data_to_hdf5(self, hdf5_group):
        """
        Write additional data specific to energy distributions.
        """
        if self.energy_data is not None:
            energy_group = hdf5_group.require_group('Parameters')
            self.energy_data.write_to_hdf5(energy_group)
    
    @staticmethod
    def read_hdf5_group(group, covariance_objects: List["CovarianceBase"]):
        for subgroup_name in group:
            subgroup = group[subgroup_name]
            
            if subgroup_name == 'EnergyDistribution':
                from .Uncertainty_Energydist import Uncertainty_Energydist
                covariance_obj = Uncertainty_Energydist.read_from_hdf5(subgroup)
                covariance_objects.append(covariance_obj)

    @abstractmethod
    def update_tape(self, tape, sample_index=1, sample_name=""):
        """
        Update the ENDF tape with sampled parameters.
        """
        pass
