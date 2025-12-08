import ENDFtk
from ENDFtk.tree import Tape
import bisect
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from ..CovarianceBase import CovarianceBase

class AngularDistributionCovariance(CovarianceBase, ABC):
    def __init__(self, mf4mt2):
        """
        Base class for angular covariance data.

        Parameters:
        - angular dsitributions: The resonance range object from MF4.
        """
        self.mf4mt2 = mf4mt2
        self.covariance_matrix = None
        self.parameters = None
        self.legendre_data = None  # Add this to hold LegendreCoefficients

    @staticmethod
    def fill_from_angular_distribution(endf_tape: Tape, covariance_objects: list, mt_covariance_dict: dict = None):
        """
        Factory method to create an angular distribution uncertainty object from an ENDF tape.
        Processes ONE MT reaction at a time.
        
        Parameters:
        - endf_tape: The ENDF tape containing the nuclear data
        - covariance_objects: List to append created uncertainty objects to
        - mt_covariance_dict: Dictionary with structure {MT: [L_orders]} for a single MT reaction
                             e.g., {2: [1, 2, 3, 4, 5, 6]} indicating which Legendre orders have covariance data
        """

        mf34 = endf_tape.MAT(endf_tape.material_numbers[0]).MF(34)
        mf4 = endf_tape.MAT(endf_tape.material_numbers[0]).MF(4)
        
        # Process single MT reaction
        if mt_covariance_dict and len(mt_covariance_dict) > 0:
            # Extract the single MT and its Legendre orders from the dictionary
            mt_number = list(mt_covariance_dict.keys())[0]
            legendre_orders = mt_covariance_dict[mt_number]
            
            print(f"Processing MT{mt_number} with Legendre orders {legendre_orders}")
        else:
            # Fallback: process all available MT reactions in MF34 (should not happen in normal flow)
            print("Warning: No MT covariance dictionary provided, processing all available MT reactions")
            mt_number = mf34.section_numbers[0] if mf34.section_numbers else None
            legendre_orders = None
            
            if mt_number is None:
                print("No MT sections found in MF34")
                return
        
        try:
            # Check if corresponding MF4 section exists
            if not mf4.has_MT(mt_number):
                print(f"Warning: MF4 MT{mt_number} not found, skipping angular distribution for MT{mt_number}")
                return
            
            # Check if MF34 section exists
            if not mf34.has_MT(mt_number):
                print(f"Warning: MF34 MT{mt_number} not found, skipping angular distribution for MT{mt_number}")
                return
            
            print(f"Creating angular distribution uncertainty for MT{mt_number}...")
            mf4mt = mf4.MT(mt_number).parse()
            mf34mt = mf34.MT(mt_number).parse()
            
            from .Uncertainty_Angular import Uncertainty_Angular
            angular_uncertainty = Uncertainty_Angular(mf4mt, mf34mt, mt_number, legendre_orders)
            covariance_objects.append(angular_uncertainty)
            print(f"✓ Created angular distribution uncertainty for MT{mt_number}")
            
        except Exception as e:
            print(f"Warning: Could not create angular distribution uncertainty for MT{mt_number}: {e}")
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
        Write additional data specific to angular distributions (Legendre coefficients).
        """
        if self.legendre_data is not None:
            leg_group = hdf5_group.require_group('Parameters')
            self.legendre_data.write_to_hdf5(leg_group)
    
    @staticmethod
    def read_hdf5_group(group, covariance_objects: List["CovarianceBase"]):
        for subgroup_name in group:
            subgroup = group[subgroup_name]
            
            if subgroup_name == 'AngularDistribution':
                from .Uncertainty_Angular import Uncertainty_Angular
                covariance_obj = Uncertainty_Angular.read_from_hdf5(subgroup)
                covariance_objects.append(covariance_obj)

    @abstractmethod
    def update_tape(self, tape, sample_index=1, sample_name=""):
        """
        Update the ENDF tape with sampled parameters.
        """
        pass