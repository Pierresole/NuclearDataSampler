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
    def fill_from_resonance_range(endf_tape: Tape, covariance_objects: list):
        mf4mt2 = endf_tape.MAT(endf_tape.material_numbers[0]).MF(4).MT(2).parse()
        mf34mt2 = endf_tape.MAT(endf_tape.material_numbers[0]).MF(34).MT(2).parse()
        from .Uncertainty_Angular import Uncertainty_Angular
        covariance_objects.append(Uncertainty_Angular(mf4mt2, mf34mt2))
        
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

    def print_parameters(self):
        """
        Prints the parameters. This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def update_tape(self, tape, sample_index=1, sample_name=""):
        """
        Update the ENDF tape with sampled parameters.
        """
        pass