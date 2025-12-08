import ENDFtk
from ENDFtk.tree import Tape
import bisect
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from ..CovarianceBase import CovarianceBase

class MultiplicityCovariance(CovarianceBase, ABC):
    def __init__(self, mf1mt):
        """
        Base class for multiplicity covariance data.

        Parameters:
        - mf1mt: The multiplicity section from MF1.
        """
        super().__init__()
        self.mf1mt = mf1mt
        self.MT = mf1mt.MT
        self.covariance_matrix = None
        self.parameters = None
        
    @staticmethod
    def fill_from_multiplicity(endf_tape: Tape, covariance_objects: list, mt_covariance_dict: dict = None):
        """
        Factory method to create multiplicity uncertainty objects from an ENDF tape.
        Processes ONE MT reaction at a time for multiplicity data (MT=455 delayed, MT=456 prompt).
        
        Parameters:
        - endf_tape: The ENDF tape containing the nuclear data
        - covariance_objects: List to append created uncertainty objects to
        - mt_covariance_dict: Dictionary with structure {MT: []} for a single MT reaction
                             e.g., {455: []} or {456: []} indicating multiplicity covariance data
        """

        mf31 = endf_tape.MAT(endf_tape.material_numbers[0]).MF(31)
        mf1 = endf_tape.MAT(endf_tape.material_numbers[0]).MF(1)
        
        # Process single MT reaction
        if mt_covariance_dict and len(mt_covariance_dict) > 0:
            # Extract the single MT from the dictionary
            mt_number = list(mt_covariance_dict.keys())[0]
            
            # Skip MT452 (total) - it will be reconstructed from MT455 + MT456
            if mt_number == 452:
                print(f"Skipping MT452 (total neutron multiplicity) - will be reconstructed from MT455 + MT456")
                return
                
            print(f"Processing multiplicity MT{mt_number}")
        else:
            # Fallback: process all available MT reactions in MF31 (should not happen in normal flow)
            print("Warning: No MT covariance dictionary provided, processing all available MT reactions")
            mt_number = mf31.section_numbers[0] if mf31.section_numbers else None
            
            if mt_number is None:
                print("No MT sections found in MF31")
                return
        
        try:
            # Check if corresponding MF1 section exists
            if not mf1.has_MT(mt_number):
                print(f"Warning: MF1 MT{mt_number} not found, skipping multiplicity for MT{mt_number}")
                return
            
            # Check if MF31 section exists
            if not mf31.has_MT(mt_number):
                print(f"Warning: MF31 MT{mt_number} not found, skipping multiplicity for MT{mt_number}")
                return
            
            print(f"Creating multiplicity uncertainty for MT{mt_number}...")
            mf1mt = mf1.MT(mt_number).parse()
            mf31mt = mf31.MT(mt_number).parse()
            
            from .Uncertainty_Multiplicity import Uncertainty_Multiplicity
            multiplicity_uncertainty = Uncertainty_Multiplicity(mf1mt, mf31mt, mt_number)
            covariance_objects.append(multiplicity_uncertainty)
            print(f"✓ Created multiplicity uncertainty for MT{mt_number}")
            
        except Exception as e:
            print(f"Warning: Could not create multiplicity uncertainty for MT{mt_number}: {e}")
            return
        
    #-----------------
    # Communication  
    #-----------------

    def write_to_hdf5(self, hdf5_group):
        """
        Writes the covariance data to an HDF5 group.
        """
        # Write L_matrix
        if hasattr(self, 'L_matrix') and self.L_matrix is not None:
            hdf5_group.create_dataset('L_matrix', data=self.L_matrix)
        # Write mean_vector
        if hasattr(self, 'mean_vector') and self.mean_vector is not None:
            hdf5_group.create_dataset('mean_vector', data=self.mean_vector)
        # Write standard deviations if available
        if hasattr(self, 'std_dev_vector') and self.std_dev_vector is not None:
            hdf5_group.create_dataset('std_dev_vector', data=self.std_dev_vector)
        # Indicate if L_matrix is a Cholesky decomposition
        hdf5_group.attrs['is_cholesky'] = self.is_cholesky
        hdf5_group.attrs['MT'] = self.MT
        # Call the derived class method to write format-specific data
        self.write_additional_data_to_hdf5(hdf5_group)

    def write_additional_data_to_hdf5(self, hdf5_group):
        """
        Write additional data specific to multiplicity (parameters).
        """
        if self.parameters is not None:
            param_group = hdf5_group.require_group('Parameters')
            self.parameters.write_to_hdf5(param_group)
    
    @staticmethod
    def read_hdf5_group(group, covariance_objects: List["CovarianceBase"]):
        for subgroup_name in group:
            subgroup = group[subgroup_name]
            
            # Look for MT subgroups (MT455, MT456, etc.)
            if subgroup_name.startswith('MT'):
                from .Uncertainty_Multiplicity import Uncertainty_Multiplicity
                covariance_obj = Uncertainty_Multiplicity.read_from_hdf5(subgroup)
                if covariance_obj is not None:
                    covariance_objects.append(covariance_obj)

    def print_parameters(self):
        """
        Prints information about the multiplicity parameters.
        """
        if self.parameters is not None:
            print(f"Multiplicity Distribution (MT={self.parameters.mt}):")
            print(f"  Energy range: [{self.parameters.energies[0]:.2e}, {self.parameters.energies[-1]:.2e}] eV")
            print(f"  Energy bins: {len(self.parameters.energies)-1}")
            if len(self.parameters.multiplicities) > 0:
                print(f"  Samples: {len(self.parameters.multiplicities)} realizations")
                print(f"  Nominal multiplicity range: [{min(self.parameters.multiplicities[0]):.4f}, {max(self.parameters.multiplicities[0]):.4f}]")
            if len(self.parameters.std_dev) > 0:
                print(f"  Relative std deviations: [{min(self.parameters.std_dev):.4f}, {max(self.parameters.std_dev):.4f}]")
        else:
            print("No multiplicity parameter data available")
        
        if hasattr(self, 'covariance_matrix') and self.covariance_matrix is not None:
            print(f"\nCovariance matrix: {self.covariance_matrix.shape[0]}×{self.covariance_matrix.shape[1]}")
    
    @abstractmethod
    def update_tape(self, tape, sample_index=1, sample_name=""):
        """
        Update the ENDF tape with sampled parameters.
        """
        pass