import ENDFtk
from ENDFtk.tree import Tape
import bisect
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from ..CovarianceBase import CovarianceBase

class ResonanceRangeCovariance(CovarianceBase, ABC):
    def __init__(self, mf2_resonance_range, NER):
        """
        Base class for resonance covariance data.

        Parameters:
        - resonance_range: The resonance range object from MF32.
        - NER: Energy range index (integer).
        """
        super().__init__()  # Initialize the CovarianceBase
        self.resonance_range = mf2_resonance_range
        self.NER = NER  # Energy range identifier
        self.LRF = mf2_resonance_range.LRF  # Resonance formalism flag
        self.LRU = mf2_resonance_range.LRU  # Resonance type (resolved or unresolved)
        self.resonance_parameters = mf2_resonance_range.parameters
        self.parameters = None
        self.AP = None  # Scattering radius
        self.DAP = None  # Scattering radius uncertainty
    
    @staticmethod
    def fill_from_resonance_range(endf_tape : Tape, covariance_objects : list, want_reduced: bool = False):
        mf2 = endf_tape.MAT(endf_tape.material_numbers[0]).MF(2).MT(151).parse()
        mf32 = endf_tape.MAT(endf_tape.material_numbers[0]).MF(32).MT(151).parse()
        
        for NER, mf32_resonance_range in enumerate(mf32.isotopes[0].resonance_ranges.to_list()):
            mf2_resonance_range = mf2.isotopes[0].resonance_ranges[NER]
            LRU = mf2_resonance_range.LRU
            LRF = mf2_resonance_range.LRF
            if LRU == 1 and LRF == 2:
                pass
                # return MultiLevelBreitWignerCovariance(resonance_range, mf2_resonance_ranges, NER)
            elif LRU == 1 and LRF == 3:
                pass
                from .ReichMoore.Uncertainty_RM_RRR import Uncertainty_RM_RRR
                covariance_objects.append(Uncertainty_RM_RRR(mf2_resonance_range, mf32_resonance_range, NER))
                # return RRRReichMooreUncertainty(mf2_resonance_range, mf32_resonance_range, NER)
            elif LRU == 1 and LRF == 7:
                from .RMatrixLimited.Uncertainty_RML_RRR import Uncertainty_RML_RRR
                covariance_objects.append(Uncertainty_RML_RRR(mf2_resonance_range, mf32_resonance_range, NER, want_reduced))
                # return RMatrixLimitedCovariance(resonance_range, mf2_resonance_ranges, NER)
                pass
            elif LRU == 2 and LRF == 2:
                from .BreitWigner.Uncertainty_BW_URR import Uncertainty_BW_URR
                covariance_objects.append(Uncertainty_BW_URR(mf2_resonance_range, mf32_resonance_range, NER))
                # return URRBreitWignerUncertainty(mf2_resonance_range, mf32_resonance_range, NER)
            else:
                raise NotImplementedError(f"Resonance covariance format not supported LRU={LRU}, LRF={LRF}")
     

    # def sample_parameters(self):
    #     """ This method will be overridden by subclasses """
    #     raise NotImplementedError("Subclasses must implement this method")

    
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
    
    # Methods now inherited from CovarianceBase:
    # - delete_parameters
    # - remove_zero_variance_parameters
    # - compute_L_matrix
    # - write_to_hdf5
           
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
        
    # @classmethod
    # def from_resonance_range(cls, resonance_range, mf2_resonance_ranges, NER):
    #     LRU = resonance_range.LRU
    #     LRF = resonance_range.LRF
    #     # if LRU == 1 and LRF == 2:
    #     #     from .RRR_RMUncertainty import RRRReichMooreUncertainty
    #     #     from .MLBWUncertainty import MultiLevelBreitWignerCovariance
    #     #     return MultiLevelBreitWignerCovariance(resonance_range, mf2_resonance_ranges, NER)
    #     if LRU == 1 and LRF == 3:
    #         from .RRR_RM_Uncertainty import ReichMooreCovariance
    #         from .RRR_RM_Uncertainty import RRRReichMooreUncertainty
    #         return ReichMooreCovariance(resonance_range, mf2_resonance_ranges, NER)
    #     # elif LRU == 1 and LRF == 7:
    #     #     from .RMatrixLimitedUncertainty import RMatrixLimitedCovariance
    #     #     return RMatrixLimitedCovariance(resonance_range, mf2_resonance_ranges, NER)
    #     elif LRU == 2 and LRF == 1:
    #         from .URR_BW_Uncertainty import URRBreitWignerUncertainty
    #         return URRBreitWignerUncertainty(resonance_range, mf2_resonance_ranges, NER)
    #     else:
    #         raise NotImplementedError("Resonance covariance format not supported")

    def _update_resonance_range(self, tape, updated_parameters : ENDFtk.MF2.MT151.ResonanceParameters):
        """
        Updates the resonance range in the tape with sampled parameters.

        Parameters:
        - tape: The ENDF tape object to update.
        - sample_index: Index of the sample to use (default is 1, since index 0 is the original value).
        """
        mf2mt151 = tape.MAT(tape.material_numbers[0]).MF(2).MT(151).parse()
        original_isotope = mf2mt151.isotopes[0]
        resonance_ranges = original_isotope.resonance_ranges.to_list()

        # Create new resonance range
        new_range = ENDFtk.MF2.MT151.ResonanceRange(
            el = resonance_ranges[self.NER].EL,
            eh = resonance_ranges[self.NER].EH,
            naps = resonance_ranges[self.NER].NAPS,
            parameters = updated_parameters
        )
        
        # Replace the parameters in the resonance range
        resonance_ranges[self.NER] = new_range

        # Create new isotope with updated resonance ranges
        new_isotope = ENDFtk.MF2.MT151.Isotope(
            zai=original_isotope.ZAI,
            abn=original_isotope.ABN,
            lfw=original_isotope.LFW,
            ranges=resonance_ranges
        )

        # Create new section with the updated isotope
        new_section = ENDFtk.MF2.MT151.Section(
            zaid=mf2mt151.ZA,
            awr=mf2mt151.AWR,
            isotopes=[new_isotope]
        )

        # Replace the existing section in the tape
        mat_num = tape.material_numbers[0]
        tape.MAT(mat_num).MF(2).insert_or_replace(new_section)
        
    def update_resonance_range(self, tape, sample_index=1):
        """
        Updates the resonance ranges in the tape with the sampled parameters for the given sample index.

        Parameters:
        - tape: The ENDF tape object to update.
        - sample_index: The index of the sample to use for updating the parameters.
        """
        # Parse the tape
        mat_num = tape.material_numbers[0]
        mf2mt151 = tape.MAT(mat_num).MF(2).MT(151).parse()
        isotope = mf2mt151.isotopes[0]
        resonance_ranges = isotope.resonance_ranges.to_list()

        # Validate NER
        if self.NER >= len(resonance_ranges):
            raise IndexError(f"NER {self.NER} is out of bounds for the resonance ranges.")

        # Update the resonance range with matching NER
        updated_ranges = []
        for idx, rr in enumerate(resonance_ranges):
            if idx == self.NER:
                # Obtain updated parameters for the sample index
                updated_parameters = self.update_resonance_parameters(sample_index)
                # Create a new resonance range with the updated parameters
                updated_rr = ENDFtk.MF2.MT151.ResonanceRange(
                    EL=rr.EL,
                    EH=rr.EH,
                    LRU=rr.LRU,
                    LRF=rr.LRF,
                    LFW=rr.LFW,
                    parameters=updated_parameters
                )
                updated_ranges.append(updated_rr)
            else:
                updated_ranges.append(rr)

        # Reconstruct the isotope and section
        new_isotope = ENDFtk.MF2.MT151.Isotope(
            zai=isotope.ZAI,
            abn=isotope.ABN,
            lfw=isotope.LFW,
            ranges=updated_ranges
        )

        new_section = ENDFtk.MF2.MT151.Section(
            zaid=mf2mt151.ZA,
            awr=mf2mt151.AWR,
            isotopes=[new_isotope]
        )

        # Replace the existing section in the tape
        tape.MAT(mat_num).MF(2).insert_or_replace(new_section)


    @staticmethod
    def read_hdf5_group(group, covariance_objects: List["CovarianceBase"]):
        for subgroup_name in group:
            subgroup = group[subgroup_name]
            
            if subgroup_name == 'URR_BreitWigner':
                from .BreitWigner.Uncertainty_BW_URR import Uncertainty_BW_URR
                covariance_obj = Uncertainty_BW_URR.read_from_hdf5(subgroup)
            elif subgroup_name == 'Uncertainty_RM_RRR':
                from .ReichMoore.Uncertainty_RM_RRR import Uncertainty_RM_RRR
                covariance_obj = Uncertainty_RM_RRR.read_from_hdf5(subgroup)
            elif subgroup_name == 'Uncertainty_RML_RRR':
                from .RMatrixLimited.Uncertainty_RML_RRR import Uncertainty_RML_RRR
                covariance_obj = Uncertainty_RML_RRR.read_from_hdf5(subgroup)
            else:
                # Handle other covariance types
                pass
            covariance_objects.append(covariance_obj)
        
    # @abstractmethod
    # def write_additional_data_to_hdf5(self, hdf5_group):
    #     """
    #     Abstract method to write format-specific data to HDF5.
    #     To be implemented by derived classes.
    #     """
    #     pass

    # @abstractmethod
    # def print_parameters(self):
    #     pass
    
    @abstractmethod
    def update_tape(self):
        pass