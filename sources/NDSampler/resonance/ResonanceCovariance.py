## Resonance Range
from ENDFtk.MF2.MT151 import ResonanceRange, Isotope, Section

import numpy as np
from abc import ABC, abstractmethod

class ResonanceCovariance(ABC):
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
    def fill_from_resonance_range(cls, resonance_range, mf2_resonance_ranges, NER):
        LRU = resonance_range.LRU
        LRF = resonance_range.LRF
        if LRU == 1 and LRF == 2:
            pass
            # return MultiLevelBreitWignerCovariance(resonance_range, mf2_resonance_ranges, NER)
        if LRU == 1 and LRF == 7:
            from .RMatrixLimited import Uncertainty_RML_RRR
            return Uncertainty_RML_RRR(resonance_range, mf2_resonance_ranges, NER)
        elif LRU == 2 and LRF == 1:
            from .BreitWigner import Uncertainty_BW_URR
            return Uncertainty_BW_URR(resonance_range, mf2_resonance_ranges, NER)
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
                updated_rr = ResonanceRange(
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
        tape.MAT(mat_num).MF(2).insert_or_replace(new_section)

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