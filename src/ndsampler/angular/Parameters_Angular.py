from dataclasses import dataclass, field
from typing import List, Optional, Dict
import ENDFtk
import numpy as np

@dataclass
class LegendreCoefficient:
    """Stores Legendre coefficient data for a single order.
    
    Follows methodology from mf34_test_clever.ipynb:
    - Stores nominal coefficients from MF4 (index 0)
    - Stores relative deviations δ where a_sample = a_nominal × (1 + δ)
    - Energy bins align with MF34 covariance structure
    """
    mt: int
    order: int  # Legendre order L (1, 2, 3, ...)
    energies: List[float] = field(default_factory=list)  # Covariance bin boundaries
    legcoeff: List[List[float]] = field(default_factory=list)  # [0] = nominal, rest computed on-demand
    rel_deviation: List[List[float]] = field(default_factory=list)  # delta per sample (1-indexed)

    def get_coefficients_for_sample(self, sample_index: int) -> List[float]:
        """Reconstruct absolute coefficients for a given sample.
        
        Args:
            sample_index: 0 for nominal, ≥1 for sampled perturbations
            
        Returns:
            List of absolute coefficient values for this energy grid
        """
        if sample_index == 0:
            return self.legcoeff[0] if self.legcoeff else []
        
        # Reconstruct from relative deviations: a_sample = a_nominal × (1 + δ)
        if sample_index < len(self.rel_deviation) and self.rel_deviation[sample_index] is not None:
            nominal = self.legcoeff[0]
            delta = self.rel_deviation[sample_index]
            return [nominal[i] * (1.0 + delta[i]) for i in range(len(nominal))]
        
        raise ValueError(f"Sample {sample_index} not available for L={self.order}")

    def get_factors_for_sample(self, sample_index: int) -> List[float]:
        """Get multiplicative factors (1 + δ) for a given sample.
        
        Args:
            sample_index: 0 for nominal (returns 1.0), ≥1 for samples
            
        Returns:
            List of multiplicative factors per energy bin
        """
        if sample_index == 0:
            n_bins = len(self.energies) - 1 if self.energies else 0
            return [1.0] * n_bins
        
        if sample_index < len(self.rel_deviation) and self.rel_deviation[sample_index] is not None:
            return [1.0 + d for d in self.rel_deviation[sample_index]]
        
        raise ValueError(f"Sample {sample_index} not available for L={self.order}")

    def write_to_hdf5(self, hdf5_group):
        """Write coefficient data to HDF5 group."""
        grp = hdf5_group.create_group(f"L{self.order}")
        grp.attrs['mt'] = self.mt
        grp.attrs['order'] = self.order
        grp.create_dataset('energies', data=self.energies)
        grp.create_dataset('legcoeff', data=np.array(self.legcoeff))
        if self.rel_deviation:
            grp.create_dataset('rel_deviation', data=np.array(self.rel_deviation))

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """Read coefficient data from HDF5 group."""
        mt = hdf5_group.attrs['mt']
        order = hdf5_group.attrs['order']
        energies = hdf5_group['energies'][()].tolist()
        legcoeff = hdf5_group['legcoeff'][()].tolist() if 'legcoeff' in hdf5_group else []
        rel_deviation = hdf5_group['rel_deviation'][()].tolist() if 'rel_deviation' in hdf5_group else []
        return cls(mt=mt, order=order, energies=energies, legcoeff=legcoeff, rel_deviation=rel_deviation)

@dataclass
class AngularDistributionData:
    """Container for angular distribution parameters from MF4/MF34.
    
    This class stores:
    - Legendre coefficients per order (with covariance bin structure)
    - Original MF4 data for interpolation during ENDF file reconstruction
    
    Name clarification:
    - LegendreCoefficient: Single Legendre order (L=1, L=2, etc.)
    - AngularDistributionData: Complete angular distribution (all orders)
    """
    coefficients: List[LegendreCoefficient] = field(default_factory=list)
    # Store original MF4 data for interpolation (avoid overwriting during sampling)
    original_mf4_energies: List[float] = field(default_factory=list)
    original_mf4_coefficients: List[List[float]] = field(default_factory=list)  # [energy_idx][coeff_idx]

    @classmethod
    def from_endftk(cls, mf4mt2, mf34mt2):
        """
        Parse Legendre coefficients from ENDFtk objects.
        Extract nominal coefficients from MF4 and compute standard deviations from MF34 covariance.
        Note: L=0 coefficient is implicit (always 1.0), first coefficient in MF4 is L=1.
        """
        coeffs = []
        
        # Extract Legendre blocks from MF34
        legendre_blocks = mf34mt2.legendre_blocks.to_list()
        
        # Create a mapping from Legendre order to energy bins and relative covariance matrices
        order_to_data = {}
        
        for block in legendre_blocks:
            l1 = block.first_legendre_order
            l2 = block.second_legendre_order
            
            # We only need diagonal blocks (l1 == l2) to get energy bins and standard deviations
            if l1 == l2:
                # Get energy boundaries from the first subblock
                first_data = block.data[0]
                if hasattr(first_data, 'energies'):
                    # SquareMatrix case
                    energies = first_data.energies[:]
                    # Extract the diagonal of the relative covariance matrix
                    values = first_data.values[:]
                    n_bins = len(energies) - 1
                    
                    # For symmetric matrix stored as triangular, extract diagonal
                    if len(values) == n_bins * (n_bins + 1) // 2:
                        # Upper triangular storage - extract diagonal elements  
                        # Element (i,i) position = sum(n_bins-j for j in range(i))
                        diagonal_rel_var = []
                        for i in range(n_bins):
                            diag_idx = sum(n_bins - j for j in range(i))
                            diagonal_rel_var.append(values[diag_idx])
                    elif len(values) == n_bins * n_bins:
                        # Full matrix storage - extract diagonal
                        diagonal_rel_var = [values[i * n_bins + i] for i in range(n_bins)]
                    else:
                        print(f"Warning: Unexpected matrix size for L={l1}")
                        continue
                        
                elif hasattr(first_data, 'first_array_energies'):
                    # CovariancePairs case (diagonal only)
                    energies = first_data.first_array_energies[:]
                    diagonal_rel_var = first_data.first_array_fvalues[:]
                else:
                    print(f"Warning: Unknown data type {type(first_data)} for L={l1}")
                    continue
                    
                order_to_data[l1] = {
                    'energies': energies,
                    'diagonal_rel_var': diagonal_rel_var
                }
        
        # Extract nominal coefficients from MF4
        if mf4mt2.LTT == 1:  # Pure Legendre case
            mf4_distributions = mf4mt2.distributions.angular_distributions.to_list()
        elif mf4mt2.LTT == 3:  # Mixed case
            mf4_distributions = mf4mt2.distributions.legendre.angular_distributions.to_list()
        else:
            print(f"Warning: LTT={mf4mt2.LTT} not implemented (only 1 and 3 are)")
            
        # mf4_distributions = mf4mt2.distributions.legendre.angular_distributions.to_list()
        mf4_energies = [dist.incident_energy for dist in mf4_distributions]
        
        # SAFEGUARD: verify covariance energy upper bounds do not exceed last nominal MF4 energy
        if mf4_energies:
            max_mf4_energy = mf4_energies[-1]
            for order_tmp, cov_data_tmp in order_to_data.items():
                try:
                    cov_last = cov_data_tmp['energies'][-1]
                except Exception:
                    continue
                if cov_last > max_mf4_energy:
                    print(
                        f"ERROR: Covariance energy upper bound {cov_last:.6e} eV for L={order_tmp} "
                        f"exceeds last MF4 incident energy {max_mf4_energy:.6e} eV. Inconsistent file energy boundaries."
                    )
        else:
            print("ERROR: No MF4 incident energies available for safeguard check.")

        # Store original MF4 data for interpolation (avoid overwriting during sampling)
        original_mf4_coefficients = []
        for dist in mf4_distributions:
            original_mf4_coefficients.append(dist.coefficients[:])

        # Create LegendreCoefficient objects for each order
        for order, cov_data in order_to_data.items():
            if order >= 1:  # Only L≥1 coefficients are stored and perturbed
                cov_energies = cov_data['energies']
                diagonal_rel_var = cov_data['diagonal_rel_var']
                n_bins = len(cov_energies) - 1
                
                # Extract nominal coefficients for this order from MF4
                nominal_coeffs = []
                std_deviations = []
                
                for bin_idx in range(n_bins):
                    # Use bin center energy for interpolation to avoid boundary issues
                    left_boundary = cov_energies[bin_idx]
                    right_boundary = cov_energies[bin_idx + 1] 
                    bin_center_energy = (left_boundary + right_boundary) / 2.0
                    
                    # Interpolate MF4 coefficient at the bin center
                    nominal_coeff = cls._interpolate_mf4_coefficient_at_energy(
                        bin_center_energy, order, mf4_energies, mf4_distributions
                    )
                    nominal_coeffs.append(nominal_coeff)
                    
                    # Compute absolute standard deviation from relative variance
                    rel_std = np.sqrt(max(0, diagonal_rel_var[bin_idx]))  # Relative standard deviation
                    abs_std = abs(nominal_coeff) * rel_std  # Absolute standard deviation
                    std_deviations.append(abs_std)
                
                coeff = LegendreCoefficient(
                    mt=2,  # Elastic scattering
                    order=order,
                    energies=cov_energies,
                    legcoeff=[nominal_coeffs],  # Initialize with nominal coefficients at index 0
                    rel_deviation=[]  # Will be populated during sampling
                )
                coeffs.append(coeff)
        
        return cls(coefficients=coeffs, original_mf4_energies=mf4_energies, original_mf4_coefficients=original_mf4_coefficients)
    
    @classmethod
    def _interpolate_mf4_coefficient_at_energy(cls, energy, order, mf4_energies, mf4_distributions):
        """
        Interpolate Legendre coefficient at given energy from MF4 data.
        
        Parameters:
        - energy: Energy at which to interpolate
        - order: Legendre order (1, 2, 3, ...)
        - mf4_energies: List of energies from MF4 distributions
        - mf4_distributions: List of MF4 LegendreCoefficients objects
        
        Returns:
        - Interpolated coefficient value
        """
        if order == 0:
            return 1.0  # L=0 coefficient is always 1.0
        
        # Find bracketing energies
        if energy <= mf4_energies[0]:
            # Below first energy - use first value
            coeffs = mf4_distributions[0].coefficients[:]
            if order <= len(coeffs):
                return coeffs[order - 1]  # order-1 because coefficients start with L=1
            else:
                return 0.0
        
        if energy >= mf4_energies[-1]:
            # Above last energy - use last value
            coeffs = mf4_distributions[-1].coefficients[:]
            if order <= len(coeffs):
                return coeffs[order - 1]
            else:
                return 0.0
        
        # Find interpolation indices
        for i in range(len(mf4_energies) - 1):
            if mf4_energies[i] <= energy <= mf4_energies[i + 1]:
                e_low = mf4_energies[i]
                e_high = mf4_energies[i + 1]
                
                # Get coefficient values at bracketing energies
                coeffs_low = mf4_distributions[i].coefficients[:]
                coeffs_high = mf4_distributions[i + 1].coefficients[:]
                
                a_low = coeffs_low[order - 1] if order <= len(coeffs_low) else 0.0
                a_high = coeffs_high[order - 1] if order <= len(coeffs_high) else 0.0
                
                # Linear interpolation
                if e_high == e_low:
                    return a_low  # Avoid division by zero
                else:
                    interpolated = a_low + (energy - e_low) / (e_high - e_low) * (a_high - a_low)
                    return interpolated
        
        # Should not reach here
        return 0.0
        
    def get_coefficients_at_energy(self, energy: float) -> List[float]:
        """
        Get Legendre coefficients at given energy, interpolating if necessary.
        Returns coefficients for L≥1 only (L=0 is implicit).
        Uses the stored original MF4 data to avoid corruption during sampling.
        """
        # If energy exists exactly in original data, use it
        if energy in self.original_mf4_energies:
            idx = self.original_mf4_energies.index(energy)
            return self.original_mf4_coefficients[idx][:]
        
        # Otherwise, interpolate
        if energy <= self.original_mf4_energies[0]:
            # Below first energy - use first coefficients
            return self.original_mf4_coefficients[0][:]
        
        if energy >= self.original_mf4_energies[-1]:
            # Above last energy - use last coefficients
            return self.original_mf4_coefficients[-1][:]
        
        # Find bracketing energies for interpolation
        for i in range(len(self.original_mf4_energies) - 1):
            if self.original_mf4_energies[i] <= energy <= self.original_mf4_energies[i + 1]:
                e_low = self.original_mf4_energies[i]
                e_high = self.original_mf4_energies[i + 1]
                coeffs_low = self.original_mf4_coefficients[i][:]
                coeffs_high = self.original_mf4_coefficients[i + 1][:]
                
                # Linear interpolation
                if e_high == e_low:
                    return coeffs_low
                
                # Interpolate each coefficient
                interpolated_coeffs = []
                max_len = max(len(coeffs_low), len(coeffs_high))
                for j in range(max_len):
                    a_low = coeffs_low[j] if j < len(coeffs_low) else 0.0
                    a_high = coeffs_high[j] if j < len(coeffs_high) else 0.0
                    interp_val = a_low + (energy - e_low) / (e_high - e_low) * (a_high - a_low)
                    interpolated_coeffs.append(interp_val)
                
                return interpolated_coeffs
        
        # Should not reach here
        return []

    def reconstruct(self, sample_index: int) -> Dict[int, List[float]]:
        """
        Returns a dict: {order: coefficients_for_this_sample}
        Now returns actual coefficients instead of factors.
        """
        return {c.order: c.get_coefficients_for_sample(sample_index) for c in self.coefficients}
    
    def reconstruct_factors(self, sample_index: int) -> Dict[int, List[float]]:
        """
        Returns a dict: {order: factors_for_this_sample} for backward compatibility
        """
        return {c.order: c.get_factors_for_sample(sample_index) for c in self.coefficients}

    def write_to_hdf5(self, hdf5_group):
        for coeff in self.coefficients:
            coeff.write_to_hdf5(hdf5_group)
        
        # Store original MF4 data for interpolation
        if self.original_mf4_energies:
            hdf5_group.create_dataset('original_mf4_energies', data=self.original_mf4_energies)
        if self.original_mf4_coefficients:
            # Store as 2D array (energy_idx, coeff_idx)  
            import numpy as np
            max_coeffs = max(len(coeffs) for coeffs in self.original_mf4_coefficients) if self.original_mf4_coefficients else 0
            padded_coeffs = np.zeros((len(self.original_mf4_coefficients), max_coeffs))
            for i, coeffs in enumerate(self.original_mf4_coefficients):
                padded_coeffs[i, :len(coeffs)] = coeffs
            hdf5_group.create_dataset('original_mf4_coefficients', data=padded_coeffs)

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        coeffs = []
        for key in hdf5_group:
            if key not in ['original_mf4_energies', 'original_mf4_coefficients']:
                coeff_grp = hdf5_group[key]
                coeffs.append(LegendreCoefficient.read_from_hdf5(coeff_grp))
        
        # Read original MF4 data if available
        original_energies = []
        original_coefficients = []
        if 'original_mf4_energies' in hdf5_group:
            original_energies = hdf5_group['original_mf4_energies'][()].tolist()
        if 'original_mf4_coefficients' in hdf5_group:
            padded_coeffs = hdf5_group['original_mf4_coefficients'][()]
            # Remove padding (trailing zeros) 
            for row in padded_coeffs:
                # Find last non-zero element
                last_nonzero = len(row) - 1
                while last_nonzero >= 0 and abs(row[last_nonzero]) < 1e-15:
                    last_nonzero -= 1
                original_coefficients.append(row[:last_nonzero+1].tolist())
                
        return cls(coefficients=coeffs, original_mf4_energies=original_energies, original_mf4_coefficients=original_coefficients)
