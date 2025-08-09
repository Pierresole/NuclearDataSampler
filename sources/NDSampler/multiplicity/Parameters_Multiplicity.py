from dataclasses import dataclass, field
from typing import List, Optional, Dict
import ENDFtk
import numpy as np

@dataclass
class Multiplicities:
    mt: int
    energies: List[float] = field(default_factory=list)
    multiplicities: List[List[float]] = field(default_factory=list)  # multiplicities[sample_index][energy_bin] - actual multiplicities
    rel_std_dev: List[float] = field(default_factory=list)  # rel_std_dev[energy_bin] - relative standard deviations
    factors: List[List[float]] = field(default_factory=list)  # factors[sample_index][energy_bin] - multiplicative factors
    relative_covariance_matrix: Optional[np.ndarray] = None  # Relative covariance matrix
    constraints: Optional[dict] = None
    
    # Backward compatibility properties
    @property
    def std_dev(self):
        """Backward compatibility: return relative std dev"""
        return self.rel_std_dev
    
    @property
    def covariance_matrix(self):
        """Backward compatibility: return relative covariance matrix"""
        return self.relative_covariance_matrix
    
    @classmethod
    def from_endftk(cls, mf1mt, mf31mt):
        """
        Create Multiplicities from ENDFtk MF1 and MF31 sections.
        Handles both SquareMatrix (MT456) and CovariancePairs (MT455) formats.
        """
        mt = mf1mt.MT
        energies = mf1mt.nubar.energies.to_list()
        nominal_multiplicities = mf1mt.nubar.multiplicities.to_list()
        
        # Extract covariance data from MF31 - depends on the structure
        reaction = mf31mt.reactions.to_list()[0]
        explicit_cov = reaction.explicit_covariances[0]
        
        if hasattr(explicit_cov, 'NE'):
            # SquareMatrix format (MT456 - prompt)
            print(f"  Using SquareMatrix format for MT{mt}")
            NE = explicit_cov.NE - 1  # Number of energy bins
            values = explicit_cov.values[:]
            
            # Reconstruct the symmetric relative covariance matrix
            relative_covariance_matrix = np.zeros((NE, NE))
            triu_indices = np.triu_indices(NE)
            relative_covariance_matrix[triu_indices] = values
            relative_covariance_matrix[(triu_indices[1], triu_indices[0])] = values  # mirror upper to lower
            
            # Get relative standard deviations from diagonal (on covariance energy grid)
            rel_variances = np.diag(relative_covariance_matrix)
            rel_std_dev_cov_grid = np.sqrt(rel_variances)
            
            print(f"    Relative covariance matrix shape: {relative_covariance_matrix.shape}")
            print(f"    Relative std dev range on cov grid: [{rel_std_dev_cov_grid.min():.6f}, {rel_std_dev_cov_grid.max():.6f}]")
            
            # Get covariance energy bin edges and centers
            cov_energies = explicit_cov.energies[:]  # bin edges, length NE+1
            cov_bin_centers = np.array([(cov_energies[i] + cov_energies[i + 1]) / 2 for i in range(len(cov_energies) - 1)])
            
            # Map relative standard deviations from covariance grid to multiplicity grid
            mult_energies = np.array(energies)
            rel_std_dev = []
            
            print(f"    Mapping rel_std_dev from covariance grid ({len(rel_std_dev_cov_grid)}) to multiplicity grid ({len(mult_energies)})")
            
            for mult_energy in mult_energies:
                if mult_energy <= cov_bin_centers[0]:
                    # Use first covariance bin
                    rel_std_dev.append(rel_std_dev_cov_grid[0])
                elif mult_energy >= cov_bin_centers[-1]:
                    # Use last covariance bin
                    rel_std_dev.append(rel_std_dev_cov_grid[-1])
                else:
                    # Linear interpolation between covariance bins
                    for i in range(len(cov_bin_centers) - 1):
                        if cov_bin_centers[i] <= mult_energy <= cov_bin_centers[i + 1]:
                            # Linear interpolation
                            x0, x1 = cov_bin_centers[i], cov_bin_centers[i + 1]
                            y0, y1 = rel_std_dev_cov_grid[i], rel_std_dev_cov_grid[i + 1]
                            
                            if x1 - x0 == 0:
                                rel_std_dev.append(y0)
                            else:
                                y = y0 + (y1 - y0) * (mult_energy - x0) / (x1 - x0)
                                rel_std_dev.append(y)
                            break
            
            print(f"    Mapped rel_std_dev range: [{min(rel_std_dev):.6f}, {max(rel_std_dev):.6f}]")
            
            # Convert to list for consistency
            rel_std_dev = [float(s) for s in rel_std_dev]
            
        elif hasattr(explicit_cov, 'first_array_energies'):
            # CovariancePairs format (MT455 - delayed)
            print(f"  Using CovariancePairs format for MT{mt}")
            # For CovariancePairs, we need to extract the relative uncertainties
            # This is more complex and depends on the specific structure
            
            # Get energy arrays and F-values
            ek_energies = explicit_cov.first_array_energies[:]  # E_k energies
            el_energies = explicit_cov.second_array_energies[:] # E_l energies  
            fk_values = explicit_cov.first_array_fvalues[:]     # F(E_k) values (relative)
            fl_values = explicit_cov.second_array_fvalues[:]    # F(E_l) values (relative)
            
            print(f"    EK energies: {len(ek_energies)}, EL energies: {len(el_energies)}")
            print(f"    FK values: {len(fk_values)}, FL values: {len(fl_values)}")
            
            # For now, create a diagonal approximation based on available data
            # This is a simplified approach - you might need to implement the full
            # covariance reconstruction based on the ENDF-6 format specifications
            
            # Map the F-values to the energy grid of the multiplicities
            # F-values represent relative standard deviations
            rel_std_dev = []
            for energy in energies:
                # Find closest energy in EK array and use corresponding FK value
                if len(ek_energies) > 0 and len(fk_values) > 0:
                    idx = np.argmin(np.abs(np.array(ek_energies) - energy))
                    relative_std = abs(fk_values[idx]) if idx < len(fk_values) else 0.01
                else:
                    relative_std = 0.01  # Default 1% relative uncertainty
                rel_std_dev.append(relative_std)
            
            # Create a simple diagonal relative covariance matrix
            relative_covariance_matrix = np.diag(np.array(rel_std_dev) ** 2)
            
        else:
            raise ValueError(f"Unknown covariance format for MT{mt}")
        
        # Start with nominal case as sample 0
        multiplicities = [nominal_multiplicities]
        factors = [[1.0] * len(nominal_multiplicities)]
        
        return cls(
            mt=mt, 
            energies=energies, 
            multiplicities=multiplicities, 
            rel_std_dev=rel_std_dev,
            factors=factors,
            relative_covariance_matrix=relative_covariance_matrix
        )
    
    def get_multiplicities_for_sample(self, sample_index: int) -> List[float]:
        """Get actual multiplicities for a given sample index."""
        if sample_index == 0:
            # Sample index 0 is always the nominal case
            return self.multiplicities[0] if self.multiplicities else []
        elif len(self.multiplicities) > sample_index and self.multiplicities[sample_index] is not None:
            return self.multiplicities[sample_index]
        else:
            # ERROR: Sample was not generated or is invalid
            raise ValueError(f"Sample index {sample_index} not available for MT={self.mt}. "
                           f"Available samples: 0-{len(self.multiplicities)-1} (excluding None entries)")
    
    def get_factors_for_sample(self, sample_index: int) -> List[float]:
        """Get multiplicative factors for a given sample index.""" 
        if sample_index == 0:
            # Sample index 0 is always the nominal case (factors = 1.0)
            return [1.0] * len(self.energies)
        elif len(self.factors) > sample_index and self.factors[sample_index] is not None:
            return self.factors[sample_index]
        else:
            # ERROR: Sample was not generated or is invalid
            raise ValueError(f"Factor for sample index {sample_index} not available for MT={self.mt}. "
                           f"Available samples: 0-{len(self.factors)-1} (excluding None entries)")
    
    def get_standard_deviations(self) -> List[float]:
        """Get relative standard deviations for each energy bin."""
        return self.rel_std_dev

    def reconstruct(self, sample_index: int) -> List[float]:
        """
        Returns the actual multiplicity values for the given sample index.
        """
        return self.get_multiplicities_for_sample(sample_index)
    
    def reconstruct_factors(self, sample_index: int) -> List[float]:
        """
        Returns multiplicative factors for the given sample index (for backward compatibility).
        """
        return self.get_factors_for_sample(sample_index)

    def write_to_hdf5(self, hdf5_group):
        hdf5_group.attrs['mt'] = self.mt
        hdf5_group.create_dataset('energies', data=self.energies)
        hdf5_group.create_dataset('multiplicities', data=np.array(self.multiplicities))
        hdf5_group.create_dataset('rel_std_dev', data=np.array(self.rel_std_dev))
        hdf5_group.create_dataset('factors', data=np.array(self.factors))
        if self.relative_covariance_matrix is not None:
            hdf5_group.create_dataset('relative_covariance_matrix', data=self.relative_covariance_matrix)

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        mt = hdf5_group.attrs['mt']
        energies = hdf5_group['energies'][()].tolist()
        multiplicities = hdf5_group['multiplicities'][()].tolist() if 'multiplicities' in hdf5_group else []
        
        # Handle both old and new formats
        if 'rel_std_dev' in hdf5_group:
            rel_std_dev = hdf5_group['rel_std_dev'][()].tolist()
        elif 'std_dev' in hdf5_group:
            # Backward compatibility: assume old std_dev was actually relative
            rel_std_dev = hdf5_group['std_dev'][()].tolist()
        else:
            rel_std_dev = []
            
        factors = hdf5_group['factors'][()].tolist() if 'factors' in hdf5_group else []
        
        relative_covariance_matrix = None
        if 'relative_covariance_matrix' in hdf5_group:
            relative_covariance_matrix = hdf5_group['relative_covariance_matrix'][()]
        elif 'covariance_matrix' in hdf5_group:
            # Backward compatibility: assume old covariance_matrix was actually relative
            relative_covariance_matrix = hdf5_group['covariance_matrix'][()]
        
        return cls(
            mt=mt,
            energies=energies,
            multiplicities=multiplicities,
            rel_std_dev=rel_std_dev,
            factors=factors,
            relative_covariance_matrix=relative_covariance_matrix
        )