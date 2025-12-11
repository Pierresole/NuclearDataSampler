from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np
import h5py

@dataclass
class EnergyBinCoefficient:
    """Stores energy distribution data for a single incident energy.
    
    Similar to LegendreCoefficient for angular distributions:
    - Stores nominal probability distribution from MF5 (index 0)
    - Stores relative deviations δ where P_sample = P_nominal × (1 + δ)
    - Energy bins align with MF35 covariance structure
    """
    mt: int
    incident_energy: float  # Incident neutron energy
    incident_energy_index: int  # Index in the incident energy list
    outgoing_energies: List[float] = field(default_factory=list)  # Covariance bin boundaries for outgoing energy
    probabilities: List[List[float]] = field(default_factory=list)  # [0] = nominal, rest computed on-demand
    rel_deviation: List[List[float]] = field(default_factory=list)  # delta per sample (1-indexed)
    covariance_type: int = 5  # LB flag: 5=relative (multiplicative), 7=absolute (additive)

    def get_probabilities_for_sample(self, sample_index: int) -> List[float]:
        """Reconstruct absolute probabilities for a given sample.
        
        Args:
            sample_index: 0 for nominal, ≥1 for sampled perturbations
            
        Returns:
            List of probability values for this outgoing energy grid
        """
        if sample_index == 0:
            return self.probabilities[0] if self.probabilities else []
        
        # Reconstruct using appropriate formula based on covariance type
        if sample_index < len(self.rel_deviation) and self.rel_deviation[sample_index] is not None:
            nominal = self.probabilities[0]
            delta = self.rel_deviation[sample_index]
            
            if self.covariance_type == 7:
                # LB=7: Absolute covariance → ADDITIVE perturbations
                # P_sample = P_nominal + δ
                return [nominal[i] + delta[i] for i in range(len(nominal))]
            else:
                # LB=5: Relative covariance → MULTIPLICATIVE perturbations
                # P_sample = P_nominal × (1 + δ)
                return [nominal[i] * (1.0 + delta[i]) for i in range(len(nominal))]
        
        raise ValueError(f"Sample {sample_index} not available for incident energy index {self.incident_energy_index}")

    def get_factors_for_sample(self, sample_index: int) -> List[float]:
        """Get perturbation factors for a given sample.
        
        For LB=5 (relative): Returns multiplicative factors (1 + δ)
        For LB=7 (absolute): Returns additive factors (δ)
        
        Args:
            sample_index: 0 for nominal (returns 1.0 or 0.0), ≥1 for samples
            
        Returns:
            List of factors per outgoing energy bin
        """
        if sample_index == 0:
            n_bins = len(self.outgoing_energies) - 1 if self.outgoing_energies else 0
            if self.covariance_type == 7:
                return [0.0] * n_bins  # Additive: no change
            else:
                return [1.0] * n_bins  # Multiplicative: no change
        
        if sample_index < len(self.rel_deviation) and self.rel_deviation[sample_index] is not None:
            if self.covariance_type == 7:
                # LB=7: Return absolute deviations directly
                return self.rel_deviation[sample_index]
            else:
                # LB=5: Return multiplicative factors
                return [1.0 + d for d in self.rel_deviation[sample_index]]
        
        raise ValueError(f"Sample {sample_index} not available for incident energy index {self.incident_energy_index}")
    
    def apply_factors_to_continuous_data(self, original_energies: np.ndarray, original_probs: np.ndarray, 
                                        sample_index: int) -> tuple:
        """Apply bin-wise perturbation factors to continuous nominal data.
        
        This creates a piecewise-constant perturbation by:
        1. Duplicating bin boundaries in the energy grid (3 points at interior boundaries)
        2. Assigning the bin's factor to all points within that bin
        3. Multiplying nominal probabilities by factors
        
        Args:
            original_energies: Continuous energy grid from original MF5 data
            original_probs: Continuous probability values from original MF5 data
            sample_index: 0 for nominal, ≥1 for perturbed sample
            
        Returns:
            (enhanced_energies, perturbed_probs): Arrays with duplicated boundaries and applied factors
        """
        if sample_index == 0:
            # No perturbation for nominal
            return original_energies, original_probs
        
        # Get bin-wise multiplicative factors (1 + δ)
        factors = self.get_factors_for_sample(sample_index)
        bin_boundaries = np.array(self.outgoing_energies)
        
        # Create enhanced energy grid with boundary duplication
        enhanced_energies = []
        enhanced_factors = []
        
        # For each original energy point, determine which bin it's in and assign factor
        for e_orig in original_energies:
            # Find bin index
            bin_idx = np.searchsorted(bin_boundaries[1:], e_orig, side='right')
            bin_idx = min(bin_idx, len(factors) - 1)  # Clamp to last bin
            enhanced_energies.append(e_orig)
            enhanced_factors.append(factors[bin_idx])
        
        # Add boundary duplications for discontinuities
        final_energies = []
        final_factors = []
        final_probs = []
        
        enhanced_energies = np.array(enhanced_energies)
        enhanced_factors = np.array(enhanced_factors)
        
        # Interpolate original probabilities onto enhanced energy grid
        enhanced_probs = np.interp(enhanced_energies, original_energies, original_probs)
        
        epsilon = 1e-10  # Small offset for boundary duplication
        
        for i, (e, f, p) in enumerate(zip(enhanced_energies, enhanced_factors, enhanced_probs)):
            # Check if this energy is near a bin boundary (except first and last)
            is_boundary = False
            boundary_idx = -1
            
            for b_idx in range(1, len(bin_boundaries) - 1):
                if abs(e - bin_boundaries[b_idx]) < epsilon * bin_boundaries[b_idx]:
                    is_boundary = True
                    boundary_idx = b_idx
                    break
            
            if is_boundary:
                # Add three points: before, at, after boundary with different factors
                left_bin_idx = boundary_idx - 1
                right_bin_idx = boundary_idx
                
                # Point just before boundary (left bin factor)
                final_energies.append(e - epsilon * e)
                final_factors.append(factors[left_bin_idx])
                final_probs.append(p)
                
                # Point at boundary (transition - use right bin factor)
                final_energies.append(e)
                final_factors.append(factors[right_bin_idx])
                final_probs.append(p)
                
                # Point just after boundary (right bin factor)
                final_energies.append(e + epsilon * e)
                final_factors.append(factors[right_bin_idx])
                final_probs.append(p)
            else:
                final_energies.append(e)
                final_factors.append(f)
                final_probs.append(p)
        
        final_energies = np.array(final_energies)
        final_factors = np.array(final_factors)
        final_probs = np.array(final_probs)
        
        # Apply factors to probabilities (depends on covariance type)
        if self.covariance_type == 7:
            # LB=7: Additive perturbations
            perturbed_probs = final_probs + final_factors
        else:
            # LB=5: Multiplicative perturbations
            perturbed_probs = final_probs * final_factors
        
        return final_energies, perturbed_probs

    def write_to_hdf5(self, hdf5_group):
        """Write coefficient data to HDF5 group."""
        grp = hdf5_group.create_group(f"E{self.incident_energy_index}")
        grp.attrs['mt'] = self.mt
        grp.attrs['incident_energy'] = self.incident_energy
        grp.attrs['incident_energy_index'] = self.incident_energy_index
        grp.create_dataset('outgoing_energies', data=self.outgoing_energies)
        grp.create_dataset('probabilities', data=np.array(self.probabilities))
        if self.rel_deviation:
            grp.create_dataset('rel_deviation', data=np.array(self.rel_deviation))

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """Read coefficient data from HDF5 group."""
        mt = hdf5_group.attrs['mt']
        incident_energy = hdf5_group.attrs['incident_energy']
        incident_energy_index = hdf5_group.attrs['incident_energy_index']
        outgoing_energies = hdf5_group['outgoing_energies'][()].tolist()
        probabilities = hdf5_group['probabilities'][()].tolist() if 'probabilities' in hdf5_group else []
        rel_deviation = hdf5_group['rel_deviation'][()].tolist() if 'rel_deviation' in hdf5_group else []
        return cls(mt=mt, incident_energy=incident_energy, incident_energy_index=incident_energy_index,
                  outgoing_energies=outgoing_energies, probabilities=probabilities, rel_deviation=rel_deviation)

@dataclass
class EnergyDistributionData:
    """Container for energy distribution parameters from MF5/MF35.
    
    This class stores:
    - Probability distributions per incident energy (with covariance bin structure)
    - Original MF5 data for interpolation during ENDF file reconstruction
    
    Name clarification:
    - EnergyBinCoefficient: Single incident energy distribution
    - EnergyDistributionData: Complete energy distributions (all incident energies)
    """
    distributions: List[EnergyBinCoefficient] = field(default_factory=list)
    # Store original MF5 data for interpolation
    original_mf5_incident_energies: List[float] = field(default_factory=list)
    original_mf5_outgoing_energies: List[List[float]] = field(default_factory=list)  # [incident_idx][outgoing_idx]
    original_mf5_probabilities: List[List[float]] = field(default_factory=list)  # [incident_idx][outgoing_idx]

    @classmethod
    def from_endftk(cls, mf5mt, mf35mt):
        """
        Parse energy distributions from ENDFtk objects.
        Extract nominal distributions from MF5 and compute standard deviations from MF35 covariance.
        """
        distributions = []
        
        # Extract covariance blocks from MF35
        energy_blocks = mf35mt.energy_blocks
        
        # Create a mapping from incident energy index to covariance data
        incident_energy_to_data = {}
        
        for block_idx in range(mf35mt.number_energy_blocks):
            block = energy_blocks[block_idx]
            
            # Block covers incident energies from E1 to E2
            E1 = block.E1
            E2 = block.E2
            
            # Get outgoing energy bins for this block
            outgoing_energy_bins = np.array(block.energies[:])
            
            # LB flag: 5 = relative covariance, 7 = absolute covariance
            LB = block.LB
            
            # Extract covariance matrix (upper triangular stored in values)
            NE = block.NE - 1  # Number of outgoing energy bins
            values = block.values[:]
            
            # Build full symmetric covariance matrix
            cov_matrix = np.zeros((NE, NE))
            triu_indices = np.triu_indices(NE)
            cov_matrix[triu_indices] = values
            cov_matrix[(triu_indices[1], triu_indices[0])] = values
            
            # Store covariance data for this incident energy range
            incident_energy_to_data[block_idx] = {
                'E1': E1,
                'E2': E2,
                'outgoing_energy_bins': outgoing_energy_bins,
                'cov_matrix': cov_matrix,
                'LB': LB
            }
        
        # Extract nominal distributions from MF5
        # For simplicity, assume we're dealing with partial distributions (subsection.distributions)
        partial_dists = mf5mt.partial_distributions
        
        if len(partial_dists) == 0:
            raise ValueError("No partial distributions found in MF5")
        
        # Use first partial distribution (typically the only one for fission)
        partial_dist = partial_dists[0]
        distribution = partial_dist.distribution
        
        # Get incident energies and outgoing distributions
        incident_energies = []
        original_outgoing_energies = []
        original_probabilities = []
        
        for dist_idx in range(distribution.number_incident_energies):
            outgoing_dist = distribution.outgoing_distributions[dist_idx]
            incident_e = outgoing_dist.incident_energy
            incident_energies.append(incident_e)
            
            # Extract outgoing energies and probabilities
            outgoing_e = np.array(outgoing_dist.outgoing_energies.to_list())
            probs = np.array(outgoing_dist.probabilities.to_list())
            
            original_outgoing_energies.append(outgoing_e.tolist())
            original_probabilities.append(probs.tolist())
        
        # Create EnergyBinCoefficient objects - ONE per covariance block
        # Use the first MF5 incident energy that falls within each block's [E1, E2] range
        print(f"\nCreating distributions for covariance blocks:")
        print("=" * 80)
        
        for block_idx, cov_data in incident_energy_to_data.items():
            E1 = cov_data['E1']
            E2 = cov_data['E2']
            covariance_outgoing_bins = cov_data['outgoing_energy_bins']
            
            # Find first MF5 incident energy within this block's range
            matching_inc_idx = None
            for inc_idx, incident_e in enumerate(incident_energies):
                if E1 <= incident_e <= E2:
                    matching_inc_idx = inc_idx
                    break
            
            if matching_inc_idx is None:
                print(f"  Block {block_idx}: [{E1:.4e}, {E2:.4e}] eV - NO matching MF5 incident energy!")
                continue
            
            incident_e = incident_energies[matching_inc_idx]
            print(f"  Block {block_idx}: [{E1:.4e}, {E2:.4e}] eV - using incident E = {incident_e:.4e} eV (index {matching_inc_idx})")
            
            # Integrate nominal probabilities over covariance bins
            nominal_outgoing_e = np.array(original_outgoing_energies[matching_inc_idx])
            nominal_probs = np.array(original_probabilities[matching_inc_idx])
            
            # Integrate probabilities over covariance bins
            integrated_probs = []
            for i in range(len(covariance_outgoing_bins) - 1):
                bin_lower = covariance_outgoing_bins[i]
                bin_upper = covariance_outgoing_bins[i+1]
                
                # Find nominal points within this bin
                mask = (nominal_outgoing_e >= bin_lower) & (nominal_outgoing_e < bin_upper)
                
                if np.any(mask):
                    E_in_bin = nominal_outgoing_e[mask]
                    P_in_bin = nominal_probs[mask]
                    if len(E_in_bin) > 1:
                        # Integrate using trapezoidal rule
                        integrated = np.trapz(P_in_bin, E_in_bin)
                    else:
                        # Single point - approximate
                        integrated = P_in_bin[0] * (bin_upper - bin_lower)
                else:
                    # No points in bin - interpolate
                    P_lower = np.interp(bin_lower, nominal_outgoing_e, nominal_probs)
                    P_upper = np.interp(bin_upper, nominal_outgoing_e, nominal_probs)
                    integrated = 0.5 * (P_lower + P_upper) * (bin_upper - bin_lower)
                
                integrated_probs.append(integrated)
            
            # Normalize to ensure sum = 1
            integrated_probs = np.array(integrated_probs)
            sum_before = integrated_probs.sum()
            if integrated_probs.sum() > 0:
                integrated_probs = integrated_probs / integrated_probs.sum()
            
            print(f"    Integrated probs: sum_before={sum_before:.6f}, sum_after={integrated_probs.sum():.6f}")
            print(f"    Range: [{integrated_probs.min():.6e}, {integrated_probs.max():.6e}]")
            
            # Create EnergyBinCoefficient
            energy_coeff = EnergyBinCoefficient(
                mt=mf5mt.MT,
                incident_energy=incident_e,
                incident_energy_index=matching_inc_idx,
                outgoing_energies=covariance_outgoing_bins.tolist(),
                probabilities=[integrated_probs.tolist()],  # [0] = nominal INTEGRATED probabilities
                rel_deviation=[],  # Will be filled during sampling
                covariance_type=cov_data['LB']  # Store LB flag (5=relative, 7=absolute)
            )
            
            distributions.append(energy_coeff)
        
        print("=" * 80)
        print(f"✓ Created {len(distributions)} distributions (one per covariance block)")
        
        return cls(
            distributions=distributions,
            original_mf5_incident_energies=incident_energies,
            original_mf5_outgoing_energies=original_outgoing_energies,
            original_mf5_probabilities=original_probabilities
        )
    
    def get_distribution_at_incident_energy(self, incident_energy: float, incident_energy_idx: int) -> tuple:
        """
        Get outgoing energy distribution at given incident energy.
        Returns (outgoing_energies, probabilities) using original MF5 data.
        """
        if incident_energy_idx < len(self.original_mf5_outgoing_energies):
            return (
                self.original_mf5_outgoing_energies[incident_energy_idx],
                self.original_mf5_probabilities[incident_energy_idx]
            )
        
        # Fallback to empty
        return ([], [])
    
    def get_continuous_distribution_for_sample(self, dist_idx: int, sample_index: int) -> tuple:
        """
        Get continuous distribution with bin-wise perturbations applied.
        
        This applies the bin-wise multiplicative factors to the original continuous MF5 data,
        creating discontinuities at bin boundaries as expected from covariance structure.
        
        Args:
            dist_idx: Index in self.distributions list (not incident_energy_index!)
            sample_index: 0 for nominal, ≥1 for perturbed sample
            
        Returns:
            (energies, probabilities): Arrays with boundary duplication and perturbations applied
        """
        if dist_idx >= len(self.distributions):
            raise ValueError(f"Distribution index {dist_idx} out of range")
        
        dist = self.distributions[dist_idx]
        incident_energy_idx = dist.incident_energy_index
        
        # Get original continuous data
        if incident_energy_idx >= len(self.original_mf5_outgoing_energies):
            return (np.array([]), np.array([]))
        
        original_energies = np.array(self.original_mf5_outgoing_energies[incident_energy_idx])
        original_probs = np.array(self.original_mf5_probabilities[incident_energy_idx])
        
        # Apply bin-wise factors with boundary duplication
        return dist.apply_factors_to_continuous_data(original_energies, original_probs, sample_index)

    def reconstruct(self, sample_index: int) -> Dict[int, List[float]]:
        """
        Returns a dict: {incident_energy_index: probabilities_for_this_sample}
        """
        return {d.incident_energy_index: d.get_probabilities_for_sample(sample_index) for d in self.distributions}
    
    def reconstruct_factors(self, sample_index: int) -> Dict[int, List[float]]:
        """
        Returns a dict: {incident_energy_index: factors_for_this_sample} for backward compatibility
        """
        return {d.incident_energy_index: d.get_factors_for_sample(sample_index) for d in self.distributions}

    def write_to_hdf5(self, hdf5_group):
        for dist in self.distributions:
            dist.write_to_hdf5(hdf5_group)
        
        # Store original MF5 data
        if self.original_mf5_incident_energies:
            hdf5_group.create_dataset('original_mf5_incident_energies', data=self.original_mf5_incident_energies)
        
        if self.original_mf5_outgoing_energies:
            # Store as ragged array using special group
            outgoing_grp = hdf5_group.create_group('original_mf5_outgoing_energies')
            for idx, energies in enumerate(self.original_mf5_outgoing_energies):
                outgoing_grp.create_dataset(f'idx_{idx}', data=energies)
        
        if self.original_mf5_probabilities:
            # Store as ragged array using special group
            probs_grp = hdf5_group.create_group('original_mf5_probabilities')
            for idx, probs in enumerate(self.original_mf5_probabilities):
                probs_grp.create_dataset(f'idx_{idx}', data=probs)

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        distributions = []
        for key in hdf5_group:
            if key.startswith('E') and key[1:].isdigit():
                dist = EnergyBinCoefficient.read_from_hdf5(hdf5_group[key])
                distributions.append(dist)
        
        # Read original MF5 data
        original_incident_energies = []
        original_outgoing_energies = []
        original_probabilities = []
        
        if 'original_mf5_incident_energies' in hdf5_group:
            original_incident_energies = hdf5_group['original_mf5_incident_energies'][()].tolist()
        
        if 'original_mf5_outgoing_energies' in hdf5_group:
            outgoing_grp = hdf5_group['original_mf5_outgoing_energies']
            for idx in range(len(original_incident_energies)):
                if f'idx_{idx}' in outgoing_grp:
                    original_outgoing_energies.append(outgoing_grp[f'idx_{idx}'][()].tolist())
        
        if 'original_mf5_probabilities' in hdf5_group:
            probs_grp = hdf5_group['original_mf5_probabilities']
            for idx in range(len(original_incident_energies)):
                if f'idx_{idx}' in probs_grp:
                    original_probabilities.append(probs_grp[f'idx_{idx}'][()].tolist())
        
        return cls(
            distributions=distributions,
            original_mf5_incident_energies=original_incident_energies,
            original_mf5_outgoing_energies=original_outgoing_energies,
            original_mf5_probabilities=original_probabilities
        )
