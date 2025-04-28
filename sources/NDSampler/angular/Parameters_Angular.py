from dataclasses import dataclass, field
from typing import List, Optional, Dict
import ENDFtk
import numpy as np

@dataclass
class LegendreCoefficient:
    mt: int
    order: int
    energies: List[float] = field(default_factory=list)
    factor: List[List[float]] = field(default_factory=list)  # factor[sample_index][energy_bin]
    constraints: Optional[dict] = None

    def get_factors_for_sample(self, sample_index: int) -> List[float]:
        if len(self.factor) > sample_index:
            return self.factor[sample_index]
        else:
            return self.factor[0]  # fallback to nominal

    def write_to_hdf5(self, hdf5_group):
        grp = hdf5_group.create_group(f"L{self.order}")
        grp.attrs['mt'] = self.mt
        grp.attrs['order'] = self.order
        grp.create_dataset('energies', data=self.energies)
        grp.create_dataset('factor', data=np.array(self.factor))
        # Optionally store constraints as attributes if needed

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        # hdf5_group is the group for this L
        mt = hdf5_group.attrs['mt']
        order = hdf5_group.attrs['order']
        energies = hdf5_group['energies'][()].tolist()
        factor = hdf5_group['factor'][()].tolist()
        # Optionally read constraints
        return cls(mt=mt, order=order, energies=energies, factor=factor)

@dataclass
class LegendreCoefficients:
    coefficients: List[LegendreCoefficient] = field(default_factory=list)

    @classmethod
    def from_endftk(cls, mf4mt2, mf34mt2):
        # Parse Legendre coefficients from ENDFtk objects
        # This is a stub; actual implementation depends on ENDFtk structure
        coeffs = []
        # ...populate coeffs from mf4mt2...
        return cls(coefficients=coeffs)

    def reconstruct(self, sample_index: int) -> Dict[int, List[float]]:
        """
        Returns a dict: {order: factors_for_this_sample}
        """
        return {c.order: c.get_factors_for_sample(sample_index) for c in self.coefficients}

    def write_to_hdf5(self, hdf5_group):
        for coeff in self.coefficients:
            coeff.write_to_hdf5(hdf5_group)

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        coeffs = []
        for key in hdf5_group:
            coeff_grp = hdf5_group[key]
            coeffs.append(LegendreCoefficient.read_from_hdf5(coeff_grp))
        return cls(coefficients=coeffs)
