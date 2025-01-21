from dataclasses import dataclass, field
from typing import List, Optional
import ENDFtk
from ENDFtk.tree import Tape

from ENDFtk.MF2.MT151 import (
    ParticlePairs,
    ResonanceChannels,
    ResonanceParameters,
    SpinGroup
)

@dataclass
class ResonanceParameter:
    """Class to store parameters of a single resonance."""
    ER: float
    AJ: float
    GN: Optional[float] = None
    GG: Optional[float] = None
    GFA: Optional[float] = None
    GFB: Optional[float] = None
    index: Optional[int] = None  # Position in the covariance matrix

@dataclass
class ParticlePair:
    """Class to store resonance parameters for a given L value."""
    L: int
    AWRI: float
    APL: Optional[float] = None
    DAPL: Optional[float] = None  # Uncertainty in APL
    resonances: List[ResonanceParameter] = field(default_factory=list)

@dataclass
class RMatrixLimited:
    """Class to store resonance data for a resonance range."""
    IFG: int
    KRL: int
    KRM: int
    LAD: Optional[int] = None
    LCOMP: Optional[int] = None
    ListParticlePairs: List[ParticlePairs] = field(default_factory=list)
    ListSpinGroups: List[SpinGroup] = field(default_factory=list)
    
    def extract_parameters(self, range: ENDFtk.MF2.MT151.ResonanceRange):
        """
        Extracts the mean parameters from MF2 and constructs the RMatrixLimited object.
        """
        self.IFG = range.parameters.IFG
        self.KRL = range.parameters.KRL
        self.KRM = range.parameters.KRM
        
        ParticlePairs( range.parameters.particle_pairs )
        
        groups = groups
        
        
        spingroups = []
        for spingroup in range.parameters.spin_groups.to_list():
            spingroups.append( SpinGroup( ResonanceChannels( spingroup.channels ),
                                          ResonanceParameters( spingroup.parameters.ER, spingroup.parameters.GAM ) ))
            
        l_values = range.parameters.l_values.to_list()

        self.L_values = []  # List to store (L, J) groups

        for l_value in l_values:
            L = l_value.L
            awri = l_value.AWRI
            j_values = l_value.j_values.to_list()
            urre_l_value = URREnergyDependentLValue(
                AWRI=awri,
                L=L
            )

            for j_value in j_values:
                J = j_value.AJ
                urre_j_value = URREnergyDependentJValue(
                    AJ=J,
                    AMUN=j_value.AMUN,
                    AMUG=j_value.AMUG,
                    AMUF=j_value.AMUF,
                    AMUX=j_value.AMUX,
                    INT=j_value.INT
                )

                energies = j_value.ES
                # For each energy, create URREnergyDependentRP
                for idx, energy in enumerate(energies):
                    rp = URREnergyDependentRP(
                        ES=energy,
                        D=[j_value.D[idx]],    # Initialize with original value
                        GN=[j_value.GN[idx]],
                        GG=[j_value.GG[idx]],
                        GF=[j_value.GF[idx]] if hasattr(j_value, 'GF') else [],
                        GX=[j_value.GX[idx]] if hasattr(j_value, 'GX') else []
                    )
                    urre_j_value.RP.append(rp)

                urre_l_value.Jlist.append(urre_j_value)
                self.L_values.append({'L': L, 'J': J})

            self.Llist.append(urre_l_value)
