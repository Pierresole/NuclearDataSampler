from dataclasses import dataclass, field
from typing import List, Optional

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
class LValue:
    """Class to store resonance parameters for a given L value."""
    L: int
    AWRI: float
    APL: Optional[float] = None
    DAPL: Optional[float] = None  # Uncertainty in APL
    resonances: List[ResonanceParameter] = field(default_factory=list)

@dataclass
class ResonanceData:
    """Class to store resonance data for a resonance range."""
    SPI: float
    AP: float
    DAP: Optional[float] = None  # Uncertainty in AP
    LAD: Optional[int] = None
    LCOMP: Optional[int] = None
    L_values: List[LValue] = field(default_factory=list)
