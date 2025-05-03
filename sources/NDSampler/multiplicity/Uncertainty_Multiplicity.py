import numpy as np
import time
from collections import defaultdict
from .MultiplicityCovariance import MultiplicityCovariance
from .Parameters_Multiplicity import Multiplicities
from ENDFtk import tree
from ENDFtk.MF4 import ResonanceRange, Isotope, Section
from scipy.linalg import block_diag  # Import block_diag function

class Uncertainty_Multiplicity(MultiplicityCovariance):
    """
    Class to handle the uncertainty in multiplicity angular distributions.
    """
    def __init__(self, mf4mt2: ResonanceRange, mf34mt2: Isotope):
        super().__init__(mf4mt2)
        self.mf34mt2 = mf34mt2
        self.parameters = Multiplicities.from_endftk(mf4mt2, mf34mt2)
        self.legendre_data = None  # Placeholder for Legendre coefficients
        self.covariance_matrix = None  # Placeholder for covariance matrix
        self.mean_vector = None  # Placeholder for mean vector
        self.std_dev_vector = None  # Placeholder for standard deviation vector