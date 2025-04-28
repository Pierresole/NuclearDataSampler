import ENDFtk
from ENDFtk.tree import Tape
import bisect
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from ..CovarianceBase import CovarianceBase

class MultiplicityCovariance(CovarianceBase, ABC):
    print("MultiplicityCovariance")