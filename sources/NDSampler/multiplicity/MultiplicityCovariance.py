import ENDFtk
from ENDFtk.tree import Tape
import bisect
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from ..CovarianceBase import CovarianceBase

class MultiplicityCovariance(CovarianceBase, ABC):
    print("MultiplicityCovariance")
    
        
    @staticmethod
    def read_hdf5_group(group, covariance_objects: List["CovarianceBase"]):
        for subgroup_name in group:
            subgroup = group[subgroup_name]
            
            if subgroup_name == 'AngularDistribution':
                from .Uncertainty_Multiplicity import Uncertainty_Multiplicity
                covariance_obj = Uncertainty_Multiplicity.read_from_hdf5(subgroup)
                covariance_objects.append(covariance_obj)