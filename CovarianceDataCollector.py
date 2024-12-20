import numpy as np
import h5py
from CovarianceResonance import ResonanceCovariance, ReichMooreCovariance, AveragedBreitWignerCovariance
from CovarianceNubar import NubarCovariance
from CovarianceCrossSection import CrossSectionCovariance
from CovarianceAngular import AngularCovariance

class CovarianceDataCollector:
    def __init__(self, hdf5_filename):
        self.hdf5_filename = hdf5_filename
        self.hdf5_file = h5py.File(hdf5_filename, 'w')

    def add_covariance(self, covariance_obj):
        # Determine the group name based on the covariance type
        if isinstance(covariance_obj, ReichMooreCovariance):
            group_name = 'ReichMooreCovariance'
        elif isinstance(covariance_obj, AveragedBreitWignerCovariance):
            group_name = 'AveragedBreitWignerCovariance'
        elif isinstance(covariance_obj, ResonanceCovariance):
            group_name = 'ResonanceCovariance'
        elif isinstance(covariance_obj, NubarCovariance):
            group_name = 'NubarCovariance'
        elif isinstance(covariance_obj, CrossSectionCovariance):
            group_name = 'CrossSectionCovariance'
        elif isinstance(covariance_obj, AngularCovariance):
            group_name = 'AngularCovariance'
        else:
            group_name = 'OtherCovariance'
        # Get or create the group
        if group_name in self.hdf5_file:
            group = self.hdf5_file[group_name]
        else:
            group = self.hdf5_file.create_group(group_name)

        # Create a unique subgroup for each covariance object
        subgroup_name = f'NER_{covariance_obj.NER}'
        subgroup = group.create_group(subgroup_name)

        # Write the covariance data
        covariance_obj.write_to_hdf5(subgroup)

    def close(self):
        self.hdf5_file.close()
