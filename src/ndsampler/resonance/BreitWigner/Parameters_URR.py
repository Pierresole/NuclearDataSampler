from dataclasses import dataclass, field
from typing import List, Optional, Dict
import ENDFtk
import numpy as np

from ENDFtk.MF2.MT151 import (
    UnresolvedEnergyDependent,
    UnresolvedEnergyDependentLValue,
    UnresolvedEnergyDependentJValue
)

@dataclass
class URREnergyDependentRP:
    ES: float  # Cannot be sampled, thus store only one float
    D: List[float] = field(default_factory=list)   # Can be sampled
    GN: List[float] = field(default_factory=list)  # Can be sampled
    GG: List[float] = field(default_factory=list)  # Can be sampled
    GF: List[float] = field(default_factory=list)  # Can be sampled
    GX: List[float] = field(default_factory=list)  # Can be sampled
    
    # Uncertainty information
    DD: float = None
    DGN: float = None
    DGG: float = None
    DGF: float = None
    DGX: float = None
    
    # Constraints information 
    constraints: Dict[str, Dict] = field(default_factory=dict)
    
    @classmethod
    def from_endftk(cls, ES: float, D: float, GN: float, GG: float, GF: Optional[float] = None, GX: Optional[float] = None, rel_variances=None):
        """
        Build a URREnergyDependentRP from ENDFtk resonance parameters.
        
        Parameters:
        -----------
        ES : float
            Energy value
        D : float
            Average level spacing
        GN : float
            Neutron width
        GG : float
            Gamma width
        GF : float, optional
            Fission width
        GX : float, optional
            Competitive width
        rel_variances : list, optional
            Relative variances for parameters [D, GN, GG, GF, GX]
        """
        instance = cls(ES=ES)
        instance.D = [D] if D is not None else []
        instance.GN = [GN] if GN is not None else []
        instance.GG = [GG] if GG is not None else []
        instance.GF = [GF] if GF is not None else []
        instance.GX = [GX] if GX is not None else []
        
        # Set uncertainties if rel_variances is provided
        if rel_variances is not None:
            # Get parameter values
            params = [
                ('D', D, 'DD'),
                ('GN', GN, 'DGN'),
                ('GG', GG, 'DGG'),
                ('GF', GF, 'DGF'),
                ('GX', GX, 'DGX')  # Special case for GX -> DGX
            ]
            
            # Set uncertainty for each parameter
            for i, (param_name, value, uncertainty_attr) in enumerate(params):
                if i < len(rel_variances) and value is not None and rel_variances[i] > 0:
                    # Calculate absolute uncertainty (std dev = sqrt(variance) * |nominal|)
                    abs_uncertainty = np.sqrt(rel_variances[i]) * abs(value)
                    setattr(instance, uncertainty_attr, abs_uncertainty)
        
        # Add constraints for each parameter
        instance.constraints = {
            'D': {'positive': True},    # Level spacing must be positive
            'GN': {'positive': True},   # Neutron width must be positive
            'GG': {'positive': True},   # Gamma width must be positive
            'GF': {'positive': True},   # Fission width must be positive
            'GX': {'positive': True}    # Competitive width must be positive
        }
        
        return instance

    def write_to_hdf5(self, hdf5_group):
        """
        Writes the URREnergyDependentRP data to the given HDF5 group.
        """
        hdf5_group.attrs['ES'] = self.ES

        # Write datasets for D, GN, GG
        hdf5_group.create_dataset('D', data=np.array(self.D))
        hdf5_group.create_dataset('GN', data=np.array(self.GN))
        hdf5_group.create_dataset('GG', data=np.array(self.GG))

        # Write GF if not empty
        if self.GF:
            hdf5_group.create_dataset('GF', data=np.array(self.GF))
        else:
            hdf5_group.attrs['GF_empty'] = True

        # Write GX if not empty
        if self.GX:
            hdf5_group.create_dataset('GX', data=np.array(self.GX))
        else:
            hdf5_group.attrs['GX_empty'] = True
            
        # Write uncertainty information if available
        if self.DD is not None:
            hdf5_group.attrs['DD'] = self.DD
        if self.DGN is not None:
            hdf5_group.attrs['DGN'] = self.DGN
        if self.DGG is not None:
            hdf5_group.attrs['DGG'] = self.DGG
        if self.DGF is not None:
            hdf5_group.attrs['DGF'] = self.DGF
        if self.DGX is not None:
            hdf5_group.attrs['DGX'] = self.DGX
            
        # Write constraints information
        if self.constraints:
            constraints_group = hdf5_group.create_group('constraints')
            for param_name, constraint_dict in self.constraints.items():
                param_group = constraints_group.create_group(param_name)
                for constraint_key, constraint_value in constraint_dict.items():
                    param_group.attrs[constraint_key] = constraint_value

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads the URREnergyDependentRP data from the given HDF5 group and returns an instance.
        """
        ES = hdf5_group.attrs['ES']

        D = hdf5_group['D'][()].tolist()
        GN = hdf5_group['GN'][()].tolist()
        GG = hdf5_group['GG'][()].tolist()

        # Check if GF exists
        if 'GF' in hdf5_group:
            GF = hdf5_group['GF'][()].tolist()
        else:
            GF = []

        # Check if GX exists
        if 'GX' in hdf5_group:
            GX = hdf5_group['GX'][()].tolist()
        else:
            GX = []

        instance = cls(ES=ES, D=D, GN=GN, GG=GG, GF=GF, GX=GX)
        
        # Read uncertainty information if available
        if 'DD' in hdf5_group.attrs:
            instance.DD = hdf5_group.attrs['DD']
        if 'DGN' in hdf5_group.attrs:
            instance.DGN = hdf5_group.attrs['DGN']
        if 'DGG' in hdf5_group.attrs:
            instance.DGG = hdf5_group.attrs['DGG']
        if 'DGF' in hdf5_group.attrs:
            instance.DGF = hdf5_group.attrs['DGF']
        if 'DGX' in hdf5_group.attrs:
            instance.DGX = hdf5_group.attrs['DGX']
            
        # Read constraints if available
        if 'constraints' in hdf5_group:
            constraints_group = hdf5_group['constraints']
            for param_name in constraints_group:
                param_group = constraints_group[param_name]
                constraint_dict = {}
                for key in param_group.attrs:
                    constraint_dict[key] = param_group.attrs[key]
                instance.constraints[param_name] = constraint_dict

        return instance

@dataclass
class URREnergyDependentJValue:
    AJ: float
    AMUN: int
    AMUG: int
    AMUF: int
    AMUX: int
    INT: int
    RP: List[URREnergyDependentRP] = field(default_factory=list)
    
    @classmethod
    def from_endftk(cls, j_value, rel_variances=None):
        """
        Build a URREnergyDependentJValue from an ENDFtk UnresolvedEnergyDependentJValue object.
        
        Parameters:
        -----------
        j_value : UnresolvedEnergyDependentJValue
            ENDFtk object containing J value parameters
        rel_variances : list, optional
            List of relative variances for parameters
            
        Returns:
        --------
        URREnergyDependentJValue instance
        """
        instance = cls(
            AJ=j_value.AJ,
            AMUN=j_value.AMUN,
            AMUG=j_value.AMUG,
            AMUF=j_value.AMUF,
            AMUX=j_value.AMUX,
            INT=j_value.INT
        )
        
        # Extract energy-dependent parameters
        energies = j_value.ES
        for idx, energy in enumerate(energies):
            rp = URREnergyDependentRP.from_endftk(
                ES=energy,
                D=j_value.D[idx],
                GN=j_value.GN[idx],
                GG=j_value.GG[idx],
                GF=j_value.GF[idx] if hasattr(j_value, 'GF') and j_value.GF is not None else None,
                GX=j_value.GX[idx] if hasattr(j_value, 'GX') and j_value.GX is not None else None,
                rel_variances=rel_variances
            )
            instance.RP.append(rp)
            
        return instance

    def write_to_hdf5(self, hdf5_group):
        """
        Writes the URREnergyDependentJValue data to the given HDF5 group.
        """
        hdf5_group.attrs['AJ'] = self.AJ
        hdf5_group.attrs['AMUN'] = self.AMUN
        hdf5_group.attrs['AMUG'] = self.AMUG
        hdf5_group.attrs['AMUF'] = self.AMUF
        hdf5_group.attrs['AMUX'] = self.AMUX
        hdf5_group.attrs['INT'] = self.INT

        # Create group for RP
        RP_group = hdf5_group.create_group('RP')

        for idx_RP, rp in enumerate(self.RP):
            RP_value_group = RP_group.create_group(f'RP_{idx_RP}')
            rp.write_to_hdf5(RP_value_group)

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads the URREnergyDependentJValue data from the given HDF5 group and returns an instance.
        """
        AJ = hdf5_group.attrs['AJ']
        AMUN = hdf5_group.attrs['AMUN']
        AMUG = hdf5_group.attrs['AMUG']
        AMUF = hdf5_group.attrs['AMUF']
        AMUX = hdf5_group.attrs['AMUX']
        INT = hdf5_group.attrs['INT']

        instance = cls(AJ=AJ, AMUN=AMUN, AMUG=AMUG, AMUF=AMUF, AMUX=AMUX, INT=INT)

        # Read RP
        RP_group = hdf5_group['RP']
        RP_list = []
        for RP_key in sorted(RP_group.keys()):
            RP_value_group = RP_group[RP_key]
            rp = URREnergyDependentRP.read_from_hdf5(RP_value_group)
            RP_list.append(rp)
        # Sort resonances by increasing resonance energy (ER)
        RP_list.sort(key=lambda r: r.ES)
        instance.RP = RP_list

        return instance
        
    def reconstruct(self, sample_index=0):
        """
        Reconstructs an ENDFtk UnresolvedEnergyDependentJValue from this object.
        
        Parameters:
        -----------
        sample_index : int
            Index of the sample to use (0 for original values)
            
        Returns:
        --------
        UnresolvedEnergyDependentJValue : ENDFtk object with sampled values
        """
        energies = []
        D = []
        GN = []
        GG = []
        GF = []
        GX = []
        
        # Sort resonances by increasing resonance energy (ER)
        self.RP.sort(key=lambda r: r.ES)
        
        for rp in self.RP:
            energies.append(rp.ES)
            # For each parameter, use the sampled value if available, otherwise use the original
            D.append(rp.D[sample_index] if len(rp.D) > sample_index else rp.D[0])
            GN.append(rp.GN[sample_index] if len(rp.GN) > sample_index else rp.GN[0])
            GG.append(rp.GG[sample_index] if len(rp.GG) > sample_index else rp.GG[0])
            
            # Handle optional parameters
            if rp.GF:
                GF.append(rp.GF[sample_index] if len(rp.GF) > sample_index else rp.GF[0])
            else:
                GF.append(None)
                
            if rp.GX:
                GX.append(rp.GX[sample_index] if len(rp.GX) > sample_index else rp.GX[0])
            else:
                GX.append(None)
        
        # Create ENDFtk UnresolvedEnergyDependentJValue
        return UnresolvedEnergyDependentJValue(
            spin=self.AJ,
            amun=self.AMUN,
            amug=self.AMUG,
            amuf=self.AMUF,
            amux=self.AMUX,
            interpolation=self.INT,
            energies=energies,
            d=D,
            gn=GN,
            gg=GG,
            gf=GF if any(gf is not None for gf in GF) else None,
            gx=GX if any(gx is not None for gx in GX) else None
        )

@dataclass
class URREnergyDependentLValue:
    AWRI: float
    L: int
    Jlist: List[URREnergyDependentJValue] = field(default_factory=list)
    
    @classmethod
    def from_endftk(cls, l_value, rel_variances=None):
        """
        Build a URREnergyDependentLValue from an ENDFtk UnresolvedEnergyDependentLValue object.
        
        Parameters:
        -----------
        l_value : UnresolvedEnergyDependentLValue
            ENDFtk object containing L value parameters
        rel_variances : list, optional
            List of relative variances for parameters for each J group
            
        Returns:
        --------
        URREnergyDependentLValue instance
        """
        instance = cls(
            AWRI=l_value.AWRI,
            L=l_value.L
        )
        
        # Extract J values
        j_values = l_value.j_values.to_list()
        for j_idx, j_value in enumerate(j_values):
            j_rel_variances = None
            if rel_variances is not None and j_idx < len(rel_variances):
                j_rel_variances = rel_variances[j_idx]
                
            urre_j_value = URREnergyDependentJValue.from_endftk(j_value, j_rel_variances)
            instance.Jlist.append(urre_j_value)
            
        return instance

    def write_to_hdf5(self, hdf5_group):
        """
        Writes the URREnergyDependentLValue data to the given HDF5 group.
        """
        hdf5_group.attrs['AWRI'] = self.AWRI
        hdf5_group.attrs['L'] = self.L

        # Create group for Jlist
        Jlist_group = hdf5_group.create_group('J_values')

        for idx_J, j_value in enumerate(self.Jlist):
            J_value_group = Jlist_group.create_group(f'J_{idx_J}')
            j_value.write_to_hdf5(J_value_group)

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads the URREnergyDependentLValue data from the given HDF5 group and returns an instance.
        """
        AWRI = hdf5_group.attrs['AWRI']
        L = hdf5_group.attrs['L']

        instance = cls(AWRI=AWRI, L=L)

        # Read Jlist
        Jlist_group = hdf5_group['J_values']
        Jlist = []
        for J_key in sorted(Jlist_group.keys()):
            J_value_group = Jlist_group[J_key]
            j_value = URREnergyDependentJValue.read_from_hdf5(J_value_group)
            Jlist.append(j_value)
        instance.Jlist = Jlist

        return instance
        
    def reconstruct(self, sample_index=0):
        """
        Reconstructs an ENDFtk UnresolvedEnergyDependentLValue from this object.
        
        Parameters:
        -----------
        sample_index : int
            Index of the sample to use (0 for original values)
            
        Returns:
        --------
        UnresolvedEnergyDependentLValue : ENDFtk object with sampled values
        """
        # Reconstruct each J value
        j_values = [j_value.reconstruct(sample_index) for j_value in self.Jlist]
        
        # Create ENDFtk UnresolvedEnergyDependentLValue
        return UnresolvedEnergyDependentLValue(
            awri=self.AWRI,
            l=self.L,
            jvalues=j_values
        )

@dataclass
class URREnergyDependent:
    SPI: float  # Target spin "I"
    AP: float   # Scattering radius
    LSSF: int   # Flag if Self-Shielding Factor
    Llist: List[URREnergyDependentLValue] = field(default_factory=list)
    
    @classmethod
    def from_endftk(cls, mf2_range: ENDFtk.MF2.MT151.ResonanceRange, mf32_range: ENDFtk.MF32.MT151.ResonanceRange = None):
        """
        Create a URREnergyDependent object from ENDFtk ResonanceRange.
        
        Parameters:
        -----------
        mf2_range : ENDFtk.MF2.MT151.ResonanceRange
            ENDFtk ResonanceRange object containing unresolved resonance parameters
            
        mf32_range : ENDFtk.MF32.MT151.ResonanceRange, optional
            ENDFtk ResonanceRange object containing covariance matrix information
            
        Returns:
        --------
        URREnergyDependent instance
        """
        instance = cls(
            SPI=mf2_range.parameters.SPI,
            AP=mf2_range.parameters.AP,
            LSSF=mf2_range.parameters.LSSF
        )
        
        # Extract relative covariance matrix if provided
        rel_covariance_matrix = None
        if mf32_range is not None:
            # Process covariance matrix
            MPAR = mf32_range.parameters.covariance_matrix.MPAR  # Number of parameters per (L,J)
            NPAR_spin = mf32_range.parameters.covariance_matrix.NPAR  # Total number of parameters
            relative_cov_matrix_upper = mf32_range.parameters.covariance_matrix.covariance_matrix
            
            # Reconstruct full covariance matrix from upper triangular
            rel_covariance_matrix = np.zeros((NPAR_spin, NPAR_spin))
            triu_indices = np.triu_indices(NPAR_spin)
            rel_covariance_matrix[triu_indices] = relative_cov_matrix_upper
            rel_covariance_matrix = rel_covariance_matrix + rel_covariance_matrix.T - np.diag(np.diag(rel_covariance_matrix))
            
            # Get the diagonal elements (variances)
            rel_variances = np.diag(rel_covariance_matrix)
        else:
            MPAR = 0
            rel_variances = None
        
        # Organize relative variances by L-value and J-value before passing to L-values
        l_rel_variances = []
        if rel_variances is not None:
            start_idx = 0
            for l_idx, l_value in enumerate(mf2_range.parameters.l_values.to_list()):
                j_rel_variances = []
                j_values = l_value.j_values.to_list()
                
                for j_idx, j_value in enumerate(j_values):
                    # Extract the variances for this (L,J) group
                    param_rel_variances = rel_variances[start_idx:start_idx + MPAR]
                    j_rel_variances.append(param_rel_variances)
                    start_idx += MPAR
                
                l_rel_variances.append(j_rel_variances)
        
        # Extract L values with their associated relative variances
        for l_idx, l_value in enumerate(mf2_range.parameters.l_values.to_list()):
            l_variance_data = l_rel_variances[l_idx] if l_rel_variances else None
            urre_l_value = URREnergyDependentLValue.from_endftk(l_value, l_variance_data)
            instance.Llist.append(urre_l_value)
            
        return instance

    def write_to_hdf5(self, hdf5_group):
        """
        Writes the URREnergyDependent data to the given HDF5 group.
        """
        # Write attributes
        hdf5_group.attrs['SPI'] = self.SPI
        hdf5_group.attrs['AP'] = self.AP
        hdf5_group.attrs['LSSF'] = self.LSSF

        # Create group for Llist
        Llist_group = hdf5_group.create_group('L_values')

        # For each L_value in Llist
        for idx_L, l_value in enumerate(self.Llist):
            L_value_group = Llist_group.create_group(f'L_{idx_L}')
            l_value.write_to_hdf5(L_value_group)

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads the URREnergyDependent data from the given HDF5 group and returns an instance.
        """
        # Read attributes
        SPI = hdf5_group.attrs['SPI']
        AP = hdf5_group.attrs['AP']
        LSSF = hdf5_group.attrs['LSSF']

        # Create an instance
        instance = cls(SPI=SPI, AP=AP, LSSF=LSSF)

        # Read Llist
        Llist_group = hdf5_group['L_values']
        Llist = []
        for L_key in sorted(Llist_group.keys()):
            L_value_group = Llist_group[L_key]
            l_value = URREnergyDependentLValue.read_from_hdf5(L_value_group)
            Llist.append(l_value)
        instance.Llist = Llist

        return instance
        
    def extract_parameters(self, mf2_resonance_range):
        """
        Extracts the mean parameters from MF2 and constructs the URREnergyDependent object.
        This is a legacy method - use from_endftk class method instead for new code.
        """
        # Use the class method for consistency
        new_obj = self.from_endftk(mf2_resonance_range)
        
        # Copy attributes to self
        self.SPI = new_obj.SPI
        self.AP = new_obj.AP
        self.LSSF = new_obj.LSSF
        self.Llist = new_obj.Llist
        self.L_values = []  # For backwards compatibility
        
        # Populate L_values for backwards compatibility
        for l_value in self.Llist:
            for j_value in l_value.Jlist:
                self.L_values.append({'L': l_value.L, 'J': j_value.AJ})
    
    def reconstruct(self, sample_index=0):
        """
        Reconstructs an ENDFtk UnresolvedEnergyDependent object from this instance.
        
        Parameters:
        -----------
        sample_index : int
            Index of the sample to use (0 for original values)
            
        Returns:
        --------
        UnresolvedEnergyDependent : ENDFtk object with sampled values
        """
        # Reconstruct each L value
        l_values = [l_value.reconstruct(sample_index) for l_value in self.Llist]
        
        # Create ENDFtk UnresolvedEnergyDependent
        return UnresolvedEnergyDependent(
            spin=self.SPI,
            ap=self.AP,
            lssf=self.LSSF,
            lvalues=l_values
        )
    
    def update_resonance_parameters(self, sample_index=1):
        """
        Returns a new UnresolvedEnergyDependent object with the sampled parameters.
        Legacy method - use reconstruct() instead for new code.

        Parameters:
        - sample_index: Index of the sample to use (default is 1, since index 0 is the original value).
        """
        # Use the new reconstruct method for consistency
        return self.reconstruct(sample_index)
    
    def get_relative_uncertainty(self) -> List[float]:
        """
        Returns a list of relative uncertainties (as fractions, not percentages) for parameters 
        that have uncertainty information and are included in the covariance matrix.
        
        Returns:
        --------
        List[float] : List of relative uncertainties (uncertainty/nominal) for all resonances
        """
        relative_uncertainties = []
        
        # Loop through all L-groups
        for l_group in self.Llist:
            # Loop through all resonances in this L-group
            for j_group in l_group.Jlist:
                for resonance in j_group.RP:
                
                    # Add parameters with uncertainty information (non-None)
                    params_with_uncertainty = [
                        (resonance.D[0], resonance.DD),
                        (resonance.GN[0] if resonance.GN is not None else None, resonance.DGN),
                        (resonance.GG[0] if resonance.GG is not None else None, resonance.DGG),
                        (resonance.GF[0] if resonance.GF is not None else None, resonance.DGF),
                        (resonance.GX[0] if resonance.GX is not None else None, resonance.DGX)
                    ]
                    
                    for nominal_value, absolute_uncertainty in params_with_uncertainty:
                        if nominal_value is not None and absolute_uncertainty is not None and absolute_uncertainty > 0:
                            # Calculate relative uncertainty: uncertainty / nominal_value
                            relative_unc = absolute_uncertainty / abs(nominal_value) if abs(nominal_value) > 1e-20 else 0.0
                            relative_uncertainties.append(relative_unc)
        
        return relative_uncertainties
    
    def get_nominal_values(self) -> List[float]:
        """
        Returns a list of nominal parameter values that have uncertainty information and are 
        therefore included in the covariance matrix.
        
        Returns:
        --------
        List[float] : List of nominal parameter values for all resonances that have uncertainties
        """
        nominal_values = []
        
        # Loop through all L-groups
        for l_group in self.Llist:
            # Loop through all resonances in this L-group
            for j_group in l_group.Jlist:
                for resonance in j_group.RP:
                
                    # Add parameters with uncertainty information (non-None)
                    params_with_uncertainty = [
                        (resonance.D[0], resonance.DD),
                        (resonance.GN[0] if resonance.GN is not None else None, resonance.DGN),
                        (resonance.GG[0] if resonance.GG is not None else None, resonance.DGG),
                        (resonance.GF[0] if resonance.GF is not None else None, resonance.DGF),
                        (resonance.GX[0] if resonance.GX is not None else None, resonance.DGX)
                    ]
                    
                    for nominal_value, absolute_uncertainty in params_with_uncertainty:
                        if nominal_value is not None and absolute_uncertainty is not None and absolute_uncertainty > 0:
                            nominal_values.append(nominal_value)
        
        return nominal_values