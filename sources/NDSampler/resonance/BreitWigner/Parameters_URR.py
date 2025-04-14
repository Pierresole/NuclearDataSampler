from dataclasses import dataclass, field
from typing import List, Optional, Dict
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
    
    @classmethod
    def from_endftk(cls, ES: float, D: float, GN: float, GG: float, GF: Optional[float] = None, GX: Optional[float] = None):
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
        """
        instance = cls(ES=ES)
        instance.D = [D] if D is not None else []
        instance.GN = [GN] if GN is not None else []
        instance.GG = [GG] if GG is not None else []
        instance.GF = [GF] if GF is not None else []
        instance.GX = [GX] if GX is not None else []
        
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
    def from_endftk(cls, j_value):
        """
        Build a URREnergyDependentJValue from an ENDFtk UnresolvedEnergyDependentJValue object.
        
        Parameters:
        -----------
        j_value : UnresolvedEnergyDependentJValue
            ENDFtk object containing J value parameters
            
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
                GX=j_value.GX[idx] if hasattr(j_value, 'GX') and j_value.GX is not None else None
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
    def from_endftk(cls, l_value):
        """
        Build a URREnergyDependentLValue from an ENDFtk UnresolvedEnergyDependentLValue object.
        
        Parameters:
        -----------
        l_value : UnresolvedEnergyDependentLValue
            ENDFtk object containing L value parameters
            
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
        for j_value in j_values:
            urre_j_value = URREnergyDependentJValue.from_endftk(j_value)
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
    def from_endftk(cls, mf2_resonance_range):
        """
        Create a URREnergyDependent object from ENDFtk ResonanceRange.
        
        Parameters:
        -----------
        mf2_resonance_range : ENDFtk.MF2.MT151.ResonanceRange
            ENDFtk ResonanceRange object containing unresolved resonance parameters
            
        Returns:
        --------
        URREnergyDependent instance
        """
        instance = cls(
            SPI=mf2_resonance_range.parameters.SPI,
            AP=mf2_resonance_range.parameters.AP,
            LSSF=mf2_resonance_range.parameters.LSSF
        )
        
        # Extract L values
        l_values = mf2_resonance_range.parameters.l_values.to_list()
        for l_value in l_values:
            urre_l_value = URREnergyDependentLValue.from_endftk(l_value)
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