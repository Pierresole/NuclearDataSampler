from dataclasses import dataclass, field
from typing import List
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
        for RP_key in RP_group:
            RP_value_group = RP_group[RP_key]
            rp = URREnergyDependentRP.read_from_hdf5(RP_value_group)
            RP_list.append(rp)
        instance.RP = RP_list

        return instance

@dataclass
class URREnergyDependentLValue:
    AWRI: float
    L: int
    Jlist: List[URREnergyDependentJValue] = field(default_factory=list)

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
        for J_key in Jlist_group:
            J_value_group = Jlist_group[J_key]
            j_value = URREnergyDependentJValue.read_from_hdf5(J_value_group)
            Jlist.append(j_value)
        instance.Jlist = Jlist

        return instance

@dataclass
class URREnergyDependent:
    SPI: float  # Target spin "I"
    AP: float   # Scattering radius
    LSSF: int   # Flag if Self-Shielding Factor
    Llist: List[URREnergyDependentLValue] = field(default_factory=list)

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
        for L_key in Llist_group:
            L_value_group = Llist_group[L_key]
            l_value = URREnergyDependentLValue.read_from_hdf5(L_value_group)
            Llist.append(l_value)
        instance.Llist = Llist

        return instance

    def extract_parameters(self, mf2_resonance_range):
        """
        Extracts the mean parameters from MF2 and constructs the URREnergyDependent object.
        """
        l_values = mf2_resonance_range.parameters.l_values.to_list()

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
    
    def update_resonance_parameters(self, sample_index=1):
        """
        Returns a new UnresolvedEnergyDependent object with the sampled parameters.

        Parameters:
        - sample_index: Index of the sample to use (default is 1, since index 0 is the original value).
        """
        new_l_values = []

        for l_value in self.Llist:
            L = l_value.L
            awri = l_value.AWRI
            new_j_values = []

            for j_value in l_value.Jlist:
                J = j_value.AJ
                amun = j_value.AMUN
                amug = j_value.AMUG
                amuf = j_value.AMUF
                amux = j_value.AMUX
                interpolation = j_value.INT

                energies = []
                D = []
                GN = []
                GG = []
                GF = []
                GX = []

                for rp in j_value.RP:
                    energies.append(rp.ES)
                    # Fetch the sampled values using sample_index
                    D.append(rp.D[sample_index])
                    GN.append(rp.GN[sample_index])
                    # Handle GG
                    if len(rp.GG) > sample_index:
                        GG.append(rp.GG[sample_index])
                    else:
                        GG.append(rp.GG[0])  # Use original value if not sampled
                    # Handle GF
                    if rp.GF:
                        if len(rp.GF) > sample_index:
                            GF.append(rp.GF[sample_index])
                        else:
                            GF.append(rp.GF[0])
                    else:
                        GF.append(None)
                    # Handle GX
                    if rp.GX:
                        if len(rp.GX) > sample_index:
                            GX.append(rp.GX[sample_index])
                        else:
                            GX.append(rp.GX[0])
                    else:
                        GX.append(None)

                # Create new UnresolvedEnergyDependentJValue
                new_j_value = UnresolvedEnergyDependentJValue(
                    spin=J,
                    amun=amun,
                    amug=amug,
                    amuf=amuf,
                    amux=amux,
                    interpolation=interpolation,
                    energies=energies,
                    d=D,
                    gn=GN,
                    gg=GG,
                    gf=GF if any(gf is not None for gf in GF) else None,
                    gx=GX if any(gx is not None for gx in GX) else None
                )

                new_j_values.append(new_j_value)

            # Create new UnresolvedEnergyDependentLValue
            new_l_value = UnresolvedEnergyDependentLValue(
                awri=awri,
                l=L,
                jvalues=new_j_values
            )

            new_l_values.append(new_l_value)

        # Create new UnresolvedEnergyDependent
        parameters = UnresolvedEnergyDependent(
            spin=self.SPI,
            ap=self.AP,
            lssf=self.LSSF,
            lvalues=new_l_values
        )

        return parameters