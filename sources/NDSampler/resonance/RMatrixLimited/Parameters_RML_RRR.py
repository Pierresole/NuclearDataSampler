from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import ENDFtk
import h5py  # ensure h5py is available if not already

@dataclass
class Resonance:
    """Class to store nominal and sampled resonance data."""
    ER: List[float] = field(default_factory=list)   # First entry can be nominal, subsequent entries are samples
    GAM: List[List[float]] = field(default_factory=list)  # Same idea for widths
    DER: float = None
    DGAM: List[Optional[float]] = field(default_factory=lambda: [None])

    def extract_parameters(self, parameters: ENDFtk.MF2.MT151.ResonanceParameters):
        # Convert any ENDFtk views to mutable lists
        self.ER = list(parameters.ER)
        self.GAM = [list(g) for g in parameters.GAM]

    def reconstruct(self) -> ENDFtk.MF2.MT151.ResonanceParameters:
        # Combine all ER, GAM into a single ResonanceParameters object
        return ENDFtk.MF2.MT151.ResonanceParameters(self.ER, self.GAM)

    def write_to_hdf5(self, hdf5_group):
        """
        Writes this Resonance's data to the given HDF5 group.
        """
        hdf5_group.create_dataset('ER', data=self.ER)
        gam_group = hdf5_group.create_group('GAM')
        for idx, gam_list in enumerate(self.GAM):
            gam_group.create_dataset(f'GAM_{idx}', data=gam_list)

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads a Resonance from the given HDF5 group.
        """
        er_data = hdf5_group['ER'][()]
        gam_data = []
        for dataset in hdf5_group['GAM']:
            gam_data.append(list(hdf5_group['GAM'][dataset][()]))
        return cls(ER=list(er_data), GAM=gam_data)

    @classmethod
    def from_endftk(cls, energy: float, denergy: float, widths: List[float], dwidths: List[float]):
        """
        Build a Resonance from ENDFtk resonance parameters.
        """
        instance = cls()
        instance.ER = [energy]
        instance.DER = denergy
        instance.GAM = [[width] for width in widths]
        instance.DGAM = dwidths
        return instance

@dataclass
class Channel:
    """
    Stores single channel properties
    """
    ppi: int
    l: int  
    s: float
    b: float
    apt: float
    ape: float

    @classmethod
    def from_endftk(cls, endf_channels, channel_index):
        """Creates a Channel instance from an ENDFtk channel at given index"""
        return cls(
            ppi=endf_channels.PPI[channel_index],
            l=endf_channels.L[channel_index],
            s=endf_channels.SCH[channel_index],
            b=endf_channels.BND[channel_index],
            apt=endf_channels.APT[channel_index],
            ape=endf_channels.APE[channel_index]
        )

    def write_to_hdf5(self, hdf5_group):
        """Writes channel data to HDF5"""
        for attr in ['ppi', 'l', 's', 'b', 'apt', 'ape']:
            hdf5_group.attrs[attr] = getattr(self, attr)

    @classmethod 
    def read_from_hdf5(cls, hdf5_group):
        """Reads channel data from HDF5"""
        return cls(**{attr: hdf5_group.attrs[attr] 
                     for attr in ['ppi', 'l', 's', 'b', 'apt', 'ape']})

@dataclass 
class SpinGroup:
    """Class to store spin group data, including channels and resonances"""
    spin: float
    parity: float
    kbk: int  
    kps: int
    channels: List[Channel] = field(default_factory=list)
    resonance_parameters: List[Resonance] = field(default_factory=list)

    def _convert_to_reduced_width(self, width: float, channel_idx: int, energy: float) -> float:
        """Convert physical width to reduced width using penetrability"""
        import numpy as np
        if width == 0.0:
            return 0.0
            
        P, _ = self.channelPenetrationAndShift(channel_idx, energy)
        if P <= 0.0:
            return 0.0
            
        return np.sign(width) * np.sqrt(abs(width)/(2*P))

    @classmethod
    def from_endftk(cls, spingroup: ENDFtk.MF2.MT151.SpinGroup, spingroup32, force_reduced: bool = False):
        """Create SpinGroup from ENDFtk objects with optional reduced width conversion"""
        instance = cls(
            spin=spingroup.channels.AJ,
            parity=spingroup.channels.PJ,
            kbk=spingroup.channels.KBK,
            kps=spingroup.channels.KPS,
            channels=[Channel.from_endftk(spingroup.channels, i) 
                     for i in range(spingroup.NCH)]
        )

        if force_reduced:
            resonances = []
            for i in range(spingroup.parameters.NRS):
                energy = spingroup.parameters.ER[i]
                widths = [instance._convert_to_reduced_width(
                            spingroup.parameters.GAM[i][ch_idx], 
                            ch_idx,
                            energy)
                         for ch_idx in range(spingroup.NCH)]
                
                resonances.append(
                    Resonance.from_endftk(
                        energy,
                        spingroup32.parameters.DER[i],
                        widths,
                        spingroup32.parameters.DGAM[i][:spingroup.NCH]
                    )
                )
        else:
            resonances = [
                Resonance.from_endftk(
                    spingroup.parameters.ER[i],
                    spingroup32.parameters.DER[i],
                    spingroup.parameters.GAM[i][:spingroup.NCH],
                    spingroup32.parameters.DGAM[i][:spingroup.NCH]
                ) for i in range(spingroup.parameters.NRS)
            ]

        instance.resonance_parameters = resonances
        return instance

    def reconstruct(self, sample_index: int = 0) -> ENDFtk.MF2.MT151.SpinGroup:
        """Reconstruct ENDFtk SpinGroup from this object"""
        channels = ENDFtk.MF2.MT151.ResonanceChannels(
            spin=self.spin,
            parity=self.parity,
            kbk=self.kbk,
            kps=self.kps,
            ppi=[ch.ppi for ch in self.channels],
            l=[ch.l for ch in self.channels],
            s=[ch.s for ch in self.channels],
            b=[ch.b for ch in self.channels],
            apt=[ch.apt for ch in self.channels],
            ape=[ch.ape for ch in self.channels]
        )

        parameters = ENDFtk.MF2.MT151.ResonanceParameters(
            energies=[res.ER[sample_index] for res in self.resonance_parameters],
            parameters=[[gam[sample_index] for gam in res.GAM] 
                       for res in self.resonance_parameters]
        )

        return ENDFtk.MF2.MT151.SpinGroup(
            channels=channels,
            parameters=parameters
        )

    def channelPenetrationAndShift(self, channel_index: int, rho: float):
        """Compute penetration and shift using channel angular momentum"""
        l = self.channels[channel_index].l
        import math

        if l < 0:
            raise ValueError("l must be a non-negative integer.")

        P, S = 0.0, 0.0
        rho2 = rho ** 2

        if rho2 >= 0:
            # Positive energy channel
            if l == 0:
                P = rho
            else:
                P_prev, S_prev = rho, 0.0
                for i in range(1, l + 1):
                    denom = (i - S_prev) ** 2 + P_prev ** 2
                    P = (rho2 * P_prev) / denom
                    S = (rho2 * (i - S_prev)) / denom - i
                    P_prev, S_prev = P, S
        else:
            # Negative energy channel
            if l == 0:
                S = -math.sqrt(-rho2)
            else:
                S_prev = -math.sqrt(-rho2)
                for i in range(1, l + 1):
                    S = (rho2 / (i - S_prev)) - i
                    S_prev = S

        return P, S

    def write_to_hdf5(self, hdf5_group):
        """Write spin group data to HDF5"""
        for attr in ['spin', 'parity', 'kbk', 'kps']:
            hdf5_group.attrs[attr] = getattr(self, attr)
            
        ch_group = hdf5_group.create_group('Channels')
        for i, channel in enumerate(self.channels):
            channel.write_to_hdf5(ch_group.create_group(f'Channel_{i}'))

        rp_group = hdf5_group.create_group('Resonances')
        for i, res in enumerate(self.resonance_parameters):
            res.write_to_hdf5(rp_group.create_group(f'Resonance_{i}'))

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """Read spin group data from HDF5"""
        channels = []
        for ch_key in sorted(hdf5_group['Channels'].keys()):
            channels.append(Channel.read_from_hdf5(hdf5_group['Channels'][ch_key]))

        resonances = []
        for res_key in sorted(hdf5_group['Resonances'].keys()):
            resonances.append(Resonance.read_from_hdf5(hdf5_group['Resonances'][res_key]))

        return cls(
            spin=hdf5_group.attrs['spin'],
            parity=hdf5_group.attrs['parity'],
            kbk=hdf5_group.attrs['kbk'],
            kps=hdf5_group.attrs['kps'],
            channels=channels,
            resonance_parameters=resonances
        )

    def getListStandardDeviation(self, spin_group_idx: int):
        index_mapping = []
        ListStandardDeviation = []
        for r_idx, resonance in enumerate(self.resonance_parameters):
            if resonance.DER is not None:
                ListStandardDeviation.append(resonance.DER)
                index_mapping.append((spin_group_idx, r_idx, 0))
            for p_idx, dgam in enumerate(resonance.DGAM):
                if dgam is not None:
                    ListStandardDeviation.append(dgam)
                    index_mapping.append((spin_group_idx, r_idx, p_idx + 1))
        return index_mapping, ListStandardDeviation


@dataclass
class ParticlePair:
    """Class to store ParticlePair."""
    ma: List[float] = field(default_factory=list)
    mb: List[float] = field(default_factory=list)
    za: List[float] = field(default_factory=list)
    zb: List[float] = field(default_factory=list)
    ia: List[float] = field(default_factory=list)
    ib: List[float] = field(default_factory=list)
    pa: List[float] = field(default_factory=list)
    pb: List[float] = field(default_factory=list)
    q: List[float] = field(default_factory=list)
    pnt: List[int] = field(default_factory=list)
    shf: List[int] = field(default_factory=list)
    mt: List[int] = field(default_factory=list)

    def extract_parameters(self, pairs):
        self.ma = list(pairs.MA)
        self.mb = list(pairs.MB)
        self.za = list(pairs.ZA)
        self.zb = list(pairs.ZB)
        self.ia = list(pairs.IA)
        self.ib = list(pairs.IB)
        self.pa = list(pairs.PA)
        self.pb = list(pairs.PB)
        self.q = list(pairs.Q)
        self.pnt = list(pairs.PNT)
        self.shf = list(pairs.SHF)
        self.mt = list(pairs.MT)

    @classmethod
    def from_endftk(cls, pairs: ENDFtk.MF2.MT151.ParticlePairs):
        """
        Creates an instance of ParticlePair from an ENDFtk ParticlePairs object.
        """
        return cls(
            ma=list(pairs.MA),
            mb=list(pairs.MB),
            za=list(pairs.ZA),
            zb=list(pairs.ZB),
            ia=list(pairs.IA),
            ib=list(pairs.IB),
            pa=list(pairs.PA),
            pb=list(pairs.PB),
            q=list(pairs.Q),
            pnt=list(pairs.PNT),
            shf=list(pairs.SHF),
            mt=list(pairs.MT)
        )

    
    def reconstruct(self):
        chunk = ENDFtk.MF2.MT151.ParticlePairs(self.ma, self.mb, self.za, self.zb, self.ia, self.ib, self.pa, self.pb, self.q, self.pnt, self.shf, self.mt)
        return chunk

    def write_to_hdf5(self, hdf5_group):
        """
        Writes the ParticlePair data to the given HDF5 group.
        """
        hdf5_group.create_dataset('ma', data=self.ma)
        hdf5_group.create_dataset('mb', data=self.mb)
        hdf5_group.create_dataset('za', data=self.za)
        hdf5_group.create_dataset('zb', data=self.zb)
        hdf5_group.create_dataset('ia', data=self.ia)
        hdf5_group.create_dataset('ib', data=self.ib)
        hdf5_group.create_dataset('pa', data=self.pa)
        hdf5_group.create_dataset('pb', data=self.pb)
        hdf5_group.create_dataset('q', data=self.q)
        hdf5_group.create_dataset('pnt', data=self.pnt)
        hdf5_group.create_dataset('shf', data=self.shf)
        hdf5_group.create_dataset('mt', data=self.mt)

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads a ParticlePair from the given HDF5 group.
        """
        # Convert each dataset to a list
        ma = list(hdf5_group['ma'][()])
        mb = list(hdf5_group['mb'][()])
        za = list(hdf5_group['za'][()])
        zb = list(hdf5_group['zb'][()])
        ia = list(hdf5_group['ia'][()])
        ib = list(hdf5_group['ib'][()])
        pa = list(hdf5_group['pa'][()])
        pb = list(hdf5_group['pb'][()])
        q = list(hdf5_group['q'][()])
        pnt = list(hdf5_group['pnt'][()])
        shf = list(hdf5_group['shf'][()])
        mt = list(hdf5_group['mt'][()])
        return cls(ma=ma, mb=mb, za=za, zb=zb, ia=ia, ib=ib, pa=pa, pb=pb, q=q,
                   pnt=pnt, shf=shf, mt=mt)

@dataclass
class RMatrixLimited:
    """Class to store resonance data for a resonance range."""
    IFG: int = 0
    KRL: int = 0
    KRM: int = 0
    ParticlePairs: ParticlePair = field(default_factory=ParticlePair)
    ListSpinGroup: List[SpinGroup] = field(default_factory=list)
    
    @classmethod
    def from_endftk(cls, range: ENDFtk.MF2.MT151.ResonanceRange, mf32_range: ENDFtk.MF32.MT151.ResonanceRange, force_reduced : bool = False):
        """
        Creates an instance of RMatrixLimited from an ENDFtk ResonanceRange object,
        calling from_endftk in cascade for ParticlePair and SpinGroup.
        """
        if range.parameters.IFG == 1:
            force_reduced = False

        particle_pairs = ParticlePair.from_endftk(range.parameters.particle_pairs)
        
        spin_groups = [
                SpinGroup.from_endftk(sgroup, mf32_range.parameters.uncertainties.spin_groups.to_list()[isg], force_reduced) 
                for isg, sgroup in enumerate(range.parameters.spin_groups.to_list()) 
            ]
                
        return cls(
            IFG=1 if force_reduced else range.parameters.IFG,
            KRL=range.parameters.KRL,
            KRM=range.parameters.KRM,
            ParticlePairs=particle_pairs,
            ListSpinGroup=spin_groups
        )
        
    def extract_parameters(self, range: ENDFtk.MF2.MT151.ResonanceRange):
        """
        Extracts the mean parameters from MF2 and constructs the RMatrixLimited object.
        """
        self.IFG = range.parameters.IFG
        self.KRL = range.parameters.KRL
        self.KRM = range.parameters.KRM
        
        self.ParticlePairs = ENDFtk.MF2.MT151.ParticlePairs(range.parameters.particle_pairs)
        
        for spingroup in range.parameters.spin_groups.to_list():
            spin_group = SpinGroup(ResonanceChannels=ENDFtk.MF2.MT151.ResonanceChannels(spingroup.channels))
            spin_group.extract_parameters(spingroup)
            self.ListSpinGroup.append(spin_group)
            
    def reconstruct(self, sample_index: int = 0) -> ENDFtk.MF2.MT151.RMatrixLimited:
        return ENDFtk.MF2.MT151.RMatrixLimited(ifg = self.IFG, 
                                                krl = self.KRL, 
                                                krm = self.KRM, 
                                                pairs = self.ParticlePairs.reconstruct(), 
                                                groups = [sg.reconstruct(sample_index) for sg in self.ListSpinGroup])

    def write_to_hdf5(self, hdf5_group):
        """
        Writes the RMatrixLimited data to the given HDF5 group,
        including particle pairs and cascading spin groups.
        """
        hdf5_group.attrs['IFG'] = self.IFG
        hdf5_group.attrs['KRL'] = self.KRL
        hdf5_group.attrs['KRM'] = self.KRM
        
        # Create a subgroup for ParticlePairs (optional example)
        self.ParticlePairs.write_to_hdf5(hdf5_group.create_group('ParticlePairs'))

        # Create a subgroup for SpinGroups
        sg_group = hdf5_group.create_group('SpinGroups')
        for idx, spin_group in enumerate(self.ListSpinGroup):
            sg_subgroup = sg_group.create_group(f'SpinGroup_{idx}')
            spin_group.write_to_hdf5(sg_subgroup)

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads RMatrixLimited data from the given HDF5 group.
        """

        ifg = hdf5_group.attrs['IFG']
        krl = hdf5_group.attrs['KRL']
        krm = hdf5_group.attrs['KRM']

        # Read pairs from 'ParticlePairs' instead of 'SpinGroups'
        pairs = ParticlePair.read_from_hdf5(hdf5_group['ParticlePairs'])

        # Rebuild SpinGroups
        spin_groups_list = []
        sg_group = hdf5_group['SpinGroups']
        for sg_key in sg_group:
            sg_subgroup = sg_group[sg_key]
            spin_groups_list.append(SpinGroup.read_from_hdf5(sg_subgroup))

        return cls(ifg, krl, krm, pairs, spin_groups_list)

    def get_nominal_parameters(self) -> List[float]:
        """
        Returns a 1D vector containing all nominal resonance parameters.
        Order: For each spin group, for each resonance: [ER, GAM1, GAM2, ..., GAMn]
        
        Returns:
            List[float]: Flattened list of all parameters
        """
        parameters = []
        for spin_group in self.ListSpinGroup:
            for resonance in spin_group.resonance_parameters:
                # Add resonance energy
                parameters.append(resonance.ER[0])
                # Add partial widths
                for gam in resonance.GAM:
                    parameters.append(gam[0])
        return parameters

    def getListStandardDeviation(self) -> Tuple[List[Tuple[int, int, int]], List[float]]:
        index_mapping = []
        ListStandardDeviation = []
        for j_idx, sg in enumerate(self.ListSpinGroup):
            sg_index_map, sg_stddev = sg.getListStandardDeviation(j_idx)
            index_mapping.extend(sg_index_map)
            ListStandardDeviation.extend(sg_stddev)
        return index_mapping, ListStandardDeviation
