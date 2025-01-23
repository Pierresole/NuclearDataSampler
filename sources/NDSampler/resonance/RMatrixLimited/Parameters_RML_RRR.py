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
class Channels:
    """
    Stores channel data (spin, parity, etc.) 
    and handles extraction, reconstruction, and HDF5 I/O.
    """
    spin: float
    parity: float
    ppi: List[int] = field(default_factory=list)
    l: List[int] = field(default_factory=list)
    s: List[float] = field(default_factory=list)
    b: List[float] = field(default_factory=list)
    apt: List[float] = field(default_factory=list)
    ape: List[float] = field(default_factory=list)
    kbk: int = 0
    kps: int = 0

    def extract_parameters(self, endf_channels):
        """
        Extract channel data from an ENDFtk object or similar structure.
        """
        # Example usage with hypothetical attributes:
        self.spin = endf_channels.AJ
        self.parity = endf_channels.PJ
        self.ppi = list(endf_channels.PPI)
        self.l = list(endf_channels.L)
        self.s = list(endf_channels.SCH)
        self.b = list(endf_channels.BND)
        self.apt = list(endf_channels.APT)
        self.ape = list(endf_channels.APE)
        self.kbk = endf_channels.KBK
        self.kps = endf_channels.KPS

    @classmethod
    def from_endftk(cls, endf_channels):
        """
        Creates a Channels instance from an ENDFtk representation.
        """
        instance = cls(
            spin=endf_channels.AJ,
            parity=endf_channels.PJ,
            ppi=list(endf_channels.PPI),
            l=list(endf_channels.L),
            s=list(endf_channels.SCH),
            b=list(endf_channels.BND),
            apt=list(endf_channels.APT),
            ape=list(endf_channels.APE),
            kbk=endf_channels.KBK,
            kps=endf_channels.KPS
        )
        return instance

    def reconstruct(self) -> ENDFtk.MF2.MT151.ResonanceChannels:
        """
        Builds an ENDFtk-like object from the stored channel data.
        """
        # Return or assemble a custom object mirroring ENDFtk's channel structure
        # Minimal placeholder:
        
        return ENDFtk.MF2.MT151.ResonanceChannels(  spin = self.spin,
                                                    parity = self.parity,
                                                    ppi = self.ppi,
                                                    l = self.l,
                                                    s = self.s,
                                                    b = self.b,
                                                    apt = self.apt,
                                                    ape = self.ape,
                                                    kbk = self.kbk,
                                                    kps = self.kps )

    def write_to_hdf5(self, hdf5_group):
        """
        Writes channel data to the given HDF5 group.
        """
        hdf5_group.attrs['spin'] = self.spin
        hdf5_group.attrs['parity'] = self.parity
        hdf5_group.create_dataset('ppi', data=self.ppi)
        hdf5_group.create_dataset('l', data=self.l)
        hdf5_group.create_dataset('s', data=self.s)
        hdf5_group.create_dataset('b', data=self.b)
        hdf5_group.create_dataset('apt', data=self.apt)
        hdf5_group.create_dataset('ape', data=self.ape)
        hdf5_group.attrs['kbk'] = self.kbk
        hdf5_group.attrs['kps'] = self.kps

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads channel data from the given HDF5 group.
        """
        spin = hdf5_group.attrs['spin']
        parity = hdf5_group.attrs['parity']
        ppi = list(hdf5_group['ppi'][()])
        l_ = list(hdf5_group['l'][()])
        s_ = list(hdf5_group['s'][()])
        b_ = list(hdf5_group['b'][()])
        apt = list(hdf5_group['apt'][()])
        ape = list(hdf5_group['ape'][()])
        kbk = hdf5_group.attrs['kbk']
        kps = hdf5_group.attrs['kps']

        return cls(
            spin=spin,
            parity=parity,
            ppi=ppi,
            l=l_,
            s=s_,
            b=b_,
            apt=apt,
            ape=ape,
            kbk=kbk,
            kps=kps
        )


@dataclass
class SpinGroup:
    """Class to store spin group data, including a list of Resonance objects."""
    ResonanceChannels: Channels
    ResonanceParameters: List[Resonance] = field(default_factory=list)

    def extract_parameters(self, spingroup: ENDFtk.MF2.MT151.SpinGroup):
        self.ResonanceChannels = ENDFtk.MF2.MT151.ResonanceChannels(spingroup.channels)
        # Create a single Resonance for this SpinGroup (or multiple if needed)
        new_resonance = Resonance()
        new_resonance.extract_parameters(spingroup.parameters)
        self.ResonanceParameters.append(new_resonance)

    @classmethod
    def from_endftk(cls, spingroup: ENDFtk.MF2.MT151.SpinGroup, spingroup32):
        """
        Creates a SpinGroup from an ENDFtk SpinGroup object, 
        cascading to Channels.from_endftk and Resonance.from_endftk.
        """
        return cls(ResonanceChannels = Channels.from_endftk(spingroup.channels), 
                   ResonanceParameters = [Resonance.from_endftk(spingroup.parameters.ER[i],
                                                                spingroup32.parameters.DER[i],
                                                                spingroup.parameters.GAM[i][:spingroup.NCH],
                                                                spingroup32.parameters.DGAM[i][:spingroup.NCH]) 
                                          for i in range(spingroup.parameters.NRS)
                                        ]
                   )
    
    def reconstruct(self, sample_index : int = 0) -> ENDFtk.MF2.MT151.SpinGroup:
        # Combine all Resonance objects into a single ENDFtk ResonanceParameters                
        
        return ENDFtk.MF2.MT151.SpinGroup(channels = self.ResonanceChannels.reconstruct(), 
                                          parameters = ENDFtk.MF2.MT151.ResonanceParameters(
                                                            energies=[res.ER[sample_index] for res in self.ResonanceParameters],
                                                            parameters=[[gam[sample_index] for gam in res.GAM] for res in self.ResonanceParameters]
                                                        ))
        
    # def reconstruct(self, sample_index) -> ENDFtk.MF2.MT151.SpinGroup:
    #     # Combine all Resonance objects into a single ENDFtk ResonanceParameters
    #     combined_ER = []
    #     combined_GAM = []
    #     for res in self.ResonanceParameters:
    #         combined_ER.append(res.ER[sample_index])
    #         for gam in res.GAM:
    #             combined_GAM.append(gam[sample_index])
    #     parameters = ENDFtk.MF2.MT151.ResonanceParameters(energies = combined_ER, parameters = combined_GAM)
        
    #     return ENDFtk.MF2.MT151.SpinGroup(channels = self.ResonanceChannels.reconstruct(), parameters = parameters)

    def write_to_hdf5(self, hdf5_group):
        """
        Writes this SpinGroup's data to the given HDF5 group.
        """
        
        self.ResonanceChannels.write_to_hdf5(hdf5_group.create_group('Channels'))

        rp_group = hdf5_group.create_group('Resonances')
        for idx, res in enumerate(self.ResonanceParameters):
            res_group = rp_group.create_group(f'Resonance_{idx}')
            res.write_to_hdf5(res_group)

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads a SpinGroup from the given HDF5 group.
        """
        # Reconstruct Channels from HDF5
        resonance_channels = Channels.read_from_hdf5(hdf5_group['Channels'])

        resonance_list = []
        if 'Resonances' in hdf5_group:
            for r_key in hdf5_group['Resonances']:
                r_subgroup = hdf5_group['Resonances'][r_key]
                resonance_list.append(Resonance.read_from_hdf5(r_subgroup))
        return cls(ResonanceChannels=resonance_channels, ResonanceParameters=resonance_list)

    def getListStandardDeviation(self, spin_group_idx: int):
        index_mapping = []
        ListStandardDeviation = []
        for r_idx, resonance in enumerate(self.ResonanceParameters):
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
    def from_endftk(cls, range: ENDFtk.MF2.MT151.ResonanceRange, mf32_range: ENDFtk.MF32.MT151.ResonanceRange):
        """
        Creates an instance of RMatrixLimited from an ENDFtk ResonanceRange object,
        calling from_endftk in cascade for ParticlePair and SpinGroup.
        """
        particle_pairs = ParticlePair.from_endftk(range.parameters.particle_pairs)
        
        spin_groups = [
                SpinGroup.from_endftk(sgroup, mf32_range.parameters.uncertainties.spin_groups.to_list()[isg]) 
                for isg, sgroup in enumerate(range.parameters.spin_groups.to_list()) 
            ]
        
        return cls(
            IFG=range.parameters.IFG,
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
            for resonance in spin_group.ResonanceParameters:
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