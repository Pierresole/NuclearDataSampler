from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import ENDFtk
import h5py  # ensure h5py is available if not already

@dataclass
class Resonance:
    """Class to store nominal (1 energy and NCH widths) and sampled resonance data. To be clear : All MF2 is here."""
    ER: List[float] = field(default_factory=list)   # First entry can be nominal, subsequent entries are samples
    GAM: List[List[float]] = field(default_factory=list)  # Same idea for widths
    DER: float = None
    DGAM: List[Optional[float]] = field(default_factory=lambda: [None])
    FissionChannels: Optional[List[int]] = None  # List of fission channels
    # def extract_parameters(self, parameters: ENDFtk.MF2.MT151.ResonanceParameters):
    #     # Convert any ENDFtk views to mutable lists
    #     self.ER = list(parameters.ER)
    #     self.GAM = [list(g) for g in parameters.GAM]

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
        # Store DER as an attribute (float or None)
        hdf5_group.attrs['DER'] = self.DER if self.DER is not None else float('nan')
        # Store DGAM as a dataset (list of floats or None)
        dgam_data = [d if d is not None else float('nan') for d in self.DGAM]
        hdf5_group.create_dataset('DGAM', data=dgam_data)

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads a Resonance from the given HDF5 group.
        """
        er_data = hdf5_group['ER'][()]
        gam_data = []
        for dataset in hdf5_group['GAM']:
            gam_data.append(list(hdf5_group['GAM'][dataset][()]))
        # Read DER, convert nan to None
        der = hdf5_group.attrs.get('DER', float('nan'))
        der = None if der != der else der  # nan check
        # Read DGAM, convert nan to None
        dgam_data = list(hdf5_group['DGAM'][()])
        dgam = [None if (x != x) else x for x in dgam_data]  # nan to None
        return cls(ER=list(er_data), GAM=gam_data, DER=der, DGAM=dgam)

    @classmethod
    def from_endftk(cls, energy: float, denergy: float, widths: List[float], dwidths: List[Optional[float]], fisschannels: Optional[List[int]]):
        """
        Build a Resonance from ENDFtk resonance parameters.
        """
        instance = cls()
        instance.ER = [energy]
        instance.DER = denergy
        instance.GAM = [[width] for width in widths]
        instance.DGAM = dwidths
        instance.FissionChannels = fisschannels
        return instance



@dataclass
class ParticlePair:
    """
    Represents a single particle pair, storing the masses, charges, Q-value, etc.
    """
    ma: float = 0.0
    mb: float = 0.0
    za: float = 0.0
    zb: float = 0.0
    ia: float = 0.0
    ib: float = 0.0
    pa: float = 0.0
    pb: float = 0.0
    q: float = 0.0
    pnt: int = 0
    shf: int = 0
    mt: int = 0

    def reducedMass(self) -> float:
        """Returns the reduced mass in some consistent mass units."""
        return (self.ma * self.mb) / (self.ma + self.mb)

    # can be negative (it is the case of negative energy channels)
    def k2(self, E_lab: float) -> float:
        """
        Computes squared wave number in center-of-mass frame in barn^-1.
        k2 [barn^-1] <=> k2 [10^-24 cm^-1] <=> k [10^12 cm-1]
        Must be in barn^-1 because channel radii are given in unity 10^-12 cm
        Ignoring relativistic terms.
        """
        
        m_neutron = 1.00866491600   # mass of a neutron in amu
        
        E_com = m_neutron * self.mb / (self.mb + self.ma) * E_lab
        
        # Convert to barn^-1
        # constantMomentum2WaveNumber2 = (m_amu * E_eV) / (hbar^2) * 1e-28
        # where m_amu is the atomic mass unit in kg, E_eV is the energy in electron volts,
        # hbar is the reduced Planck constant in J*s, and 1e-28 converts m^-2 to barn^-1.
        constant_momentum_to_wavenumber_squared = 2.392253439955726e-6  # m_amu * eV / (hbar^2) * 1e-28
        
        return 2.0 * self.reducedMass() * (E_com + self.q) * constant_momentum_to_wavenumber_squared

    def write_to_hdf5(self, hdf5_group):
        # Minimal example: store each float in attrs
        for attr in ['ma','mb','za','zb','ia','ib','pa','pb','q','pnt','shf','mt']:
            hdf5_group.attrs[attr] = getattr(self, attr)

    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        kwargs = {attr: hdf5_group.attrs[attr] for attr in ['ma','mb','za','zb','ia','ib','pa','pb','q','pnt','shf','mt']}
        return cls(**kwargs)



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

    # def PenetrationFactor(self, energy_lab: float, particlePair: ParticlePair) -> float:
    #     """Compute the channel penetration factor based on the orbital momentum l"""
    #     import math
        
    #     rho2 = abs(particlePair.k2(energy_lab)) * (self.apt ** 2)
        
    #     # if rho2 < 0.0:
    #     #     return 0.0
        
    #     rho = math.sqrt(rho2)
    #     if self.l == 0:
    #         return rho
    #     elif self.l == 1:
    #         return rho**3 / (1.0 + rho2)
    #     elif self.l == 2:
    #         return rho**5 / (9.0 + 3.0*rho2 + rho2**2)
    #     elif self.l == 3:
    #         return rho**7 / (225.0 + 45.0*rho2 + 6.0*rho2**2 + rho2**3)
    #     else:
    #         raise ValueError("PenetrationFactor only implemented for l=0,1,2,3.")



@dataclass 
class SpinGroup:
    """Class to store spin group data, including channels and resonances"""
    spin: float
    parity: float
    kbk: int  
    kps: int
    channels: List[Channel] = field(default_factory=list)
    ResonanceParameters: List[Resonance] = field(default_factory=list)
    all_pairs: List[ParticlePair] = field(default_factory=list)
    
    @classmethod
    def from_endftk(cls, spingroup2: ENDFtk.MF2.MT151.SpinGroup, spingroup32 = None, want_reduced: bool = False,
                    entrance_pair: Optional[ParticlePair] = None, all_pairs: Optional[List[ParticlePair]] = None):
        """
        Create SpinGroup from ENDFtk objects with optional reduced width conversion.
        The entrance_pair is the particle pair with mt=2 by default (elastic).
        """
        instance = cls(
            spin=spingroup2.channels.AJ,
            parity=spingroup2.channels.PJ,
            kbk=spingroup2.channels.KBK,
            kps=spingroup2.channels.KPS,
            channels=[Channel.from_endftk(spingroup2.channels, i) for i in range(spingroup2.NCH)], 
            all_pairs=all_pairs if all_pairs else []
        )
        
        # Fission channels may have negative widths, we want to keep this information
        ppairs_fission = [i for i, pp in enumerate(instance.all_pairs) if pp.mt == 18]
        index_fission_channels = [i for i, ch in enumerate(instance.channels) if ch.ppi in ppairs_fission]

        if want_reduced and entrance_pair is not None:
            resonances = []
            for i in range(spingroup2.parameters.NRS):
                energy = spingroup2.parameters.ER[i]
                widths = []
                widths_uncertainty = []
                for ch_idx in range(spingroup2.NCH):
                    reduced_width, reduced_width_uncertainty = instance._convert_to_reduced_width(spingroup2.parameters.GAM[i][ch_idx], spingroup32.parameters.DGAM[i][ch_idx], ch_idx, energy, entrance_pair)
                    widths.append(reduced_width)
                    # Pierre 30/04/25 - append reduced, and conversion matrix on correlation ok ?
                    widths_uncertainty.append(reduced_width_uncertainty)
                    # widths_uncertainty.append(spingroup32.parameters.DGAM[i][ch_idx])

                resonances.append( Resonance.from_endftk(energy, spingroup32.parameters.DER[i], widths, widths_uncertainty) )
        else:
            resonances = [ Resonance.from_endftk(spingroup2.parameters.ER[i],
                                                 spingroup32.parameters.DER[i],
                                                 spingroup2.parameters.GAM[i][:spingroup2.NCH],
                                                 spingroup32.parameters.DGAM[i][:spingroup2.NCH], 
                                                 index_fission_channels ) for i in range(spingroup2.parameters.NRS) ]

        instance.ResonanceParameters = resonances
        return instance
    
    def derivative_gamma_by_Gamma_in_Gamma0(self, resonance_idx: int, channel_idx: int, entrance_pair: ParticlePair) -> float:
        """
        Computes the derivative of gamma with respect to Gamma.
        """
        import numpy as np
        
        E_resonance_lab = self.ResonanceParameters[resonance_idx].ER[0]
        
        # Find the specific pair for this channel
        pair_index = self.channels[channel_idx].ppi
        channel_pair = self.all_pairs[pair_index-1]
        # Width is reduced
        gamma0 = self.ResonanceParameters[resonance_idx].GAM[channel_idx][0]
        
        # Penetrability for a photon is P=1 (but why ?)
        if channel_pair.pnt == -1 or (channel_pair.pnt == 0 and channel_pair.mt in [19, 102]):
            P = 1.0
        else:
            P,_ = self.channelPenetrationAndShift(E_resonance_lab, channel_idx, entrance_pair)
            # P = self.channels[channel_idx].PenetrationFactor(E_resonance_lab, entrance_pair)
            
        if P <= 0 or gamma0 == 0:
            return 0
        
        # return 1 / (2 * np.sqrt(2 * P * abs(Gamma0)))
        return 1 / ( 4 * P * gamma0 )


    def _convert_to_reduced_width(self, width: float, width_uncertainty: Optional[float], channel_idx: int, E_lab: float, entrance_pair: ParticlePair) -> Tuple[float, float]:
        """
        Convert physical width to reduced width using PenetrationFactor
        from the associated channel and entrance pair.
        Also calculates the uncertainty on the reduced width.
        """
        import numpy as np
        if width == 0.0:
            return 0.0, width_uncertainty

        # Find the specific pair for this channel
        pair_index = self.channels[channel_idx].ppi
        channel_pair = self.all_pairs[pair_index-1]

        # Penetrability for a photon is P=1 (but why ?)
        if channel_pair.pnt == -1 or (channel_pair.pnt == 0 and channel_pair.mt in [19, 102]):
            P = 1.0
        else:
            P,_ = self.channelPenetrationAndShift(np.abs(E_lab), channel_idx, entrance_pair)
            # P = self.channels[channel_idx].PenetrationFactor(E_lab, entrance_pair)

        if P <= 0.0:
            return 0.0, 0.0

        reduced_width = np.sign(width) * np.sqrt(abs(width) / (2.0 * P))
        
        if width_uncertainty is None:
            uncertainty = 0.0
        else:
            uncertainty = width_uncertainty / (2 * np.sqrt(2 * P * abs(width)))

        return reduced_width, uncertainty


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
            energies=[res.ER[sample_index] if len(res.ER) > 1 else res.ER[0] for res in self.ResonanceParameters],
            parameters=[[gam[sample_index] if len(gam) > 1 else gam[0] for gam in res.GAM] 
                   for res in self.ResonanceParameters]
        )

        return ENDFtk.MF2.MT151.SpinGroup(
            channels=channels,
            parameters=parameters
        )


    def channelPenetrationAndShift(self, E_lab: float, channel_index: int, entrance_pair: ParticlePair):
        """Compute penetration and shift using channel angular momentum"""
        import math
        
        l = self.channels[channel_index].l
        rho2 = abs(entrance_pair.k2(E_lab)) * (self.channels[channel_index].apt ** 2)

        if l < 0:
            raise ValueError("l must be a non-negative integer.")

        P, S = 0.0, 0.0
        rho = math.sqrt(rho2)

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
        for i, res in enumerate(self.ResonanceParameters):
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
        resonances.sort(key=lambda res: res.ER[0])

        return cls(
            spin=hdf5_group.attrs['spin'],
            parity=hdf5_group.attrs['parity'],
            kbk=hdf5_group.attrs['kbk'],
            kps=hdf5_group.attrs['kps'],
            channels=channels,
            ResonanceParameters=resonances
        )




@dataclass
class RMatrixLimited:
    """Class to store resonance data for a resonance range."""
    IFG: int = 0
    KRL: int = 0
    KRM: int = 0
    ListParticlePairs: List[ParticlePair] = field(default_factory=list)
    EntranceParticlePair : ParticlePair = None
    ListSpinGroup: List[SpinGroup] = field(default_factory=list)
    
    @classmethod
    def from_endftk(cls, mf2_range: ENDFtk.MF2.MT151.ResonanceRange, mf32_range: ENDFtk.MF32.MT151.ResonanceRange, want_reduced: bool = False):
        """
        Creates an instance of RMatrixLimited from an ENDFtk ResonanceRange object,
        calling from_endftk in cascade for ParticlePair and SpinGroup.
        """
        # Determine if reduction is needed
        already_reduced = mf2_range.parameters.IFG == 1
        to_reduced = want_reduced and not already_reduced
        
        all_pairs = []
        for i in range(mf2_range.parameters.particle_pairs.NPP):
            pp = ParticlePair(
                ma= mf2_range.parameters.particle_pairs.MA[i],
                mb= mf2_range.parameters.particle_pairs.MB[i],
                za= mf2_range.parameters.particle_pairs.ZA[i],
                zb= mf2_range.parameters.particle_pairs.ZB[i],
                ia= mf2_range.parameters.particle_pairs.IA[i],
                ib= mf2_range.parameters.particle_pairs.IB[i],
                pa= mf2_range.parameters.particle_pairs.PA[i],
                pb= mf2_range.parameters.particle_pairs.PB[i],
                q= mf2_range.parameters.particle_pairs.Q[i],
                pnt= mf2_range.parameters.particle_pairs.PNT[i],
                shf= mf2_range.parameters.particle_pairs.SHF[i],
                mt= mf2_range.parameters.particle_pairs.MT[i]
            )
            all_pairs.append(pp)

        # Find the entrance pair, the pair with mt=2 (elastic)
        EntrancePair = next((p for p in all_pairs if p.mt == 2), None)

        spin_groups = [
            SpinGroup.from_endftk(
                sping_group_mf2,
                mf32_range.parameters.uncertainties.spin_groups.to_list()[isg],
                to_reduced,
                EntrancePair,
                all_pairs
            )
            for isg, sping_group_mf2 in enumerate(mf2_range.parameters.spin_groups.to_list())
        ]

        return cls(
            IFG=1 if to_reduced else 0,  # IFG is 1 if force reduced, otherwise 0
            KRL=mf2_range.parameters.KRL,
            KRM=mf2_range.parameters.KRM,
            ListSpinGroup=spin_groups,
            ListParticlePairs=all_pairs,
            EntranceParticlePair=EntrancePair
        )
            
    def reconstruct(self, sample_index: int = 0) -> ENDFtk.MF2.MT151.RMatrixLimited:
        
        ppairs = ENDFtk.MF2.MT151.ParticlePairs(
            ma = [pp.ma  for pp in self.ListParticlePairs], 
            mb = [pp.mb  for pp in self.ListParticlePairs], 
            za = [pp.za  for pp in self.ListParticlePairs], 
            zb = [pp.zb  for pp in self.ListParticlePairs], 
            ia = [pp.ia  for pp in self.ListParticlePairs], 
            ib = [pp.ib  for pp in self.ListParticlePairs], 
            pa = [pp.pa  for pp in self.ListParticlePairs], 
            pb = [pp.pb  for pp in self.ListParticlePairs], 
            q  = [pp.q   for pp in self.ListParticlePairs],
            pnt= [pp.pnt for pp in self.ListParticlePairs],
            shf= [pp.shf for pp in self.ListParticlePairs],
            mt = [pp.mt  for pp in self.ListParticlePairs],
        )
        
        return ENDFtk.MF2.MT151.RMatrixLimited(ifg = self.IFG, 
                                                krl = self.KRL, 
                                                krm = self.KRM, 
                                                pairs = ppairs, 
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
        pp_group = hdf5_group.create_group('ParticlePairs')
        for idx, pair in enumerate(self.ListParticlePairs):
            pair.write_to_hdf5(pp_group.create_group(f'ParticlePair_{idx}'))

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
        # Read basic attributes
        ifg = hdf5_group.attrs['IFG']
        krl = hdf5_group.attrs['KRL']
        krm = hdf5_group.attrs['KRM']

        # Read and reconstruct particle pairs
        pairs = []
        pp_group = hdf5_group['ParticlePairs']
        for pp_key in sorted(pp_group.keys()):
            pair = ParticlePair.read_from_hdf5(pp_group[pp_key])
            pairs.append(pair)

        # Reconstruct entrance pair
        entrance_pair = next((p for p in pairs if p.mt == 2), None)

        # Read and reconstruct spin groups
        spin_groups = []
        sg_group = hdf5_group['SpinGroups']
        for sg_key in sorted(sg_group.keys()):  # Sort to maintain order
            sg_subgroup = sg_group[sg_key]
            spin_group = SpinGroup.read_from_hdf5(sg_subgroup)
            # Set all_pairs reference for each spin group
            spin_group.all_pairs = pairs
            spin_groups.append(spin_group)

        return  cls(IFG=ifg,
                    KRL=krl,
                    KRM=krm,
                    ListParticlePairs=pairs,
                    EntranceParticlePair=entrance_pair,
                    ListSpinGroup=spin_groups)

    def get_nominal_parameters(self) -> List[float]:
        """
        Returns a 1D vector containing all nominal resonance parameters.
        Order: For each spin group, for each resonance: [ER, GAM1, GAM2, ..., GAMn]
        
        Returns:
            List[float]: Flattened list of all parameters
        """
        parameters = []
        for spin_group in self.ListSpinGroup:
            for res_idx, resonance in enumerate(spin_group.ResonanceParameters):
                # Add resonance energy
                parameters.append(resonance.ER[0])
                # Add partial widths
                for gam in resonance.GAM:
                    parameters.append(gam[0])
        return parameters

    def get_standard_deviations(self, non_null_only: bool = False) -> Tuple[List[Tuple[int, int, int]], List[float]]:
        """
        Returns a tuple (index_mapping, std_devs) where:
        - index_mapping: list of (spin_group_idx, resonance_idx, param_idx)
        - std_devs: list of standard deviations (DER and DGAM)
        If non_null_only is True, only non-None and non-zero values are included.
        If False, all values are included, replacing None with 0.
        """
        index_mapping = []
        std_devs = []
        for j_idx, sg in enumerate(self.ListSpinGroup):
            for r_idx, resonance in enumerate(sg.ResonanceParameters):
                # DER
                der = resonance.DER
                if non_null_only:
                    if der is not None and der != 0:
                        index_mapping.append((j_idx, r_idx, 0))
                        std_devs.append(der)
                else:
                    index_mapping.append((j_idx, r_idx, 0))
                    std_devs.append(der if der is not None else 0.0)
                # DGAM
                for p_idx, dgam in enumerate(resonance.DGAM):
                    if non_null_only:
                        if dgam is not None and dgam != 0:
                            index_mapping.append((j_idx, r_idx, p_idx + 1))
                            std_devs.append(dgam)
                    else:
                        index_mapping.append((j_idx, r_idx, p_idx + 1))
                        std_devs.append(dgam if dgam is not None else 0.0)
        return index_mapping, std_devs

    def get_nominal_parameters_with_uncertainty(self) -> Tuple[List[Tuple[int, int, int]], List[float]]:
        """
        Returns a tuple (index_mapping, nominal_values) where:
        - index_mapping: list of (spin_group_idx, resonance_idx, param_idx) for parameters with non-null/non-zero uncertainty.
        - nominal_values: list of nominal parameter values (ER[0] or GAM[param_idx-1][0]) corresponding to the index_mapping.
        """
        index_mapping = []
        nominal_values = []
        for j_idx, sg in enumerate(self.ListSpinGroup):
            for r_idx, resonance in enumerate(sg.ResonanceParameters):
                # DER
                if resonance.DER is not None and resonance.DER != 0:
                    index_mapping.append((j_idx, r_idx, 0))
                    nominal_values.append(resonance.ER[0])
                # DGAM
                for p_idx, dgam in enumerate(resonance.DGAM):
                    if dgam is not None and dgam != 0:
                        # Ensure the corresponding GAM exists before accessing
                        if p_idx < len(resonance.GAM):
                            index_mapping.append((j_idx, r_idx, p_idx + 1))
                            nominal_values.append(resonance.GAM[p_idx][0])
                        else:
                            # This case might indicate an inconsistency in data structure
                            print(f"Warning: DGAM index {p_idx} has uncertainty but no corresponding GAM in SG {j_idx}, Res {r_idx}.")

        return index_mapping, nominal_values

    def get_jacobian_diagonal(self) -> List[float]:
        """
        Returns a list representing the diagonal of the Jacobian for parameter conversion.
        Each energy contributes '1', each width contributes derivative_gamma_by_Gamma_in_Gamma0().
        """
        diag = []
        for sg_idx, sg in enumerate(self.ListSpinGroup):
            for r_idx, resonance in enumerate(sg.ResonanceParameters):
                # Energy partial derivative == 1
                diag.append(1.0)
                # Width partial derivatives
                for ch_idx in range(len(sg.channels)):
                    val = sg.derivative_gamma_by_Gamma_in_Gamma0(r_idx, ch_idx, self.EntranceParticlePair)
                    diag.append(val)
        return diag

    def extract_covariance_matrix_LCOMP2(self, covariance_matrix):
        """
        Fix the previous J @ covariance_matrix @ J.T by scaling
        each element with the corresponding diagonal entries of J.
        """
        J = self.get_jacobian_diagonal()  # J is just a list of diagonal values
        for i in range(len(J)):
            for j in range(len(J)):
                covariance_matrix[i][j] *= J[i] * J[j]
        return covariance_matrix
