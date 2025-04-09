from dataclasses import dataclass, field
from typing import List, Optional, Dict
import ENDFtk
import numpy as np

@dataclass
class ResonanceParameter:
    """Class to store nominal and uncertainty of parameters of a single resonance."""
    ER: List[float] = field(default_factory=list)   # First entry can be nominal, subsequent entries are samples
    AJ: float = None
    GN: List[float] = None
    GG: List[float] = None
    GFA: List[float] = None
    GFB: List[float] = None
    index: List[int] = None  # Position in the covariance matrix
    
    # Uncertainty information
    DER:  float = None
    DGN:  float = None
    DGG:  float = None
    DGFA: float = None
    DGFB: float = None
    
    # Parameter constraints and distribution information
    constraints: Dict[str, Dict] = field(default_factory=dict)  # Stores constraints for each parameter
    
    @classmethod
    def from_endftk(cls, ER: float, DER: float, AJ: float, GN: float, DGN: float, GG: float, DGG: float, GFA: float, DGFA: float, GFB: float, DGFB: float, index: int):
        """
        Build a Resonance from ENDFtk resonance parameters.
        """
        instance = cls()
        instance.ER = [ER]
        instance.AJ = AJ
        instance.GN = [GN] if GN is not None else None
        instance.GG = [GG] if GG is not None else None
        instance.GFA = [GFA] if GFA is not None else None
        instance.GFB = [GFB] if GFB is not None else None
        instance.index = [index] if index is not None else None
        
        # Set uncertainties, ensuring zero values are replaced with None
        instance.DER = DER if DER is not None and DER > 0 else None
        instance.DGN = DGN if DGN is not None and DGN > 0 else None
        instance.DGG = DGG if DGG is not None and DGG > 0 else None
        instance.DGFA = DGFA if DGFA is not None and DGFA > 0 else None
        instance.DGFB = DGFB if DGFB is not None and DGFB > 0 else None
        
        # Add constraints for the parameters
        instance.constraints = {
            'ER': {'positive': True},     # Resonance energy must be positive
            'GN': {'positive': True},     # Neutron width must be positive
            'GG': {'positive': True},     # Gamma width must be positive
            'GFA': {'signed': True},      # Fission widths can be signed
            'GFB': {'signed': True}       # Fission widths can be signed
        }
        
        return instance
    
    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads a Resonance from the given HDF5 group.
        """
        instance = cls()
        
        instance.ER = list(hdf5_group['ER'][()])
        
        if 'AJ' in hdf5_group:
            instance.AJ = list(hdf5_group['AJ'][()])
        elif 'AJ' in hdf5_group.attrs:
            instance.AJ = hdf5_group.attrs['AJ']
            
        if 'GN' in hdf5_group:
            instance.GN = list(hdf5_group['GN'][()])
        if 'GG' in hdf5_group:
            instance.GG = list(hdf5_group['GG'][()])
        if 'GFA' in hdf5_group:
            instance.GFA = list(hdf5_group['GFA'][()])
        if 'GFB' in hdf5_group:
            instance.GFB = list(hdf5_group['GFB'][()])
        if 'index' in hdf5_group:
            instance.index = list(hdf5_group['index'][()])
        
        # Read uncertainty values from attributes
        if 'DER' in hdf5_group.attrs:
            instance.DER = hdf5_group.attrs['DER']
        if 'DGN' in hdf5_group.attrs:
            instance.DGN = hdf5_group.attrs['DGN']
        if 'DGG' in hdf5_group.attrs:
            instance.DGG = hdf5_group.attrs['DGG']
        if 'DGFA' in hdf5_group.attrs:
            instance.DGFA = hdf5_group.attrs['DGFA']
        if 'DGFB' in hdf5_group.attrs:
            instance.DGFB = hdf5_group.attrs['DGFB']
            
        # Load constraints if available
        if 'constraints' in hdf5_group:
            constraints_group = hdf5_group['constraints']
            for param_name in constraints_group:
                param_group = constraints_group[param_name]
                constraint_dict = {}
                for key in param_group.attrs:
                    constraint_dict[key] = param_group.attrs[key]
                instance.constraints[param_name] = constraint_dict
                
        return instance

    def write_to_hdf5(self, hdf5_group):
        """
        Writes this Resonance's data to the given HDF5 group.
        """
        hdf5_group.create_dataset('ER', data=self.ER)
        if isinstance(self.AJ, list):
            hdf5_group.create_dataset('AJ', data=self.AJ)
        else:
            hdf5_group.attrs['AJ'] = self.AJ
        
        if self.GN is not None:
            hdf5_group.create_dataset('GN', data=self.GN)
        if self.GG is not None:
            hdf5_group.create_dataset('GG', data=self.GG)
        if self.GFA is not None:
            hdf5_group.create_dataset('GFA', data=self.GFA)
        if self.GFB is not None:
            hdf5_group.create_dataset('GFB', data=self.GFB)
        if self.index is not None:
            hdf5_group.create_dataset('index', data=self.index)
        
        # Store uncertainty values as attributes if they exist
        if self.DER is not None:
            hdf5_group.attrs['DER'] = self.DER
        if self.DGN is not None:
            hdf5_group.attrs['DGN'] = self.DGN
        if self.DGG is not None:
            hdf5_group.attrs['DGG'] = self.DGG
        if self.DGFA is not None:
            hdf5_group.attrs['DGFA'] = self.DGFA
        if self.DGFB is not None:
            hdf5_group.attrs['DGFB'] = self.DGFB
        
        # Store constraints information
        if self.constraints:
            constraints_group = hdf5_group.create_group('constraints')
            for param_name, constraint_dict in self.constraints.items():
                param_group = constraints_group.create_group(param_name)
                for constraint_key, constraint_value in constraint_dict.items():
                    param_group.attrs[constraint_key] = constraint_value

@dataclass
class LGroup:
    """Class to store resonance parameters for a given L value."""
    L: int = 0
    AWRI: float = 0.0
    APL: List[float] = field(default_factory=list)  # Changed to List for storing samples
    DAPL: float = None  # Uncertainty in APL
    resonances: List[ResonanceParameter] = field(default_factory=list)
    
    @classmethod
    def from_endftk(cls, mf2_lvalue: ENDFtk.MF2.MT151.ReichMooreLValue, scattering_radius_uncertainty: Optional[float] = None, mf32_range: Optional[ENDFtk.MF32.MT151.ResonanceRange] = None):
        
        # Create a mapping from MF2 resonance energies to MF32 indices
        mf32_indices = {}
        mf32_energies = []
        MPAR = 0
        diag = None
        
        # Extract diagonal uncertainties directly from MF32 if available
        if mf32_range is not None and mf32_range.parameters.LCOMP == 1 and hasattr(mf32_range.parameters, 'short_range_blocks') and len(mf32_range.parameters.short_range_blocks) > 0:
            block = mf32_range.parameters.short_range_blocks[0]
            MPAR = block.MPAR
            mf32_energies = block.ER[:]
            
            # Extract diagonal elements from covariance matrix
            covariance_order = block.NPARB
            cm = np.array(block.covariance_matrix)
            cov_matrix = np.zeros((covariance_order, covariance_order))
            triu_indices = np.triu_indices(covariance_order)
            cov_matrix[triu_indices] = cm
            
            # Get standard deviations
            diag = np.sqrt(np.diag(cov_matrix))
            
        # For each MF2 resonance energy, find the matching MF32 index
        for mf2_idx, mf2_er in enumerate(mf2_lvalue.ER):
            # Find the closest matching energy in MF32
            for mf32_idx, mf32_er in enumerate(mf32_energies):
                if abs(mf2_er - mf32_er) < 1e-6:  # Using a small tolerance for floating-point comparison
                    mf32_indices[mf2_idx] = mf32_idx
                    break

        resonances = []
        for iResonance in range(mf2_lvalue.NRS):
            # Get MF32 index if available
            mf32_idx = mf32_indices.get(iResonance, None)
            
            # Initialize uncertainties
            der, dgn, dgg, dgfa, dgfb = None, None, None, None, None
            
            # Set uncertainties based on MPAR value and diagonal elements
            if mf32_idx is not None and diag is not None:
                base_idx = mf32_idx * MPAR
                
                # Get uncertainties based on MPAR value
                if MPAR >= 1 and base_idx < len(diag):
                    der = diag[base_idx]
                if MPAR >= 2 and base_idx + 1 < len(diag):
                    dgn = diag[base_idx + 1]
                if MPAR >= 3 and base_idx + 2 < len(diag):
                    dgg = diag[base_idx + 2]
                if MPAR >= 4 and base_idx + 3 < len(diag):
                    dgfa = diag[base_idx + 3]
                if MPAR >= 5 and base_idx + 4 < len(diag):
                    dgfb = diag[base_idx + 4]
            
            # Create resonance parameter
            resonance = ResonanceParameter.from_endftk(
                ER=mf2_lvalue.ER[iResonance],
                DER=der,
                AJ=mf2_lvalue.spin_values[iResonance],
                GN=mf2_lvalue.neutron_widths[iResonance],
                DGN=dgn,
                GG=mf2_lvalue.gamma_widths[iResonance],
                DGG=dgg,
                GFA=mf2_lvalue.first_fission_widths[iResonance],
                DGFA=dgfa,
                GFB=mf2_lvalue.second_fission_widths[iResonance],
                DGFB=dgfb,
                index=mf32_idx
            )
            resonances.append(resonance)
        
        # Sort resonances by increasing resonance energy (ER)
        resonances.sort(key=lambda r: r.ER[0])
        
        return cls(
            L=mf2_lvalue.L,
            AWRI=mf2_lvalue.AWRI,
            APL=[mf2_lvalue.APL],  # Store as list with initial value
            DAPL=scattering_radius_uncertainty,
            resonances=resonances
        )
       
    def reconstruct(self, sample_index: int = 0) -> ENDFtk.MF2.MT151.ReichMooreLValue:
        """Reconstruct ENDFtk ReichMooreLValue from this object"""
        
        # Sort resonances by increasing resonance energy (ER)
        self.resonances.sort(key=lambda r: r.ER[0])
        
        # Use sampled APL value if available, otherwise use nominal value
        apl_value = self.APL[sample_index] if len(self.APL) > sample_index else self.APL[0]
        
        return ENDFtk.MF2.MT151.ReichMooreLValue(
            awri=self.AWRI,
            l=self.L,
            apl=apl_value,  # Use sampled value
            energies=[res.ER[sample_index] if len(res.ER) > sample_index else res.ER[0] for res in self.resonances],
            spins=[res.AJ for res in self.resonances],
            gn=[res.GN[sample_index] if len(res.GN) > sample_index else res.GN[0] for res in self.resonances],
            gg=[res.GG[sample_index] if len(res.GG) > sample_index else res.GG[0] for res in self.resonances],
            gfa=[res.GFA[sample_index] if len(res.GFA) > sample_index else res.GFA[0] for res in self.resonances],
            gfb=[res.GFB[sample_index] if len(res.GFB) > sample_index else res.GFB[0] for res in self.resonances]
        ) 
     
    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """Read l group data from HDF5"""
        l_value = hdf5_group.attrs['L']
        awri = hdf5_group.attrs['AWRI']
        
        # Read APL as a dataset or create a list from the attribute
        if 'APL' in hdf5_group:
            apl = list(hdf5_group['APL'][()])
        elif 'APL' in hdf5_group.attrs:
            apl = [hdf5_group.attrs['APL']]
        else:
            apl = []
            
        dapl = hdf5_group.attrs['DAPL'] if 'DAPL' in hdf5_group.attrs else None
        
        resonances = []
        if 'Resonances' in hdf5_group:
            rp_group = hdf5_group['Resonances']
            for res_name in sorted(rp_group.keys()):
                resonances.append(ResonanceParameter.read_from_hdf5(rp_group[res_name]))
                
        # Sort resonances by increasing resonance energy (ER)
        resonances.sort(key=lambda r: r.ER[0])
        
        return cls(L=l_value, AWRI=awri, APL=apl, DAPL=dapl, resonances=resonances)
        
    def write_to_hdf5(self, hdf5_group):
        """Write l group data to HDF5"""
        hdf5_group.attrs['L'] = self.L
        hdf5_group.attrs['AWRI'] = self.AWRI
        
        # Store APL as a dataset if it's a list with multiple values
        if len(self.APL) > 1:
            hdf5_group.create_dataset('APL', data=self.APL)
        elif len(self.APL) == 1:
            hdf5_group.attrs['APL'] = self.APL[0]
            
        if self.DAPL is not None:
            hdf5_group.attrs['DAPL'] = self.DAPL
        
        rp_group = hdf5_group.create_group('Resonances')
        for i, res in enumerate(self.resonances):
            res.write_to_hdf5(rp_group.create_group(f'Resonance_{i}'))
        

@dataclass
class ReichMooreData:
    """Class to store resonance data for a resonance range."""
    SPI: float = 0.0
    AP: List[float] = field(default_factory=list)  # Changed to List for storing samples
    DAP: float = None  # Uncertainty in AP
    LAD: int = 0
    LGroups: List[LGroup] = field(default_factory=list)
    NLSC: int = 0  # Number of L values to converge angular distribution
    
        
    @classmethod
    def from_endftk(cls, mf2_range: ENDFtk.MF2.MT151.ResonanceRange, mf32_range: ENDFtk.MF32.MT151.ResonanceRange):
        """
        Creates an instance of ReichMooreData from an ENDFtk ResonanceRange object,
        calling from_endftk in cascade for ParticlePair and SpinGroup.
        """
        if mf32_range.parameters.LCOMP == 2:
            raise NotImplementedError("LCOMP=2 not implemented")
                
        if mf32_range.parameters.ISR != 0:
            vDAPL = mf32_range.parameters.DAP.DAPL.to_list()
            if not vDAPL:
                vDAPL = [mf32_range.parameters.DAP.default_uncertainty] * mf2_range.parameters.NLS
        
        L_groups = [
            LGroup.from_endftk(
                mf2_lvalue=l_group_mf2,
                scattering_radius_uncertainty=vDAPL[ilg] if mf32_range.parameters.ISR != 0 else None,
                mf32_range=mf32_range  # Pass the complete mf32_range
            )
            for ilg, l_group_mf2 in enumerate(mf2_range.parameters.l_values.to_list())
        ]

        return cls(
            SPI=mf2_range.parameters.SPI,
            AP=[mf2_range.parameters.AP],  # Store as list with initial value
            DAP=mf32_range.parameters.DAP.DAP if mf32_range.parameters.ISR != 0 else None,
            LAD=mf2_range.parameters.LAD,
            LGroups=L_groups,
            NLSC=mf2_range.parameters.NLSC
        )
        
    def reconstruct(self, sample_index: int = 0) -> ENDFtk.MF2.MT151.ReichMoore:
        
        # Use sampled AP value if available, otherwise use nominal value
        ap_value = self.AP[sample_index] if len(self.AP) > sample_index else self.AP[0]
        
        return ENDFtk.MF2.MT151.ReichMoore(spin = self.SPI, 
                                            ap = ap_value,  # Use sampled value
                                            lad = self.LAD, 
                                            nlsc = self.NLSC, 
                                            lvalues = [lg.reconstruct(sample_index) for lg in self.LGroups])
        
        
    @classmethod
    def read_from_hdf5(cls, hdf5_group):
        """
        Reads ReichMooreData from the given HDF5 group.
        """
        spi = hdf5_group.attrs['SPI']
        
        # Read AP as a dataset or create a list from the attribute
        if 'AP' in hdf5_group:
            ap = list(hdf5_group['AP'][()])
        elif 'AP' in hdf5_group.attrs:
            ap = [hdf5_group.attrs['AP']]
        else:
            ap = []
            
        dap = hdf5_group.attrs['DAP'] if 'DAP' in hdf5_group.attrs else None
        lad = hdf5_group.attrs['LAD'] if 'LAD' in hdf5_group.attrs else None
        nlsc = hdf5_group.attrs['NLSC']
        
        l_groups = []
        if 'LGroups' in hdf5_group:
            lg_group = hdf5_group['LGroups']
            for lg_name in sorted(lg_group.keys()):
                l_groups.append(LGroup.read_from_hdf5(lg_group[lg_name]))
        
        return cls(SPI=spi, AP=ap, DAP=dap, LAD=lad, LGroups=l_groups, NLSC=nlsc)
    
    def write_to_hdf5(self, hdf5_group):
        """
        Writes the ReichMooreData to the given HDF5 group.
        """
        hdf5_group.attrs['SPI'] = self.SPI
        
        # Store AP as a dataset if it's a list with multiple values
        if len(self.AP) > 1:
            hdf5_group.create_dataset('AP', data=self.AP)
        elif len(self.AP) == 1:
            hdf5_group.attrs['AP'] = self.AP[0]
            
        if self.DAP is not None:
            hdf5_group.attrs['DAP'] = self.DAP
            
        hdf5_group.attrs['LAD'] = self.LAD
        hdf5_group.attrs['NLSC'] = self.NLSC

        # Create a subgroup for LGroups
        lg_group = hdf5_group.create_group('LGroups')
        for idx, l_group in enumerate(self.LGroups):
            sg_subgroup = lg_group.create_group(f'LGroup_{idx}')
            l_group.write_to_hdf5(sg_subgroup)
    
    def get_correlated_nominal_parameters(self) -> List[float]:
        """
        Returns a list of nominal parameter values that have uncertainty information and are 
        therefore included in the covariance matrix.
        
        Returns:
        --------
        List[float] : List of nominal parameter values for all resonances that have uncertainties
        """
        correlated_params = []
        
        # Loop through all L-groups
        for l_group in self.LGroups:
            # Loop through all resonances in this L-group
            for resonance in l_group.resonances:
                # Skip resonances without uncertainty information
                if resonance.index is None:
                    continue
                
                # Add parameters with uncertainty information (non-None)
                params_with_uncertainty = [
                    (resonance.ER[0], resonance.DER),
                    (resonance.GN[0] if resonance.GN is not None else None, resonance.DGN),
                    (resonance.GG[0] if resonance.GG is not None else None, resonance.DGG),
                    (resonance.GFA[0] if resonance.GFA is not None else None, resonance.DGFA),
                    (resonance.GFB[0] if resonance.GFB is not None else None, resonance.DGFB)
                ]
                
                for value, uncertainty in params_with_uncertainty:
                    if value is not None and uncertainty is not None and uncertainty > 0:
                        correlated_params.append(value)
        
        return correlated_params
    
    def get_uncorr_nominal_parameters(self) -> List[float]:
        """
        Returns a list of nominal parameter values that are not included in the covariance matrix,
        but can have uncertainties (AP and APL for each L-group).
        
        Returns:
        --------
        List[float] : List of nominal values for AP and APL parameters
        """
        uncorr_params = []
        
        # Add AP if it has uncertainty
        if self.AP and len(self.AP) > 0 and self.DAP is not None and self.DAP > 0:
            uncorr_params.append(self.AP[0])
        
        # Add APL for each L-group if it has uncertainty
        for l_group in self.LGroups:
            if l_group.APL and len(l_group.APL) > 0 and l_group.DAPL is not None and l_group.DAPL > 0:
                uncorr_params.append(l_group.APL[0])
        
        return uncorr_params