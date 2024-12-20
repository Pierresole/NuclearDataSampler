## Resonance Range
from ENDFtk.MF2.MT151 import ResonanceRange
from ENDFtk.MF2.MT151 import Isotope, Section
## R-Matrix Limited
from ENDFtk.MF2.MT151 import ParticlePairs, ResonanceChannels, ResonanceParameters, SpinGroup
from ENDFtk.MF2.MT151 import RMatrixLimited
## Reich Moore
from ENDFtk.MF2.MT151 import ReichMooreLValue
from ENDFtk.MF2.MT151 import ReichMoore
## Breit Wigner
from ENDFtk.MF2.MT151 import SingleLevelBreitWigner
## Averaged Breit Wigner
from ENDFtk.MF2.MT151 import UnresolvedEnergyDependent, UnresolvedEnergyDependentLValue, UnresolvedEnergyDependentJValue


class ResonanceUpdater:
    def update_resonance_parameters(self, tape, resonance_range, sampled_params):
        """
        Updates the resonance parameters in the tape for the given resonance range.

        Parameters:
        - tape: The ENDF tape object to update.
        - resonance_range: The resonance range object from the tape.
        - sampled_params: The sampled parameters vector.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class ReichMooreUpdater(ResonanceUpdater):
    def update_resonance_parameters(self, tape, resonance_range, sampled_params, cov_data, sampled_AP=None, sampled_APLs=[]):
        """
        Updates the resonance parameters in the tape for the Reich-Moore format.

        Parameters:
        - tape: The ENDF tape object to update.
        - resonance_range: The resonance range object from the tape.
        - sampled_params: The sampled parameters vector.
        - cov_data: The covariance data dictionary from the HDF5 file.
        - sampled_AP: Sampled value of AP (scattering radius), if available.
        - sampled_APLs: List of sampled APL values, if available.
        """
        # Extract mapping information from cov_data
        L_values = cov_data['parameters_L_values']
        resonance_indices = cov_data['resonance_indices']
        param_types = cov_data['param_types']

        # Reconstruct the parameters list from the sampled_params and mapping
        parameters = []
        for idx, sampled_value in enumerate(sampled_params):
            L = L_values[idx]
            resonance_index = resonance_indices[idx]
            param_type = param_types[idx]
            parameters.append({
                'L': L,
                'resonance_index': resonance_index,
                'type': param_type,
                'value': sampled_value
            })

        # Update the resonance parameters in the tape
        l_values = resonance_range.parameters.l_values.to_list()
        # Create a mapping from L to l_value
        l_value_dict = {l_value.L: l_value for l_value in l_values}

        # Initialize new l_values with copies of the original parameters
        new_l_values = []
        for idx_l_value, l_value in enumerate(l_values):
            L = l_value.L
            NRS = l_value.NRS
            ER_new = l_value.ER.copy()
            GN_new = l_value.GN.copy()
            GG_new = l_value.GG.copy()
            GFA_new = l_value.GFA.copy()
            GFB_new = l_value.GFB.copy()
            AJ_new = l_value.AJ  # Assuming AJ remains unchanged

            # Replace parameters with sampled values if available
            for param in parameters:
                if param['L'] == L and param['resonance_index'] is not None:
                    idx_res = param['resonance_index']
                    param_type = param['type']
                    value = param['value']
                    if param_type == 1:
                        ER_new[idx_res] = value
                    elif param_type == 2:
                        GN_new[idx_res] = value
                    elif param_type == 3:
                        GG_new[idx_res] = value
                    elif param_type == 4:
                        GFA_new[idx_res] = value
                    elif param_type == 5:
                        GFB_new[idx_res] = value

            # Use sampled APL if available
            if idx_l_value < len(sampled_APLs):
                sampled_APL = sampled_APLs[idx_l_value]
            else:
                sampled_APL = l_value.APL  # Original APL

            # Create new l_value with updated APL
            new_l_value = ReichMooreLValue(
                awri = l_value.AWRI,
                apl = sampled_APL,
                l = L,
                energies = ER_new,
                spins = AJ_new,
                gn = GN_new,
                gg = GG_new,
                gfa = GFA_new,
                gfb = GFB_new
            )
            new_l_values.append(new_l_value)

        # Create new parameters object with sampled AP
        new_parameters = ReichMoore(
            spin = resonance_range.parameters.SPI,
            ap = sampled_AP if sampled_AP is not None else resonance_range.parameters.AP,
            lad = resonance_range.parameters.LAD,
            nlsc = resonance_range.parameters.NLSC,
            lvalues = new_l_values
        )

        # Create new resonance range
        new_resonance_range = ResonanceRange(
            el = resonance_range.EL,
            eh = resonance_range.EH,
            naps = resonance_range.NAPS,
            parameters = new_parameters
        )

        return new_resonance_range



class RMatrixLimitedUpdater(ResonanceUpdater):
    def update_resonance_parameters(self, tape, resonance_range, sampled_params):
        """
        Updates the resonance parameters in the tape for R-Matrix Limited format.
        """
        # Extract necessary information
        isotope = resonance_range.isotope
        NAPS = resonance_range.NAPS
        EL = resonance_range.EL
        EH = resonance_range.EH

        parameters = resonance_range.parameters
        pairs = ParticlePairs(parameters.particle_pairs)
        spin_groups = parameters.spin_groups.to_list()

        # Update the resonance parameters with sampled values
        new_spin_groups = []
        idx = 0  # Index in the sampled_params vector
        for sg in spin_groups:
            channels = ResonanceChannels(sg.channels)
            resonances = []
            for res_idx in range(sg.parameters.NRSA):
                # Extract sampled ER
                ER_sampled = sampled_params[idx]
                idx += 1
                # Extract sampled GAMs
                GAM_sampled = []
                for ch_idx in range(sg.NCH):
                    GAM_sampled.append(sampled_params[idx])
                    idx +=1
                # Create new ResonanceParameters object
                resonance_parameters = ResonanceParameters([ER_sampled], [GAM_sampled])
                resonances.append(resonance_parameters)

            # Combine all resonance parameters into one ResonanceParameters object
            ER_list = [rp.ER[0] for rp in resonances]
            GAM_list = [rp.GAM[0] for rp in resonances]
            resonance_parameters = ResonanceParameters(ER_list, GAM_list)
            new_spin_group = SpinGroup(channels, resonance_parameters)
            new_spin_groups.append(new_spin_group)

        # Create new RMatrixLimited parameters
        new_parameters = RMatrixLimited(
            ifg = parameters.IFG,
            krl = parameters.KRL,
            krm = parameters.KRM,
            pairs = pairs,
            groups = new_spin_groups
        )

        # Create new ResonanceRange
        new_resonance_range = ResonanceRange(
            el = EL,
            eh = EH,
            naps = NAPS,
            parameters = new_parameters
        )

        # Create new Isotope
        new_isotope = Isotope(
            zai = isotope.ZAI,
            abn = isotope.ABN,
            lfw = isotope.LFW,
            ranges = [new_resonance_range]
        )

        # Create new Section
        new_section = Section(
            zaid = resonance_range.ZA,
            awr = resonance_range.AWR,
            isotopes = [new_isotope]
        )

        # Replace the existing section in the tape
        mat_num = tape.material_numbers[0]
        tape.MAT(mat_num).MF(2).insert_or_replace(new_section)



class AveragedBreitWignerUpdater(ResonanceUpdater):
    def update_resonance_parameters(self, tape, resonance_range, sampled_groups):
        """
        Updates the resonance parameters in the tape for the Averaged Breit-Wigner format.

        Parameters:
        - tape: The ENDF tape object to update.
        - resonance_range: The resonance range object from the tape.
        - sampled_groups: List of dictionaries containing sampled parameters per group.
        """
        # Build new l_values with updated parameters
        l_values = resonance_range.parameters.l_values.to_list()
        new_l_values = []

        # Map (L, J) to sampled parameters
        sampled_params_dict = {}
        for group in sampled_groups:
            L = group['L']
            J = group['J']
            sampled_parameters = group['sampled_parameters']
            sampled_params_dict[(L, J)] = sampled_parameters

        for l_value in l_values:
            L = l_value.L
            j_values = l_value.j_values.to_list()
            new_j_values = []

            for j_value in j_values:
                J = j_value.AJ

                # Get the sampled parameters for this (L, J)
                key = (L, J)
                if key in sampled_params_dict:
                    sampled_parameters = sampled_params_dict[key]

                    # Create new j_value with updated parameters
                    new_j_value = UnresolvedEnergyDependentJValue(
                        spin = J,
                        amun = j_value.AMUN,
                        amug = j_value.AMUG,
                        amuf = j_value.AMUF,
                        amux = j_value.AMUX,
                        interpolation = j_value.INT,
                        energies = j_value.ES[:],  # Energies remain the same
                        d = sampled_parameters.get('D', j_value.D).tolist(),
                        gn = sampled_parameters.get('GN', j_value.GN).tolist(),
                        gg = j_value.GG[:],  # Assuming GG is not sampled
                        gf = j_value.GF[:],
                        gx = j_value.GX[:]
                    )
                else:
                    # No sampled parameters for this (L, J), keep original
                    new_j_value = j_value

                new_j_values.append(new_j_value)

            # Create new l_value with updated j_values
            new_l_value = UnresolvedEnergyDependentLValue(
                awri = l_value.AWRI,
                l = L,
                jvalues = new_j_values
            )

            new_l_values.append(new_l_value)

        # Create new parameters object
        new_parameters = UnresolvedEnergyDependent(
            spin = resonance_range.parameters.SPI,
            ap = resonance_range.parameters.AP,
            lssf = resonance_range.parameters.LSSF,
            lvalues = new_l_values
        )

        # Create new resonance range
        new_resonance_range = ResonanceRange(
            el = resonance_range.EL,
            eh = resonance_range.EH,
            naps = resonance_range.NAPS,
            parameters = new_parameters,
            scatteringRadius = resonance_range.scattering_radius
        )

        return new_resonance_range
