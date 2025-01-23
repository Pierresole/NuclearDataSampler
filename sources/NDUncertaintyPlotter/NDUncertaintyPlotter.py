import ENDFtk
from ENDFtk.Tree import Tape
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from ..NDSampler.resonance.RMatrixLimited.Uncertainty_RML_RRR import Uncertainty_RML_RRR
from ..NDSampler.resonance.RMatrixLimited.Parameters_RML_RRR import SpinGroup

def plot_uncertainty(endf_tape : Tape):
    
    
    tape = Tape.from_file('/home/sole-pie01/ndlib/endfb8-neutron/n-029_Cu_063.endf')# OK

    MATMF2MT151 = tape.MAT(tape.material_numbers[0]).MF(2).MT(151).parse()
    MATMF32MT151 = tape.MAT(tape.material_numbers[0]).MF(32).MT(151).parse()

    isotopeIndex = 0 # Always zero
    rangeIndex = 0 # User defined
    
    latex_column_names = {
        'energy': r'$E$',
        'Cu64 + photon [inclusive] width': r'$\Gamma_\gamma$',
        'n + Cu63 width': r'$\Gamma_n$',
        'n + Cu63 width_2': r'$\Gamma_n2$'
    }
    
    object = Uncertainty_RML_RRR(MATMF2MT151.isotopes[isotopeIndex].resonance_ranges[rangeIndex], 
                                      MATMF32MT151.isotopes[isotopeIndex].resonance_ranges[rangeIndex], 
                                      rangeIndex)

    # Assuming spin_group_data and parameterCov are already defined
    plot_all_spin_groups(object.rml_data.ListSpinGroup, object.covariance_matrix, latex_column_names, show_labels=False)
    
def plot_parameters_with_relative_uncertainty(ax, df, variances, max_params, start_param_index=0, show_y_label=False, show_x_label=False):
    num_resonances = df.shape[0]
    param_index = start_param_index

    for i, column in enumerate(df.columns):
        relative_uncertainties = np.abs(np.sqrt(variances[:, i]) / df[column]) * 100  # Relative uncertainty in percent
        for j in range(num_resonances):
            # print(df[column][j] ,np.sqrt(variances[:, i][j]), relative_uncertainties[j])
            ax[i].bar(j, relative_uncertainties[j], color='blue')
            param_index += 1
        
        if show_y_label and i == 0:  # Ensure y labels are only set for the first column
            ax[i].set_ylabel('Rel. Unc. (%)')
        if show_x_label:
            ax[i].set_xlabel('Resonance Number')
        
        # Add shaded area to indicate unused part of the x-axis
        if num_resonances < max_params:
            ax[i].axvspan(num_resonances - 0.5, max_params - 0.5, color='red', alpha=0.1, hatch='/')

        ax[i].set_xlim(-0.5, max_params - 0.5)
        # ax[i].set_xticks(range(max_params))  # Display all x ticks
        
        # Format y-tick labels
        #ax[i].ticklabel_format(axis='y', scilimits=[-3, 3])
    
    return param_index

def plot_relative_uncertainty(spin_group: SpinGroup, variance_vector: np.ndarray, param_type: int):
    """
    Plots the relative uncertainty vs resonance number.
    param_type = 0 -> resonance energies
    param_type = i > 0 -> width of the i-th channel
    """
    # Gather data
    values = []
    for idx, resonance in enumerate(spin_group.ResonanceParameters):
        if param_type == 0:
            nominal = resonance.ER[0] if resonance.ER else 0.0
        else:
            ch_idx = param_type - 1
            nominal = resonance.GAM[idx][ch_idx] if resonance.GAM and idx < len(resonance.GAM) else 0.0
        values.append(nominal)

    values = np.array(values)
    # Extract the relevant variances
    # (Assuming a simple 1D slice from variance_vector per resonance.)
    # If each resonance has 1 parameter for energies or for i-th width, then:
    uncertainties = np.sqrt(variance_vector[:len(values)])
    # Compute relative (percent) uncertainty
    rel_unc = np.where(values != 0, (uncertainties / np.abs(values))*100.0, 0.0)

    # Plot
    plt.figure()
    plt.bar(range(len(values)), rel_unc, color='blue', alpha=0.7)
    plt.xlabel("Resonance Number")
    plt.ylabel("Relative Uncertainty (%)")
    plt.title("SpinGroup Relative Uncertainty" if param_type == 0 else f"Channel {param_type} Width Relative Unc.")
    plt.show()

def plot_all_spin_groups(object: Uncertainty_RML_RRR, latex_column_names=None, show_labels=True):
    num_groups = len(object.rml_data.ListSpinGroup)
    
    object.rml_data.ListSpinGroup[0].ResonanceChannels.spin
    
    # Find the maximum number of parameters across all groups
    max_params = max([group_data['data'].shape[0] for group_data in spin_group_data.values()])
    num_parameters = 4  # We know each group has 3 parameters

    # Create a grid with an extra row for titles and an extra column for row names
    fig = plt.figure(figsize=(15, 2 * (num_groups + 1)))  # Adjust the 3 to control the overall height
    # The first element in width_ratios=[1.5] + [4] * num_parameters is use to tune the width of the first column
    gs = GridSpec(num_groups + 1, num_parameters + 1, figure=fig, height_ratios=[0.000001] + [1] * num_groups, width_ratios=[2.5] + [4] * num_parameters)
    
    # Set up column titles
    first_group = next(iter(spin_group_data.values()))['data']
    param_names = first_group.columns
    for j, param_name in enumerate(param_names):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.set_title(latex_column_names.get(param_name, param_name))
        ax.axis('off')

    # Set up row names and plots for each spin group
    global_param_index = 0
    for i, (group_label, group_data) in enumerate(spin_group_data.items()):
        df = group_data['data']
        # print("row number ", i)
        # print(df)
        # Extract the diagonal of the covariance matrix
        start_index = sum([spin_group_data[key]['data'].shape[0] * spin_group_data[key]['data'].shape[1] for key in spin_group_data if key < group_label])
        end_index = start_index + (df.shape[1] * df.shape[0])
        variances = np.array(variance_vector[start_index:end_index]).reshape((df.shape[0], df.shape[1]))

        # Set row names
        spin = group_data['spin']
        parity = group_data['parity']
        parity_symbol = "+" if int(parity) > 0 else "-"
        row_name = f"$J^{{\Pi}} = {spin}^{parity_symbol}$"
        row_ax = fig.add_subplot(gs[i + 1, 0])
        row_ax.text(0.5, 0.5, row_name, ha='center', va='center', transform=row_ax.transAxes, fontsize=14)
        row_ax.axis('off')
        
        # Plot parameters
        group_axes = [fig.add_subplot(gs[i + 1, j + 1]) for j in range(df.shape[1])]
        for j in range(df.shape[1]):
            show_y = (j == 1)  # Show y-axis label only for the second column
            show_x = (i == num_groups - 1)  # Show x-axis label only for the last row
            global_param_index = plot_parameters_with_relative_uncertainty(group_axes, df, variances, max_params, global_param_index, show_y, show_x)
    
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    plt.show()
    
    
    
# def plot_spin_groups(range: ENDFtk.MF2.MT151.ResonanceRange, uncertertainty: ENDFtk.MF2.MT151.ResonanceRange):

    
#     for spingroup in range.parameters.spin_groups.to_list():
#         spin_group = SpinGroup(ResonanceChannels=ENDFtk.MF2.MT151.ResonanceChannels(spingroup.channels))
#         def extract_parameters(self, spingroup: ENDFtk.MF2.MT151.SpinGroup):
#             self.ResonanceChannels = ENDFtk.MF2.MT151.ResonanceChannels(spingroup.channels)
#             # Create a single Resonance for this SpinGroup (or multiple if needed)
#             new_resonance = Resonance()
#             new_resonance.extract_parameters(spingroup.parameters)
#             self.ResonanceParameters.append(new_resonance)