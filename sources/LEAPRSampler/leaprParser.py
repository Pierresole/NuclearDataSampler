from .LeaprInterface import LeaprInterface, TemperatureDependentData
from scipy.stats import qmc
from scipy.optimize import curve_fit
import numpy as np
import random
from copy import deepcopy
from pyDOE3 import lhs
import os
import glob
import subprocess
import multiprocessing

# LeaprInterface : parses, stores and writes LEAPR input.
# PerturbLeaprInput : - perturbs LEAPR input and generates 'N' perturbed files.
#                     - interfaces with LEAPR/THERMR to reconstruct cross sections (parallel or sequential).

class PerturbLeaprInput:
    def __init__(self, path: str, perturb_params: dict, sampling_type: str = "LHS"):
        self.path = path # Path to the LEAPR input file
        self.perturb_params = perturb_params # How and which paremeters to perturb
        self.sampling_type = sampling_type
        self.authorized_perturbations = {
            "twt_perturb": {"distribution": "uniform", "range": (-0.3, 0.3)},
            "c_perturb": {"distribution": "uniform", "range": (-0.3, 0.3)},
            # ...existing or additional authorized parameters...
        }
        self.leapr_input = LeaprInterface.from_file(self.path)  # Use from_file method

    def perturb_all(self, n_files: int, output_template: str):
        """
        Generates multiple perturbed LEAPR files using Sobol or LHS sampling.
        :param n_files: Number of perturbed files to generate.
        :param output_template: Template for output filenames (e.g., "perturbed_{}.leapr").
        """
        if self.sampling_type == "Sobol":
            # Initialize Sobol sampler
            total_oscillators = len(self.perturb_params.get("osc_energies_perturb", []))
            sampler = qmc.Sobol(d=total_oscillators, scramble=True)
            samples = sampler.random_base2(m=int(np.log2(n_files)))
        elif self.sampling_type == "LHS":
            # Constrained LHS for TWT, WOsc1, WOsc2
            if "twt_perturb" in self.perturb_params and "osc_weights_perturb" in self.perturb_params:
                twt_range = self.perturb_params["twt_perturb"]["range"]
                osc_weights_range = self.perturb_params["osc_weights_perturb"]["range"]
                tbeta = self.perturb_params.get("tbeta", 0.2)  # Default TBETA value
                variable_ranges = [twt_range, osc_weights_range, osc_weights_range]

                lhs_samples = lhs(2, samples=n_files)  # Generate LHS for TWT and WOsc1
                TWT = lhs_samples[:, 0] * (twt_range[1] - twt_range[0]) + twt_range[0]
                WOsc1 = lhs_samples[:, 1] * (osc_weights_range[1] - osc_weights_range[0]) + osc_weights_range[0]
                WOsc2 = 1 - tbeta - TWT - WOsc1

                # Validate WOsc2 range
                if np.any(WOsc2 < osc_weights_range[0]) or np.any(WOsc2 > osc_weights_range[1]):
                    raise ValueError("Some WOsc2 values are out of range. Adjust variable_ranges or TBETA.")
            else:
                samples = lhs(len(self.perturb_params.get("osc_energies_perturb", [])), samples=n_files)
        else:
            raise ValueError(f"Unsupported sampling type: {self.sampling_type}")

        # Create the "random" folder if it doesn't exist, delete if exists
        if os.path.exists("random"):
            subprocess.run(["rm", "-rf", "random"])
            os.makedirs("random")

        for i in range(n_files):
            perturbed_data = deepcopy(self.leapr_input)

            # Perturb all temperature dependant parameters of a LEAPR input
            for temp_data in perturbed_data.temp_parameters:
                # Perturb oscillator energies
                if "osc_energies_perturb" in self.perturb_params:
                    for j, osc_perturb in enumerate(self.perturb_params["osc_energies_perturb"]):
                        osc_number = osc_perturb["osc_number"] - 1  # Convert to 0-based index
                        perturb_range = osc_perturb["range"]

                        if 0 <= osc_number < len(temp_data.osc_energies):
                            if self.sampling_type == "Sobol":
                                # Map Sobol sample to the specified range
                                temp_data.osc_energies[osc_number] = (
                                    perturb_range[0] + samples[i, j] * (perturb_range[1] - perturb_range[0])
                                )
                            elif self.sampling_type == "LHS":
                                # Map LHS sample to the specified range
                                temp_data.osc_energies[osc_number] = (
                                    perturb_range[0] + samples[i, j] * (perturb_range[1] - perturb_range[0])
                                )

                # Perturb other parameters (e.g., TWT, C) using uniform random sampling
                if "twt_perturb" in self.perturb_params:
                    if self.sampling_type == "Sobol":
                        temp_data.twt *= 1 + np.random.uniform(*self.perturb_params["twt_perturb"]["range"])
                    elif self.sampling_type == "LHS":
                        temp_data.twt = TWT[i]
                if "c_perturb" in self.perturb_params:
                    temp_data.c *= 1 + np.random.uniform(*self.perturb_params["c_perturb"]["range"])

                # Apply constrained LHS sampling to oscillator weights
                if self.sampling_type == "LHS" and "osc_weights_perturb" in self.perturb_params:
                    if len(temp_data.osc_weights) >= 2:
                        temp_data.osc_weights[0] = WOsc1[i]
                        temp_data.osc_weights[1] = WOsc2[i]

                # Perturb the spectrum
                if "spectrum" in self.perturb_params:
                    self.perturb_spectrum(temp_data)

            # Write the perturbed file
            output_file = os.path.join("random", output_template.format(f"random{i+1}"))
            perturbed_data.write_to_file(output_file)
    

    ###################################################
    #  PARALLEL PROCESSING (WITH GAIA but adaptable)  #
    ###################################################
    
    def detect_slurm(self):
        """
        Detects if the platform supports SLURM by checking for 'sbatch' or relevant SLURM environment variables.
        """
        try:
            subprocess.run(["sbatch", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return "SLURM_CPUS_ON_NODE" in os.environ

    def get_available_cpus(self):
        """
        Returns the number of available CPUs on the machine.
        """
        try:
            if self.detect_slurm():
                return int(os.environ.get("SLURM_CPUS_ON_NODE", multiprocessing.cpu_count()))
            else:
                return multiprocessing.cpu_count()
        except Exception as e:
            print(f"Error detecting CPUs: {e}")
            return 1  # Default to 1 CPU in case of error
        
    def process(self, gaiapath: str, njoypath: str, endfpath: str, cpus_per_job=3):
        """
        Submits a single SLURM job or runs locally, checking for required executables.
        Launches Gaia with a provided input file.
        """
        # Check if gaiapath and njoypath exist
        if not os.path.isfile(gaiapath):
            raise FileNotFoundError(f"Gaia executable not found at: {gaiapath}")
        if not os.path.isfile(njoypath):
            raise FileNotFoundError(f"NJOY executable not found at: {njoypath}")

        if os.path.exists("processing"):
            subprocess.run(["rm", "-rf", "processing"])
        os.makedirs("processing")
            
        # Create a symbolic link to the ENDF file
        os.makedirs("processing/endf", exist_ok=True)
        endf_link_path = os.path.join("processing/endf", os.path.basename(endfpath))
        os.symlink(endfpath, endf_link_path)
        
        # Prepare the Gaia input file
        gaia_input_content = f"""gaia2:
    - title: test_tsl
      endf: endf
      leapr: ../random
      njoy: {njoypath}
      modules:
          leapr:
          dop:
          thermr:
          acer:
"""
        gaia_input_file = "processing/gaia_input.txt"
        with open(gaia_input_file, "w") as f:
            f.write(gaia_input_content)
        print(f"Generated Gaia input file: {gaia_input_file}")

        # Check for SLURM
        if self.detect_slurm():
            print("SLURM detected. Submitting a single job to SLURM...")
            sbatch_command = [
                "sbatch",
                f"--cpus-per-task={cpus_per_job}",
                "--wrap", f"mpirun -n {cpus_per_job} {gaiapath} gaia_input.txt"
            ]
            try:
                subprocess.run(sbatch_command, check=True, cwd="processing")
                print(f"Submitted job with Gaia input file: gaia_input.txt")
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job: {e}")
        else:
            print("SLURM not detected. Running locally...")
            try:
                subprocess.run(
                    ["mpirun", "-n", str(cpus_per_job), gaiapath, "gaia_input.txt"],
                    check=True,
                    cwd="processing"
                )
                print(f"Gaia executed successfully with input file: gaia_input.txt")
            except subprocess.CalledProcessError as e:
                print(f"Error running Gaia locally: {e}")

    
    #########################################
    #  FUNCTIONS FOR SPECTRUM PERTURBATION  #
    #########################################
            
    def fit_spectrum(self, E_values, rho_values):
        """
        Fits the spectrum to a sum of Gaussians.
        :param E_values: List of energy values (x-axis).
        :param rho_values: Corresponding density values (y-axis).
        :return: Optimal Gaussian parameters (popt).
        """
        # Initial guess for 6 Gaussians
        p0 = [
            2, 0.005, 0.003,  # Gaussian 1: Amplitude, Mean, Std Dev
            2, 0.015, 0.005,  # Gaussian 2
            2, 0.025, 0.008,  # Gaussian 3
            8, 0.060, 0.015,  # Gaussian 4
            4, 0.090, 0.020,  # Gaussian 5
            2, 0.145, 0.010   # Gaussian 6
        ]

        # Bounds for parameters
        bounds = (
            [  # Lower bounds
                0, 0.001, 0.001,
                0, 0.005, 0.001,
                0, 0.015, 0.005,
                2, 0.030, 0.008,
                1, 0.060, 0.010,
                0, 0.110, 0.005
            ],
            [  # Upper bounds
                4,  0.010, 0.006,
                4,  0.020, 0.010,
                6,  0.030, 0.015,
                12, 0.070, 0.030,
                8,  0.100, 0.030,
                3,  0.180, 0.050
            ]
        )

        # Mask energy values above 1 meV (0.001 eV)
        min_energy = 0.001
        mask = [value >= min_energy for value in E_values]

        # Apply mask to E_values and rho_values
        masked_E_values = [E_values[i] for i in range(len(E_values)) if mask[i]]
        masked_rho_values = [rho_values[i] for i in range(len(rho_values)) if mask[i]]

        # Fit the sum of Gaussians to the masked data
        popt, pcov = curve_fit(
            self.sum_of_gaussians,  # Function to fit
            masked_E_values,        # Filtered x values
            masked_rho_values,      # Filtered y values
            p0=p0,                  # Initial guess
            bounds=bounds           # Parameter bounds
        )

        return popt

    def derivative_sum_of_gaussians(self, x, *params):
        """
        Derivative of the sum of Gaussians.
        :param x: Energy values.
        :param params: Flattened list of Gaussian parameters.
        :return: Derivative of the summed Gaussian value at x.
        """
        n = len(params) // 3  # Number of Gaussians
        result = np.zeros_like(x)
        for i in range(n):
            a = params[3 * i]
            b = params[3 * i + 1]
            c = params[3 * i + 2]
            result += a * np.exp(-((x - b)**2) / (2 * c**2)) * (b - x) / (c**2)
        return result

    @staticmethod
    def sum_of_gaussians(x, *params):
        """
        Sum of multiple Gaussian functions.
        :param x: Input x values.
        :param params: Flattened list of Gaussian parameters (a, b, c for each Gaussian).
        :return: Sum of Gaussians evaluated at x.
        """
        n = len(params) // 3
        result = np.zeros_like(x)
        for i in range(n):
            a = params[3 * i]
            b = params[3 * i + 1]
            c = params[3 * i + 2]
            result += a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
        return result
    
    def adjusted_sum_of_gaussians(self, x, *params):
        """
        Adjusts the sum of Gaussians to transition to a quadratic form below 3 meV.
        :param x: Energy value.
        :param params: Flattened list of Gaussian parameters.
        :return: Adjusted value of the sum of Gaussians.
        """
        if x >= 0.003:  # 3 meV
            return self.sum_of_gaussians(x, *params)
        else:
            val_2ev = self.sum_of_gaussians(0.003, *params)
            deriv_2ev = self.derivative_sum_of_gaussians(0.003, *params)

            # Compute coefficients for the quadratic form
            a = deriv_2ev / 0.003 - val_2ev / (0.003**2)
            b = 2 * val_2ev / 0.003 - deriv_2ev

            return a * x**2 + b * x

    def perturb_spectrum(self, params: TemperatureDependentData):
        """
        Perturbs the phonon spectrum by modifying Gaussian parameters and transitioning
        to a quadratic form below 3 meV.
        :param params: Temperature-dependent parameters containing `delta`, `ni`, and `rho_values`.
        """
        delta = params.delta
        ni = params.ni
        E_values = [delta * i for i in range(ni)]
        rho_values = params.rho_values

        # Fit the spectrum
        gaussians_parameters = self.fit_spectrum(E_values, rho_values)

        # Identify main and second Gaussian indices
        index_main_gaussian_mean = 3 * 3 + 1  # Fourth Gaussian mean
        index_second_gaussian_mean = index_main_gaussian_mean + 3  # Fifth Gaussian mean

        # Compute the shift between the spectrum maximum and the main Gaussian mean
        index_max_rho = np.argmax(rho_values)
        shift_main = E_values[index_max_rho] - gaussians_parameters[index_main_gaussian_mean]
        shift_second = E_values[index_max_rho] - gaussians_parameters[index_second_gaussian_mean]

        # Perturb the main Gaussian mean
        old_value_main = gaussians_parameters[index_main_gaussian_mean]
        min_value_main = self.perturb_params["spectrum"][0]["gauss_min"] * 1e-3 - shift_main
        max_value_main = self.perturb_params["spectrum"][0]["gauss_max"] * 1e-3 - shift_main
        new_value_main = np.random.uniform(min_value_main, max_value_main)
        gaussians_parameters[index_main_gaussian_mean] = new_value_main

        # Recalculate the spectrum with perturbed Gaussian parameters
        new_rho_values = [
            self.adjusted_sum_of_gaussians(E_value, *gaussians_parameters) for E_value in E_values
        ]

        # Update the params with the perturbed spectrum
        params.rho_values = new_rho_values

