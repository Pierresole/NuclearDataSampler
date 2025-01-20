from dataclasses import dataclass, field
from typing import List, Union

@dataclass
class TemperatureDependentData:
    temp: float
    reused: bool
    delta: Union[float, None] = None
    ni: Union[int, None] = None
    rho_values: List[float] = field(default_factory=list)
    twt: Union[float, None] = None
    c: Union[float, None] = None
    tbeta: Union[float, None] = None
    nd: int = 0
    osc_energies: List[float] = field(default_factory=list)
    osc_weights: List[float] = field(default_factory=list)

@dataclass
class LeaprInterface:
    nout: int
    title: str
    ntempr: int
    iprint: int
    nphon: int
    mat: int
    za: int
    isabt: int
    ilog: int
    smin: float
    awr: float
    spr: float
    npr: int
    iel: int
    ncold: int
    nsk: int
    nss: int
    b7: int
    aws: float
    sps: float
    mss: int
    nalpha: int
    nbeta: int
    lat: int
    alpha_values: List[float]
    beta_values: List[float]
    temp_parameters: List[TemperatureDependentData] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str):
        data = {}
        with open(path, "r") as input_file:
            input_lines = input_file.readlines()

        # Clean lines: remove comments and newlines, stop at the 'stop' line
        input_lines = [
            line.split("/")[0].strip() for line in input_lines if not line.startswith("stop")
        ]

        if input_lines[0] != "leapr":
            raise ValueError("Invalid input file format.")

        # Populate the data dictionary
        data["nout"] = int(input_lines[1])
        data["title"] = input_lines[2]

        # Move parse_run_parameters, parse_alpha_beta_values, parse_temperature_dependent_parameters here
        cls._parse_run_parameters(data, input_lines)
        line_index = cls._parse_alpha_beta_values(data, input_lines, start_index=8)
        cls._parse_temperature_dependent_parameters(data, input_lines, line_index)

        return cls(
            nout=data["nout"],
            title=data["title"],
            ntempr=data["ntempr"],
            iprint=data["iprint"],
            nphon=data["nphon"],
            mat=data["mat"],
            za=data["za"],
            isabt=data["isabt"],
            ilog=data["ilog"],
            smin=data["smin"],
            awr=data["awr"],
            spr=data["spr"],
            npr=data["npr"],
            iel=data["iel"],
            ncold=data["ncold"],
            nsk=data["nsk"],
            nss=data["nss"],
            b7=data["b7"],
            aws=data["aws"],
            sps=data["sps"],
            mss=data["mss"],
            nalpha=data["nalpha"],
            nbeta=data["nbeta"],
            lat=data["lat"],
            alpha_values=data["alpha_values"],
            beta_values=data["beta_values"],
            temp_parameters=[
                TemperatureDependentData(**temp) for temp in data["temp_parameters"]
            ],
        )

    @staticmethod
    def _parse_run_parameters(data, lines):
        current_line = lines[3].split()
        data["ntempr"] = int(current_line[0])
        data["iprint"] = 1  # Setting default verbosity
        data["nphon"] = int(current_line[2])

        current_line = lines[4].split()
        data.update({
            "mat": int(current_line[0]),
            "za": int(current_line[1]),
            "isabt": int(current_line[2]),
            "ilog": int(current_line[3]),
            "smin": float(current_line[4]),
        })

        current_line = lines[5].split()
        data.update({
            "awr": float(current_line[0]),
            "spr": float(current_line[1]),
            "npr": int(current_line[2]),
            "iel": int(current_line[3]),
            "ncold": int(current_line[4]),
            "nsk": int(current_line[5]),
        })

        current_line = lines[6].split()
        data.update({
            "nss": int(current_line[0]),
            "b7": int(current_line[1]),
            "aws": float(current_line[2]),
            "sps": float(current_line[3]),
            "mss": int(current_line[4]),
        })

        current_line = lines[7].split()
        data.update({
            "nalpha": int(current_line[0]),
            "nbeta": int(current_line[1]),
            "lat": int(current_line[2])
        })

    @staticmethod
    def _parse_alpha_beta_values(data, lines, start_index):
        line_index = start_index
        alpha_values = []
        nalpha = data["nalpha"]
        nbeta = data["nbeta"]

        # Read alpha values sequentially
        while len(alpha_values) < nalpha:
            current_line = lines[line_index].split()
            alpha_values.extend(map(float, current_line[:nalpha - len(alpha_values)]))
            line_index += 1
        data["alpha_values"] = alpha_values

        # Read beta values sequentially
        beta_values = []
        while len(beta_values) < nbeta:
            current_line = lines[line_index].split()
            beta_values.extend(map(float, current_line[:nbeta - len(beta_values)]))
            line_index += 1
        data["beta_values"] = beta_values

        return line_index

    @staticmethod
    def _parse_temperature_dependent_parameters(data, lines, line_index):
        ntempr = data["ntempr"]
        data["temp_parameters"] = []

        for _ in range(ntempr):
            temp_info = {}
            current_line = lines[line_index].split()
            temp = float(current_line[0])
            temp_info["temp"] = abs(temp)
            temp_info["reused"] = temp < 0
            line_index += 1

            if temp >= 0:
                # Parse rho grid parameters
                current_line = lines[line_index].split()
                temp_info.update({"delta": float(current_line[0]), "ni": int(current_line[1])})
                line_index += 1

                # Parse rho values
                rho_values = []
                while len(rho_values) < temp_info["ni"]:
                    current_line = lines[line_index].split()
                    rho_values.extend(map(float, current_line[:temp_info["ni"] - len(rho_values)]))
                    line_index += 1
                temp_info["rho_values"] = rho_values

                # Parse continuous distribution parameters
                current_line = lines[line_index].split()
                temp_info.update({
                    "twt": float(current_line[0]),
                    "c": float(current_line[1]),
                    "tbeta": float(current_line[2]),
                })
                line_index += 1

                # Parse discrete oscillator parameters if present
                current_line = lines[line_index].split()
                temp_info["nd"] = int(current_line[0])
                line_index += 1

                if temp_info["nd"] > 0:
                    temp_info["osc_energies"] = list(map(float, lines[line_index].split()))
                    line_index += 1
                    temp_info["osc_weights"] = list(map(float, lines[line_index].split()))
                    line_index += 1

            data["temp_parameters"].append(temp_info)

    def write_to_file(self, output_path: str):
        """
        Writes LeaprInterface to a LEAPR inpu format.
        :param leapr: Optional LEAPRInput object to write.
        :param output_path: Path to the output file.
        """

        with open(output_path, "w") as output_file:
            # Write the fixed parts of the file
            output_file.write("leapr\n")
            output_file.write(f"{self.nout} / NOUT\n")
            output_file.write(f"{self.title} / TITLE\n")
            output_file.write(f"{self.ntempr} {self.iprint} {self.nphon} / NTEMPR IPRINT NPHON\n")
            output_file.write(f"{self.mat} {self.za} {self.isabt} {self.ilog} {self.smin} / MAT ZA ISABT ILOG SMIN\n")
            output_file.write(f"{self.awr} {self.spr} {self.npr} {self.iel} {self.ncold} {self.nsk} / AWR SPR NPR IEL NCOLD NSK\n")
            output_file.write(f"{self.nss} {self.b7} {self.aws} {self.sps} {self.mss} / NSS B7 AWS SPS MSS\n")
            output_file.write(f"{self.nalpha} {self.nbeta} {self.lat} / NALPHA NBETA LAT\n")

            # Write alpha and beta values
            for i in range(0, len(self.alpha_values), 5):
                output_file.write(" ".join(map(str, self.alpha_values[i:i + 5])) + " ")
            output_file.write("/ end of alpha grid \n")
            for i in range(0, len(self.beta_values), 5):
                output_file.write(" ".join(map(str, self.beta_values[i:i + 5])) + " ")
            output_file.write("/ end of beta grid \n")

            # Write temperature-dependent parameters
            for temp_data in self.temp_parameters:
                temp = -temp_data.temp if temp_data.reused else temp_data.temp
                output_file.write(f"{temp} / temperature (K)\n")

                if not temp_data.reused:
                    output_file.write(f"{temp_data.delta} {temp_data.ni} / frequency distribution: DELTA NI\n")
                    for i in range(0, len(temp_data.rho_values), 5):
                        output_file.write(" ".join(map(str, temp_data.rho_values[i:i + 5])) + " ")
                    output_file.write("/ end of frequency spectrum \n")
                    output_file.write(f"{temp_data.twt} {temp_data.c} {temp_data.tbeta} / TWT C TBETA\n")
                    output_file.write(f"{temp_data.nd} / ND\n")
                    if temp_data.nd > 0:
                        output_file.write(" ".join(map(str, temp_data.osc_energies)) + " / oscillator energies\n")
                        output_file.write(" ".join(map(str, temp_data.osc_weights)) + " / oscillator weights\n")

            # Add the "stop" keyword
            output_file.write("/ end leapr\n")
            output_file.write("stop\n")
            
    def Spectrum(self, temperature: float):
        """
        Retrieves the energy grid and rho values for a specified temperature.
        :param temperature: The temperature for which the spectrum is retrieved.
        :return: Tuple (energy_grid, rho_values).
        """
        for temp_data in self.temp_parameters:
            if temp_data.temp == temperature:
                delta = temp_data.delta
                ni = temp_data.ni
                rho_values = temp_data.rho_values
                energy_grid = [delta * i for i in range(ni)]
                return energy_grid, rho_values
        raise ValueError(f"No data found for temperature {temperature}.")