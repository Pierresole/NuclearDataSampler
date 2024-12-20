import numpy as np
import ENDFtk
from ENDFtk.MF3 import Section
from bisect import bisect_left, bisect_right


# Define XSInterpolator class with AddRange
class XSInterpolator:
    def __init__(self, section : Section):
        # Copy attributes from the section without modifying the original
        self.mt = section.MT
        self.zaid = section.ZA
        self.lr = section.LR
        self.awr = section.AWR
        self.qm = section.QM
        self.qi = section.QI
        self.energies = section.energies[:]
        self.cross_sections = section.cross_sections[:]
        self.boundaries = [b - 1 for b in section.boundaries]
        self.interpolants = section.interpolants[:]

    def InterpolateAtEnergy(self, e):
        # Determine which range the energy falls into based on boundaries
        for i, boundary in enumerate(self.boundaries):
            if e < self.energies[boundary]:
                interp_type = self.interpolants[i]
                start_index = 0 if i == 0 else self.boundaries[i - 1]
                end_index = boundary
                break
        else:
            interp_type = self.interpolants[-1]
            start_index = self.boundaries[-1]
            end_index = len(self.energies)
        
        # Get the relevant energies and cross sections for interpolation
        x_vals = self.energies[start_index:end_index]
        y_vals = self.cross_sections[start_index:end_index]

        # Perform interpolation based on the interp_type
        if interp_type == 1:  # Constant in x (histogram)
            for j in range(len(x_vals) - 1):
                if x_vals[j] <= e < x_vals[j + 1]:
                    return [y_vals[j], e]
            return [y_vals[-1], e]  # if e is at the last energy boundary

        elif interp_type == 2:  # Linear in x (linear-linear)
            f = np.interp(e, x_vals, y_vals)
            return [f, e]

        elif interp_type == 3:  # Linear in ln(x) (linear-log)
            f = np.interp(np.log(e), np.log(x_vals), y_vals)
            return [f, e]

        elif interp_type == 4:  # ln(y) is linear in x (log-linear)
            log_y_vals = np.log(y_vals)
            f_log = np.interp(e, x_vals, log_y_vals)
            return [np.exp(f_log), e]

        elif interp_type == 5:  # ln(y) is linear in ln(x) (log-log)
            log_x_vals = np.log(x_vals)
            log_y_vals = np.log(y_vals)
            f_log = np.interp(np.log(e), log_x_vals, log_y_vals)
            return [np.exp(f_log), e]

        else:
            raise ValueError(f"Unsupported interpolation type: {interp_type}")

    def AddRange(self, interp, energies_new, xs_new):
            energies_new = list(energies_new)
            xs_new = list(xs_new)
            a = energies_new[0]
            b = energies_new[-1]

            # Find insertion indices
            start_index = bisect_left(self.energies, a)
            end_index = bisect_right(self.energies, b)

            # Build new energies and xs
            new_energies = []
            new_xs = []

            # Part before new range
            new_energies.extend(self.energies[:start_index])
            new_xs.extend(self.cross_sections[:start_index])

            # Insert duplicate at start
            if start_index < len(self.energies):
                new_energies.append(self.energies[start_index])
                new_xs.append(self.cross_sections[start_index])
            else:
                new_energies.append(self.energies[-1])
                new_xs.append(self.cross_sections[-1])

            # New range
            new_energies.extend(energies_new)
            new_xs.extend(xs_new)

            # Insert duplicate at end
            if end_index < len(self.energies):
                new_energies.append(self.energies[end_index - 1])
                new_xs.append(self.cross_sections[end_index - 1])
            else:
                new_energies.append(self.energies[-1])
                new_xs.append(self.cross_sections[-1])

            # Part after new range
            new_energies.extend(self.energies[end_index:])
            new_xs.extend(self.cross_sections[end_index:])

            # Compute boundaries
            boundary1 = start_index + 1  # After the duplicate at the start
            boundary2 = boundary1 + len(energies_new)
            boundary3 = len(new_energies) - 1

            print(new_energies)
            print(self.boundaries)
            print([boundary1, boundary2, boundary3])
            print(self.interpolants)
            print([self.interpolants[0], interp, self.interpolants[-1]])
            self.boundaries = [boundary1, boundary2, boundary3]
            self.interpolants = [self.interpolants[0], interp, self.interpolants[-1]]
            
            self.energies = new_energies
            self.cross_sections = new_xs

    def GetLinearizedDataInRange(self, a, b, tol=1e-3, max_points=1000):
        """
        Returns energies and cross sections between a and b such that linear interpolation between
        them reproduces the cross section according to the original Section's interpolation schemes.
        """
        energies = []
        cross_sections = []

        # Ensure a and b are within the energies range
        if a < self.energies[0] or b > self.energies[-1]:
            raise ValueError("Requested range is outside the energies in the Section.")

        # Find overlapping ranges
        ranges = []
        for i, boundary in enumerate(self.boundaries):
            range_start = self.energies[0] if i == 0 else self.energies[self.boundaries[i - 1] + 1]
            range_end = self.energies[boundary]
            if range_end < a or range_start > b:
                continue  # No overlap
            else:
                # Overlapping range
                overlap_start = max(range_start, a)
                overlap_end = min(range_end, b)
                ranges.append((i, overlap_start, overlap_end))

        # Generate points for each overlapping range
        for i, start_e, end_e in ranges:
            interp_type = self.interpolants[i]
            # For simplicity, we will generate points adaptively
            e_range, xs_range = self._adaptive_sampling(start_e, end_e, interp_type, tol, max_points)
            energies.extend(e_range)
            cross_sections.extend(xs_range)

        # Sort energies and cross_sections
        sorted_indices = np.argsort(energies)
        energies = np.array(energies)[sorted_indices].tolist()
        cross_sections = np.array(cross_sections)[sorted_indices].tolist()

        return energies, cross_sections

    def _adaptive_sampling(self, start_e, end_e, interp_type, tol, max_points):
        """
        Perform adaptive sampling within the interval [start_e, end_e] to approximate the
        interpolation scheme with linear segments within a specified tolerance.
        """
        # Initialize with start and end points
        energies = [start_e, end_e]
        cross_sections = [self.InterpolateAtEnergy(start_e)[0], self.InterpolateAtEnergy(end_e)[0]]

        # Start adaptive sampling
        points_to_check = [(0, 1)]  # Indices in energies list
        while points_to_check and len(energies) < max_points:
            i, j = points_to_check.pop()
            mid_e = (energies[i] + energies[j]) / 2
            mid_xs = self.InterpolateAtEnergy(mid_e)[0]

            # Linear interpolation between energies[i] and energies[j]
            lin_xs = cross_sections[i] + (cross_sections[j] - cross_sections[i]) * \
                     (mid_e - energies[i]) / (energies[j] - energies[i])

            # Check if the difference is within tolerance
            if abs(mid_xs - lin_xs) > tol * max(abs(mid_xs), 1e-10):
                # Insert mid-point
                energies.insert(j, mid_e)
                cross_sections.insert(j, mid_xs)
                # Add new intervals to check
                points_to_check.append((i, j))
                points_to_check.append((j, j + 1))
                # Sort points_to_check based on interval size (optional)
            # Else, no need to insert; the linear approximation is good enough

        return energies, cross_sections
    
    def create_new_section(self):
        # Convert boundaries back to the original format by shifting by +1
        boundaries = [b + 1 for b in self.boundaries]
        return Section(
            mt=self.mt,
            zaid=self.zaid,
            lr=self.lr,
            awr=self.awr,
            qm=self.qm,
            qi=self.qi,
            interpolants=self.interpolants,
            boundaries=boundaries,
            energies=self.energies,
            xs=self.cross_sections
        )
