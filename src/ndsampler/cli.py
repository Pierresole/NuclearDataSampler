#!/usr/bin/env python3
"""
Command-line interface for Nuclear Data Sampler.

Examples:
    # Sample angular distributions (MF34)
    python -m ndsampler --input n_92-U-235.endf --mf34 --n 100 --output samples/

    # Sample energy distributions (MF35)
    python -m ndsampler --input n_92-U-235.endf --mf35 --mt 18 --n 50

    # Sample resonance parameters (MF32)
    python -m ndsampler --input n_92-U-235.endf --mf32 --n 200

    # Sample multiple sections
    python -m ndsampler --input n_92-U-235.endf --mf34 --mf35 --n 100
"""

import argparse
import sys
import os
from pathlib import Path
import ENDFtk

# Import sampler modules
from ndsampler.angular.Uncertainty_Angular import Uncertainty_Angular
from ndsampler.energy.Uncertainty_Energydist import Uncertainty_Energydist
from ndsampler.resonance.BreitWigner.Uncertainty_BW_URR import Uncertainty_BW_URR
from ndsampler.resonance.ReichMoore.Uncertainty_RM_RRR import Uncertainty_RM_RRR
from ndsampler.multiplicity.Uncertainty_Multiplicity import Uncertainty_Multiplicity


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Sample nuclear data files with uncertainty propagation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input/Output
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input ENDF file path')
    parser.add_argument('-o', '--output', type=str, default='./samples',
                        help='Output directory for sampled files (default: ./samples)')
    
    # Sample count
    parser.add_argument('-n', '--n-samples', type=int, default=100,
                        help='Number of samples to generate (default: 100)')
    
    # File sections to sample
    parser.add_argument('--mf34', action='store_true',
                        help='Sample angular distributions (MF4 with MF34 covariance)')
    parser.add_argument('--mf35', action='store_true',
                        help='Sample energy distributions (MF5 with MF35 covariance)')
    parser.add_argument('--mf32', action='store_true',
                        help='Sample resonance parameters (MF2 with MF32 covariance)')
    parser.add_argument('--mf31', action='store_true',
                        help='Sample multiplicities (MF1 with MF31 covariance)')
    
    # MT selection
    parser.add_argument('--mt', type=int, nargs='+',
                        help='Specific MT reactions to sample (default: all available)')
    
    # Sampling options
    parser.add_argument('--method', type=str, default='Simple',
                        choices=['Simple', 'LHS', 'Sobol'],
                        help='Sampling method (default: Simple)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    # HDF5 options
    parser.add_argument('--save-hdf5', action='store_true',
                        help='Save covariance data to HDF5')
    parser.add_argument('--hdf5-file', type=str, default='covariance_data.hdf5',
                        help='HDF5 filename (default: covariance_data.hdf5)')
    
    # Verbosity
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    return parser.parse_args()


def validate_input_file(filepath):
    """Validate that input file exists and is readable."""
    if not os.path.exists(filepath):
        print(f"ERROR: Input file not found: {filepath}")
        sys.exit(1)
    
    try:
        tape = ENDFtk.tree.Tape.from_file(filepath)
        return tape
    except Exception as e:
        print(f"ERROR: Failed to parse ENDF file: {e}")
        sys.exit(1)


def create_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def sample_angular_distributions(tape, mt_list, n_samples, method, output_dir, verbose):
    """Sample angular distributions (MF4/MF34)."""
    print("\n" + "="*80)
    print("SAMPLING ANGULAR DISTRIBUTIONS (MF4/MF34)")
    print("="*80)
    
    mat = tape.material(tape.material_numbers[0])
    
    # Check if MF4 and MF34 exist
    if not mat.has_MF(4):
        print("WARNING: No MF4 (angular distributions) found in file")
        return
    if not mat.has_MF(34):
        print("WARNING: No MF34 (angular covariance) found in file")
        return
    
    mf4 = mat.MF(4)
    mf34 = mat.MF(34)
    
    # Get available MT reactions
    available_mts = list(mf4.section_numbers)
    
    if mt_list:
        mts_to_process = [mt for mt in mt_list if mt in available_mts]
    else:
        mts_to_process = available_mts
    
    print(f"Processing MT reactions: {mts_to_process}")
    
    for mt in mts_to_process:
        if not mf34.has_MT(mt):
            continue
        
        print(f"\n  Found covariance for MT{mt}")
        print(f"  Generating {n_samples} samples...")

        
        try:
            # Parse sections
            mf4mt = mf4.MT(mt).parse()
            mf34mt = mf34.MT(mt).parse()
            
            # Create uncertainty object
            covariance_objects = []
            Uncertainty_Angular.fill_from_angular_distribution(
                tape, covariance_objects, mt_covariance_dict={mt: None}
            )
            
            if len(covariance_objects) == 0:
                print(f"    ERROR: Failed to create uncertainty object for MT{mt}")
                continue
            
            uncertainty = covariance_objects[0]
            
            # Generate samples
            samples = uncertainty.sample_parameters(
                num_samples=n_samples,
                sampling_method=method,
                mode='replace'
            )
            
            # Write sampled ENDF files
            for sample_idx in range(n_samples):
                output_file = os.path.join(output_dir, f'sample_random{sample_idx}.endf')
                uncertainty.update_tape(tape, sample_index=sample_idx+1)
                tape.to_file(output_file)
            
            print(f"  ✓ Wrote {n_samples} samples to {output_dir}/")
            
        except Exception as e:
            print(f"    ERROR sampling MT{mt}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()


def sample_energy_distributions(tape, mt_list, n_samples, method, output_dir, verbose):
    """Sample energy distributions (MF5/MF35)."""
    print("\n" + "="*80)
    print("SAMPLING ENERGY DISTRIBUTIONS (MF5/MF35)")
    print("="*80)
    
    mat = tape.material(tape.material_numbers[0])
    
    if not mat.has_MF(5):
        print("WARNING: No MF5 (energy distributions) found in file")
        return
    if not mat.has_MF(35):
        print("WARNING: No MF35 (energy covariance) found in file")
        return
    
    mf5 = mat.MF(5)
    mf35 = mat.MF(35)
    
    available_mts = list(mf5.section_numbers)
    
    if mt_list:
        mts_to_process = [mt for mt in mt_list if mt in available_mts]
    else:
        mts_to_process = available_mts
    
    print(f"Processing MT reactions: {mts_to_process}")
    
    for mt in mts_to_process:
        if not mf35.has_MT(mt):
            continue
        
        print(f"\n  Found covariance for MT{mt}")
        print(f"  Generating {n_samples} samples...")

        
        try:
            mf5mt = mf5.MT(mt).parse()
            mf35mt = mf35.MT(mt).parse()
            
            # Create uncertainty object
            uncertainty = Uncertainty_Energydist(mf5mt, mf35mt, mt_number=mt)
            
            # Generate samples
            samples = uncertainty.sample_parameters(
                num_samples=n_samples,
                sampling_method=method,
                mode='replace'
            )
            
            # Write sampled ENDF files
            for sample_idx in range(n_samples):
                output_file = os.path.join(output_dir, f'sample_random{sample_idx}.endf')
                uncertainty.update_tape(tape, sample_index=sample_idx+1)
                tape.to_file(output_file)
            
            print(f"  ✓ Wrote {n_samples} samples to {output_dir}/")
            
        except Exception as e:
            print(f"    ERROR sampling MT{mt}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()


def sample_resonance_parameters(tape, n_samples, method, output_dir, verbose):
    """Sample resonance parameters (MF2/MF32)."""
    print("\n" + "="*80)
    print("SAMPLING RESONANCE PARAMETERS (MF2/MF32)")
    print("="*80)
    
    mat = tape.material(tape.material_numbers[0])
    
    if not mat.has_MF(2):
        print("WARNING: No MF2 (resonance parameters) found in file")
        return
    if not mat.has_MF(32):
        print("WARNING: No MF32 (resonance covariance) found in file")
        return
    
    print("  Generating samples for resonance parameters...")
    
    try:
        # Detect formalism and create appropriate uncertainty object
        # This is a simplified version - you may need to add logic to detect formalism
        
        print("    Note: Resonance sampling requires additional implementation")
        print("    Please use the specific resonance modules directly for now")
        
    except Exception as e:
        print(f"    ERROR: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


def sample_multiplicities(tape, mt_list, n_samples, method, output_dir, verbose):
    """Sample multiplicities (MF1/MF31)."""
    print("\n" + "="*80)
    print("SAMPLING MULTIPLICITIES (MF1/MF31)")
    print("="*80)
    
    mat = tape.material(tape.material_numbers[0])
    
    if not mat.has_MF(1):
        print("WARNING: No MF1 (multiplicities) found in file")
        return
    if not mat.has_MF(31):
        print("WARNING: No MF31 (multiplicity covariance) found in file")
        return
    
    mf1 = mat.MF(1)
    mf31 = mat.MF(31)
    
    available_mts = list(mf1.section_numbers)
    
    if mt_list:
        mts_to_process = [mt for mt in mt_list if mt in available_mts]
    else:
        mts_to_process = available_mts
    
    print(f"Processing MT reactions: {mts_to_process}")
    
    for mt in mts_to_process:
        if not mf31.has_MT(mt):
            continue
        
        print(f"\n  Found covariance for MT{mt}")
        print(f"  Generating {n_samples} samples...")

        
        try:
            mf1mt = mf1.MT(mt).parse()
            mf31mt = mf31.MT(mt).parse()
            
            # Create uncertainty object
            uncertainty = Uncertainty_Multiplicity(mf1mt, mf31mt)
            
            # Generate samples
            samples = uncertainty.sample_parameters(
                num_samples=n_samples,
                sampling_method=method
            )
            
            # Write sampled ENDF files
            for sample_idx in range(n_samples):
                output_file = os.path.join(output_dir, f'sample_random{sample_idx}.endf')
                uncertainty.update_tape(tape, sample_index=sample_idx+1)
                tape.to_file(output_file)
            
            print(f"  ✓ Wrote {n_samples} samples to {output_dir}/")
            
        except Exception as e:
            print(f"    ERROR sampling MT{mt}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()


def main():
    """Main CLI entry point."""
    args = parse_arguments()
    
    print("="*80)
    print("NUCLEAR DATA SAMPLER - CLI")
    print("="*80)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Sampling method: {args.method}")
    if args.seed:
        print(f"Random seed: {args.seed}")
    
    # Validate input
    tape = validate_input_file(args.input)
    output_dir = create_output_directory(args.output)
    
    # Set random seed if provided
    if args.seed:
        import numpy as np
        np.random.seed(args.seed)
    
    # Check if at least one section is requested
    if not (args.mf34 or args.mf35 or args.mf32 or args.mf31):
        print("\nERROR: No sections specified. Use --mf34, --mf35, --mf32, or --mf31")
        sys.exit(1)
    
    # Sample requested sections
    if args.mf34:
        sample_angular_distributions(tape, args.mt, args.n_samples, args.method, 
                                     output_dir, args.verbose)
    
    if args.mf35:
        sample_energy_distributions(tape, args.mt, args.n_samples, args.method,
                                    output_dir, args.verbose)
    
    if args.mf32:
        sample_resonance_parameters(tape, args.n_samples, args.method,
                                    output_dir, args.verbose)
    
    if args.mf31:
        sample_multiplicities(tape, args.mt, args.n_samples, args.method,
                             output_dir, args.verbose)
    
    print("\n" + "="*80)
    print("✓ SAMPLING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
