import sys
import os
import numpy as np
import torch

from ase import Atoms
from ase.io import read, write
from ase.data import vdw_radii
from ase.build import make_supercell
from ase.optimize import BFGS, LBFGS
from ase.filters import UnitCellFilter
from ase.io.trajectory import Trajectory

from time import time
from deep_mdmc import DeepMDMC
from molmod.units import kelvin, bar
from time import time

from utilities import PREOS

from multiprocessing import Pool
import argparse

from utilities import calculate_fugacity_with_coolprop, getBoolStr

torch.set_num_threads(6)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

parser = argparse.ArgumentParser(description="MD-GCMC simulation with DeepMDMC")
# Required arguments
parser.add_argument("-sim_type",
                    type=str,
                    choices=["rigid", "gcmc", "gcmcmd", "tmmcmd"],
                    required=True,
                    metavar="SIM_TYPE",
                    help="Type of simulation to be performed. Choose from 'rigid', 'gcmc', 'gcmcmd', or 'tmmcmd'.")
parser.add_argument("-model_gcmc_path",
                    type=str,
                    required=True,
                    metavar="MODEL_GCMC_PATH",
                    help="Path to the Nequip MLP model to use on the GCMC moves.")
parser.add_argument("-model_md_path",
                    type=str,
                    required=True,
                    metavar="MODEL_MD_PATH",
                    help="Path to the Nequip MLP model to use on the MD simulations.")
parser.add_argument("-struc_path",
                    type=str,
                    required=True,
                    metavar="STRUCTURE_PATH",
                    help="Path to the initial structure as .cif file.")
parser.add_argument("-molecule_path",
                    type=str,
                    required=True,
                    metavar="MOLECULE_PATH",
                    help="Path to the adsorbate molecule as .dat file.")
parser.add_argument("-pressure",
                    type=float,
                    required=True,
                    metavar="PRESSURE",
                    help="Pressure of the gas phase in bar.")
parser.add_argument("-temperature",
                    type=float,
                    required=True,
                    metavar="TEMPERATURE",
                    help="Temperature of the gas phase in K.")
parser.add_argument("-timestep",
                    type=float,
                    required=True,
                    metavar="TIMESTEP",
                    help="Timestep for MD simulation in ps.")
parser.add_argument("-totalsteps",
                    type=int,
                    required=True,
                    metavar="TOTALSTEPS",
                    help="Total number of steps for the complete simulation.")
parser.add_argument("-nmdsteps",
                    type=int,
                    required=True,
                    metavar="NMDSTEPS",
                    help="Number of steps for MD simulation between each GCMC step.")
parser.add_argument("-neqsteps",
                    type=int,
                    required=True,
                    metavar="NEQSTEPS",
                    help="Number of steps for equilibration before GCMC.")
parser.add_argument("-nmcswap",
                    type=int,
                    required=True,
                    metavar="NMCSWAP",
                    help="Average number of GCMC exchanges to attempt every nmdsteps steps.")
parser.add_argument("-nmcmoves",
                    type=int,
                    required=True,
                    metavar="NMCMOVES",
                    help="Average number of GCMC moves to attempt every nmdsteps steps.")
parser.add_argument("-framework_atom_types",
                    type=str,
                    required=True,
                    metavar="FRAMEWORK_ATOM_TYPES",
                    help="Atom types for the framework atoms as a csv strung. Ex.: 'Mg,O,C,H'.")
parser.add_argument("-framework_atom_masses",
                    type=str,
                    required=True,
                    metavar="FRAMEWORK_ATOM_MASSES",
                    help="Masses for the framework atoms as a csv strung. Ex.: '24.3050,15.9994,12.0107,1.00794'.")
parser.add_argument("-adsorbate_atom_types",
                    type=str,
                    required=True,
                    metavar="ADSORBATE_ATOM_TYPES",
                    help="Atom types for the adsorbate atoms as a csv strung. Ex.: 'C,O'.")
parser.add_argument("-adsorbate_atom_masses",
                    type=str,
                    required=True,
                    metavar="ADSORBATE_ATOM_MASSES",
                    help="Masses for the adsorbate atoms as a csv strung. Ex.: '12.0107,15.9994'.")
# Optional arguments
parser.add_argument("-flex_ads",
                    type=str,
                    required=False,
                    default="False",
                    metavar="FLEX_ADS",
                    help="Whether to use flexible adsorbate. Choose from 'True/False' or 'yes/no'.")
parser.add_argument("-opt",
                    type=str,
                    required=False,
                    default="False",
                    help="Whether to perform geometry optimization on the initial structure.'True/False' or 'yes/no'.")
parser.add_argument("-interval",
                    type=int,
                    required=False,
                    default=100,
                    metavar="INTERVAL",
                    help="Interval for printing the simulation infomration on Lammps simulations.")
args = parser.parse_args()


#  torch.set_default_dtype(torch.float32)

#  from nequip.ase.nequip_calculator import NequIPCalculator
#
#  def load_model(model_path):
#      #  Modify species for Mg-MOF-74 (see training yaml file)
#      return NequIPCalculator.from_deployed_model(
#          model_path = model_path, #'./MgF2_nonbonded_v10_nnp1_e10.pth',
#          species_to_type_name = {"C" : "C", "H" : "H", "O" : "O", "Mg" : "Mg"}, device=device,
#          #  set_global_options=True
#      )

# Process command line arguments
args.sim_type = args.sim_type.lower()
args.temperature *= kelvin
pressure = args.pressure * bar

args.flex_ads = getBoolStr(args.flex_ads)
args.opt = getBoolStr(args.opt)

# Preferably run on GPUs
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#  calc_gcmc = load_model(model_gcmc_path)
#  calc_md = load_model(model_md_path)

vdw_radii = vdw_radii.copy()
vdw_radii[1] = 1.0
vdw_radii[6] = 1.0
vdw_radii[8] = 1.25
vdw_radii[12] = 1.25

#  fugacity = calculate_fugacity_with_coolprop("HEOS", "CO2", temperature, pressure)
eos = PREOS.from_name('carbondioxide')
fugacity = eos.calculate_fugacity(args.temperature, pressure)

results_dir = "{}_results_N{}_X{}{}_{}bar_{}K".format(args.sim_type,
                                                      args.nmdsteps,
                                                      args.nmcswap + args.nmcmoves,
                                                      "_flexAds" if args.flex_ads else "",
                                                      args.pressure/bar,
                                                      int(args.temperature))

# Create the results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)


#  atom_type_pairs = {"Mg": 1, "O": 2,  "C": 3, "H": 4}
#  atom_type_pairs = {"Mg": 1, "O": 2,  "C": 3, "H": 4, "C2": 5, "O2": 6}
atom_type_pairs_frame = {atom: [i + 1, float(mass)] for i, (atom, mass) in enumerate(
    zip(args.framework_atom_types.split(","),
        args.framework_atom_masses.split(",")))
    }
atom_type_pairs_ads = {atom: [i + 1 + len(atom_type_pairs_frame), float(mass)] for i, (atom, mass) in enumerate(
    zip(args.adsorbate_atom_types.split(","),
        args.adsorbate_atom_masses.split(",")))
    }

masses = {i + 1: float(mass) for i, mass in enumerate(
    args.framework_atom_masses.split(",") + args.adsorbate_atom_masses.split(","))
    }

tdump = 500 * args.timestep
pdump = 5000 * args.timestep

if args.sim_type == "tmmcmd":
    #  pass
    #  atoms_frame0 = read(struc_path)
    #  replica = [1, 1, 1]
    #  P = [[0, 0, -replica[0]], [0, -replica[1], 0], [-replica[2], 0, 0]]
    #  atoms_frame = make_supercell(atoms_frame, P)
    #  write("frame0.extxyz", atoms_frame)

    atoms_ads = read(args.molecule_path)
    vdw_radii = vdw_radii.copy()
    # Mg radius is set to 1.0 A
    vdw_radii[12] = 1.0
    atoms_frame0 = read("frame0.extxyz")
    atoms_frame = read("loaded_frame1.extxyz")
    Z_ads = int((len(atoms_frame) - len(atoms_frame0)) / len(atoms_ads))
    #  print(N_ads)
    #  quit()
    deep_mdmc = DeepMDMC(
        args.model_gcmc_path,
        args.model_md_path,
        results_dir,
        args.interval,
        atoms_frame,
        atoms_ads,
        args.flex_ads,
        Z_ads,
        args.temperature,
        pressure,
        fugacity,
        device,
        vdw_radii
        )

    #  deep_mdmc.init_gcmc()
    deep_mdmc.init_md(args.timestep,
                      atom_type_pairs_frame,
                      atom_type_pairs_ads,
                      units_lmp="metal",
                      tdump=tdump,
                      pdump=pdump,
                      md_type="npt",
                      opt=False,
                      equ_steps=0)
    deep_mdmc.run_tmmcmd(nmdsteps=180000)

    # Exit after TMMCMD
    sys.exit(0)

else:
    atoms_frame = read(args.struc_path)
    replica = [1, 1, 1]
    P = [[0, 0, -replica[0]], [0, -replica[1], 0], [-replica[2], 0, 0]]
    atoms_frame = make_supercell(atoms_frame, P)
    #  if opt:
    #      atoms_frame.calc = calc_md
    #      #  write("framebeforeopt.cif", atoms_frame)
    #      # geom opt frame based on model
    #      # to account relaxation of cell
    #      #  traj = Trajectory('frames.traj', 'w', atoms_frame)
    #      ucf = UnitCellFilter(atoms_frame)
    #      optimizer = LBFGS(ucf)
    #      #  optimizer.attach(traj)
    #      optimizer.run(fmax=0.001)
    write("frame0.extxyz", atoms_frame)
    #  write("frame0.cif", atoms_frame)
    #  atoms_frame = read("frame0.cif")
    #  quit()
    # C and O were renamed to Co and Os to differentiate them from framework atoms during training
    #  atoms_ads = read('./co2_v2.xyz')
    atoms_ads = read(args.molecule_path)
    Z_ads = 0
    deep_mdmc = DeepMDMC(
        args.model_gcmc_path,
        args.model_md_path,
        results_dir,
        args.interval,
        atoms_frame,
        atoms_ads,
        args.flex_ads,
        Z_ads,
        args.temperature,
        pressure,
        fugacity,
        device,
        vdw_radii
        )


if args.nmdsteps == 0:
    args.sim_type = "rigid"

if args.sim_type.lower() == "rigid":
    deep_mdmc.init_gcmc()
    deep_mdmc.run_gcmc(args.nmcswap, args.nmcmoves)

elif args.sim_type.lower() == "gcmcmd":

    deep_mdmc.init_md(args.timestep,
                      atom_type_pairs_frame,
                      atom_type_pairs_ads,
                      units_lmp="metal",
                      tdump=tdump,
                      pdump=pdump,
                      md_type="npt",
                      opt=True,
                      equ_steps=args.neqsteps)

    deep_mdmc.init_gcmc()
    deep_mdmc.run_gcmcmd(args.totalsteps, args.nmdsteps, args.nmcswap, args.nmcmoves)