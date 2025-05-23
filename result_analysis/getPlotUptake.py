
from ase.io import read
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os



BAR2PASCAL = 1e5
N_A = 6.022e+23  # molecules per mole


def calcRhoCoolProp(pressure, temperature, M_fluid, fluid="CO2"):
    import CoolProp.CoolProp as CP
    # in mol/m^3
    return CP.PropsSI("D", "T", temperature, "P", pressure, fluid) / M_fluid


def calcBulkFluid(rho_bulk, V_void, frame_mass):

    """

    Parameters:
    - rho_bulk (float): density of bulk fluid (in mol/m^3).
    - V_void (float): Volume of void in solid (in m^3).

    Returns:
    - amount_bulk (float): (in mmol/g).
    """

    n_bulk = rho_bulk * V_void * 1000  # mmol
    return n_bulk * N_A / frame_mass # in mmol/g



def getUptake(file_path, plot=False):

    if file_path.endswith(".exyxyz"):
        # from gas loaded exyxyz file
        fl_base = file_path.split("/")[-1].replace(".extxyz", "")
        atoms_list = read(extxyz_path, index=":")
        n_atoms_frame = len(atoms_frame)
        nAds_list = [(len(atoms)-n_atoms_frame)/3 for atoms in atoms_list]

    elif file_path.endswith(".csv"):
        # from status file
        df = pd.read_csv(csv_path)
        succ_insert = df[" succ_insertion"].tolist()
        succ_del = df["succ_deletion"].tolist()
        nAds_list = [n_insert-n_del for n_insert, n_del in zip(succ_insert,succ_del)]

    elif file_path.endswith(".npy"):
        # from npy file
        fl_base = results_dir
        nAds_list = np.load(file_path).tolist()

    abs_avg_nAds = np.array(nAds_list[int(len(nAds_list)/4):]).mean()
    #  abs_avg_nAds = np.array(nAds_list[int(len(nAds_list)/1.25):]).mean()

    if plot:
        plt.plot(np.array(range(len(nAds_list))), nAds_list)
        plt.xlabel(r"Steps")
        plt.ylabel(r"Number of Molecules")
        plt.savefig(f"{fl_base}.png")
        plt.clf()
        #  plt.show()
    abs_uptake = (abs_avg_nAds)/sum(atoms_frame.get_masses())* 1000 # in mmol/g

    excess_uptake = abs_uptake - calcBulkFluid(rho_bulk_co2, V_void, frame_mass_tot) # in mmol/g

    return abs_uptake, excess_uptake




results_dir_list = [it for it in os.listdir("./") if os.path.isdir(it) and "results" in it]
atoms_frame = read("./frame0.extxyz")


#  void_volume = (he_void_frac * atoms_frame.get_volume()) / 1e30 # in m^3

M_CO2 = 0.04401  # kg/mol for CO2
void_fraction = 0.826  # Helium void fraction from CoRE MOF database
frame_volume_tot =  atoms_frame.get_volume() # in Ang^3
frame_mass_tot = sum(atoms_frame.get_masses())
V_void = void_fraction * frame_volume_tot / 1e30 # in m3


#  atoms_frame = read("./MgMOF74_clean_frame0.extxyz")
fl = open("uptakes.csv", "w")
print("FileName,Pressure(Pascal),Temperature(K),AbsUptake(mmol/g),ExcesUptake(mmol/g)", file=fl)
for results_dir in results_dir_list:
    pressure = float(results_dir.split("bar")[0].split("_")[-1]) # in bar
    temperature = float(results_dir.split("K")[0].split("_")[-1])
    print(pressure, temperature)
    #  extxyz_path = f"{results_dir}/trajectory_{pressure}bar.extxyz"
    #  csv_path = f"{results_dir}/status.csv"
    npy_path = f"{results_dir}/uptake_{pressure}bar.npy"
    pressure *= BAR2PASCAL # in pascal
    rho_bulk_co2 = calcRhoCoolProp(pressure, temperature, M_CO2, fluid="CO2")
    abs_uptake, excess_uptake = getUptake(npy_path, plot=False)
    print(f"{results_dir},{pressure},{temperature},{abs_uptake},{excess_uptake}", file=fl)

