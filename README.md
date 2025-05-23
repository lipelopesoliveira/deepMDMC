# DeepMDMC: Molecular Dynamics and Monte Carlo Simulations with Deep Learning Potentials

## Setting up the environment

First, create a conda environment with the required packages:

```bash
conda env create --file environment.yml
```

To use conda with MKL you need to set up the environment variables setted based on the conda packages:

```bash
export MKLROOT=$CONDA_PREFIX
export LIBRARY_PATH=$CONDA_PREFIX/lib/:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
export PATH=$CONDA_PREFIX/bin/:$PATH
```

## Building LAMMPS with this pair style

### Download LAMMPS

```bash
git clone --depth=1 https://github.com/lammps/lammps
```

or your preferred method. (`--depth=1` prevents the entire history of the LAMMPS repository from being downloaded.)

### Download this repository

```bash
git clone https://github.com/mir-group/pair_nequip
```

### Patch LAMMPS

#### Automatically (Strongly recommended)

From the `pair_nequip` directory, run:

```bash
./patch_lammps.sh /path/to/lammps/
```

#### Manually

First copy the source files of the pair style:

```bash
cp /path/to/pair_nequip/*.cpp /path/to/lammps/src/
cp /path/to/pair_nequip/*.h /path/to/lammps/src/
```

Then make the following modifications to `lammps/cmake/CMakeLists.txt`:

- Append the following lines:

```cmake
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
```

### Configure LAMMPS

If you have PyTorch installed:

```bash
cd lammps

mkdir build && cd build

cmake -C ../cmake/presets/most.cmake ../cmake  -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -D CMAKE_INSTALL_PREFIX=$HOME/lib/lammps -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX/bin/ -DMKL_INCLUDE_DIR=$CONDA_PREFIX/include/ -D BUILD_SHARED_LIBS=yes -D BUILD_TOOLS=yes -D BUILD_OMP=yes
```

If you don't have PyTorch installed, you need to download LibTorch from the [PyTorch download page](https://pytorch.org/get-started/locally/). Unzip the downloaded file, then configure LAMMPS with the `CMAKE_PREFIX_PATH` pointing to the unzipped LibTorch directory. For example, if you downloaded and unzipped it to `/path/to/libtorch`, then run:

```bash
-DCMAKE_PREFIX_PATH=/path/to/libtorch
```

CMake will look for MKL and, optionally, CUDA and cuDNN. You may have to explicitly provide the path for your CUDA installation (e.g. `-DCUDA_TOOLKIT_ROOT_DIR=/usr/lib/cuda/`) and your MKL installation (e.g. `-DMKL_INCLUDE_DIR=/usr/include/`).

Pay attention to warnings and error messages.

**MKL:** If `MKL_INCLUDE_DIR` is not found and you are using a Python environment, a simple solution is to run `conda install mkl-include` or `pip install mkl-include` and append:

```bash
-DMKL_INCLUDE_DIR="$CONDA_PREFIX/include"
```

to the `cmake` command if using a `conda` environment, or

```bash
-DMKL_INCLUDE_DIR=`python -c "import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()[\"include\"]).parent)"`
```

if using plain Python and `pip`.

**CUDA:** Note that the CUDA that comes with PyTorch when installed with `conda` (the `cudatoolkit` package) is usually insufficient (see [here](https://github.com/pytorch/extension-cpp/issues/26), for example) and you may have to install full CUDA seperately. A minor version mismatch between the available full CUDA version and the version of `cudatoolkit` is usually *not* a problem, as long as the system CUDA is equal or newer. (For example, PyTorch's requested `cudatoolkit==11.3` with a system CUDA of 11.4 works, but a system CUDA 11.1 will likely fail.)

### Build LAMMPS

```bash
make -j9
make install
```

Then, install the lammps python package:

```bash
make install-python
```

## Running the code

Copy the `CO2.dat` file from the `data` folder to the directory that you want to run the code. This file contains the parameters for the CO2 molecule for the LAMMPS simulation. You can also use your own file, but make sure it is in the same format as `CO2.dat`.

```bash
python /home/flopes/MCMD/deepMDMC-main/mdmc/runDeepMDMC.py -sim_type 'gcmcmd' \
                                                            -pressure 1 \
                                                            -temperature 298 \
                                                            -totalsteps 400000 \
                                                            -nmdsteps 1000 \
                                                            -nmcswap 250000 \
                                                            -nmcmoves 250000 \
                                                            -neqsteps 10000 \
                                                            -model_gcmc_path /home/flopes/MCMD/deepMDMC-main/MgMOF74_CO2_models/MgMOF74_CO2_float32.nequip.pth \
                                                            -model_md_path /home/flopes/MCMD/deepMDMC-main/MgMOF74_CO2_models/MgMOF74_CO2.nequip.pth \
                                                            -molecule_path /home/flopes/MCMD/deepMDMC-main/data/CO2.xyz \
                                                            -struc_path /home/flopes/MCMD/deepMDMC-main/data/mg-mof-74.cif \
                                                            -timestep 0.0005 \
                                                            -flex_ads 'no' \
                                                            -opt 'no' \
                                                            -interval 50
```
