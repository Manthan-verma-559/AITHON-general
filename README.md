# <p align="center">AITHON <br> Asynchronous Implementation of Turbulence for Hydro-Astrophysical Applications using Optimized Numerics</p>

**Developers**:  
- *Manthan Verma (PhD student at IIT Kanpur)*  
- *Prof. Mahendra Verma (IIT Kanpur)*

## Overview
AITHON is a code developed for solving general-purpose Direct Numerical Simulations (DNS) of Hydro-Astrophysical turbulence problems. The code leverages optimized numerics for efficient and scalable computation.

## Problems It Can Solve
AITHON is capable of solving the following types of problems:
- **Hydrodynamics** (with and without anisotropy)
- **Magnetohydrodynamics (MHD)** (with and without anisotropy)
- **Scalar flows**

## Scalability and Testing
AITHON has been rigorously tested and scaled on various high-performance computing platforms, including:

- **Frontier (Oak Ridge National Laboratory)**: 32,768 AMD MI250X GPUs across 4,096 nodes.
- **Selene (NVIDIA)**: 512 nodes.
- **ALCF (Polaris)**: 256 nodes.

The scalability and performance of the code make it suitable for handling large-scale turbulence simulations in astrophysical contexts.

## Advanced GPU and Communication Technologies
<p align="justify"> AITHON employs the latest GPU technologies such as CUDA-aware MPI for efficient communication. Additionally, it can be built with advanced technologies like NVSHMEMs or ROCSHMEMs, which allow for kernel-to-kernel communication. These shared memory (SHMEM) technologies, which are rarely utilized in turbulence codes, enable AITHON to achieve nearly linear scaling, significantly enhancing its performance in terms of both time and memory efficiency.</p>

## FFTs INTEGRATED
For single GPU solver it uses normal cufft/rocfft. But For Multi-GPU/multi-node following FFTs are integrated in solver
- GPU-FFT (By Verma et al) (https://github.com/Manthan-Verma/GPU_FFT)
- cuFFTMp (By NVIDIA)

## Installation

### 1. Libraries Requirements

#### Single GPU Solver

- **CUDA/Rocm**: Install the appropriate toolkit for GPU acceleration. (Minimum version 10.2 for cuda/ rocm5.0 for Rocm)
- **Python3**: Python 3.x is required for running scripts and code.
- **Python3_packages**: Python 3.x packages are required for running scripts and code (h5py, pybind11, numpy, cupy, prettytable).
- **Microsoft Visual Studio (Windows)**: Microsoft Visual studio of version 19 or above is required for Windows based installation

#### Multi-GPU/Multi-node Solver

- **CUDA/Rocm**: Install the appropriate toolkit for GPU acceleration.
- **Python3**: Python 3.x is still required.
- **Python3_packages**: Python 3.x packages are still required (mpi4py, parallel h5py, pybind11, numpy, cupy and prettytable).
- **GPU-AWARE MPI**: For communication between GPUs, ensure GPU-aware MPI is set up.
- **NVSHMEMs/ROCSHMEMs** (Optional): If you want to utilize SHMEM technology for kernel-to-kernel communication, install NVSHMEM or ROCSHMEM.
- **NCCL/RCCL** (Optional): If you plan to use the NCCL/RCCL libraries for collective communication, ensure these libraries are installed.

### 2. Libraries Installations

<div style="margin-left: 20px;"> 

#### For Linux 
- **CUDA/Rocm**<br>
   For Installation please refer to ( https://developer.nvidia.com/cuda-downloads (For Nvidia GPUs) / https://rocm.docs.amd.com/projects/install-on-linux/en/latest/ ( For AMD GPUs) )
   <br>Add CUDA/Rocm Binaries (nvcc/hipcc) in PATH by <br>
   ```bash
   export PATH=<Path_to_binaries>:$PATH
   ``` 

- **GPU-AWARE MPI  (For Multi-GPU/Multi-node installation)**<br>
    - Download openmpi version 4.x from https://www.open-mpi.org/software/ompi/v4.1/ 
    - Now go to folder where you have downloaded this
    ```bash
        tar -xzf openmpi-4.1.6.tar.gz
        cd openmpi-4.1.6
        ./configure --prefix=<Path_to_install_openmpi> --disable-mpi-fortran --disable-mpi-java --with-cuda=<Path_to_cuda/rocm_toolkit>
        make -j
        make install
    ``` 
    - Now add binaries **mpiexec/mpirun** to PATH

- **NVSHMEMs/ROCSHMEMs  (For Multi-GPU/Multi-node installation)**<br>
    - NVSHMEMs can be installed by just downloading the NVIDIA HPC_SDK from  https://developer.nvidia.com/hpc-sdk-downloads
    - For ROCSHMEMs is still experimental, so you need to have a working installation of ROCSHMMEMs.

- **NCCL/RCCL**<br>
    - NCCL can be installed by just downloading the NVIDIA HPC_SDK from  https://developer.nvidia.com/hpc-sdk-downloads
    - For RCCL can be installed from https://techdocs.broadcom.com/us/en/storage-and-ethernet-connectivity/ethernet-nic-controllers/bcm957xxx/adapters/Configuration-adapter/configuring-nccl-and-gpudirect-with-bcm5750x-network-adapters/configuring-peer-memory-direct-with-amd-gpus/installing-and-running-rccl-collectives.html .

- **Python3**<br>
    - If you are sudo user then you can do
    ```bash
    sudo apt-get install python3
    ```
    - or you can download anaconda3 from - https://www.anaconda.com/download (Install instructions is on its site)

- **Python3_packages**<br>
    - numpy, pybind11, prettytable
        ```bash
        pip3 install numpy pybind11 prettytable
        ```
    - cupy
        - For Nvidia GPU's
            ```bash
            pip3 install cupy-cuda12x
            ```
            Here you can change 12x to 11x depending on version of cuda toolkit installed in system.

        - For AMD GPU's
            ```bash
            export ROCM_HOME=/opt/rocm/
            export CUPY_INSTALL_USE_HIP=1
            export HCC_AMDGPU_TARGET=gfx90a
            pip3 install cupy
            ```
            Here you can change gfx90a to any other gpu arch.

    - mpi4py (Optional -- only for multi-GPU multi-Node required)
        - Make Sure cuda-aware MPI binaries are in PATH and LD_LIBRARY_PATH variables
        - Now type and enter :
            ```bash
            CC=mpicc pip3 install mpi4py
            ```
    
    - h5py
        - For Single Process installation
            ```bash
            pip3 install h5py
            ```

        - For Multi GPU Process installation
            - First install parallel h5py in c++
                - *HDF5 Parallel* ( For multi-GPU/multi-node Solver) <br>
                    - Make sure MPI binaries is in PATH
                    - Go to HDF5 website and Download hdf5-1.14.x.tar.gz
                    - Now go to directory where it is downloaded and type - 
                        ```bash
                        tar -xzf hdf5-1.14.x.tar.gz
                        cd hdf5-1.14.x
                        ./configure --prefix=<Path_Where_you_want_to_install_the_HDF5> --enable-parallel --enable-shared
                        make -j
                        make check
                        make install
                        ```
                - *Parallel H5py* ( For multi-GPU/multi-node Solver) <br>
                    - Make sure this parallel installed HDF5 binaries are in PATH and LD_LIBRARY_PATH variables
                        ```bash
                        CC=mpicc HDF5_MPI="ON" HDF5_DIR=<path_to_HDF5_library_installed_form_source_for_C++_or_C> pip3 install h5py --no-binary=h5py
                        ```

#### For WINDOWS
- **Microsoft Visual Studio (Windows)**<br>
    - Download the Microsoft Visual Studio 2019 or above from https://visualstudio.microsoft.com/downloads/
    - Install this along with Visual C++ package.
    - Add the Visual C++ compiler **cl.exe** in PATH of windows envoirment variable.

- **CUDA/Rocm**<br>
   For Installation please refer to ( https://developer.nvidia.com/cuda-downloads (For Nvidia GPUs) / https://rocm.docs.amd.com/projects/install-on-linux/en/latest/ ( For AMD GPUs) )
   <br>Add CUDA/Rocm Binaries (nvcc/hipcc) in PATH envoirment variable by going to envoirmental variable settings<br>

    Rocm is still not supported on windows

- **GPU-AWARE MPI  (For Multi-GPU/Multi-node installation)**<br>
    - For WINDOWS GPU AWARE MPI is not avilaible at the moment (MICROSFT MPI). So, Multi-GPU/Multi-node is not possible in windows yet.

- **NVSHMEMs/ROCSHMEMs  (For Multi-GPU/Multi-node installation)**<br>
    - For WINDOWS NVSHMEMs/ROCHSMEMs is not avilaible at the moment. So, Multi-GPU/Multi-node is not possible in windows yet.

- **NCCL/RCCL**<br>
    - For WINDOWS NCCL/RCCL is not avilaible at the moment. So, Multi-GPU/Multi-node is not possible in windows yet.

- **Python3_packages**<br>
    - numpy, pybind11, prettytable
        ```bash
        pip3 install numpy pybind11 prettytable
        ```
    - cupy
        - For Nvidia GPU's
            ```bash
            pip3 install cupy-cuda12x
            ```
            Here you can change 12x to 11x depending on version of cuda toolkit installed in system.

        - For AMD GPU's
            ```bash
            export ROCM_HOME=/opt/rocm/
            export CUPY_INSTALL_USE_HIP=1
            export HCC_AMDGPU_TARGET=gfx90a
            pip3 install cupy
            ```
            Here you can change gfx90a to any other gpu arch.

    - mpi4py
        - Not availaible in windows as cuda aware is not in windows
    
    - h5py
        - For Single Process installation
            ```bash
            pip3 install h5py
            ```

        - For Multi GPU Process installation
            - Not avialiable for windows as it des not have working cuda aware MPI

- **HDF5 Parallel** ( For multi-GPU/multi-node Solver) <br>
    - It rquires MPI Installation and as we know for WINDOWS GPU-AWARE MPI is not availaible. So, cannot be installed in GPUs.

- **Python3**<br>
    - Download Python3 form https://www.python.org/downloads/release/python-3114/ .
    - Install it in windows and add its binaries and libraries in PATH envoirment variable.

</div>



### 3. AITHON INSTALLATION
- After installing all the libraries and adding then i PATH and LD_LIBRARY_PATH proceed as follows :

<div style="margin-left: 20px;"> 

- Clone this repository 
```bash
git clone https://github.com/manver-iitk/AITHON.git
cd AITHON
mkdir bin
```

#### For Linux 
- For single GPU Installation
    ``` bash
    cd AITHON/cmakelist/
    mkdir build
    cd build
    cmake -S ../ -B . -DCMAKE_INSTALL_PREFIX=<Path_to_bin_directory_created_above> -DSOURCE_DIR=<Path_to_Release_5_directory>
    make -j 4
    make install
    ```
- For Multi-GPU/Multi-node installation
    ```bash
    cd AITHON/cmakelist/
    mkdir build
    cd build
    cmake -S ../ -B . -DCMAKE_INSTALL_PREFIX=<Path_to_bin_directory_created_above> -DSOURCE_DIR=<Path_to_Release_5_directory> -DMPI=ON
    make -j 4
    make install

After this code will be installed in bin folder where you can go and run it.

#### For WINDOWS
- For single GPU Installation (Use powershell)
    ``` bash
    cd AITHON/cmakelist/
    mkdir build
    cd build
    cmake -S ../ -B . -DCMAKE_INSTALL_PREFIX=<Path_to_bin_directory_created_above> -DSOURCE_DIR=<Path_to_Release_5_directory>
    ```
    - This will create a Visual studio project file in build folder. 
    - Now open the File AITHON.sln (Will open in Visual studio)
    - Now in Visual studio change from DEBUG to Release to all build and then build. It will install the code.
    - Now build the INSTALL. This will install the code in bin directory.

- For Multi-GPU/Multi-node installation
    - Not availaible for WINDOWS because of lack of GPU-AWARE MPI libararies in windows.




### 4. RUNNING AITHON FOR "SCALING" 
- You can also run the code just for testing or scaling pourpose without any para file just using command line to input parameters. Format is as follows::
    - ./AITHON validate solver precision internal_feild_type kind dimension Nx Ny Nz time_scheme fixed_dt_or_not solving_basis
    ```
    precision = double or single
    internal_feild_type = modes or random 
    kind = hydro or mhd or scalar 
    dimesnion = 3 or 2 
    Nx = grid dimension in x
    Ny = grid dimension in y (for 2d its 1)
    Nz = grid dimension in z
    time_scheme = rk4 or rk2 or euler 
    fixed_dt_or_not = true or false
    solver_basis = true or false (only in mhd case needed. That is if true it will solve mhd equations in elssaser basis, else for false in U and B 
                                  basis)
    ```

    - Single GPU example
    ```bash
        CUDA_VISIBLE_DEVICES=0 ./AITHON validate solver double random mhd 3 512 512 512 rk2 true true
    ```

    - Multiple GPU example
    ```bash
        CUDA_VISIBLE_DEVICES=1,2 mpiexec -np 2 ./AITHON validate solver double random mhd 3 512 512 512 rk2 true true
    ```

### 5. RUNNING AITHON FOR "VALIDATION/TESTING" 
- You can also run the full code authenticator for testing of code if it is installed correctly or not. <br>
    Go to code_authenticator README file for furthur instructions.
