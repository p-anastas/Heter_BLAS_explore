# Towards a Heterogeneous data-centric framework for efficient Linear algebra

The code used for the aforementioned work (https://2020.isc-program.com/presentation/?id=phd104&sess=sess328").
An attempt to explore automatic CPU-GPU hybrid execution for BLAS with transfer awareness.
The backend GPU microbencmarks are performed using cuBLAS, while the CPU ones use Intel mkl (Can be replaced by OpenBLAS with minor changes).

*The CUDA code and models used in this work have been cleaned and (partially) integrated in https://github.com/p-anastas/CoCoPeLia-Framework.* 
# Requirements

- CUDA (Tested with 7.0 and 9.2, but should work with older/newer versions)
- CUBLAS library (default = installed along with CUDA)
- NVIDIA GPU with support for double operations. 
- Intel CPU and Intel MKL (parallel studio 1.1 tested)
- Cmake 3.10 or greater

# Deployment

**Build:**
- Modify the library paths at CmakeLists.txt for your system.
- *mkdir build && cd build*
- *cmake ../ && make -j 4*

**Experiment:**
- Run *run-benchmark.sh  machinename*
- Use the python scripts to proccesss the logs.
- Example results for our 2 testbeds are available in the repo.
