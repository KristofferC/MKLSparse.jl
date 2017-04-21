# MKLSparse.jl

`MKLSparse.jl` is a Julia package to seamlessly use the sparse functionality in MKL to speed up operations on sparse arrays in Julia.
In order to use `MKLSparse.jl`you need to have MKL installed and the environment variables `MKLROOT` correctly set, see the [MKL getting started guide]( https://software.intel.com/en-us/articles/intel-mkl-103-getting-started) for a guide. You do not need to have a julia built with MKL.

### Matrix multiplication

Loading `MKLSparse.jl` will make sparse-dense matrix operations be computed using MKL.

### Solving linear systems

Solving linear systems with a triangular sparse matrix is supported

For solving general sparse linear systems using MKL we refer to [Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl)

### Possible TODO's

* Wrap BLAS1 (`SparseVector`)
* Wrap DSS
* Wrap Incomplete LU preconditioners
