# MKLSparse.jl

`MKLSparse.jl` is a Julia package to override sparse-dense operations when MKL is available. It also provides an interface to MKL's Direct Sparse Solver which can be used to factorize and solve sparse matrices.

In order to use `MKLSparse.jl`you need to have built Julia with MKL as the BLAS library. For instructions how to do so, please see [this link](https://github.com/JuliaLang/julia#intel-compilers-and-math-kernel-library-mkl)

### Matrix multiplication
`MKLSparse.jl` provides wra

```julia
A = sprand(10^4, 10^4, 50 / 10^4)
b = rand(10^4)
# Normal matrix multiplication
A * b
A' * b
A.' * b

# 
A1 = 

```

### Factorization
