# # Matrix Inversion

# Load the packages
using SubmatrixMethod
using BenchmarkTools
SubmatrixMethod.disable_benchmarks() # hide

# A sparse symmetric input matrix
M = SubmatrixMethod.generate_input_matrix(1000, 0.001)
# M = SubmatrixMethod.generate_input_matrix(10_000, 0.0001)

# The operation that we want to consider is a matrix inversion.
# However, directly calling `inv` doesn't work
inv(M)

# To "fix" this and establish a baseline, we create a dense copy of the matrix `M`
Mdense = Matrix(M);

# Baseline:
@btime inv($Mdense);

# Submatrix for dense input
@btime submatrix_apply($inv, $Mdense);

# Submatrix for sparse input
@btime submatrix_apply($inv, $M);

# Submatrix for sparse input
@btime submatrix_apply($inv, $M; multithreading=false);
@btime submatrix_apply($sqrt, $M; multithreading=false);

# We can speed up things further by enabling multithreading
@btime submatrix_apply($inv, $M; multithreading=true);
@btime submatrix_apply($sqrt, $M; multithreading=true);

# Other multithreading
@btime submatrix_apply($inv, $M; multithreading=:threads);
@btime submatrix_apply($sqrt, $M; multithreading=:threads);

# Make sure that Julia is actually running with multiple threads
Threads.nthreads()

# asd
using LinearAlgebra: BLAS
BLAS.get_num_threads()

# ASD
BLAS.set_num_threads(1)
@btime submatrix_apply($inv, $M; multithreading=false);

# test
@btime submatrix_apply($inv, $M; multithreading=true);

# Fin.