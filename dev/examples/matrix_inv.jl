# # Speeding up Matrix Inversion

# Load the packages
using SubmatrixMethod
using BenchmarkTools
SubmatrixMethod.disable_benchmarks() # hide

# A sparse symmetric input matrix
M = SubmatrixMethod.generate_input_matrix(1000, 0.001)
# M = SubmatrixMethod.generate_input_matrix(10_000, 0.0001)

# The operation that we want to consider is a matrix inversion.
# However, directly calling `inv` doesn't work. Since the inverse of
# a sparse matrix is generally dense, we need to convert `M` to a
# dense matrix first.
Mdense = Matrix(M);

# Establishing a baseline:
@btime inv($Mdense);

# Submatrix for dense input
@btime submatrix_apply($inv, $Mdense);

# Submatrix for sparse input
@btime submatrix_apply($inv, $M);

# We can speed up things further by enabling multithreading.
# Of course, this only works if we've started Julia with multiple threads.
Threads.nthreads()

# Also, to avoid conflicts with BLAS's built-in multithreading, we should
# set the number of BLAS threads to one.
BLAS.set_num_threads(1)

# Let's now benchmark the multithreaded variant
@btime submatrix_apply($inv, $M; multithreading=true);

# Hope this was helpful!