# # Speeding up Matrix Inversion

# ## Regular inversion
# First, we need to generate a random input matrix `M` that we want to invert.
# However, since the submatrix method expects a sparse symmetric
# positive definite matrix, we can't just use `rand`. Instead,
# we use the utility function [`SubmatrixMethod.generate_input_matrix`](@ref) to
# generate our input matrix.
using SubmatrixMethod
SubmatrixMethod.disable_benchmarks() # hide
M = SubmatrixMethod.generate_input_matrix(1000, 0.001)

# Note that `M` isn't just sparse in the value sense but actually
# a `SparseMatrixCSC` datastructure.
typeof(M)

# If we try to naively invert this matrix, i.e. calling `inv(M)`, Julia will throw an
# error, reminding us of the fact that the inverse of a sparse matrix is generally dense.
# To circumvent this error, we first need to convert `M` to a dense `Matrix{Float64}` first.
Mdense = Matrix(M);

# Now, `inv(M)` does what it's supposed to do.
using LinearAlgebra, Test
@test inv(Mdense) * Mdense ≈ I

# Alright, let's load [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) and
# benchmark how long the conventional inversion takes.
# This will serve as a baseline that we can compare to the submatrix method.
using BenchmarkTools
@btime inv($Mdense);

# ## Submatrix method

# Using the function `submatrix_apply` by SubmatrixMethod.jl, we can get a (good) approximation of `inv(Mdense)` via `submatrix_apply(inv, Mdense)`.
M̃inv = submatrix_apply(inv, Mdense)
Minv = inv(Mdense)
maximum(abs.(M̃inv .- Minv))

# However, note that the computation based on the submatrix method is much faster
@btime submatrix_apply($inv, $Mdense);

# And it is even faster if we provide the sparse matrix (`SparseMatrixCSC`) directly
@btime submatrix_apply($inv, $M);

# ## Multithreading

# Depending on the size/sparsity of the input matrix (see [Scaling](@ref)), we can sometimes
# speed things up even further by enabling the multithreading functionality of SubmatrixMethod.jl.
# Of course, this only works if we've started Julia with multiple threads in the first place.
Threads.nthreads()

# To avoid conflicts with BLAS's built-in multithreading, it is generally recommended
# to set the number of BLAS threads to one.
BLAS.set_num_threads(1)

# Alright, here comes a benchmark that shows a case where multithreading gives a significant speedup.
M = SubmatrixMethod.generate_input_matrix(1000, 0.01)
@btime submatrix_apply($inv, $M; multithreading=false);
@btime submatrix_apply($inv, $M; multithreading=true);

# ## Scaling
#
# * [As a function of the density of the input matrix (and for various sizes)](https://git.uni-paderborn.de/pc2/julia/submatrixmethod.jl/-/tree/master/analysis/scaling_density)
# * [As a function of the size of the input matrix (and for various densities)](https://git.uni-paderborn.de/pc2/julia/submatrixmethod.jl/-/tree/master/analysis/scaling_matrixsize)
# * [As a function of the number of Julia / BLAS threads](https://git.uni-paderborn.de/pc2/julia/submatrixmethod.jl/-/tree/master/analysis/scaling_multithreading)

# So much about speeding up matrix inversion with the submatrix method. See you in the next tutorial!