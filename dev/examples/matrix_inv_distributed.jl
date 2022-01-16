# # Distributed Matrix Inversion

# ## No parallelism

# Preparation.
using SubmatrixMethod, MKL, LinearAlgebra, BenchmarkTools
BLAS.set_num_threads(1)

# We need a random input matrix.
A = sprandsymposdef(1000, 0.001)

# We can get a (good) approximation of the inverse of `A` via `submatrix_apply(inv, A)`.
Ainv = submatrix_apply(inv, A)
maximum(abs.(Ainv .- inv(Matrix(A))))

# Let's benchmark the serial submatrix method
@btime submatrix_apply($inv, $A);

# ## `JLDistributed`

# Add a few Julia workers.
# ```julia
# using Distributed
# withenv("JULIA_PROJECT" => @__DIR__) do
#     addprocs(5)
# end
# ```

# Prepare all of them.
# ```julia
# @everywhere begin
#     using SubmatrixMethod
#     using MKL
#     using LinearAlgebra
#     BLAS.set_num_threads(1)
# end
# ```

# Let's see if things work correctly.
# ```julia
# Ainv = submatrix_apply(inv, A, JLDistributed())
# maximum(abs.(Ainv .- inv(Matrix(A))))
# ```

# And now let's benchmark.
# ```julia
# @btime submatrix_apply($inv, $A, $(JLDistributed()));
# ```

# ### `HDF5Input`
#
# You can also use [`HDF5Input`](@ref) to make `submatrix_apply` use a sparse matrix stored
# in a HDF5 file (should have been stored with [`write_hdf5`](@ref)).
# In this case, the input matrix isn't send to workers but instead read / mmapped from disk
# on every worker. This lowers the inter-worker communication at the cost of (parallel)
# disk accesses.
# ```julia
# # if an A::SparseMatrixCSC is provided to the HDF5Input constructor
# # the matrix is written to file before the HDF5Input wrapper is created
# Ah5 = HDF5Input(A; fname="A.h5", path="A", mmap=mmap, overwrite=true)
# R = submatrix_apply(inv, Ah5, JLDistributed())
# ```

# That's it.
# ```julia
# nworkers() > 1 && rmprocs(workers()) # hide
# ```