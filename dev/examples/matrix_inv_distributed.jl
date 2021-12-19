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

# ## `DistributedSerial`

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
# Ainv = submatrix_apply(inv, A, DistributedSerial())
# maximum(abs.(Ainv .- inv(Matrix(A))))
# ```

# And now let's benchmark.
# ```julia
# @btime submatrix_apply($inv, $A, $(DistributedSerial()));
# ```

# That's it.
# ```julia
# nworkers() > 1 && rmprocs(workers()) # hide
# ```