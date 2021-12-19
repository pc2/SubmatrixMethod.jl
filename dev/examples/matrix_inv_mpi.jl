# # Distributed Matrix Inversion (MPI)

# ```julia
# using SubmatrixMethod
# using MPI
# using LinearAlgebra
# BLAS.set_num_threads(1)

# MPI.Init()
# comm = MPI.COMM_WORLD
# rank = MPI.Comm_rank(comm)
# pinthread(rank) # compact pinning

# A = sprandsymposdef(10_000, 0.01);

# MPI.Barrier(comm)
# R = submatrix_apply(inv, A, MPISerial())
# if rank == 0 # test on master
#     @show maximum(abs.(R .- R̃))
# end

# MPI.Barrier(comm)
# MPI.Finalize()
# ```