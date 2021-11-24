var documenterSearchIndex = {"docs":
[{"location":"utility/#Utility","page":"Utility","title":"Utility","text":"","category":"section"},{"location":"utility/#Index","page":"Utility","title":"Index","text":"","category":"section"},{"location":"utility/","page":"Utility","title":"Utility","text":"Pages   = [\"utility.md\"]\nOrder   = [:function, :type]","category":"page"},{"location":"utility/#References","page":"Utility","title":"References","text":"","category":"section"},{"location":"utility/","page":"Utility","title":"Utility","text":"Modules = [SubmatrixMethod]\nPages   = [\"utility.jl\", \"input_matrix_generation.jl\"]","category":"page"},{"location":"utility/#SubmatrixMethod.density-Tuple{AbstractArray{var\"#s41\", N} where {var\"#s41\"<:Number, N}}","page":"Utility","title":"SubmatrixMethod.density","text":"Number of non-zero elements over the total number of elements.\n\n\n\n\n\n","category":"method"},{"location":"utility/#SubmatrixMethod.inv!-Tuple{Any}","page":"Utility","title":"SubmatrixMethod.inv!","text":"inv!(M)\n\nIn-place matrix inversion. Overwrites the input matrix M with the result.\n\n\n\n\n\n","category":"method"},{"location":"utility/#SubmatrixMethod.sparsity-Tuple{AbstractArray{var\"#s42\", N} where {var\"#s42\"<:Number, N}}","page":"Utility","title":"SubmatrixMethod.sparsity","text":"Number of zero-valued elements over the total number of elements.\n\n\n\n\n\n","category":"method"},{"location":"utility/#SubmatrixMethod.spectral_norm-Tuple{AbstractMatrix{var\"#s41\"} where var\"#s41\"<:Number}","page":"Utility","title":"SubmatrixMethod.spectral_norm","text":"Computes the spectral norm of the input matrix:\n\nA_2=sqrtlambda_max left(A^H Aright)\n\ni.e. the square root of the maximal eigenvalue of A'*A.\n\n\n\n\n\n","category":"method"},{"location":"utility/#SubmatrixMethod.generate_input_matrix-Tuple{Any, Any}","page":"Utility","title":"SubmatrixMethod.generate_input_matrix","text":"generate_input_matrix(n,density)\n\nGenerate a sparse n x n symmetric, positive definite matrix with a density of non-zero elements of approximately density.\n\nTODO: Improve this so that we can control the condition number. TODO: Compare to MATLABs sprandsym(size,density,1/condition,kind) with kind=1\n\nInspired by\n\nhttps://www.mathworks.com/help/matlab/ref/sprandsym.html\nhttps://stackoverflow.com/questions/46930325/create-a-sparse-symmetric-random-matrix-in-julia\n\n\n\n\n\n","category":"method"},{"location":"devel/benchmarking/#Benchmarking","page":"Benchmarking","title":"Benchmarking","text":"","category":"section"},{"location":"devel/benchmarking/","page":"Benchmarking","title":"Benchmarking","text":"By default, the benchmarking facilities are turned off entirely to avoid any performance overhead.","category":"page"},{"location":"devel/benchmarking/#Usage","page":"Benchmarking","title":"Usage","text":"","category":"section"},{"location":"devel/benchmarking/","page":"Benchmarking","title":"Benchmarking","text":"note: Note\nOnly the serial variant, i.e. with multithreading=false (default), can be benchmarked for now. For multithreading=true, you'll likely get lot's of errors.","category":"page"},{"location":"devel/benchmarking/","page":"Benchmarking","title":"Benchmarking","text":"Enable the built-in time measurements with SubmatrixMethod.enable_benchmarks().\nRun the functionality that you want to benchmark.\nCall SubmatrixMethod.print_benchmarks() to see the timing results.","category":"page"},{"location":"devel/benchmarking/","page":"Benchmarking","title":"Benchmarking","text":"using SubmatrixMethod\nSubmatrixMethod.enable_benchmarks()\nA = SubmatrixMethod.generate_input_matrix(1000, 0.01);\nsubmatrix_apply(inv, A);\nsubmatrix_apply(inv, A);\nSubmatrixMethod.print_benchmarks()","category":"page"},{"location":"devel/benchmarking/","page":"Benchmarking","title":"Benchmarking","text":"Benchmarks can be reset by calling SubmatrixMethod.reset_benchmarks() or turned off again via SubmatrixMethod.disable_benchmarks().","category":"page"},{"location":"devel/benchmarking/#Index","page":"Benchmarking","title":"Index","text":"","category":"section"},{"location":"devel/benchmarking/","page":"Benchmarking","title":"Benchmarking","text":"Pages   = [\"benchmarking.md\"]\nOrder   = [:function, :type]","category":"page"},{"location":"devel/benchmarking/#References","page":"Benchmarking","title":"References","text":"","category":"section"},{"location":"devel/benchmarking/","page":"Benchmarking","title":"Benchmarking","text":"Modules = [SubmatrixMethod]\nPages   = [\"debugging.jl\"]","category":"page"},{"location":"devel/benchmarking/#SubmatrixMethod.disable_benchmarks-Tuple{}","page":"Benchmarking","title":"SubmatrixMethod.disable_benchmarks","text":"disable_benchmarks()\n\nDisables benchmarking.\n\nSee: enable_benchmarks\n\n\n\n\n\n","category":"method"},{"location":"devel/benchmarking/#SubmatrixMethod.enable_benchmarks-Tuple{}","page":"Benchmarking","title":"SubmatrixMethod.enable_benchmarks","text":"enable_benchmarks()\n\nEnables benchmarking. This affects all SubmatrixMethod.@bench macro applications.\n\nResults can be printed via SubmatrixMethod.print_benchmarks() and reset via SubmatrixMethod.reset_benchmarks().\n\nSee: disable_benchmarks\n\n\n\n\n\n","category":"method"},{"location":"devel/benchmarking/#SubmatrixMethod.print_benchmarks-Tuple{}","page":"Benchmarking","title":"SubmatrixMethod.print_benchmarks","text":"Print benchmark results.\n\nSee: enable_benchmarks\n\n\n\n\n\n","category":"method"},{"location":"devel/benchmarking/#SubmatrixMethod.reset_benchmarks-Tuple{}","page":"Benchmarking","title":"SubmatrixMethod.reset_benchmarks","text":"Reset benchmark results.\n\n\n\n\n\n","category":"method"},{"location":"devel/benchmarking/#SubmatrixMethod.@bench-Tuple{Any, Any}","page":"Benchmarking","title":"SubmatrixMethod.@bench","text":"Usage: @bench \"some description\" 3+3\n\nSee: enable_benchmarks\n\n\n\n\n\n","category":"macro"},{"location":"#SubmatrixMethod.jl","page":"SubmatrixMethod","title":"SubmatrixMethod.jl","text":"","category":"section"},{"location":"","page":"SubmatrixMethod","title":"SubmatrixMethod","text":"Approximately compute matrix functions of large sparse matrices by using the submatrix method.","category":"page"},{"location":"","page":"SubmatrixMethod","title":"SubmatrixMethod","text":"TLDR: Simply replace the call f(A) by submatrix_apply(f, A) to obtain an approximation of the result in (hopefully) less time.","category":"page"},{"location":"","page":"SubmatrixMethod","title":"SubmatrixMethod","text":"Technically, we support arbitrary input matrices A::AbstractMatrix but","category":"page"},{"location":"","page":"SubmatrixMethod","title":"SubmatrixMethod","text":"the approximation works best / at all for large sparse matrices,\na specialized, and thus faster, implementation can be used for A::SparseMatrixCSC (not available yet).","category":"page"},{"location":"#Installation","page":"SubmatrixMethod","title":"Installation","text":"","category":"section"},{"location":"","page":"SubmatrixMethod","title":"SubmatrixMethod","text":"You can simply add SubmatrixMethod.jl to your Julia environment with the command","category":"page"},{"location":"","page":"SubmatrixMethod","title":"SubmatrixMethod","text":"] add https://git.uni-paderborn.de/pc2/julia/submatrixmethod.jl","category":"page"},{"location":"","page":"SubmatrixMethod","title":"SubmatrixMethod","text":"Note: The minimal required Julia version is 1.6 but we recommend using Julia ≥ 1.7.","category":"page"},{"location":"#Example-usage","page":"SubmatrixMethod","title":"Example usage","text":"","category":"section"},{"location":"","page":"SubmatrixMethod","title":"SubmatrixMethod","text":"using SubmatrixMethod\n\n# Generate a random, sparse, symmetric, positive definite matrix of\n# size 1000 x 1000 with a density of non-zero elements of approximately 0.05.\nA = SubmatrixMethod.generate_input_matrix(1000, 0.05)\n\n# Use the submatrix method to approximate inv(A)\nX = submatrix_apply(inv, A; multithreading=true)\n\n# Check some properties\nusing Test\n@test findnz(sparse(X))[1] == findnz(A)[1] # same sparsity pattern\n@test maximum(abs.(X .- inv(Matrix(A)))) ≤ 1e-7\n@test spectral_norm(X * A - I) ≤ 1e-3","category":"page"},{"location":"#Resources","page":"SubmatrixMethod","title":"Resources","text":"","category":"section"},{"location":"","page":"SubmatrixMethod","title":"SubmatrixMethod","text":"A Massively Parallel Algorithm for the Approximate Calculation of Inverse p-th Roots of Large Sparse Matrices   Michael Lass, Stephan Mohr, Hendrik Wiebeler, Thomas D. Kühne, Christian Plessl   GitHub repository: https://github.com/pc2/SubmatrixMethod","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"EditURL = \"https://git.uni-paderborn.de/pc2/julia/submatrixmethod.jl/blob/main/docs/docs/src/examples/matrix_inv.jl\"","category":"page"},{"location":"examples/matrix_inv/#Matrix-Inversion","page":"Matrix Inversion","title":"Matrix Inversion","text":"","category":"section"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"Load the packages","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"using SubmatrixMethod\nusing BenchmarkTools\nSubmatrixMethod.disable_benchmarks() # hide","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"A sparse symmetric input matrix","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"M = SubmatrixMethod.generate_input_matrix(1000, 0.001)","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"M = SubmatrixMethod.generateinputmatrix(10_000, 0.0001)","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"The operation that we want to consider is a matrix inversion. However, directly calling inv doesn't work","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"inv(M)","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"To \"fix\" this and establish a baseline, we create a dense copy of the matrix M","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"Mdense = Matrix(M);\nnothing #hide","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"Baseline:","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"@btime inv($Mdense);\nnothing #hide","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"Submatrix for dense input","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"@btime submatrix_apply($inv, $Mdense);\nnothing #hide","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"Submatrix for sparse input","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"@btime submatrix_apply($inv, $M);\nnothing #hide","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"Submatrix for sparse input","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"@btime submatrix_apply($inv, $M; multithreading=false);\n@btime submatrix_apply($sqrt, $M; multithreading=false);\nnothing #hide","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"We can speed up things further by enabling multithreading","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"@btime submatrix_apply($inv, $M; multithreading=true);\n@btime submatrix_apply($sqrt, $M; multithreading=true);\nnothing #hide","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"Other multithreading","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"@btime submatrix_apply($inv, $M; multithreading=:threads);\n@btime submatrix_apply($sqrt, $M; multithreading=:threads);\nnothing #hide","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"Make sure that Julia is actually running with multiple threads","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"Threads.nthreads()","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"asd","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"using LinearAlgebra: BLAS\nBLAS.get_num_threads()","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"ASD","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"BLAS.set_num_threads(1)\n@btime submatrix_apply($inv, $M; multithreading=false);\nnothing #hide","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"test","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"@btime submatrix_apply($inv, $M; multithreading=true);\nnothing #hide","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"Fin.","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"","category":"page"},{"location":"examples/matrix_inv/","page":"Matrix Inversion","title":"Matrix Inversion","text":"This page was generated using Literate.jl.","category":"page"},{"location":"submatrix/#Submatrix-Method","page":"Submatrix Method","title":"Submatrix Method","text":"","category":"section"},{"location":"submatrix/#Index","page":"Submatrix Method","title":"Index","text":"","category":"section"},{"location":"submatrix/","page":"Submatrix Method","title":"Submatrix Method","text":"Pages   = [\"submatrix.md\"]\nOrder   = [:function, :type]","category":"page"},{"location":"submatrix/#References","page":"Submatrix Method","title":"References","text":"","category":"section"},{"location":"submatrix/","page":"Submatrix Method","title":"Submatrix Method","text":"Modules = [SubmatrixMethod]\nPages   = [\"submatrix.jl\"]","category":"page"},{"location":"submatrix/#SubmatrixMethod.construct_submatrix-Tuple{AbstractMatrix{T} where T, Any}","page":"Submatrix Method","title":"SubmatrixMethod.construct_submatrix","text":"construct_submatrix(A, j) -> submatrix, indices\n\nConstruct the j-th submatrix form the input matrix A.\n\n\n\n\n\n","category":"method"},{"location":"submatrix/#SubmatrixMethod.submatrix_apply-Union{Tuple{T}, Tuple{Any, AbstractMatrix{T}}} where T<:Number","page":"Submatrix Method","title":"SubmatrixMethod.submatrix_apply","text":"submatrix_apply(f, A)\n\nUses the submatrix method to approximately compute f(A), i.e. the application of the matrix function f to the input matrix A.\n\nThe input matrix A\n\nmust be symmetric and positive definite (check with isposdef),\nshould be sparse (check with sparsity).\n\nThe keyword argument multithreading can be used to enable parallel processing of different submatrices. The available options are:\n\nfalse: no multithreading (default)\ntrue: composable, load-balanced multithreading (Threads.@spawn)\n\nFurther keyword arguments:\n\ninplace (default: false): if true, expects that f is an in-place\n\nmatrix function that overwrites the input matrix with the result, i.e. inv! instead of inv, for example.\n\n\n\n\n\n","category":"method"},{"location":"submatrix/#SubmatrixMethod.submatrix_computation!-Tuple{AbstractMatrix{T} where T, Any, AbstractMatrix{T} where T, Any}","page":"Submatrix Method","title":"SubmatrixMethod.submatrix_computation!","text":"submatrix_computation!(X, f, A, j)\n\nThis is the \"computation kernel\". Constructs the j-th submatrix from the input matrix A, applies the matrix function f to it, and fills the corresponding column of the result matrix X.\n\n\n\n\n\n","category":"method"},{"location":"submatrix/#SubmatrixMethod.submatrix_computation_inplace!-Tuple{AbstractMatrix{T} where T, Any, AbstractMatrix{T} where T, Any}","page":"Submatrix Method","title":"SubmatrixMethod.submatrix_computation_inplace!","text":"submatrix_computation_inplace!(X, f_inplace, A, j)\n\nThis is the \"computation kernel\". Constructs the j-th submatrix from the input matrix A, applies the matrix function f_inplace to it (in-place), and fills the corresponding column of the result matrix X.\n\n\n\n\n\n","category":"method"}]
}
