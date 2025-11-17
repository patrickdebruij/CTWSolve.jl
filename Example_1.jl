"""
Note: this is an example of how to use the code. However, I'd be very surprised if the answers
it gave were not nonsense due to difficulties with arpack being not as good as I'd like it to be.

This implementation does allow for viscous boundary conditions and non-constant viscosity. These
lead to a damping of the modes so a negative imaginary part for ω. They can be commented out/
deleted to recover the usual inviscid problem.

This script determines the dispersion curve ω = ω(k) for 
k ∈ [0.1, 3]. We use ω₀ as an initial guess for ω and take
ω₀ = 0.5*tanh.(k/2) here. Simpler ω₀ will also work, e.g.
ω₀ = 0.5*ones(length(k)) but will generally be slower.

"""

include("CTWSolve.jl")

# Grid parameters:

Ny, Nz = 31, 21
Ly = [0, 4]
type = :laguerre

# Numerical EVP parameters:

k = 0.1:0.1:3
ω₀ = 0.5*tanh.(k/2)
n = 6

# Problem parameters:

f = 1
H₀ = 0.7
H(y) = H₀ + (1 - H₀) * tanh.(y)
U(y, z) = exp.(-y)
M²(y, z) = 0
N²(y, z) = 1
νv(y, z), νh(y, z) = 0.01, 0.01
κv(y, z), κh(y, z) = 0, 0

# Boundary conditions [top, bottom, left, right]:

# NormalFlowBCs = [:noflow, :noflow, :noflow, :noflow]
# NormalStressBCs = [:nostress, :nostress, :nostress, :noflow]
# NormalFluxBCs = [:none, :none, :none, :none]

# Create grid:

println("Creating grid ...")
grid = CreateGrid(Ny, Nz, Ly, H; type)

# Create EVP:

println("Building EVP ...")
prob = CreateProblem(grid; f, U, N², νv, νh, κv, κh, NormalFlowBCs, NormalStressBCs, NormalFluxBCs)

# Solve EVP for dispersion curves:

println("Solving EVP ...")
ω = DispersionCurve(prob, k; n, ω₀, method = :all)

# Plot a mode:
using Plots
heatmap(grid.y[:, 1], grid.z[1, :], real(p[:, :, 1]))
