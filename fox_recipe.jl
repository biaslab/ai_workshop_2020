# This script implements Hamiltonian Dynamics based on Ma et al (2015) - A complete recipe for MCMC
# It iterates the update EQ's of Hamiltonian MCMC - without any sampling - and
# shows that it converges to the correct posterior. It's mainly written as a didactic exercise to
# test the formalism

using Pkg;Pkg.activate(".");Pkg.instantiate()
using Distributions, Random, DifferentialEquations, Zygote, Plots
Random.seed!(666)

# Because it looks good :)
∑ = sum

# Generate data
data = rand(Normal(10,1),100);

# Matrices
Q = [ 0 -1;
     1 0]
Γ = [ 0 0;
     0 0.5]


# Define model
prior = Normal(0,10)
likelihood(x) = Normal(x,1)

# Potential Function
U(θ,data) = -∑(
	      logpdf.(likelihood(θ),data) .-
	      logpdf(prior,θ)
	      )

# Hamiltonian
function H(θ,r,data)
    M = 1 # Mass Matrix. Since the model is so small it's just a scalar
    U(θ,data) + 0.5 * ( r * M * r)
end

# DiffEq Section starts here
function dynamics!(du,u,p,t)
    θ,r = u

    ∇_H = gradient( (θ_,r_) -> H(θ_,r_,data), θ,r)
    ∇_H = [∇_H[1],∇_H[2]]

    dt = -(Q + Γ) * ∇_H

    # probably a smarter way to do bookkeeping. For some reason it crashes with du=dt
    du[1] = dt[1]
    du[2] = dt[2]
end

# Initialize parameters. Momentum r is randomly initialized.
θ = rand(prior)
r = rand(Normal(0,1))

# Set up diffeqproblem and solve
u0 = [θ,r]
tspan= (0.0,50.0)
prob = ODEProblem(dynamics!,u0,tspan)
sol = solve(prob);

# Shows θ over r
plot(sol,vars=(1,2))

