using Pkg;Pkg.activate(".");Pkg.instantiate()
using OhMyREPL,Zygote, Random, DifferentialEquations, Plots, LaTeXStrings
enable_autocomplete_brackets(false)

Random.seed!(666);
# TODO
# Add f and g functions to make the code easier to mess around with.
# Find a better way to change the Hamiltonian

# Set up callback and timepoint to switch environments
tswitch = 25.0
condition(u,t,integrator) = t==tswitch
function affect!(integrator)
    # New position and velocity
    integrator.u[1] = 10.0
    integrator.u[2] = 2.0
end
cb = DiscreteCallback(condition,affect!)

# System dynamics
function dynamics!(du,u,p,t)
    x,x_dot,μ_0,μ_1,a = u # Unpacke the state vector

    # Link functions
    h(μ) = μ
    g(ϕ) = ϕ

    # Parameters for environment. This change should be doable through p in the input instead.
    if t<tswitch
	m = 1.
	k = 1.
    else
	m = 10.
	k = 0.1
    end

    # Parameters for agent. 2 precisions and action limiting term
    σ_1 = 0.1
    σ_2 = 0.1
    force = 0.5

    # Hamiltonian for the environmental process
    H(x,x_dot,a,m,k) = 1/(2*m) * x^2 + 1/2 * (k * x_dot^2) - x_dot * force *tanh(a)

    # VFE as potential function. With the right choice of p and q it boils down to precision weighed prediction errors. See Buckley for derivation
    J(x,μ_0,μ_1) = 0.5 * ( 1/σ_2 * (μ_1 - h(μ_0))^2 + 1/σ_1 * (x - g(μ_0))^2)

    # Inverse model. Derivative of x (observations) with respect to a (action)
    dxda(a) = -force * (sech(a) * sech(a))

    # Get gradients for both the agent and the generative process
    # We use autodiff to make the code easier to adapt
    ∇_J = gradient( (x_,μ_0_,μ_1_) 	 -> J(x_,μ_0_,μ_1_),	   x,μ_0,μ_1)
    ∇_H = gradient( (x_,x_dot_,a_,m_,k_) -> H(x_,x_dot_,a_,m_,k_), x,x_dot,a,m,k)

    # Combine gradients to get ∇_J'. Final entry is for action. Split through chain rule by using inverse model
    ∇_J_prime = [∇_H[1],∇_H[2],∇_J[2],∇_J[3], -∇_J[1] * dxda(a)]

    # Define shared Q and Γ matrices for the whole system. First 2 rows follow H, the Hamiltonian of a simple harmonic oscillator. Rows 3 and 4 governs dynamics of the agent. Final row is action which is not part of the generative model as per Friston and Ao (2012).
    Q = [
	 0.0 -1.0 0.0 0.0 0.0;
	 1.0 0.0 0.0 0.0 0.0;
	 0.0 0.0 0.0 -1.0 0.0;
	 0.0 0.0 1.0 0.0 0.0;
	 0.0 0.0 0.0 0.0 0.0
	]

    # Random fluctuations govern entries of Γ. This performs gradient descent on potential function
    Γ = [
	 0.0 0.0 0.0 0.0 0.0;
	 0.0 0.0 0.0 0.0 0.0;
	 0.0 0.0 0.1 0.0 0.0;
	 0.0 0.0 0.0 0.1 0.0;
	 0.0 0.0 0.0 0.0 1.0
	]

    # Calculate actual time derivatives. There has to be a better way to do bookkeeping
    dudt = -(Q + Γ) * ∇_J_prime
    du[1] = dudt[1]
    du[2] = dudt[2]
    du[3] = dudt[3]
    du[4] = dudt[4]
    du[5] = dudt[5]
end

# Uncorrelated Wiener noise
function noise!(du,u,p,t)
    du[1] = 0.1
    du[2] = 0.1
    du[3] = 0.0
    du[4] = 0.0
    du[5] = 0.0
end

# Initial conditions and timespan to solve. This is the vector state of the system
u0 = [2.0,2.0,0.0,0.0,0.0]
tspan = (0.0,50.0)
prob = SDEProblem(dynamics!,noise!,u0,tspan,callback=cb,tstops=[tswitch])

sol = solve(prob);

# Vars (1,2) show the environment. (3,4) is the agent and (5) is the action.

# Plot all states. This generate Fig. 1
plot(sol,dpi=600,labels=[L"x" L"\dot{x}" L"\mu_0" L"\mu_1" L"a"])
#savefig("viz/model.png")

# Predicted / actual states. This should be close to a straight line - except for the spike when the environment is reset
#plot(sol,vars=(1,3),dpi=600)
#savefig("viz/tracking.png")
