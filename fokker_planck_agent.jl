using Pkg;Pkg.activate(".");Pkg.instantiate()
using Zygote, Random, DifferentialEquations
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
    s,v,x,x_prime,a = u

    # Parameters. 2 precisions and powerscaling
    ϵ_1 = 0.1
    ϵ_2 = 0.1
    force = 1

    # Hamiltonians. Set m and k to 1.0/0.1.
    H(x,v) = 1/2 * x^2 + 1/2 * v^2
    H_2(x,v) = 1/(2*10) * x^2 + 1/2 * 0.1 * v^2

    # VFE as potential function. With the right choice of p and q in boils down
    # to precision weighed prediction errors. See Buckley for derivation
    U(s,x,x_prime) = 0.5 * ( 1/ϵ_1 * (s - x)^2 + 1/ϵ_2 * (x_prime - x)^2)

    # Derivative of softmax function for action
    dxda(a) =-force * (sech(a) * sech(a))

    # Get gradients for both the agent and the generative process
    ∇_U = gradient( (s_,x_,x_prime_) -> U(s_,x_,x_prime_), s,x,x_prime)

    # Hack in a switch in the Hamiltonian of the generative process
    if t<tswitch
	∇_H = gradient( (s_,v_) -> H(s_,v_), s,v)
    else
	∇_H = gradient( (s_,v_) -> H_2(s_,v_), s,v)
    end

    # Final entry is for action. Split through chain rule by virtue of Markov Blanket
    ∇ = [∇_H[1],∇_H[2],∇_U[2],∇_U[3], ∇_U[1] *  dxda(a)]

    # Define shared Q and Γ matrices for the whole system. First 2 rows follow H, the Hamiltonian of a simple harmonic oscillator. Rows 3 and 4 governs dynamics of the agent. Final row is action which is not part of the generative model as per Friston and Ao (2012).
    Q = [
	 0.0 -1.0 0.0 0.0 0.0;
	 1.0 0.0 0.0 0.0 0.0;
	 0.0 0.0 0.0 -1.0 0.0;
	 0.0 0.0 1.0 0.0 0.0;
	 0.0 0.0 0.0 0.0 0.0
	]

    # ϕ(a,a_prime) = ( force / a_prime) *tanh(a). We add this later to avoid errors when a_prime=0
    # Random fluctuations govern entries of Γ. This performs gradient descent on potential function
    Γ = [
	 0.0 0.0 0.0 0.0 0.0;
	 0.0 0.0 0.0 0.0 0.0;
	 0.0 0.0 0.1 0.0 0.0;
	 0.0 0.0 0.0 0.1 0.0;
	 0.0 0.0 0.0 0.0 1.0
	]

    # Calculate actual time derivatives. There has to be a better way to do bookkeeping
    dudt = -(Q + Γ) * ∇
    du[1] = dudt[1]
    du[2] = dudt[2] - force * tanh(a) # We apply action here to avoid accidentally dividing by 0 when calculating ϕ
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
    du[5] = 0.0 # Note that we don't add noise to active states.

end

# Initial conditions and timespan to solve
u0 = [2.0,2.0,0.0,0.0,0.0]
tspan = (0.0,50.0)
prob = SDEProblem(dynamics!,noise!,u0,tspan,callback=cb,tstops=[tswitch])

sol = solve(prob);


# Vars (1,2) show the environments. (3,4) are the agent and (5) is the action.
using Plots

# Plot all states
plot(sol,dpi=600);savefig("viz/model.png")

# Predicted / actual states. This should be close to a straight line - except for the spike when the environment is reset
plot(sol,vars=(1,3),dpi=600); savefig("viz/tracking.png")
