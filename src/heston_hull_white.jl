using Statistics
using Random
using Distributions
using DifferentialEquations
using Base.Threads
using LinearAlgebra


"""
    heston_hull_white_option(;
        S0::Float64,
        v0::Float64,
        r0::Float64,
        K::Float64,
        T::Float64,
        q::Float64,
        κ_v::Float64,
        θ_v::Float64,
        σ_v::Float64,
        κ_r::Float64,
        θ_r::Float64,
        σ_r::Float64,
        ρ_sv::Float64,
        ρ_sr::Float64,
        ρ_vr::Float64,
        option_type::Symbol,
        M::Int,
        dt::Float64 = 1/252
    )::Tuple{Float64, Float64}

Price a European option (call or put) using the Heston-Hull-White stochastic volatility and stochastic interest rate model, incorporating a continuous dividend yield.

# Model
The Heston-Hull-White model with dividend yield:

dS(t) = (r(t) - q)S(t)dt + √v(t)S(t)dW₁(t)
dv(t) = κ_v(θ_v - v(t))dt + σ_v√v(t)dW₂(t)
dr(t) = κ_r(θ_r - r(t))dt + σ_rdW₃(t)

where W₁, W₂, and W₃ are correlated Brownian motions with correlations ρ_sv, ρ_sr, and ρ_vr.

# Assumptions
- The underlying asset follows a geometric Brownian motion with stochastic volatility and stochastic interest rate.
- Volatility follows the CIR (Cox-Ingersoll-Ross) process.
- Interest rate follows the Ornstein-Uhlenbeck process.
- Continuous dividend yield is modeled by adjusting the drift term of the stock price process.
- No transaction costs or taxes.

# Implementation
- Uses Euler-Maruyama discretization for numerical simulation of the SDEs.
- Implements a full truncation scheme to handle potential negative values in the variance process.
- Uses Cholesky decomposition to generate correlated Brownian motions.
- Employs Monte Carlo simulation with a specified number of paths.

# Arguments
- `S0::Float64`: Initial stock price
- `v0::Float64`: Initial variance
- `r0::Float64`: Initial interest rate
- `K::Float64`: Strike price
- `T::Float64`: Time to maturity in years
- `q::Float64`: Continuous dividend yield
- `κ_v::Float64`: Mean reversion speed of variance
- `θ_v::Float64`: Long-term mean of variance
- `σ_v::Float64`: Volatility of variance
- `κ_r::Float64`: Mean reversion speed of interest rate
- `θ_r::Float64`: Long-term mean of interest rate
- `σ_r::Float64`: Volatility of interest rate
- `ρ_sv::Float64`: Correlation between stock price and variance
- `ρ_sr::Float64`: Correlation between stock price and interest rate
- `ρ_vr::Float64`: Correlation between variance and interest rate
- `option_type::Symbol`: Type of option, either :call or :put
- `M::Int`: Number of Monte Carlo simulations
- `dt::Float64`: Time step size for discretization (default: 1/252, approximately one trading day)

# Returns
- `Tuple{Float64, Float64}`: A tuple containing the estimated option price and its standard error

# Note
This implementation uses multi-threading for improved performance. Ensure Julia is started with multiple threads for optimal performance.
"""
function heston_hull_white_option(;
    S0::Float64,
    v0::Float64,
    r0::Float64,
    K::Float64,
    T::Float64,
    q::Float64,
    κ_v::Float64,
    θ_v::Float64,
    σ_v::Float64,
    κ_r::Float64,
    θ_r::Float64,
    σ_r::Float64,
    ρ_sv::Float64,
    ρ_sr::Float64,
    ρ_vr::Float64,
    option_type::Symbol,
    M::Int,
    dt::Float64 = 1/252
)::Tuple{Float64, Float64}
    @assert option_type in [:call, :put] "Option type must be either :call or :put"
    
    # Calculate number of time steps
    N = ceil(Int, T / dt)
    dt = T / N  # Recalculate dt to ensure it divides T evenly
    
    # Define the drift function
    function drift!(du, u, p, t)
        S, v, r = u
        du[1] = (r - q) * S  # Adjusted for dividend yield
        du[2] = κ_v * (θ_v - v)
        du[3] = κ_r * (θ_r - r)
    end
    
    # Define the diffusion function
    function diffusion!(du, u, p, t)
        S, v, r = u
        du[1] = sqrt(max(v, 0)) * S
        du[2] = σ_v * sqrt(max(v, 0))
        du[3] = σ_r
    end
    
    # Define the noise correlations
    Γ = [1.0       ρ_sv    ρ_sr;
         ρ_sv      1.0     ρ_vr;
         ρ_sr      ρ_vr    1.0]
    
    # Compute the Cholesky decomposition
    C = cholesky(Symmetric(Γ)).L
    
    # Initial conditions
    u0 = [S0, v0, r0]
    tspan = (0.0, T)
    
    # Define the problem
    prob = SDEProblem(drift!, diffusion!, u0, tspan, noise_rate_prototype=C)
    
    # Simulate paths
    function sim_path()
        sol = solve(prob, EM(), dt=dt, adaptive=false)
        S_T = sol[end][1]
        r_avg = mean(sol[i][3] for i in 1:length(sol))
        return S_T, r_avg
    end
    
    # Monte Carlo simulation
    payoffs = zeros(M)
    @threads for i in 1:M
        S_T, r_avg = sim_path()
        payoff = option_type == :call ? max(S_T - K, 0) : max(K - S_T, 0)
        payoffs[i] = payoff * exp(-r_avg * T)
    end
    
    # Calculate price and standard error
    price = mean(payoffs)
    se = std(payoffs) / sqrt(M)
    
    return price, se
end






function least_squares_regression(X::Matrix{Float64}, y::Vector{Float64})
    try
        return X \ y
    catch e
        if isa(e, SingularException)
            # Use ridge regression with a small regularization parameter
            λ = 1e-6
            return (X'X + λ*I) \ (X'y)
        else
            rethrow(e)
        end
    end
end

"""
    heston_hull_white_american_option(;
        S0::Float64,
        v0::Float64,
        r0::Float64,
        K::Float64,
        T::Float64,
        q::Float64,
        κ_v::Float64,
        θ_v::Float64,
        σ_v::Float64,
        κ_r::Float64,
        θ_r::Float64,
        σ_r::Float64,
        ρ_sv::Float64,
        ρ_sr::Float64,
        ρ_vr::Float64,
        option_type::Symbol,
        M::Int,
        dt::Float64 = 1/252
    )::Tuple{Float64, Float64}

Price an American option (call or put) using the Heston-Hull-White model with the Least Squares Monte Carlo method.

# Model
The Heston-Hull-White model combines stochastic volatility and stochastic interest rates:

dS(t) = (r(t) - q)S(t)dt + √v(t)S(t)dW₁(t)
dv(t) = κ_v(θ_v - v(t))dt + σ_v√v(t)dW₂(t)
dr(t) = κ_r(θ_r - r(t))dt + σ_rdW₃(t)

where W₁, W₂, and W₃ are correlated Brownian motions.

# Method
- Uses Euler-Maruyama discretization for simulating paths
- Implements Longstaff-Schwartz Least Squares Monte Carlo for American option valuation
- Employs an enhanced regression model for continuation value estimation
- Includes fallback to ridge regression for numerical stability

# Implementation Details
- Path Simulation:
  - Ensures non-negative variance by using max(v, 0) in both drift and diffusion terms
  - Uses multi-threading for parallel path generation
- Least Squares Monte Carlo:
  - Uses an expanded set of basis functions: [1, S, S², √v, log(v), r]
  - Implements a wider in-the-money (ITM) range for regression: S > 0.8K for calls, S < 1.2K for puts
  - Only performs regression if there are at least 5 ITM paths
  - Uses ridge regression as a fallback if the design matrix is singular
    (adds a small regularization term λ = 1e-6 to the diagonal of X'X)
- Discounting:
  - Uses the average simulated interest rate for discounting in both LSM and final payoff calculation

# Arguments
- `S0::Float64`: Initial stock price
- `v0::Float64`: Initial variance
- `r0::Float64`: Initial interest rate
- `K::Float64`: Strike price
- `T::Float64`: Time to maturity in years
- `q::Float64`: Continuous dividend yield
- `κ_v::Float64`: Mean reversion speed of variance
- `θ_v::Float64`: Long-term mean of variance
- `σ_v::Float64`: Volatility of variance
- `κ_r::Float64`: Mean reversion speed of interest rate
- `θ_r::Float64`: Long-term mean of interest rate (Note: Ideally should be time-dependent)
- `σ_r::Float64`: Volatility of interest rate
- `ρ_sv::Float64`: Correlation between stock price and variance
- `ρ_sr::Float64`: Correlation between stock price and interest rate
- `ρ_vr::Float64`: Correlation between variance and interest rate
- `option_type::Symbol`: Type of option, either :call or :put
- `M::Int`: Number of Monte Carlo simulations
- `dt::Float64`: Time step size for discretization (default: 1/252, approximately one trading day)

# Returns
- `Tuple{Float64, Float64}`: Estimated option price and its standard error

# Notes
- This function uses multi-threading for path simulation. Ensure Julia is started with multiple threads for optimal performance.
- The implementation is designed to be robust against numerical instabilities, but extreme parameter values may still cause issues.
- The Hull-White part of the model uses a constant long-term mean (θ_r), which is a simplification. For more accuracy, this should ideally be a time-dependent function.
- The regression model and ITM conditions are heuristics and may need adjustment for specific use cases or extreme parameter sets.
"""
function heston_hull_white_american_option(;
    S0::Float64,
    v0::Float64,
    r0::Float64,
    K::Float64,
    T::Float64,
    q::Float64,
    κ_v::Float64,
    θ_v::Float64,
    σ_v::Float64,
    κ_r::Float64,
    θ_r::Float64,
    σ_r::Float64,
    ρ_sv::Float64,
    ρ_sr::Float64,
    ρ_vr::Float64,
    option_type::Symbol,
    M::Int,
    dt::Float64 = 1/252
)::Tuple{Float64, Float64}
    @assert option_type in [:call, :put] "Option type must be either :call or :put"
    
    N = ceil(Int, T / dt)
    dt = T / N

    function drift!(du, u, p, t)
        S, v, r = u
        du[1] = (r - q) * S
        du[2] = κ_v * (θ_v - max(v, 0))
        du[3] = κ_r * (θ_r - r)
    end
    
    function diffusion!(du, u, p, t)
        S, v, r = u
        du[1] = sqrt(max(v, 0)) * S
        du[2] = σ_v * sqrt(max(v, 0))
        du[3] = σ_r
    end
    
    Γ = [1.0 ρ_sv ρ_sr; ρ_sv 1.0 ρ_vr; ρ_sr ρ_vr 1.0]
    C = cholesky(Symmetric(Γ)).L
    
    u0 = [S0, v0, r0]
    tspan = (0.0, T)
    prob = SDEProblem(drift!, diffusion!, u0, tspan, noise_rate_prototype=C)

    # Simulate paths
    paths = Matrix{Float64}(undef, M, N+1)
    vars = Matrix{Float64}(undef, M, N+1)
    rates = Matrix{Float64}(undef, M, N+1)
    @threads for i in 1:M
        sol = solve(prob, EM(), dt=dt, adaptive=false)
        paths[i, :] = [sol[j][1] for j in 1:N+1]
        vars[i, :] = [max(sol[j][2], 0) for j in 1:N+1]  # Ensure non-negative variance
        rates[i, :] = [sol[j][3] for j in 1:N+1]
    end

    # Initialize payoffs at maturity
    payoffs = option_type == :call ? max.(paths[:, end] .- K, 0) : max.(K .- paths[:, end], 0)

    # Backward induction for LSM
    for t in N:-1:2
        S = paths[:, t]
        v = vars[:, t]
        r = rates[:, t]
        
        # Correct ITM condition for both call and put
        itm = option_type == :call ? S .> K * 0.8 : S .< K * 1.2  # Wider ITM range
        
        if sum(itm) > 5  # Ensure enough samples for regression
            X = hcat(ones(sum(itm)), S[itm], S[itm].^2, sqrt.(v[itm]), log.(v[itm]), r[itm])
            y = payoffs[itm] .* exp.(-mean(rates[itm, t:end], dims=2) * dt)
            β = least_squares_regression(X, vec(y))
            continuation_value = X * β
            immediate_exercise = option_type == :call ? max.(S[itm] .- K, 0) : max.(K .- S[itm], 0)
            exercise = immediate_exercise .> continuation_value
            payoffs[itm][exercise] = immediate_exercise[exercise]
        end
    end

    # Discount final payoffs to time 0 using average rate
    discount_factors = exp.(-mean(rates, dims=2) * T)
    discounted_payoffs = payoffs .* vec(discount_factors)

    price = mean(discounted_payoffs)
    se = std(discounted_payoffs) / sqrt(M)
    
    return price, se
end
