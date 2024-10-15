using Base.Threads
using Random
using CUDA


"""
    black_scholes_option(;
        S0::Float64,
        K::Float64,
        T::Float64,
        r::Float64,
        q::Float64,
        σ::Float64,
        option_type::Symbol,
        M::Int = 100000
    )::Tuple{Float64, Float64}

Price a European option (call or put) using the Black-Scholes model, incorporating a continuous dividend yield.

# Model
The Black-Scholes model with dividend yield:

dS(t) = (r - q)S(t)dt + σS(t)dW(t)

where W(t) is a standard Brownian motion.

# Assumptions
- The underlying asset follows a geometric Brownian motion with constant volatility.
- Constant risk-free interest rate.
- Continuous dividend yield.
- No transaction costs or taxes.
- European-style option (can only be exercised at expiration).

# Implementation
- Uses the analytical Black-Scholes formula for option pricing.
- Implements a Monte Carlo simulation for comparison and error estimation.

# Arguments
- `S0::Float64`: Initial stock price
- `K::Float64`: Strike price
- `T::Float64`: Time to maturity in years
- `r::Float64`: Risk-free interest rate (annualized)
- `q::Float64`: Continuous dividend yield (annualized)
- `σ::Float64`: Volatility of the underlying asset (annualized)
- `option_type::Symbol`: Type of option, either :call or :put
- `M::Int`: Number of Monte Carlo simulations (default: 100000)

# Returns
- `Tuple{Float64, Float64}`: A tuple containing the analytical Black-Scholes price and the Monte Carlo price
"""
function black_scholes_option(;
    S0::Float64,
    K::Float64,
    T::Float64,
    r::Float64,
    q::Float64,
    σ::Float64,
    option_type::Symbol,
    M::Int = 100000
)::Tuple{Float64, Float64}
    @assert option_type in [:call, :put] "Option type must be either :call or :put"

    # Black-Scholes formula
    d1 = (log(S0/K) + (r - q + 0.5σ^2)*T) / (σ*sqrt(T))
    d2 = d1 - σ*sqrt(T)
    
    if option_type == :call
        bs_price = S0 * exp(-q*T) * cdf(Normal(), d1) - K * exp(-r*T) * cdf(Normal(), d2)
    else  # put
        bs_price = K * exp(-r*T) * cdf(Normal(), -d2) - S0 * exp(-q*T) * cdf(Normal(), -d1)
    end

    # Monte Carlo simulation
    dt = T / 252  # Approximately daily steps
    drift = (r - q - 0.5σ^2) * dt
    diffusion = σ * sqrt(dt)
    
    payoffs = zeros(M)
    for i in 1:M
        ST = S0 * exp(drift * 252 + diffusion * sum(randn(252)))
        payoff = option_type == :call ? max(ST - K, 0) : max(K - ST, 0)
        payoffs[i] = payoff * exp(-r * T)
    end
    
    mc_price = mean(payoffs)
    
    return bs_price, mc_price
end










# Option Type:

# European-style option (can only be exercised at expiration)
# Call option (right to buy the underlying asset)
# Up-and-out barrier option (becomes worthless if the asset price reaches or exceeds a certain barrier level)


# Model:
# The code uses a Monte Carlo simulation based on the Black-Scholes-Merton model.
# Key Assumptions:

# The underlying asset price follows a geometric Brownian motion
# Constant risk-free interest rate (r)
# Constant dividend yield (q)
# Constant volatility (sigma)
# No transaction costs or taxes
# Continuous time model discretized into small time steps (dt)



function monte_carlo_european_barrier_option_cpu(
    r::Float64,
    q::Float64,
    sigma::Float64,
    S0::Float64,
    K::Float64,
    T::Float64,
    B::Float64,
    M::Int,
    dt::Float64
)::Tuple{Float64, Float64}

    N::Int = round(Int, T/dt) + 1
    
    # Precompute constants
    drift::Float64 = (r - q - 0.5*sigma^2) * dt
    vol::Float64 = sigma * sqrt(dt)
    discount::Float64 = exp(-r * T)
    log_B::Float64 = log(B)
    log_S0::Float64 = log(S0)

    sum_payoff = zeros(Float64, nthreads())
    sum_barrier_payoff = zeros(Float64, nthreads())

    paths_per_thread = cld(M, nthreads())

    @threads for t in 1:nthreads()
        rng = Random.default_rng()
        local_sum_payoff::Float64 = 0.0
        local_sum_barrier_payoff::Float64 = 0.0

        for _ in 1:paths_per_thread
            log_S_up::Float64 = log_S0
            log_S_down::Float64 = log_S0
            max_log_S_up::Float64 = log_S0
            max_log_S_down::Float64 = log_S0
            
            for _ in 2:N
                Z::Float64 = randn(rng)
                log_S_up += drift + vol * Z
                log_S_down += drift - vol * Z
                max_log_S_up = max(max_log_S_up, log_S_up)
                max_log_S_down = max(max_log_S_down, log_S_down)
            end
            
            S_up = exp(log_S_up)
            S_down = exp(log_S_down)
            payoff_up = max(S_up - K, 0)
            payoff_down = max(S_down - K, 0)
            
            local_sum_payoff += payoff_up + payoff_down
            local_sum_barrier_payoff += ifelse(max_log_S_up < log_B, payoff_up, 0.0) +
                                        ifelse(max_log_S_down < log_B, payoff_down, 0.0)
        end

        sum_payoff[t] = local_sum_payoff
        sum_barrier_payoff[t] = local_sum_barrier_payoff
    end

    V::Float64 = discount * sum(sum_payoff) / (2 * M)
    V_barrier::Float64 = discount * sum(sum_barrier_payoff) / (2 * M)

    return V, V_barrier
end

function monte_carlo_european_barrier_option_gpu(
    r::Float32,
    q::Float32,
    sigma::Float32,
    S0::Float32,
    K::Float32,
    T::Float32,
    B::Float32,
    M::Int,
    dt::Float32
)::Tuple{Float32, Float32}

    N::Int32 = round(Int32, T/dt) + 1

    # Precompute constants
    drift::Float32 = (r - q - 0.5f0*sigma^2) * dt
    vol::Float32 = sigma * sqrt(dt)
    discount::Float32 = exp(-r * T)
    log_B::Float32 = log(B)
    log_S0::Float32 = log(S0)

    # CUDA kernel
    function mc_kernel!(payoff, barrier_payoff, M, N, drift, vol, log_S0, log_B, K)
        idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = blockDim().x * gridDim().x

        local_sum_payoff = 0.0f0
        local_sum_barrier_payoff = 0.0f0

        for i in idx:stride:M
            log_S_up = log_S0
            log_S_down = log_S0
            max_log_S_up = log_S0
            max_log_S_down = log_S0

            for _ in 2:N
                Z = randn(Float32)
                log_S_up += drift + vol * Z
                log_S_down += drift - vol * Z
                max_log_S_up = max(max_log_S_up, log_S_up)
                max_log_S_down = max(max_log_S_down, log_S_down)
            end

            S_up = exp(log_S_up)
            S_down = exp(log_S_down)
            payoff_up = max(S_up - K, 0.0f0)
            payoff_down = max(S_down - K, 0.0f0)

            local_sum_payoff += payoff_up + payoff_down
            local_sum_barrier_payoff += ifelse(max_log_S_up < log_B, payoff_up, 0.0f0) +
                                        ifelse(max_log_S_down < log_B, payoff_down, 0.0f0)
        end

        CUDA.@atomic payoff[] += local_sum_payoff
        CUDA.@atomic barrier_payoff[] += local_sum_barrier_payoff
        return
    end

    # Setup CUDA arrays
    d_payoff = CUDA.zeros(Float32, 1)
    d_barrier_payoff = CUDA.zeros(Float32, 1)

    # Launch kernel
    threads = 512  # This is a multiple of 32 and allows for 3 blocks per SM
    blocks = min(46 * 3, ceil(Int, M / threads))  # Ensure we don't exceed GPU limits
    @cuda threads=threads blocks=blocks mc_kernel!(d_payoff, d_barrier_payoff, M, N, drift, vol, log_S0, log_B, K)

    # Retrieve results
    h_payoff = Array(d_payoff)
    h_barrier_payoff = Array(d_barrier_payoff)

    V::Float32 = discount * h_payoff[1] / (2 * M)
    V_barrier::Float32 = discount * h_barrier_payoff[1] / (2 * M)

    return V, V_barrier
end


"""
    monte_carlo_european_barrier_option(;
        r::Float64,
        q::Float64,
        sigma::Float64,
        S0::Float64,
        K::Float64,
        T::Float64,
        B::Float64,
        M::Int,
        dt::Float64 = 1/252,
        use_cuda::Bool = false
    )::Tuple{Float64, Float64}

Monte Carlo option pricing function, supporting both CPU and GPU (CUDA) implementations.

This function calculates the price of a European call option and a barrier option using Monte Carlo simulation.
It can use either a multi-threaded CPU implementation or a CUDA GPU implementation based on the `use_cuda` parameter.

# Arguments
- `r::Float64`: Risk-free interest rate
- `q::Float64`: Dividend yield
- `sigma::Float64`: Volatility
- `S0::Float64`: Initial stock price
- `K::Float64`: Strike price
- `T::Float64`: Time to maturity in years
- `B::Float64`: Barrier price for the barrier option
- `M::Int`: Number of Monte Carlo simulations
- `dt::Float64`: Time step size (default: 1/252, approximately one trading day)
- `use_cuda::Bool`: Whether to use the CUDA implementation (default: false)

# Returns
- `Tuple{Float64, Float64}`: (V, V_barrier)
  - `V`: Price of the European call option
  - `V_barrier`: Price of the barrier option

# Note
When `use_cuda` is `true`, the function uses single-precision floating-point numbers (`Float32`)
for compatibility with the CUDA implementation. This may result in slight differences in the
output compared to the CPU version, which uses double-precision (`Float64`).

# Examples
```julia
# CPU version
V, V_barrier = monte_carlo_option(r=0.03, q=0.02, sigma=0.2, S0=100.0, K=100.0, T=1.0, B=120.0, M=10^7)

# GPU (CUDA) version
V_cuda, V_barrier_cuda = monte_carlo_option(r=0.03, q=0.02, sigma=0.2, S0=100.0, K=100.0, T=1.0, B=120.0, M=10^7, use_cuda=true)
```
"""
function monte_carlo_european_barrier_option(;
    r::Float64,
    q::Float64,
    sigma::Float64,
    S0::Float64,
    K::Float64,
    T::Float64,
    B::Float64,
    M::Int,
    dt::Float64 = 1/252,
    use_cuda::Bool = false
)::Tuple{Float64, Float64}
    # Assertions
    @assert r >= 0 "Risk-free rate must be non-negative"
    @assert q >= 0 "Dividend yield must be non-negative"
    @assert sigma > 0 "Volatility must be positive"
    @assert S0 > 0 "Initial stock price must be positive"
    @assert K > 0 "Strike price must be positive"
    @assert T > 0 "Time to maturity must be positive"
    @assert B > S0 "Barrier price must be greater than initial stock price"
    @assert M > 0 "Number of simulations must be positive"
    @assert dt > 0 "Time step must be positive"

    if use_cuda
        return monte_carlo_european_barrier_option_gpu(
            Float32(r),
            Float32(q),
            Float32(sigma),
            Float32(S0),
            Float32(K),
            Float32(T),
            Float32(B),
            M,
            Float32(dt)
        )
    else
        return monte_carlo_european_barrier_option_cpu(
            r,
            q,
            sigma,
            S0,
            K,
            T,
            B,
            M,
            dt
        )
    end
end
