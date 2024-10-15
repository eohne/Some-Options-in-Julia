# Options Pricing Models

This repository contains various example implementations of option pricing models in Julia. These models cover a range of complexities, from the basic Black-Scholes model to more advanced models like Heston-Hull-White, including both European and American options, as well as barrier options.

## Disclaimer

**IMPORTANT**: This code is provided for educational and illustrative purposes only. It has not been thoroughly tested for accuracy or performance in real-world scenarios. Do not use this code for actual financial decision-making or trading without extensive validation and testing.

## Models and Implementations

### 1. Black-Scholes Model
- European options pricing
- Includes dividend yield
- Both analytical solution and Monte Carlo simulation

The Black-Scholes model assumes that the underlying asset price follows a geometric Brownian motion:

$dS(t) = (r - q)S(t)dt + \sigma S(t)dW(t)$

where $r$ is the risk-free rate, $q$ is the dividend yield, $\sigma$ is the volatility, and $W(t)$ is a standard Brownian motion.

```julia
function black_scholes_option(;
    S0::Float64,        # Initial stock price
    K::Float64,         # Strike price
    T::Float64,         # Time to maturity in years
    r::Float64,         # Risk-free interest rate
    q::Float64,         # Dividend yield
    σ::Float64,         # Volatility
    option_type::Symbol,# Type of option (:call or :put)
    M::Int = 100000     # Number of Monte Carlo simulations
)::Tuple{Float64, Float64}
```

### 2. Monte Carlo Simulation for European and Barrier Options
- Supports both European call options and up-and-out barrier options
- Multi-threaded CPU and CUDA GPU implementations

This implementation uses the risk-neutral valuation formula:

$V = e^{-rT} \mathbb{E}[\max(S(T) - K, 0)]$

For the barrier option, it becomes:

$V_{barrier} = e^{-rT} E[\max(S(T) - K, 0) \cdot I_{\max_{0 \leq t \leq T} S(t) < B}]$  

where $B$ is the barrier level.

```julia
function monte_carlo_european_barrier_option(;
    r::Float64,         # Risk-free interest rate
    q::Float64,         # Dividend yield
    sigma::Float64,     # Volatility
    S0::Float64,        # Initial stock price
    K::Float64,         # Strike price
    T::Float64,         # Time to maturity in years
    B::Float64,         # Barrier price
    M::Int,             # Number of Monte Carlo simulations
    dt::Float64 = 1/252,# Time step size
    use_cuda::Bool = false # Whether to use CUDA GPU implementation
)::Tuple{Float64, Float64}
```

### 3. Monte Carlo Simulation for American Options
- Least Squares Monte Carlo (LSM) method
- Multi-threaded implementation

The LSM method uses regression to estimate the continuation value of the option at each time step, allowing for optimal early exercise decisions.

```julia
function monte_carlo_option_american_cpu(;
    r::Float64,         # Risk-free interest rate
    q::Float64,         # Dividend yield
    sigma::Float64,     # Volatility
    S0::Float64,        # Initial stock price
    K::Float64,         # Strike price
    T::Float64,         # Time to maturity in years
    M::Int,             # Number of Monte Carlo simulations
    option_type::Symbol,# Type of option (:call or :put)
    dt::Float64 = 1/252 # Time step size
)::Float64
```

### 4. Binomial Tree Model for American Options
- Discrete-time approximation of the Black-Scholes model

The binomial tree model discretizes both time and possible stock price movements, creating a tree of potential future stock prices and working backwards to determine the option price.

```julia
function binomial_tree_american_option(;
    r::Float64,         # Risk-free interest rate
    q::Float64,         # Dividend yield
    sigma::Float64,     # Volatility
    S0::Float64,        # Initial stock price
    K::Float64,         # Strike price
    T::Float64,         # Time to maturity in years
    M::Int,             # Number of time steps in the tree
    option_type::Symbol,# Type of option (:call or :put)
    dt::Float64 = 1/252 # Time step size
)::Float64
```

### 5. Finite Difference Method for American Options
- Uses Crank-Nicolson scheme
- Solves the Black-Scholes PDE numerically

The finite difference method solves the Black-Scholes partial differential equation:

$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2} + (r-q)S\frac{\partial V}{\partial S} - rV = 0$

```julia
function finite_difference_american_option(;
    r::Float64,         # Risk-free interest rate
    q::Float64,         # Dividend yield
    sigma::Float64,     # Volatility
    S0::Float64,        # Initial stock price
    K::Float64,         # Strike price
    T::Float64,         # Time to maturity in years
    M::Int,             # Number of spatial steps
    option_type::Symbol,# Type of option (:call or :put)
    dt::Float64 = 1/252 # Time step size
)::Float64
```

### 6. Heston-Hull-White Model for European Options
- Stochastic volatility and stochastic interest rate
- Monte Carlo simulation

The Heston-Hull-White model extends the Black-Scholes model by incorporating stochastic volatility and stochastic interest rates:

$dS(t) = (r(t) - q)S(t)dt + \sqrt{v(t)}S(t)dW_1(t)$  
$dv(t) = \kappa_v(\theta_v - v(t))dt + \sigma_v\sqrt{v(t)}dW_2(t)$  
$dr(t) = \kappa_r(\theta_r - r(t))dt + \sigma_rdW_3(t)$  

where $W_1$, $W_2$, and $W_3$ are correlated Brownian motions.

```julia
function heston_hull_white_option(;
    S0::Float64,        # Initial stock price
    v0::Float64,        # Initial variance
    r0::Float64,        # Initial interest rate
    K::Float64,         # Strike price
    T::Float64,         # Time to maturity in years
    q::Float64,         # Dividend yield
    κ_v::Float64,       # Mean reversion speed of variance
    θ_v::Float64,       # Long-term mean of variance
    σ_v::Float64,       # Volatility of variance
    κ_r::Float64,       # Mean reversion speed of interest rate
    θ_r::Float64,       # Long-term mean of interest rate
    σ_r::Float64,       # Volatility of interest rate
    ρ_sv::Float64,      # Correlation between stock price and variance
    ρ_sr::Float64,      # Correlation between stock price and interest rate
    ρ_vr::Float64,      # Correlation between variance and interest rate
    option_type::Symbol,# Type of option (:call or :put)
    M::Int,             # Number of Monte Carlo simulations
    dt::Float64 = 1/252 # Time step size
)::Tuple{Float64, Float64}
```

### 7. Heston-Hull-White Model for American Options
- Stochastic volatility and stochastic interest rate
- Least Squares Monte Carlo (LSM) method

This implementation combines the Heston-Hull-White model with the LSM method for pricing American options, allowing for early exercise in a complex stochastic environment.

```julia
function heston_hull_white_american_option(;
    S0::Float64,        # Initial stock price
    v0::Float64,        # Initial variance
    r0::Float64,        # Initial interest rate
    K::Float64,         # Strike price
    T::Float64,         # Time to maturity in years
    q::Float64,         # Dividend yield
    κ_v::Float64,       # Mean reversion speed of variance
    θ_v::Float64,       # Long-term mean of variance
    σ_v::Float64,       # Volatility of variance
    κ_r::Float64,       # Mean reversion speed of interest rate
    θ_r::Float64,       # Long-term mean of interest rate
    σ_r::Float64,       # Volatility of interest rate
    ρ_sv::Float64,      # Correlation between stock price and variance
    ρ_sr::Float64,      # Correlation between stock price and interest rate
    ρ_vr::Float64,      # Correlation between variance and interest rate
    option_type::Symbol,# Type of option (:call or :put)
    M::Int,             # Number of Monte Carlo simulations
    dt::Float64 = 1/252 # Time step size
)::Tuple{Float64, Float64}
```

## Implementation Details

1. **Monte Carlo Methods**: These use random sampling to simulate many possible price paths of the underlying asset. The option price is then estimated as the discounted expected payoff. For American options, the Least Squares Monte Carlo method is used to estimate the optimal exercise boundary.

2. **Binomial Tree**: This method creates a discrete-time model of the possible price paths, allowing for simple calculation of early exercise value at each node. It converges to the Black-Scholes model as the number of time steps increases.

3. **Finite Difference**: This numerical method discretizes the Black-Scholes PDE and solves it backwards in time. The Crank-Nicolson scheme is used for its stability and accuracy.

4. **Heston-Hull-White**: This advanced model incorporates stochastic volatility and interest rates. It's implemented using Euler-Maruyama discretization for the SDEs and uses Cholesky decomposition to generate correlated Brownian motions.

## Notes

- All implementations support both call and put options.
- Most functions use a default time step (`dt`) of 1/252, approximating one trading day.
- The Heston-Hull-White implementations use multi-threading for improved performance. Ensure Julia is started with multiple threads for optimal performance.
- The American option implementations allow for early exercise, while European options can only be exercised at expiration.
- The Monte Carlo method for European and barrier options supports both CPU and GPU (CUDA) implementations.

## Usage Examples

Here are some basic examples of how to use these functions:

```julia
# Black-Scholes European Option
price, mc_price = black_scholes_option(
    S0 = 100.0,
    K = 100.0,
    T = 1.0,
    r = 0.05,
    q = 0.02,
    σ = 0.2,
    option_type = :call
)

# Monte Carlo European and Barrier Option
V, V_barrier = monte_carlo_european_barrier_option(
    r = 0.03,
    q = 0.02,
    sigma = 0.2,
    S0 = 100.0,
    K = 100.0,
    T = 1.0,
    B = 120.0,
    M = 10^7
)

# Monte Carlo American Option
price = monte_carlo_option_american_cpu(
    r = 0.05,
    q = 0.02,
    sigma = 0.2,
    S0 = 100.0,
    K = 100.0,
    T = 1.0,
    M = 100000,
    option_type = :put
)

# Heston-Hull-White American Option
price, se = heston_hull_white_american_option(
    S0 = 100.0,
    v0 = 0.04,
    r0 = 0.03,
    K = 100.0,
    T = 1.0,
    q = 0.02,
    κ_v = 1.0,
    θ_v = 0.04,
    σ_v = 0.2,
    κ_r = 0.5,
    θ_r = 0.04,
    σ_r = 0.1,
    ρ_sv = -0.5,
    ρ_sr = 0.1,
    ρ_vr = 0.0,
    option_type = :call,
    M = 100000
)
```