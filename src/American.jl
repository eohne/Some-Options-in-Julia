using Statistics
using LinearAlgebra
using Base.Threads
using Random


# Option Type:

# American-style option (can be exercised at any time up to expiration)
# Can price both call and put options (specified by the option_type parameter)


# Model:
# The code uses a Monte Carlo simulation based on the Black-Scholes-Merton model, combined with the Longstaff-Schwartz method for early exercise decisions.
# Key Assumptions:

# The underlying asset price follows a geometric Brownian motion
# Constant risk-free interest rate (r)
# Constant dividend yield (q)
# Constant volatility (sigma)
# No transaction costs or taxes
# Discrete time model with specified time steps (dt)


# Method Details:

# Uses Monte Carlo simulation to generate price paths
# Implements the Longstaff-Schwartz algorithm for early exercise decisions
# Uses least squares regression to estimate continuation values
# Implements parallel computing using Julia's multi-threading capabilities


"""
    monte_carlo_option_american_cpu(;
        r::Float64,
        q::Float64,
        sigma::Float64,
        S0::Float64,
        K::Float64,
        T::Float64,
        M::Int,
        option_type::Symbol,
        dt::Float64 = 1/252
    )::Float64

Conservative multi-threaded Monte Carlo simulation for pricing American call and put options using the Least Squares Monte Carlo (LSM) method.

# Arguments
- `r::Float64`: Risk-free interest rate
- `q::Float64`: Continuous dividend yield
- `sigma::Float64`: Volatility
- `S0::Float64`: Initial stock price
- `K::Float64`: Strike price
- `T::Float64`: Time to maturity in years
- `M::Int`: Number of Monte Carlo simulations
- `option_type::Symbol`: Type of option, either :call or :put
- `dt::Float64`: Time step size (default: 1/252, approximately one trading day)

# Returns
- `Float64`: Price of the American option (call or put)
"""
function monte_carlo_option_american_cpu(;
    r::Float64,
    q::Float64,
    sigma::Float64,
    S0::Float64,
    K::Float64,
    T::Float64,
    M::Int,
    option_type::Symbol,
    dt::Float64 = 1/252
)::Float64
    @assert option_type in [:call, :put] "Option type must be either :call or :put"
    
    N = ceil(Int, T / dt)
    dt = T / N
    
    nudt = (r - q - 0.5*sigma^2) * dt
    sigsdt = sigma * sqrt(dt)
    discount = exp(-r * dt)

    # Payoff function
    payoff = option_type == :call ? (S) -> max.(S .- K, 0) : (S) -> max.(K .- S, 0)

    # Number of threads
    nthreads = Threads.nthreads()
    M_per_thread = cld(M, nthreads)

    # Arrays to store results from each thread
    thread_results = Vector{Float64}(undef, nthreads)

    # Multi-threaded Monte Carlo simulation
    Threads.@threads for thread_id in 1:nthreads
        # Determine the number of paths for this thread
        start_idx = (thread_id - 1) * M_per_thread + 1
        end_idx = min(thread_id * M_per_thread, M)
        M_thread = end_idx - start_idx + 1

        # Generate paths
        paths = zeros(M_thread, N+1)
        paths[:, 1] .= S0
        for t in 2:N+1
            paths[:, t] = paths[:, t-1] .* exp.(nudt .+ sigsdt .* randn(M_thread))
        end

        option_values = payoff(paths[:, end])

        for t in N:-1:2
            S = paths[:, t]
            time_to_maturity = (N - t + 1) * dt
            df = exp(-r * time_to_maturity)
            dq = exp(-q * time_to_maturity)
            
            intrinsic_values = payoff(S)
            itm = intrinsic_values .> 0
            
            if sum(itm) > 0
                X = [ones(sum(itm)) S[itm] S[itm].^2 intrinsic_values[itm] sqrt.(S[itm])]
                cont_values = zeros(M_thread)
                cont_values[itm] = X * (X \ (option_values[itm] ./ df))
                
                exercise = itm .& (intrinsic_values .>= cont_values .* dq)
                
                option_values = discount * option_values
                option_values[exercise] = intrinsic_values[exercise]
            else
                option_values = discount * option_values
            end
        end

        thread_results[thread_id] = mean(max.(payoff(paths[:, 1]), option_values))
    end

    # Combine results from all threads
    return mean(thread_results)
end



















# Option Type:

# American-style option (can be exercised at any time up to expiration)
# Can price both call and put options (specified by the option_type parameter)


# Model:
# The code uses a binomial tree model, which is a discrete-time approximation of the continuous-time Black-Scholes-Merton model.
# Key Assumptions:

# The underlying asset price follows a binomial process (can move up or down by a fixed percentage in each time step)
# Constant risk-free interest rate (r)
# Constant dividend yield (q)
# Constant volatility (sigma)
# No transaction costs or taxes
# Discrete time model with specified time steps (dt)


# Method Details:

# Constructs a binomial tree of asset prices
# Uses backward induction to calculate option values at each node
# Allows for early exercise at each node of the tree
"""
    binomial_tree_american_option(;
        r::Float64,
        q::Float64,
        sigma::Float64,
        S0::Float64,
        K::Float64,
        T::Float64,
        M::Int,
        option_type::Symbol,
        dt::Float64 = 1/252
    )::Float64

Price an American option (call or put) using a binomial tree model.

# Arguments
- `r::Float64`: Risk-free interest rate
- `q::Float64`: Continuous dividend yield
- `sigma::Float64`: Volatility
- `S0::Float64`: Initial stock price
- `K::Float64`: Strike price
- `T::Float64`: Time to maturity in years
- `M::Int`: Not used in this implementation, included for compatibility
- `option_type::Symbol`: Type of option, either :call or :put
- `dt::Float64`: Time step size (default: 1/252, approximately one trading day)

# Returns
- `Float64`: Price of the American option (call or put)
"""
function binomial_tree_american_option(;
    r::Float64,
    q::Float64,
    sigma::Float64,
    S0::Float64,
    K::Float64,
    T::Float64,
    M::Int,
    option_type::Symbol,
    dt::Float64 = 1/252
)::Float64
    @assert option_type in [:call, :put] "Option type must be either :call or :put"
    
    N = ceil(Int, T / dt)
    dt = T / N  # Recalculate dt to ensure it divides T evenly
    
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp((r - q) * dt) - d) / (u - d)

    # Initialize asset prices at maturity
    S = S0 * d.^(N:-1:0) .* u.^(0:N)

    # Initialize option values at maturity
    if option_type == :call
        V = max.(S .- K, 0)
    else  # put
        V = max.(K .- S, 0)
    end

    # Backward induction through the tree
    for i in N:-1:1
        S = S0 * d.^(i-1:-1:0) .* u.^(0:i-1)
        hold_values = exp(-r * dt) * (p * V[2:i+1] + (1 - p) * V[1:i])
        if option_type == :call
            exercise_values = max.(S .- K, 0)
        else  # put
            exercise_values = max.(K .- S, 0)
        end
        V = max.(hold_values, exercise_values)
    end

    return V[1]
end















# Option Type:

# American-style option (can be exercised at any time up to expiration)
# Can price both call and put options (specified by the option_type parameter)


# Model:
# The code uses a finite difference method to solve the Black-Scholes partial differential equation (PDE) numerically.
# Key Assumptions:

# The underlying asset price follows a geometric Brownian motion
# Constant risk-free interest rate (r)
# Constant dividend yield (q)
# Constant volatility (sigma)
# No transaction costs or taxes
# Continuous time model discretized into small time steps (dt)


# Method Details:

# Uses the Crank-Nicolson scheme, which is an implicit finite difference method
# Implements a spatial grid for stock prices
# Uses the Thomas algorithm (tridiagonal matrix algorithm) to solve the linear system at each time step
# Incorporates early exercise feature for American options

"""
    finite_difference_american_option(;
        r::Float64,
        q::Float64,
        sigma::Float64,
        S0::Float64,
        K::Float64,
        T::Float64,
        M::Int,
        option_type::Symbol,
        dt::Float64 = 1/252
    )::Float64

Price an American option (call or put) using the Finite Difference Method with Crank-Nicolson scheme.

# Arguments
- `r::Float64`: Risk-free interest rate
- `q::Float64`: Continuous dividend yield
- `sigma::Float64`: Volatility
- `S0::Float64`: Initial stock price
- `K::Float64`: Strike price
- `T::Float64`: Time to maturity in years
- `M::Int`: Number of spatial steps (stock price steps)
- `option_type::Symbol`: Type of option, either :call or :put
- `dt::Float64`: Time step size (default: 1/252, approximately one trading day)

# Returns
- `Float64`: Price of the American option (call or put)
"""
function finite_difference_american_option(;
    r::Float64,
    q::Float64,
    sigma::Float64,
    S0::Float64,
    K::Float64,
    T::Float64,
    M::Int,
    option_type::Symbol,
    dt::Float64 = 1/252
)::Float64
    # Ensure that the option type is either a call or put
    @assert option_type in [:call, :put] "Option type must be either :call or :put"
    
    # 1. Setup Grid for Time-Stepping
    # ----------------------------------------------
    # Calculate the number of time steps N by dividing time to maturity T by dt
    N = ceil(Int, T / dt)
    dt = T / N  # Recalculate dt to ensure it divides T evenly

    # Set up the spatial grid for stock prices, with the stock price range [0, 2K]
    S_max = 2 * K  # Maximum stock price considered on the grid
    dS = S_max / M # Spatial step size (ΔS)
    
    # 2. Initialize Option Value at Expiry
    # ----------------------------------------------
    # At expiry (time T), the value of the option is determined by its intrinsic value
    # For a call option: max(S - K, 0); for a put option: max(K - S, 0)
    V = [max(option_type == :call ? (i-1)*dS - K : K - (i-1)*dS, 0.0) for i in 1:M+1]

    # 3. Precompute Coefficients for the PDE
    # ----------------------------------------------
    # These coefficients correspond to the discretization of the Black-Scholes PDE using the Crank-Nicolson method
    # - j represents grid points in the stock price dimension
    # - alpha, beta, and gamma represent the finite difference coefficients
    #   for the Crank-Nicolson scheme
    j = 0:M  # Grid points index
    alpha = 0.25 * dt * (sigma^2 * j.^2 .- (r - q) * j)  # From discretization of Black-Scholes PDE
    beta = -0.5 * dt * (sigma^2 * j.^2 .+ r)             # Diagonal elements in the implicit matrix
    gamma = 0.25 * dt * (sigma^2 * j.^2 .+ (r - q) * j)  # Crank-Nicolson weighting for future values

    # 4. Boundary Conditions for the Option
    # ----------------------------------------------
    # Boundary conditions are critical for finite difference schemes
    # At S = 0: The value of the put is K (since you can sell for K), and the call is 0 (since the stock has no value)
    # At S = S_max: For a call, it's simply S_max - K, and for a put it's essentially 0.
    boundary_put = K * exp(-r * dt)  # Put option boundary condition (lower bound)
    boundary_call = S_max - K * exp(-r * dt)  # Call option boundary condition (upper bound)

    # 5. Preallocate Arrays for Tridiagonal Matrix
    # ----------------------------------------------
    # Arrays a, b, c correspond to the tridiagonal matrix system to be solved at each time step
    # - a represents the sub-diagonal coefficients
    # - b represents the main diagonal coefficients
    # - c represents the super-diagonal coefficients
    # - d represents the right-hand side of the system (option values at the previous time step)
    a = zeros(M+1)  # Sub-diagonal elements
    b = ones(M+1)   # Main diagonal elements
    c = zeros(M+1)  # Super-diagonal elements
    d = zeros(M+1)  # Right-hand side of the equation

    # 6. Time-Stepping Loop Using Crank-Nicolson
    # ----------------------------------------------
    # The Crank-Nicolson method is an implicit finite difference method that is unconditionally stable
    # for solving PDEs like the Black-Scholes equation. It uses a weighted average of the forward and backward
    # Euler methods, which gives it second-order accuracy in both time and space.
    for _ in 1:N
        # 6.1 Set Up Tridiagonal System for Each Time Step
        # --------------------------------------------------
        # The system of equations comes from discretizing the PDE.
        # The system is tridiagonal due to the finite difference discretization of second-order derivatives.
        @threads for i in 2:M
            a[i] = -alpha[i]  # Sub-diagonal element
            b[i] = 1 - beta[i]  # Main diagonal element
            c[i] = -gamma[i]  # Super-diagonal element

            # Compute the right-hand side (d[i]) based on the option value from the previous time step
            d[i] = alpha[i] * V[i-1] + (1 + beta[i]) * V[i] + gamma[i] * V[i+1]
        end

        # 6.2 Adjust Boundary Conditions
        # --------------------------------------------------
        # Apply boundary conditions to ensure the correct option price at the grid boundaries
        d[1] = option_type == :put ? boundary_put : 0.0  # Lower boundary for put option
        d[M+1] = option_type == :call ? boundary_call : 0.0  # Upper boundary for call option

        # 6.3 Forward Sweep (Thomas Algorithm)
        # --------------------------------------------------
        # Thomas algorithm for solving tridiagonal systems. The forward sweep eliminates the sub-diagonal
        # elements, transforming the matrix into an upper triangular form.
        for i in 2:M+1
            w = a[i] / b[i-1]  # Compute the multiplier for elimination
            b[i] -= w * c[i-1]  # Update the main diagonal
            d[i] -= w * d[i-1]  # Update the right-hand side
        end

        # 6.4 Backward Substitution
        # --------------------------------------------------
        # After the forward sweep, the matrix is upper triangular, and we can use backward substitution
        # to solve for the option values at the current time step.
        V[M+1] = d[M+1] / b[M+1]  # Solve for the last variable
        for i in M:-1:1
            V[i] = (d[i] - c[i] * V[i+1]) / b[i]  # Solve for the remaining variables
        end # added end here
        # 6.5 Apply Early Exercise Condition (American Option)
        # --------------------------------------------------
        # For American options, the option holder can exercise at any time, so we must compare the option's value
        # at each time step with its intrinsic value (early exercise payoff).
        @threads for i in 1:M+1
            intrinsic = option_type == :call ? (i-1)*dS - K : K - (i-1)*dS
            V[i] = max(V[i], intrinsic)  # Early exercise condition: max(V, intrinsic value)
        end
    end

    # 7. Interpolation to Get Option Price at S0
    # ----------------------------------------------
    # The final step is to interpolate the option price for the initial stock price S0
    # - We locate the grid point closest to S0 and linearly interpolate the option price.
    idx = floor(Int, S0 / dS) + 1
    if 1 < idx < M+1
        # Linear interpolation between two nearest points
        return V[idx] + (V[idx+1] - V[idx]) * (S0 - (idx-1)*dS) / dS
    elseif idx <= 1
        # Return the boundary value if S0 is too small
        return V[1]
    else
        # Return the boundary value if S0 is too large
        return V[M+1]
    end
end

