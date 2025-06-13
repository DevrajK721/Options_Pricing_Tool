import numpy as np
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.integrate import quad

def bs_options_pricer(S, K, T, r, sigma, q):
    """
    Calculate the Black-Scholes option price for a vanilla call or put option.

    Parameters:
    S (float): Current stock price
    K (float): Option strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying stock (annualized)

    Returns:
    float: The price of the option
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    t_d1 = norm.cdf(d1)
    t_d2 = norm.cdf(d2)

    call_price = (S * np.exp(-q * T) * t_d1) - (K * np.exp(-r * T) * t_d2)
    put_price = call_price + K * np.exp(-r * T) - S * np.exp(-q * T)

    return call_price, put_price

def mc_pricer(S, r, q, sigma, T, payoff_fn, payoff_args=(), n_paths=500000, discrete_steps=100):
    """
    Monte Carlo simulation for option pricing for European Style Options with exotic payoff functions.

    Parameters:
    S (float): Current stock price
    r (float): Risk-free interest rate (annualized)
    q (float): Dividend yield (annualized)
    sigma (float): Volatility of the underlying stock (annualized)
    T (float): Time to expiration in years
    payoff_fn (callable): Function to calculate the payoff of the option
    payoff_args (tuple): Additional arguments for the payoff function
    n_paths (int): Number of simulated paths

    Returns:
    float: The estimated price of the option
    """
    dt = T / discrete_steps
    paths = np.zeros(shape=(n_paths, discrete_steps + 1), dtype=np.float64)
    paths[:, 0] = S

    for i in range(1, discrete_steps + 1):
        Z = np.random.normal(size=n_paths)
        paths[:, i] = paths[:, i - 1] * np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
   
    payoffs = payoff_fn(paths[:, -1], *payoff_args)
    discounted_payoffs = np.exp(-r * T) * payoffs
    option_price = np.mean(discounted_payoffs)
    
    return option_price


def quad_price(S, r, q, sigma, T, payoff_fn, payoff_args=(), strike=None):
    """
    Calculate the option price using numerical integration (quadrature method).

    Parameters:
    S (float): Current stock price
    r (float): Risk-free interest rate (annualized)
    q (float): Dividend yield (annualized)
    sigma (float): Volatility of the underlying stock (annualized)
    T (float): Time to expiration in years
    payoff_fn (callable): Function to calculate the payoff of the option
    payoff_args (tuple): Additional arguments for the payoff function

    Returns:
    float: The estimated price of the option
    """
    m = np.log(S) + (r - q - 0.5 * sigma**2) * T
    dist = lognorm(s=sigma * np.sqrt(T), scale=np.exp(m))

    integrand = lambda s: payoff_fn(s, *payoff_args) * dist.pdf(s)

    # If we know the payoff is zero below strike, start there:
    lower = strike if (strike is not None) else 0.0

    integral, error = quad(
        integrand,
        lower,
        np.inf,
        epsabs=1e-8,
        epsrel=1e-8
    )

    return np.exp(-r * T) * integral

if __name__ == "__main__":
    # Example usage
    S = 100  # Current stock price
    K = 100  # Strike price
    T = 1    # Time to expiration in years
    r = 0.05 # Risk-free interest rate
    sigma = 0.2 # Volatility
    q = 0.01 # Dividend yield

    call_price, put_price = bs_options_pricer(S, K, T, r, sigma, q)
    print(f"Call Price: {call_price:.2f}, Put Price: {put_price:.2f}")

    def european_call_payoff(S_T, K):
        return np.maximum(S_T - K, 0)
    
    def european_put_payoff(S_T, K):
        return np.maximum(K - S_T, 0)

    call_mc_price = mc_pricer(S, r, q, sigma, T, european_call_payoff, (K,))
    put_mc_price = mc_pricer(S, r, q, sigma, T, european_put_payoff, (K,))
    print(f"Monte Carlo Call Price: {call_mc_price:.2f}, Put Price: {put_mc_price:.2f}")

    call_quad_price = quad_price(S, r, q, sigma, T, european_call_payoff, (K,), K)
    put_quad_price = quad_price(S, r, q, sigma, T, european_put_payoff, (K,))
    print(f"Quadrature Call Price: {call_quad_price:.2f}, Put Price: {put_quad_price:.2f}")
    