import numpy as np
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.integrate import quad

def bs_analytic(S, K, T, r, sigma, q):
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

def precompute_gh_nodes_weights(n):
    """
    Precompute the Gauss-Hermite nodes and weights for numerical integration.

    Parameters:
    n (int): Number of nodes

    Returns:
    tuple: (nodes, weights)
    """
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    return nodes, weights

def gh_quad_price(S, r, q, sigma, T, payoff_fn, payoff_args=(), strike=None, t = 0):
    """
    Calculate the option price using Gauss-Hermite Quadrature.

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

    # Precompute nodes and weights for Gauss-Hermite quadrature
    nodes, weights = precompute_gh_nodes_weights(256) # Weights (w_i) and nodes (x_i)
    tau = T - t  # Remaining time to expiration

    integral_coeff = np.exp(-r * tau) / np.sqrt(np.pi)

    def integrand(u):
        S_T = S * np.exp((r - q - 0.5 * sigma ** 2) * tau + sigma * np.sqrt(2 * tau) * u)
        term = payoff_fn(S_T, *payoff_args)
        return term

    integral = np.sum(weights * integrand(nodes)) # Gauss-Hermite quadrature 

    return integral_coeff * integral

    
if __name__ == "__main__":
    # Example usage
    S = 100  # Current stock price
    K = 100  # Strike price
    T = 1    # Time to expiration in years
    r = 0.05 # Risk-free interest rate
    sigma = 0.2 # Volatility
    q = 0.0 # Dividend yield

    call_price, put_price = bs_analytic(S, K, T, r, sigma, q)
    print(f"Call Price: {call_price:.4f}, Put Price: {put_price:.4f}")

    def european_call_payoff(S_T, K):
        return np.maximum(S_T - K, 0)
    
    def european_put_payoff(S_T, K):
        return np.maximum(K - S_T, 0)

    call_quad_price = gh_quad_price(S, r, q, sigma, T, european_call_payoff, (K,), K)
    put_quad_price = gh_quad_price(S, r, q, sigma, T, european_put_payoff, (K,))
    print(f"Quadrature Call Price: {call_quad_price:.4f}, Put Price: {put_quad_price:.4f}")

    assert np.isclose(call_price, call_quad_price, atol=0.1), "Call prices do not match!"
    assert np.isclose(put_price, put_quad_price, atol=0.1), "Put prices do not match!"
    