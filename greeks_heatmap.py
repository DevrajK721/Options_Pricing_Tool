from european_option_pricer import gh_quad_price
import numpy as np
import matplotlib.pyplot as plt

# Define European call and put payoff functions just for testing
def european_call_payoff(S_T, K):
    return np.maximum(S_T - K, 0)

def european_put_payoff(S_T, K):
    return np.maximum(K - S_T, 0)

def compute_greek_estimates(spot_price, volatility, risk_free_rate, time_to_expiration, strike_price, dividend_yield=0.0, payoff_fn=None, payoff_args=()):
    """
    Compute option price and Greeks using complex-step differentiation.
    Returns a dict with keys: price, delta, gamma, nu (vol sensitivity), rho, theta.
    """
    def derivative_complex_step(f, x, h=1e-20):
        return (f(x + 1j*h).imag) / h

    # Price wrappers
    def price_spot(s):
        return gh_quad_price(s, risk_free_rate, dividend_yield, volatility, time_to_expiration, payoff_fn, payoff_args)
    def price_vol(sig):
        return gh_quad_price(spot_price, risk_free_rate, dividend_yield, sig, time_to_expiration, payoff_fn, payoff_args)
    def price_rate(r):
        return gh_quad_price(spot_price, r, dividend_yield, volatility, time_to_expiration, payoff_fn, payoff_args)
    def price_time(t):
        return gh_quad_price(spot_price, risk_free_rate, dividend_yield, volatility, t, payoff_fn, payoff_args)

    price = price_spot(spot_price)
    delta = derivative_complex_step(price_spot, spot_price)
    gamma = derivative_complex_step(lambda s: derivative_complex_step(price_spot, s), spot_price)
    nu    = derivative_complex_step(price_vol, volatility)
    rho   = derivative_complex_step(price_rate, risk_free_rate)
    theta = -derivative_complex_step(price_time, time_to_expiration)

    greeks = {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'nu': nu,
        'rho': rho,
        'theta': theta
    }

    return greeks

def compute_grids(spot_price_range=[50, 150], volatility_range=[0.1, 0.5], risk_free_rate_range=[0.01, 0.1], time_to_expiration_range=[0, 2.0]):
    """
    Compute mesh grids for spot prices, volatilities, risk-free rates, and time to expiration.
    
    Parameters:
    spot_price_range (list[2]): Range of spot prices [min, max]
    volatility_range (list[2]): Range of volatilities [min, max]
    risk_free_rate_range (list[2]): Range of risk-free rates [min, max]
    time_to_expiration_range (list[2]): Range of time to expiration [min, max]

    Returns:
    tuple: Mesh grids for spot prices, volatilities, risk-free rates, and time to expiration
    """

    spot_prices = np.linspace(spot_price_range[0], spot_price_range[1], 10)
    volatilities = np.linspace(volatility_range[0], volatility_range[1], 10)
    risk_free_rates = np.linspace(risk_free_rate_range[0], risk_free_rate_range[1], 10)
    time_to_expirations = np.linspace(time_to_expiration_range[0], time_to_expiration_range[1], 10)

    sv_meshgrid = np.meshgrid(spot_prices, volatilities)
    sr_meshgrid = np.meshgrid(spot_prices, risk_free_rates)
    st_meshgrid = np.meshgrid(spot_prices, time_to_expirations)
    
    return sv_meshgrid, sr_meshgrid, st_meshgrid

def plot_price_heatmaps(meshgrids, strike_price, risk_free_rate, dividend_yield, volatility, time_to_expiration, payoff_fn_call=None, payoff_fn_put=None, payoff_args=()):
    """
    Plot heatmaps of call and put prices for given parameter grids.
    """
    sv_mesh, sr_mesh, st_mesh = meshgrids
    cmap = plt.get_cmap("RdYlGn_r")

    def compute_grid(mesh1, mesh2, var_pair):
        grid_call = np.zeros_like(mesh1)
        grid_put  = np.zeros_like(mesh1)
        rows, cols = mesh1.shape
        for i in range(rows):
            for j in range(cols):
                S = mesh1[i, j]
                var = mesh2[i, j]
                if var_pair == "sv":
                    c = gh_quad_price(S, risk_free_rate, dividend_yield, var, time_to_expiration, payoff_fn_call, payoff_args)
                    p = gh_quad_price(S, risk_free_rate, dividend_yield, var, time_to_expiration, payoff_fn_put, payoff_args)
                elif var_pair == "st":
                    c = gh_quad_price(S, risk_free_rate, dividend_yield, volatility, var, payoff_fn_call, payoff_args)
                    p = gh_quad_price(S, risk_free_rate, dividend_yield, volatility, var, payoff_fn_put, payoff_args)
                elif var_pair == "sr":
                    c = gh_quad_price(S, var, dividend_yield, volatility, time_to_expiration, payoff_fn_call, payoff_args)
                    p = gh_quad_price(S, var, dividend_yield, volatility, time_to_expiration, payoff_fn_put, payoff_args)
                grid_call[i, j] = c
                grid_put[i, j]  = p
        return grid_call, grid_put

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    # pitch-black background for figure and axes
    fig.patch.set_facecolor('black')
    for ax in axes.flatten():
        ax.set_facecolor('black')

    # Spot vs Volatility
    c_sv, p_sv = compute_grid(sv_mesh[0], sv_mesh[1], "sv")
    # Spot vs Time
    c_st, p_st = compute_grid(st_mesh[0], st_mesh[1], "st")
    # Spot vs Risk-Free Rate
    c_sr, p_sr = compute_grid(sr_mesh[0], sr_mesh[1], "sr")

    # common color scale across all subplots
    vmin = min(c_sv.min(), p_sv.min(), c_st.min(), p_st.min(), c_sr.min(), p_sr.min())
    vmax = max(c_sv.max(), p_sv.max(), c_st.max(), p_st.max(), c_sr.max(), p_sr.max())

    axes[0, 0].imshow(c_sv, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Call Price", color='white')
    axes[0, 1].imshow(p_sv, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Put Price", color='white')

    axes[1, 0].imshow(c_st, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Call Price", color='white')
    axes[1, 1].imshow(p_st, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("Put Price", color='white')

    axes[2, 0].imshow(c_sr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2, 0].set_title("Call Price", color='white')
    axes[2, 1].imshow(p_sr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2, 1].set_title("Put Price", color='white')

    # annotate each cell with its price
    for i in range(c_sv.shape[0]):
        for j in range(c_sv.shape[1]):
            axes[0,0].text(j, i, f"{c_sv[i,j]:.2f}", ha='center', va='center', color='black')
            axes[0,1].text(j, i, f"{p_sv[i,j]:.2f}", ha='center', va='center', color='black')
            axes[1,0].text(j, i, f"{c_st[i,j]:.2f}", ha='center', va='center', color='black')
            axes[1,1].text(j, i, f"{p_st[i,j]:.2f}", ha='center', va='center', color='black')
            axes[2,0].text(j, i, f"{c_sr[i,j]:.2f}", ha='center', va='center', color='black')
            axes[2,1].text(j, i, f"{p_sr[i,j]:.2f}", ha='center', va='center', color='black')

    # set axis labels and white ticks for each subplot
    for ax in axes.flatten():
        ax.tick_params(colors='white')
    # Spot vs Volatility (first row)
    for ax in axes[0]:
        ax.set_xlabel("Spot Price", color='white')
        ax.set_ylabel("Volatility", color='white')
    # Spot vs Time (second row)
    for ax in axes[1]:
        ax.set_xlabel("Spot Price", color='white')
        ax.set_ylabel("Time to Expiration", color='white')
    # Spot vs Risk-Free Rate (third row)
    for ax in axes[2]:
        ax.set_xlabel("Spot Price", color='white')
        ax.set_ylabel("Risk-Free Rate", color='white')

        # ----------- custom tick labels -----------
    # X‑axis: spot prices are common
    spot_vals = sv_mesh[0][0]  # 1‑D array of spot values (len = cols)
    # Y‑axes for each row
    vol_vals  = sv_mesh[1][:,0]
    time_vals = st_mesh[1][:,0]
    rate_vals = sr_mesh[1][:,0]

    def _apply_ticks(ax_pair, y_vals):
        left_ax, right_ax = ax_pair
        # X
        x_idx = np.arange(len(spot_vals))
        left_ax.set_xticks(x_idx)
        right_ax.set_xticks(x_idx)
        left_ax.set_xticklabels([f"{v:.0f}" for v in spot_vals], rotation=90, color='white')
        right_ax.set_xticklabels([f"{v:.0f}" for v in spot_vals], rotation=90, color='white')
        # Y
        y_idx = np.arange(len(y_vals))
        left_ax.set_yticks(y_idx)
        right_ax.set_yticks(y_idx)
        left_ax.set_yticklabels([f"{v:.3f}" for v in y_vals], color='white')
        right_ax.set_yticklabels([f"{v:.3f}" for v in y_vals], color='white')

    _apply_ticks((axes[0,0], axes[0,1]), vol_vals)
    _apply_ticks((axes[1,0], axes[1,1]), time_vals)
    _apply_ticks((axes[2,0], axes[2,1]), rate_vals)


    plt.tight_layout()
    plt.close(fig)
    
    return fig

if __name__ == "__main__":
    # Example parameters
    strike = 100
    r      = 0.05
    q      = 0.0
    vol0   = 0.2
    T0     = 1.0

    # Generate parameter grids
    grids = compute_grids([50, 150], [0.1, 0.5], [0.01, 0.1], [0, 2.0])

    # Plot call/put heatmaps
    plot_price_heatmaps(grids, strike, r, q, vol0, T0, payoff_fn=european_call_payoff, payoff_args=(strike,))

    # Compute Greeks at a sample point
    greeks = compute_greek_estimates(100, vol0, r, T0, strike, q)
    print("Greeks at S=100:", greeks)