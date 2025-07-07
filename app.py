import streamlit as st
import numpy as np
from scipy.stats import norm, lognorm  # unused imports; remove if not needed
from scipy.integrate import quad        # unused import; remove if not needed
from european_option_pricer import bs_analytic, gh_quad_price
from greeks_heatmap import *
# Optional: install streamlit-ace for a nicer code editor
# pip install streamlit-ace
from streamlit_ace import st_ace

# --- Streamlit UI ---
st.set_page_config(page_title="Options Pricer", layout="centered")

# Sidebar Branding and Title
st.sidebar.markdown(
    "<div style='text-align:left; padding-bottom:15px;'>"
    "<h2 style='margin:0; font-size:1.6em;'>Options Pricing Tool 📊</h2>"
    "<p style='margin:5px 0; font-size:1.3em;'>"
    "<a href='https://linkedin.com/in/devrajkatkoria' target='_blank' style='vertical-align:middle;'>"
    "<img src='https://cdn-icons-png.flaticon.com/512/174/174857.png'"
    " width='24' style='margin-right:8px;'/></a>"
    "<strong>Created by: Devraj Katkoria</strong></p>"
    "</div>", unsafe_allow_html=True
)

# Sidebar inputs
st.sidebar.header("Model Parameters")
S = st.sidebar.number_input("Spot Price $S$", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price $K$", value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Expiry $T$ (years)", value=1.0, min_value=0.0, step=0.1)
r = st.sidebar.number_input("Risk-free Rate $r$", value=0.05, format="%.4f")
sigma = st.sidebar.number_input("Volatility $\\sigma$", value=0.2, format="%.4f")
q = st.sidebar.number_input("Dividend Yield $q$", value=0.0, format="%.4f")
method = st.sidebar.selectbox("Pricing Method", ["Analytic (Vanilla)", "Gauss-Hermite Quadrature"])

# Heatmap parameter ranges (±10% defaults)
st.sidebar.header("Heatmap Parameter Ranges")
vol_range = st.sidebar.slider(
    "Volatility range",
    min_value=0.0,
    max_value=sigma * 20,
    value=(sigma * 0.9, sigma * 1.1),
    step=0.001
)
r_max = min(r * 20, 0.2)
r_range = st.sidebar.slider(
    "Risk-free Rate range",
    min_value=0.0,
    max_value=r_max,
    value=(max(0.0, r * 0.9), r * 1.1),
    step=0.001
)
T_max = 300.0
time_range = st.sidebar.slider(
    "Time to Expiry range",
    min_value=0.0,
    max_value=T_max,
    value=(0.0, T * 1.1),
    step=0.1
)
stock_price_range = st.sidebar.slider(
    "Spot Price range",
    min_value=0.0,
    max_value=S * 20,
    value=(S * 0.8, S * 1.2),
    step=0.01
)


# Main content title
st.markdown("# Options Pricing Dashboard 💹")
st.markdown("This tool alllows you to price European, American and exotic options using various methods. To start please enter your parameters and choose an options pricing method. **Note: If you are using custom Payoff functions Options, please use Gauss-Hermite Quadrature.**")

with st.expander("How to Use"):
    st.markdown("### European Vanilla Options")
    st.markdown(
        "Simply enter the parameters in the sidebar and select **Analytic** "
        "to compute the option prices using the Black-Scholes formula. "
        "This method is fast and efficient for standard European options."
    )

    st.markdown("### European Options with custom payoffs")
    st.markdown(
        "For European Style options with custom payoffs, you can use the **Gauss-Hermite Quadrature** "
        "method. This methods allow you to define custom payoff functions for your options, "
        "enabling pricing of various derivatives other than just vanilla."
    )

# Custom payoff section in main area for Quad
def get_custom_payoff_editor():
    st.subheader("Custom Payoffs (Quadrature)")
    st.markdown("Define payoff functions below. 👨🏽‍💻")
    default_call = (
        "# Define your call payoff function below:\n"
        "def payoff_call(S, K):\n"
        "    # S: array of terminal prices, K: strike price\n"
        "    return np.maximum(S - K, 0.0)"
    )
    default_put = (
        "# Define your put payoff function below:\n"
        "def payoff_put(S, K):\n"
        "    # S: array of terminal prices, K: strike price\n"
        "    return np.maximum(K - S, 0.0)"
    )
    call_code = st_ace(value=default_call, language='python', theme='github', key='call')
    put_code  = st_ace(value=default_put,  language='python', theme='github', key='put')
    return call_code, put_code

if method == "Gauss-Hermite Quadrature":
    call_code, put_code = get_custom_payoff_editor()
else:
    call_code = put_code = None

# Compute trigger
def run_pricing():
    grids = compute_grids(stock_price_range, vol_range, r_range, time_range)
    if method == "Analytic (Vanilla)":
        # 1) Get analytic BS prices
        call_price, put_price = bs_analytic(S, K, T, r, sigma, q)
        # 2) Compute Greeks via the same complex‐step routine,
        #    using the standard call/put payoff from greeks_heatmap.py
        greeks_call = compute_greek_estimates(
            S, sigma, r, T, K,
            q,
            european_call_payoff,  # payoff_fn
            (K,),                  # payoff_args
        )
        greeks_put = compute_greek_estimates(
            S, sigma, r, T, K,
            q,
            european_put_payoff,
            (K,)
        )

        fig = plot_price_heatmaps(grids, K, r, q, sigma, T, european_call_payoff, european_put_payoff, (K,))
        return call_price, put_price, greeks_call, greeks_put, fig

    # For Quad, execute the user-defined functions
    local_vars = {}
    exec(call_code, {"np": np}, local_vars)
    exec(put_code, {"np": np}, local_vars)
    payoff_call = local_vars.get("payoff_call")
    payoff_put  = local_vars.get("payoff_put")

    call = gh_quad_price(S, r, q, sigma, T, payoff_call, (K,), strike=K)
    put  = gh_quad_price(S, r, q, sigma, T, payoff_put,  (K,), strike=0)

    # Compute Greeks
    greeks_call = compute_greek_estimates(S, sigma, r, T, K, q, payoff_call, (K,))
    greeks_put = compute_greek_estimates(S, sigma, r, T, K, q, payoff_put, (K,))

    fig = plot_price_heatmaps(grids, K, r, q, sigma, T, payoff_call, payoff_put, (K,))

    return call, put, greeks_call, greeks_put, fig

# Styled Compute Button
if st.button("💥 **Compute Option Prices**"):
    call_price, put_price, greeks_call, greeks_put, fig = run_pricing()
    # display results
    st.markdown(
        f"<div style='background-color:green; color:black; border-radius:5px; padding:15px; margin:15px 0;'>"
        f"<h3 style='margin:0;'>🟢 Call Price: {call_price:.4f}</h3></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='background-color:red; color:white; border-radius:5px; padding:15px; margin:15px 0;'>"
        f"<h3 style='margin:0;'>🔴 Put Price: {put_price:.4f}</h3></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "### The Greeks and Heatmaps"
    )
    st.latex(r"\Delta = \frac{\partial V}{\partial S}")
    st.latex(r"\Gamma = \frac{\partial^2 V}{\partial S^2}")
    st.latex(r"\nu = \frac{\partial V}{\partial \sigma}")
    st.latex(r"\rho = \frac{\partial V}{\partial r}")
    st.latex(r"\Theta = \frac{\partial V}{\partial T}")

    st.markdown("#### Call Option Greek Estimates")
    # Greek estimates
    st.latex(rf"\Delta = {greeks_call['delta']:.4f}")
    st.latex(rf"\Gamma = {greeks_call['gamma']:.4f}")
    st.latex(rf"\nu = {greeks_call['nu']:.4f}")
    st.latex(rf"\rho = {greeks_call['rho']:.4f}")
    st.latex(rf"\theta = {greeks_call['theta']:.4f}")

    st.markdown("#### Put Option Greek Estimates")
    st.latex(rf"\Delta = {greeks_put['delta']:.4f}")
    st.latex(rf"\Gamma = {greeks_put['gamma']:.4f}")
    st.latex(rf"\nu = {greeks_put['nu']:.4f}")
    st.latex(rf"\rho = {greeks_put['rho']:.4f}")
    st.latex(rf"\theta = {greeks_put['theta']:.4f}")

    # Price Heatmaps
    st.markdown("### Price Heatmaps")
    st.pyplot(fig)
