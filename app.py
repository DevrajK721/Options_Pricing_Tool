import streamlit as st
import numpy as np
from scipy.stats import norm, lognorm
from scipy.integrate import quad
from european_option_pricer import bs_analytic, mc_price, quad_price_log
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
method = st.sidebar.selectbox("Pricing Method", ["Analytic", "Monte Carlo", "Quadrature"])

# Main content title
st.markdown("# Options Pricing Dashboard 💹")
st.markdown("This tool alllows you to price European, American and exotic options using various methods. To start please enter your parameters and choose an options pricing method. **Note: If you are using custom Payoff functions and/or American Style Options, you cannot use the base analytical solver.**")

with st.expander("How to Use"):
    st.markdown("### European Vanilla Options")
    st.markdown(
        "Simply enter the parameters in the sidebar and select **Analytic** "
        "to compute the option prices using the Black-Scholes formula. "
        "This method is fast and efficient for standard European options."
    )

    st.markdown("### American Options with or without custom payoffs")
    st.markdown(
        "For American options, you can use the **Binomial Tree Method** or "
        "**Finite Difference PDE** methods. These methods allow for more complex "
        "payoff structures and can handle early exercise features of American options."
    )

    st.markdown("### European Exotic Options with custom payoffs")
    st.markdown(
        "For European exotic options, you can use the **Monte Carlo** or **Quadrature** "
        "methods. These methods allow you to define custom payoff functions for your options, "
        "enabling pricing of complex derivatives."
    )

# Custom payoff section in main area for MC/Quad
def get_custom_payoff_editor():
    st.subheader("Custom Payoffs (Monte Carlo/Quadrature)")
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

if method in ["Monte Carlo", "Quadrature"]:
    call_code, put_code = get_custom_payoff_editor()
else:
    call_code = put_code = None

# Compute trigger
def run_pricing():
    if method == "Analytic":
        return bs_analytic(S, K, T, r, sigma, q)

    # For MC/Quad, execute the user-defined functions
    local_vars = {}
    exec(call_code, {"np": np}, local_vars)
    exec(put_code, {"np": np}, local_vars)
    payoff_call = local_vars.get("payoff_call")
    payoff_put  = local_vars.get("payoff_put")

    if method == "Monte Carlo":
        call = mc_price(S, r, q, sigma, T, payoff_call, (K,))
        put  = mc_price(S, r, q, sigma, T, payoff_put,  (K,))
    else:
        call = quad_price_log(S, r, q, sigma, T, payoff_call, (K,), strike=K)
        put  = quad_price_log(S, r, q, sigma, T, payoff_put,  (K,), strike=0)
    return call, put

# Styled Compute Button
if st.button("💥 **Compute Option Prices**"):
    call_price, put_price = run_pricing()
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