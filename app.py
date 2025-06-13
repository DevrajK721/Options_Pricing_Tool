import streamlit as st
import numpy as np
from scipy.stats import norm, lognorm
from scipy.integrate import quad
from european_option_pricer import bs_analytic, mc_price, quad_price_log

# --- Streamlit UI ---
st.set_page_config(page_title="Options Pricer", layout="centered")
st.title("Options Pricing Dashboard")

# Sidebar inputs
st.sidebar.header("Model Parameters")
S = st.sidebar.number_input("Spot Price S", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price K", value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Expiry (years)", value=1.0, min_value=0.0, step=0.1)
r = st.sidebar.number_input("Risk-free Rate r", value=0.05, format="%.4f")
sigma = st.sidebar.number_input("Volatility sigma", value=0.2, format="%.4f")
q = st.sidebar.number_input("Dividend Yield q", value=0.0, format="%.4f")
method = st.sidebar.selectbox("Pricing Method", ["Analytic", "Monte Carlo", "Quadrature"] )

# Custom payoff section
st.sidebar.header("Custom Payoffs (MC/Quad)")
if method in ["Monte Carlo", "Quadrature"]:
    st.sidebar.markdown("Define payoff functions. Use vectorized numpy operations.")
    default_call = "# Change Function Arguments Here \ndef payoff_call(S, K):\n  # Change Payoff Function Here\n  return np.maximum(S - K, 0.0)"
    default_put  = "# Change Function Arguments Here\ndef payoff_put(S, K):\n  # Change Payoff Function Here\n  return np.maximum(K - S, 0.0)"
    call_code = st.sidebar.text_area("Call payoff code", value=default_call, height=100)
    put_code  = st.sidebar.text_area("Put payoff code", value=default_put, height=100)

# Compute trigger
def run_pricing():
    # choose payoff functions
    if method == "Analytic":
        return bs_analytic(S, K, T, r, sigma, q)
    # for MC/Quad, exec user code
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

if st.button("Compute Option Prices"):
    call_price, put_price = run_pricing()
    # display results
    st.markdown(f"<div style='border:2px solid green; border-radius:5px; padding:10px; margin:10px 0;'>"
                f"<strong>Call Price:</strong> {call_price:.4f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='border:2px solid red; border-radius:5px; padding:10px; margin:10px 0;'>"
                f"<strong>Put Price:</strong> {put_price:.4f}</div>", unsafe_allow_html=True)
