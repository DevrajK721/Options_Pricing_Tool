# Options 
## Options Overview
Options are a contract that give the bearer the right but not the obligation to buy or sell a given asset at a given price in the future.
- A call option is the right to buy a particular asset for an agreed price, $K$ (the Strike Price) at a specific time in the future $T$. 
- A put option is the right to sell a particular asset for an agreed price, $K$ at a specific time in the future, $T$.

### European vs. American Options
European Call Options may only be exercised at a single pre-defined point in time which is the $T$ expiration date while American Call Options may be exercised at any time, $t < T$ before the expiration date. 

### Put vs. Call Options
- We buy a call option when we speculate that the underlying asset will rise in value
- We buy a put option when we speculate that the underlying asset will decrease in value

Since you do not have the obligation to exercise the option, it makes sense that you only exercise when positions are favourable in which case your loss is purely the price of the option itself, $V$. This is the parameter we aim to compute when dealing with options pricing. 

### Parameters in Option Pricing 
There are a few key parameters in option pricing:
- $K$: Strike Price 
- $T$: Expiration Date or Expiry 
- $S$: Current underlying price 
- $\sigma$: Volatility of the underlying asset 
- $r$: Risk-Free Rate
- $V(S, t)$: Price of the option as a function of the underlying asset price and time
- $f(S_T)$: Payoff Function

### Exotic Options 
Exotic options are a class of options which are not vanilla. An option is vanilla if it has the standard payoff functions, $$\begin{cases} S_T - K \text{  Call Option} \\
K - S_T \text{ Put Option}\end{cases}$$

Therefore, any other payoff function would deem the option to be considered exotic. While some exotic options can also be solved analytically alongside vanilla options, e.g. Call Option with payoff function $f(S_T) = S_T^2$, many are special functions which cannot be analytically evaluated and thus present a need for numerical approximations. This also applies for American Style Options. 

## Black-Scholes Equation
### Derivation

We start from the dynamics of the underlying asset price under the risk-neutral measure $\mathbb{Q}$:

$$\frac{dS}{S} = (r - q)\,dt + \sigma\,dW_t^{\mathbb{Q}}$$

where;
- $S$ is the asset price at time $t$,
- $r$ is the continuously-compounded risk-free rate,
- $q$ is the continuous dividend yield,
- $\sigma$ is the asset’s volatility,
- $W_t^{\mathbb{Q}}$ is a standard Brownian motion under $\mathbb{Q}$.

Let $V(S,t)$ denote the price of a European derivative with payoff $f(S_T)$ at expiry $T$.  By Itô’s lemma:

$$dV = \left( \frac{\partial V}{\partial t} + (r-q)S\frac{\partial V}{\partial S} + \tfrac12\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} \right)dt + \sigma S \frac{\partial V}{\partial S} \,dW_t^{\mathbb{Q}}$$


To eliminate risk (i.e. hedge the stochastic term), we form a replicating portfolio: long one derivative, short $\Delta=\tfrac{\partial V}{\partial S}$ shares of the underlying.  The portfolio value

$$\Pi = V - \Delta S$$

has dynamics (since the $dW$ term cancels):

$$d\Pi = dV - \Delta\,dS = \left( \frac{\partial V}{\partial t} + \tfrac12\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - qS\frac{\partial V}{\partial S} \right)dt$$

To avoid arbitrage, this must earn the risk-free rate:

$$d\Pi = r\,\Pi\,dt = r\,(V - S\tfrac{\partial V}{\partial S})\,dt$$

Equate the $dt$ terms and rearrange to obtain the Black–Scholes PDE:
$$\frac{\partial V}{\partial t} + (r - q)S\frac{\partial V}{\partial S} + \tfrac12\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - rV = 0$$


With the terminal condition,

$$V(S,T) = f(S) \quad\text{(e.g. }(S-K)\text{ for a call).}$$

Solving this PDE (via change of variables to the heat equation) yields the closed-form Black–Scholes formulas:

$$\displaystyle
\begin{aligned}
C &= S e^{-qT}N(d_1) - K e^{-rT}N(d_2), \\[6pt]
P &= K e^{-rT}N(-d_2) - S e^{-qT}N(-d_1),
\end{aligned}$$

where,

$$
d_{1,2} = \frac{\ln(S/K) + (r - q \pm \tfrac12\sigma^2)T}{\sigma\sqrt T},
$$

and $N(\cdot)$ is the standard normal CDF.

## Volatility Estimation

Robust volatility estimates are important to ensure that the result from the options pricing tool are accurate.  Some common approaches include:

### Historical (Realized) Volatility
- Collect log-returns $r_i = \ln(S_{t_i}/S_{t_{i-1}})$ over a window of $n$ observations.
- Sample standard deviation:
$\hat\sigma_{\rm hist} = \sqrt{ \frac{1}{n-1} \sum_{i=1}^n (r_i - \bar r)^2 } \times \sqrt{\frac{N}{\Delta t}}$,
where $\Delta t$ is the sampling interval (e.g. 1 day) and $N$ is the number of such intervals in a year (e.g. 252).

### Exponential Weighted Moving Average (EWMA)
- Gives more weight to recent returns:
$$\sigma_t^2 = \lambda\sigma_{t-1}^2 + (1-\lambda)r_{t}^2,$$
- where $0<\lambda<1$. Then, annualize by $\sigma=\sqrt{\text{avg daily var} \times 252}.$

### GARCH Models
Fit a $\text{GARCH}(p,q)$ model to capture volatility clustering:

$$\sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i r_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2$$

