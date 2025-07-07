# Gauss-Hermite Quadrature Method for European Style Option Pricing 
## Derivation of Gauss-Hermite Option Pricing Equation Form
This set of notes starts with the integral form for computing the options price, $V(S, t)$. $$V(S, t) = e^{-r(T - t)} \int_0^\infty f(S_T) p_{RN} (S_T) dS_T$$ where, $$p_{RN} (S_T) = \frac{1}{S_T \sqrt{2 \pi  \sigma^2 (T - t)}} \exp\left[\frac{(\ln(\frac{S_T}{S}) - (r - \frac{1}{2} \sigma^2) (T - t))^2}{2\sigma^2 (T - t)}\right]$$

Note here that $f(S_T)$ is our terminal payoff function. 

From this, we apply the following substitutions alongside a change in variables, $\tau = T - t$, $m = (r - \frac{1}{2} \sigma^2) \tau$, $$x = \frac{\ln(\frac{S_T}{S}) - m}{\sigma \sqrt{\tau}} \implies \ln(S_T) = \ln(S) + m + \sigma \sqrt{\tau} \;x$$

Rearranging, we get, $$S_T = S e^{m + \sigma \sqrt{\tau} \; x}, \;\; dS_T = S \sigma \sqrt{\tau} e^{m + \sigma \sqrt{\tau} \; x}$$

Note that with these transformations, we can simplify, $$\exp\left[-\frac{(\ln(\frac{S_T}{S}) - m)^2}{2 \sigma^2 \tau}\right] = \exp \left[ - \frac{x^2}{2}\right]$$

Now, we can substitute and simplify into our original equation, $$\begin{gathered} 
V = e^{-r \tau} \int_{x = -\infty}^{\infty} f(S e^{m + \sigma \sqrt{\tau} \; x}) \frac{1}{\cancel{S e^{m + \sigma \sqrt{\tau} \; x}} \sqrt{2 \pi \sigma^2 \tau}} e^{-frac{x^2}{2}} \cancel{Se^{m + \sigma \sigma \sqrt{\tau} \; x}} \; dx \\
= \frac{e^{-r \tau}}{\sqrt{2 \pi}} \int_{x = -\infty}^{\infty} f(S e^{m + \sigma \sqrt{\tau} \; x}) e^{-\frac{x^2}{2}} \; dx
\end{gathered}$$

The Gauss-Hermite quadrature method requires an integral of the form, $$\int_{-\infty}^{\infty} e^{-u^2} g(u) \; du$$

In order to get our integral into this form, we use the substitution $x = \sqrt{2} u$ which gives us, $$V = \frac{e^{-r \tau}}{\sqrt{\pi}} \int_{-\infty}^{\infty} e^{-u^2} f(S e^{m + \sigma \sqrt{\tau} \sqrt{2} u}) \; du$$

This is now in the form the Gauss-Hermite quadrature method expects and so now we can (approximately) solve for $V$ using, $$\int_{-\infty}^{\infty} e^{-u^2} g(u) \; du \approx \sum_{i = 1}^N w_i g(u_i)$$ where the weights ($w_i$) and nodes ($h_i$) come from the Hermite polynomials. The higher the value of $N$ the more accurate our solution becomes. 

## Hermite Polynomials and Efficient Computation of Weights and Nodes
The Hermite Polynomials come from the recursive sequence, $$\begin{gathered} 
H_0(x) = 1 \\
H_1(x) = 2x \\
H_{n + 1}(x) = 2x H_n(x) - 2n(H_{n - 1} (x)), \; \forall \; n \geq 1
\end{gathered}$$

$u_i$ is computed by solving the $N^{th}$ Hermite Polynomial with $u_i$ subsitituted in, i.e. $H_N(u_i) = 0$. Then, $w_i$ is computed simply using, $$w_i = \frac{2^{N - 1} N! \sqrt{\pi}}{N^2 [H_{N - 1} (u_i)]^2}$$

The *Golub-Welsch* method efficiently computes the nodes and weights by first forming the symmetric tri-diagonal matrix, $$J = \begin{bmatrix}
0 & \sqrt{\frac{1}{2}} & 0 & \dots & \dots & 0 \\
\sqrt{\frac{1}{2}} & 0 & \sqrt{\frac{2}{2}} & \dots & \dots 0 \\
0 & \sqrt{\frac{2}{2}} & 0 & \sqrt{\frac{3}{2}} & \dots & \dots \\
: & : & : & : & : & : \\
0 & \dots & 0 & \dots & \sqrt{\frac{N-1}{2}} & 0 
\end{bmatrix}$$

Now from this, we compute the eigenvalues and normalized eigenvectors of $J$. 
- The eigenvalues are precisely the quadrature nodes $u_i$
- The weights can be found using $w_i = \sqrt{\pi} (v_1^{(i)})^2$

Finally, we can subsitute and use these values to compute $V$.