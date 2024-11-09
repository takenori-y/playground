# Stats
See this [page](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance) for details.
The general formula was derived based on the equations up to the fourth order.

## 0th order
```math
L = M + N
```

## 1st order
```math
\begin{align}
\mu_{AB}
&= \frac{1}{L} \sum_{i=1}^L x_i \\
&= \frac{1}{L} \left( \sum_{i=1}^M x_i + \sum_{i=1}^N x_{i+M} \right) \\
&= \frac{1}{L} \left( M\mu_A + N\mu_B \right) \\
&= \frac{1}{L} \left( M\mu_A + N\mu_B \right) - \frac{1}{L} \left( N\mu_A - N\mu_A \right) \\
&= \mu_A + \frac{N}{L} \left( \mu_B - \mu_A\right) 
\end{align}
```

## $`p`$-th order
```math
\begin{align}
\mathcal{M}_{AB}^{(p)} &= \sum_{i=1}^L ( x_i - \mu_{AB} )^p \\
&= \sum_{i=1}^{L} x_i^p - \sum_{q=1}^{p-2} \binom{p}{q} \mathcal{M}_{AB}^{(p-q)} \mu_{AB}^q - L \mu_{AB}^p \\
&= \mathcal{M}_{A}^{(p-1)} + \mathcal{M}_{B}^{(p-1)}
+ \sum_{q=1}^{p-2} \binom{p}{q} \left( M^q \mathcal{M}_{B}^{(p-q)} + (-N)^q \mathcal{M}_{A}^{(p-q)} \right) \left( \frac{\mu_{B} - \mu_{A}}{L} \right)^q
+ LMN \left( \sum_{q=0}^{p-2} M^{p - 2 - q} (-N)^q \right) \left( \frac{\mu_{B} - \mu_{A}}{L} \right)^p
\end{align}
```
