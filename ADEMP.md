# ADEMP — Simulation Study Design

**Project:** Reproduction of Wainwright (2009), Figure 2  
**Course:** STATS 607 — Advanced Statistical Computing  
**Study Type:** Option A — Reproduction of an Existing Study


## A — Aims

This simulation study aims to reproduce the results presented in Figure 2 of Wainwright (2009), *Sharp Thresholds for High-Dimensional and Noisy Sparsity Recovery Using $\ell_1$-Constrained Quadratic Programming (Lasso)*.

The primary objective is to investigate the probability that the Lasso correctly recovers the signed support of a sparse regression coefficient vector under a standard Gaussian design.

Specifically, this study aims to:

1. Reproduce the sharp transition in signed-support recovery probability as the rescaled sample size $\theta = n/[2k\log(p-k)]$ increases.
2. Examine whether the transition occurs near the theoretical threshold $\theta=1$.
3. Compare recovery behavior across three sparsity regimes (linear, sublinear, and fractional-power sparsity) and three problem dimensions ($p=128,256,512$).

The reproduced results will be compared with the original Figure 2, focusing on the location and sharpness of the transition and the consistency of recovery patterns across problem sizes.



## D — Data-generating Mechanisms

We reproduce the standard Gaussian design simulations from Figure 2 of Wainwright (2009).

### Linear Regression Model

For each simulation replication, data are generated according to

$$
y = X\beta^* + \varepsilon,
$$

where $X\in\mathbb{R}^{n\times p}$ is the design matrix, $\beta^*\in\mathbb{R}^{p}$ is the true sparse coefficient vector, and $\varepsilon\in\mathbb{R}^n$ is the observation noise.

The design matrix has independent standard Gaussian entries:

$$
X_{ij}\sim N(0,1).
$$

The noise terms are independently generated from

$$
\varepsilon_i\sim N(0,0.5^2).
$$

### Sparsity Regimes

We consider three sparsity regimes:

1. **Linear sparsity:** $k(p)=\lceil\alpha p\rceil$.
2. **Sublinear sparsity:** $k(p)=\left\lceil\frac{\alpha p}{\log(\alpha p)}\right\rceil$.
3. **Fractional-power sparsity:** $k(p)=\lceil\alpha p^\delta\rceil$.

Here, $\lceil x\rceil$ denotes the ceiling function, which rounds a value upward to the nearest integer.

Following the original paper, we set $\alpha=0.40$ and $\delta=0.75$, and consider problem dimensions $p\in\{128,256,512\}$.

### True Coefficient Vector

For each replication, a support set $S$ of size $k$ is randomly selected from the $p$ predictor indices. Coefficients in the support set are assigned $+0.5$ or $-0.5$ with equal probability. All remaining coefficients are zero.

Thus, the minimum absolute value of a nonzero coefficient is 0.5.

### Sample Size and Experimental Conditions

The sample size is determined by

$$
n=2\theta k\log(p-k),
$$

where the control parameter $\theta$ varies over the interval $(0,2.4)$.

The simulations examine all combinations of three sparsity regimes, three problem dimensions, and the selected values of $\theta$.

The original paper averages the signed-support recovery results over 200 trials per plotted condition.

The exact grid of $\theta$ values, the integer conversion rule for the sample size $n$, and the random seeds are not fully specified in the paper. The sparsity level $k$ is rounded upward using the ceiling function, as specified in Section VII.

### Simulation Design Matrix

The design is organized by the following experimental factors:

| Factor | Values |
|---|---|
| Design covariance | Identity matrix |
| Problem dimension $p$ | 128, 256, 512 |
| Sparsity regime | Linear, Sublinear, Fractional-power |
| Sparsity coefficient $\alpha$ | 0.40 |
| Fractional exponent $\delta$ | 0.75 |
| Noise standard deviation | 0.50 |
| Nonzero coefficient magnitude | 0.50 |
| Rescaled sample size $\theta$ | 14 values in [0.2, 2.4] |
| Monte Carlo replications | 200 per condition |

### Final Replication Design Choices

Since the original paper does not provide the exact numerical grid of rescaled sample sizes, we select the following values:

$$
\theta \in
\{0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1,
1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4\}.
$$

The grid contains additional points near the theoretical threshold $\theta=1$ to better examine the transition in recovery probability. The endpoint $\theta=2.4$ is included as a plotting boundary choice.

Following the sparsity definitions in Section VII, the values of $k$ are rounded upward using the ceiling function.

Since sample size must be an integer, we also choose

$$
n = \left\lceil 2\theta k\log(p-k)\right\rceil.
$$

This sample-size rounding rule is a replication choice rather than an explicitly specified rule in the original paper.

### Scenario Combinations

| Sparsity regime | $p$ | $k$ |
|---|---:|---:|
| Linear | 128 | 52 |
| Linear | 256 | 103 |
| Linear | 512 | 205 |
| Sublinear | 128 | 14 |
| Sublinear | 256 | 23 |
| Sublinear | 512 | 39 |
| Fractional-power | 128 | 16 |
| Fractional-power | 256 | 26 |
| Fractional-power | 512 | 44 |

Each of these nine combinations is evaluated at all 14 selected values of $\theta$, giving 126 simulation scenarios.

With 200 independent replications per scenario, the full study consists of 25,200 Lasso fits.

A complete scenario-level design matrix, including the calculated integer sample size $n$ for every combination, will be generated and saved with the simulation configuration.


## E — Estimands / Targets

The primary target of this simulation study is the true signed support of the sparse regression coefficient vector $\beta^*$.

Following Wainwright (2009), the signed support is defined by the elementwise sign function:

$$
\operatorname{sign}(\beta_j^*) =
\begin{cases}
+1, & \beta_j^* > 0, \\
0, & \beta_j^* = 0, \\
-1, & \beta_j^* < 0.
\end{cases}
$$

For each simulation replication, the Lasso estimate $\hat{\beta}$ is evaluated according to whether its signed support exactly matches that of the true coefficient vector:

$$
\operatorname{sign}(\hat{\beta})
=
\operatorname{sign}(\beta^*).
$$

This target requires both correct identification of nonzero coefficient positions and correct recovery of their signs.

The study focuses on exact signed-support recovery rather than coefficient estimation error or prediction accuracy.



## M — Methods

### Lasso Estimator

We reproduce the Lasso estimator used in Wainwright (2009). The Lasso estimates the sparse regression coefficient vector by minimizing a squared-error loss with an L1 penalty.

Following Equation (3) of the original paper, the estimator is defined as

$$
\hat{\beta}
=
\arg\min_{\beta\in\mathbb{R}^p}
\left\{
\frac{1}{2n}\|y-X\beta\|_2^2
+
\lambda_n\|\beta\|_1
\right\}.
$$

Here, the first term measures the residual sum of squares, normalized by the sample size, while the second term encourages sparsity in the estimated coefficients.

### Regularization Parameter

Following Section VII of Wainwright (2009), the regularization parameter is specified as

$$
\lambda_n
=
\sqrt{
\frac{2\sigma^2\log(k)\log(p-k)}{n}
}.
$$

We set $\sigma=0.5$, consistent with the original simulation study.

The regularization parameter is calculated separately for each simulation scenario using its sample size $n$, sparsity level $k$, and dimension $p$.

### Implementation Considerations

We plan to use the Lasso implementation in scikit-learn, whose objective function matches the form used in the original paper.

The penalty parameter will be set to $\alpha_{\mathrm{sklearn}}=\lambda_n$, rather than using the library's default value.

The estimator will be fitted without an intercept, and the Gaussian design matrix will not be standardized, in order to preserve the data-generating mechanism specified in the original paper.

All method settings, including the solver tolerance and maximum number of iterations, will be documented before the final simulation run.

The Lasso estimator will receive only the observed design matrix $X$ and response vector $y$. The true coefficient vector $\beta^*$ will be used only for evaluation and will not be available to the fitting procedure.


## P — Performance Measures

### Exact Signed-support Recovery

The primary performance measure is the probability that the Lasso correctly recovers the signed support of the true sparse coefficient vector.

For each simulation replication $b$, we define the success indicator as

$$
I_b =
\begin{cases}
1, & \operatorname{sign}(\hat{\beta}^{(b)})
     = \operatorname{sign}(\beta^{*(b)}),\\
0, & \text{otherwise}.
\end{cases}
$$

A replication is considered successful only when all coefficient positions and signs are recovered correctly. Any false positive, false negative, or incorrect sign results in a failure.

### Estimated Recovery Probability

For each simulation scenario, the recovery probability is estimated by averaging the success indicators across $B$ independent replications:

$$
\hat{P}_{\mathrm{success}}
=
\frac{1}{B}\sum_{b=1}^{B} I_b.
$$

Following the original paper, we use $B=200$ replications per scenario.

The estimated recovery probability ranges from 0 to 1, with larger values indicating more reliable signed-support recovery.

### Monte Carlo Standard Error

To quantify uncertainty arising from the finite number of simulation replications, we report the Monte Carlo standard error (MCSE) of the estimated recovery probability:

$$
\widehat{\mathrm{MCSE}}
=
\sqrt{
\frac{
\hat{P}_{\mathrm{success}}
(1-\hat{P}_{\mathrm{success}})
}{
B-1
}
}.
$$

The MCSE measures the simulation uncertainty in the estimated recovery probability. When the observed success rate is exactly 0 or 1, the plug-in MCSE is zero and should not be interpreted as proof of zero uncertainty.

MCSE is included as an additional reporting measure following the simulation reporting principles discussed in STATS 607 Lecture 5.

### Comparison with the Original Study

The reproduced recovery probabilities will be compared with Figure 2 of Wainwright (2009).

The main comparisons will focus on:

1. Whether recovery probability increases sharply near the theoretical threshold $\theta=1$.
2. Whether the transition becomes sharper as the problem dimension $p$ increases.
3. Whether the three sparsity regimes exhibit similar threshold behavior after rescaling the sample size.
4. Whether the reproduced patterns are qualitatively consistent with the original results.

Differences from the original figure will be discussed in relation to Monte Carlo uncertainty, numerical solver settings, and implementation choices not fully specified in the paper.

