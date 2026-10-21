
# ADEMP — Simulation Study Design

**Project:** Reproduction of Wainwright (2009), Figure 2  
**Course:** STATS 607 — Advanced Statistical Computing  
**Study Type:** Option A — Reproduction of an Existing Study

## A — Aims

This simulation study reproduces Figure 2 of Wainwright (2009), *Sharp Thresholds for High-Dimensional and Noisy Sparsity Recovery Using L1-Constrained Quadratic Programming (Lasso)*.

The primary aim is to investigate the probability that the Lasso correctly recovers the signed support of a sparse regression coefficient vector under a standard Gaussian design.

Specifically, the study aims to:

1. Reproduce the sharp transition in signed-support recovery probability as the rescaled sample size $\theta=n/[2k\log(p-k)]$ increases.
2. Examine whether the transition occurs near the theoretical threshold $\theta=1$.
3. Compare recovery behavior across three sparsity regimes (linear, sublinear, and fractional-power) and three problem dimensions ($p=128,256,512$).

The reproduced results are compared with the original Figure 2, focusing on the location and sharpness of the transition and the consistency of recovery patterns across problem sizes.

## D — Data-generating Mechanisms

We reproduce the standard Gaussian design simulations described in Section VII of Wainwright (2009).

### Linear Regression Model

For each simulation replication, data are generated according to

$$
y=X\beta^{*}+\varepsilon.
$$

Here, $X\in\mathbb{R}^{n\times p}$ is the design matrix, $\beta^{*}\in\mathbb{R}^{p}$ is the true sparse coefficient vector, and $\varepsilon\in\mathbb{R}^{n}$ is the observation noise.

The design matrix has independent standard Gaussian entries:

$$
X_{ij}\sim N(0,1).
$$

The noise terms are independently generated from

$$
\varepsilon_i\sim N(0,0.5^2).
$$

The design matrix and noise terms are generated independently for each replication.

### Sparsity Regimes

We consider three sparsity regimes:

1. **Linear sparsity:** $k(p)=\lceil\alpha p\rceil$.
2. **Sublinear sparsity:** $k(p)=\left\lceil\frac{\alpha p}{\log(\alpha p)}\right\rceil$.
3. **Fractional-power sparsity:** $k(p)=\lceil\alpha p^{\delta}\rceil$.

Here, $\lceil x\rceil$ denotes the ceiling function, which rounds a value upward to the nearest integer.

Following the original paper, we set $\alpha=0.40$ and $\delta=0.75$, and consider problem dimensions $p\in\{128,256,512\}$.

### True Coefficient Vector

For each replication, a support set $S$ of size $k$ is randomly selected from the $p$ predictor indices.

Coefficients in the support set are assigned $+0.5$ or $-0.5$ with equal probability. All remaining coefficients are zero.

Thus, the minimum absolute value of a nonzero coefficient is 0.5.

### Sample Size and Experimental Conditions

Following Section VII of the paper, the sample size is parameterized by

$$
n=2\theta k\log(p-k).
$$

The control parameter $\theta$ takes selected values in the interval $(0,2.4]$.

The simulations examine combinations of three sparsity regimes, three problem dimensions, and the selected values of $\theta$.

The original paper reports 200 Monte Carlo trials per plotted condition.

The exact grid of $\theta$ values, the integer conversion rule for $n$, and the random seeds are not fully specified in the paper. We therefore document these choices explicitly.

### Simulation Design Matrix

The design is organized by the following experimental factors:

| Factor | Values |
|---|---|
| Design covariance | Identity matrix |
| Problem dimension $p$ | 128, 256, 512 |
| Sparsity regime | Linear, Sublinear, Fractional-power |
| Sparsity coefficient $\alpha$ | 0.40 |
| Fractional exponent $\delta$ | 0.75 |
| Noise standard deviation $\sigma$ | 0.50 |
| Nonzero coefficient magnitude | 0.50 |
| Rescaled sample size $\theta$ | 14 values from 0.2 to 2.4 |
| Monte Carlo replications | 200 per scenario |

### Final Replication Design Choices

Since the original paper does not provide the exact numerical grid of rescaled sample sizes, we select

$$
\theta\in\{0.2,0.4,0.6,0.8,0.9,1.0,1.1,1.2,1.4,1.6,1.8,2.0,2.2,2.4\}.
$$

The grid includes additional points near the theoretical threshold $\theta=1$ to examine the transition in recovery probability.

The endpoint $\theta=2.4$ is included as a plotting boundary choice.

Following Section VII, sparsity levels are rounded upward using the ceiling function. Because the sample size must also be an integer, we use

$$
n=\left\lceil2\theta k\log(p-k)\right\rceil.
$$

This sample-size rounding rule is a documented replication choice rather than an explicitly specified rule in the original paper.

A fixed master random seed of `6072026` is used for reproducibility. Independent random streams are generated for individual replications using NumPy's random-number generation utilities.

### Scenario Combinations

The resulting sparsity levels are:

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

A complete scenario-level design matrix, including the calculated integer sample size $n$ for every combination, is saved in `results/scenario_design.csv`.

## E — Estimands / Targets

The primary estimand is the probability of exact signed-support recovery under each simulation scenario.

For a given scenario, let $\beta^{*}$ denote the true coefficient vector and $\hat{\beta}$ the Lasso estimate.

For each coefficient, the sign function is defined as follows:

Positive coefficient: $\mathrm{sign}(\beta_j^{*}) = 1$ when $\beta_j^{*} > 0$.

Zero coefficient: $\mathrm{sign}(\beta_j^{*}) = 0$ when $\beta_j^{*} = 0$.

Negative coefficient: $\mathrm{sign}(\beta_j^{*}) = -1$ when $\beta_j^{*} < 0$.

Exact signed-support recovery occurs when

$$
\mathrm{sign}(\hat{\beta})=\mathrm{sign}(\beta^{*}).
$$

This requires identifying all nonzero coefficient positions correctly and recovering their signs, with no false positives or false negatives.

The scenario-specific recovery probability is

$$
P_{\mathrm{success}}=\Pr\left(\mathrm{sign}(\hat{\beta})=\mathrm{sign}(\beta^{*})\right).
$$

The probability is taken over the randomness in the design matrix, coefficient support and signs, and observation noise.

The study focuses on exact signed-support recovery rather than coefficient estimation error or prediction accuracy.

## M — Methods

### Lasso Estimator

We reproduce the Lasso estimator used in Wainwright (2009). The Lasso estimates a sparse regression coefficient vector by minimizing a squared-error loss with an L1 penalty.

Following Equation (3) of the original paper, the estimator is defined as


$$
\hat{\beta}=\underset{\beta\in\mathbb{R}^{p}}{\mathrm{argmin}}\left(\frac{1}{2n}\|y-X\beta\|_2^2+\lambda_n\|\beta\|_1\right).
$$


The first term measures residual squared error, normalized by sample size, while the second term encourages sparsity in the estimated coefficients.

### Regularization Parameter

Following Section VII of Wainwright (2009), the regularization parameter is specified as

$$
\lambda_n=\sqrt{\frac{2\sigma^2\log(k)\log(p-k)}{n}}.
$$

We set $\sigma=0.5$, consistent with the original simulation study.

The regularization parameter is calculated separately for each scenario using its sample size $n$, sparsity level $k$, and dimension $p$.

### Implementation Considerations

We use the Lasso implementation in scikit-learn, whose objective function matches the loss scaling in Equation (3) of the original paper.

The penalty parameter is set to $\alpha_{\mathrm{sklearn}}=\lambda_n$, rather than using the library's default value.

The estimator is fitted without an intercept, and the Gaussian design matrix is not standardized, preserving the original data-generating mechanism.

The implementation uses the following solver settings:

- `fit_intercept=False`
- `selection="cyclic"`
- `tol=1e-7`
- `max_iter=20000`

The Lasso estimator uses only the observed design matrix $X$ and response vector $y$. The true coefficient vector $\beta^{*}$ is used only for evaluation and is not available to the fitting procedure.

The simulation records the number of solver iterations and the final duality gap for numerical diagnostics.

## P — Performance Measures

### Exact Signed-support Recovery

The primary performance measure is whether the Lasso correctly recovers the signed support of the true sparse coefficient vector.

For each replication $b$, define the success indicator as

$$
I_b=
\begin{cases}
1, & \mathrm{sign}(\hat{\beta}^{(b)})=\mathrm{sign}(\beta^{*(b)}),\\
0, & \text{otherwise}.
\end{cases}
$$

A replication is successful only when every coefficient position and sign is recovered correctly.

In the numerical implementation, an estimated coefficient with absolute value at most $10^{-8}$ is treated as zero. This is an implementation choice for evaluating the fitted signed support, rather than a threshold specified by the original paper.

Any false positive, false negative, or incorrect sign results in a failure.

### Estimated Recovery Probability

For each simulation scenario, recovery probability is estimated by averaging the success indicators across $B$ independent replications:

$$
\hat{P}_{\mathrm{success}}=\frac{1}{B}\sum_{b=1}^{B}I_b.
$$

Following the original paper, we use $B=200$ replications per scenario.

The estimated recovery probability ranges from 0 to 1, with larger values indicating more reliable signed-support recovery.

### Monte Carlo Standard Error

To quantify uncertainty arising from the finite number of simulation replications, we report the Monte Carlo standard error (MCSE) of the estimated recovery probability:

$$
\widehat{\mathrm{MCSE}}=\sqrt{\frac{\hat{P}_{\mathrm{success}}(1-\hat{P}_{\mathrm{success}})}{B-1}}.
$$

The denominator $B-1$ corresponds to estimating the Bernoulli variance using the sample variance across replications.

When the observed success rate is exactly zero or one, the plug-in MCSE is zero. This should not be interpreted as proof that the true recovery probability has no uncertainty.

For visualization, approximate uncertainty bars are constructed as

$$
\hat{P}_{\mathrm{success}}\pm1.96\widehat{\mathrm{MCSE}}.
$$

These are normal-approximation intervals, which are less reliable when estimated probabilities are close to zero or one.

### Comparison with the Original Study

The reproduced recovery probabilities are compared with Figure 2 of Wainwright (2009).

The main comparisons focus on:

1. Whether recovery probability increases sharply near the theoretical threshold $\theta=1$.
2. Whether the transition becomes sharper as problem dimension $p$ increases.
3. Whether the three sparsity regimes show similar threshold behavior after sample-size rescaling.
4. Whether the reproduced curves qualitatively agree with the original results.

Differences from the original figure are discussed in `ANALYSIS.md`, including Monte Carlo uncertainty, numerical solver settings, and implementation choices not fully specified in the paper.

### Additional Computational Diagnostics

In addition to recovery probability and MCSE, the simulation records the solver iteration count and final duality gap for each Lasso fit.

These diagnostics help identify potential computational issues and provide context for interpreting the numerical results. They are supplementary to the primary statistical performance measure.

The main results are summarized in `results/recovery_summary.csv` and visualized in `results/figures/recovery_curves.png`. Solver diagnostics are displayed in `results/figures/solver_diagnostics.png`.
