# Analysis — Reproduction of Wainwright (2009), Figure 2

## 1. Replication Results

I reproduced Figure 2 of Wainwright (2009), which studies exact signed-support recovery by the Lasso under an independent Gaussian design. The study covers linear, sublinear, and fractional-power sparsity, with $p\in\{128,256,512\}$ and 14 values of the rescaled sample size $\theta$. Across 126 scenarios, I ran 200 replications per scenario (25,200 fits). The full simulation design is documented in `ADEMP.md`.

The main result is clear: recovery probability is near zero when $\theta$ is small, rises quickly around the threshold region, and approaches one at larger sample sizes. For $p=512$, recovery probability at $\theta=0.8$ versus $\theta=2.0$ increases from 0.125 to 0.985 under linear sparsity, from 0.070 to 0.965 under sublinear sparsity, and from 0.085 to 0.955 under fractional-power sparsity. However, the finite-sample curves do not jump abruptly at $\theta=1$.

To compare transition locations, I estimated $\theta_{50}$, the value of $\theta$ where the estimated recovery probability first reaches 0.5, using linear interpolation between neighboring grid points:

| Sparsity regime | $p=128$ | $p=256$ | $p=512$ |
|---|---:|---:|---:|
| Linear | 1.068 | 1.022 | 1.000 |
| Sublinear | 1.250 | 1.165 | 1.149 |
| Fractional-power | 1.206 | 1.145 | 1.112 |

In each regime, the estimated midpoint moves closer to one as $p$ increases. This is consistent with the theoretical threshold, but three finite dimensions cannot establish asymptotic convergence. The interpolated midpoints also have Monte Carlo uncertainty that I have not separately quantified.

## 2. Comparison with the Original Figure

My recovery curves reproduce the S-shaped pattern in the original Figure 2 and show roughly aligned curves across dimensions after rescaling. The main disagreement is that my transitions generally occur later, especially for sublinear and fractional-power sparsity. At $\theta=1$ and $p=512$, my recovery probabilities are 0.500 (linear), 0.295 (sublinear), and 0.325 (fractional-power); the corresponding points appear higher in the published figure. Because the paper does not supply the plotted numerical results, this comparison is visual rather than an exact point-by-point assessment.

I checked the statistical model against Section VII and the Lasso objective against Equation (3). In particular, the penalty used in my implementation follows

$$
\lambda_n=\sqrt{\frac{2\sigma^2\log(k)\log(p-k)}{n}},
$$

and scikit-learn's Lasso uses the same $1/(2n)$ loss scaling as the paper. Thus, I found no mismatch in the stated penalty formula or objective scaling.

One complication is that Section VII specifies $n=2\theta k\log(p-k)$, whereas the Figure 2 caption defines $\theta=n/[k\log(p-k)]$. I followed the Section VII convention:

$$
\theta=\frac{n}{2k\log(p-k)},\qquad
n=\left\lceil 2\theta k\log(p-k)\right\rceil.
$$

The caption's alternative definition would double the horizontal coordinates of my results, so relabeling the axis that way would not fix the observed rightward shift. I cannot determine from the published figure alone which convention was used to generate its plotted coordinates. Other unresolved differences include the exact sample-size grid, random seeds, solver settings, and how near-zero fitted coefficients were classified.

## 3. Critique of the Simulation Design

The original design has a clear purpose: it varies sparsity and dimension while keeping the Gaussian model controlled, making the theoretical threshold easier to examine. It also evaluates the exact quantity addressed by the theory—signed-support recovery—rather than relying on prediction error as a substitute.

At the same time, Figure 2 studies only the Lasso. It is not a head-to-head comparison with alternative variable-selection methods or earlier approaches, so it cannot establish that Lasso performs better than competing estimators. Independent Gaussian predictors and nonzero coefficients fixed at $\pm0.5$ make the setting favorable for isolating the predicted phenomenon, but less representative of datasets with correlated variables, unequal signals, or departures from Gaussian assumptions. The paper does investigate correlated designs elsewhere; my replication is deliberately limited to its Figure 2.

If extending the design, I would vary signal strength and predictor correlation and compare Lasso with at least one alternative selection method under the same generated datasets. I would also add more $\theta$ values close to the transition and check sensitivity to the numerical zero threshold. These changes would test robustness and practical relevance, rather than simply improve agreement with the published curves.

## 4. Visualizations and Implementation

I recreated the three-panel recovery plot (`results/figures/recovery_curves.png`) and added error bars equal to $1.96\times\widehat{\mathrm{MCSE}}$. With 200 replications, the largest plug-in MCSE is about 0.035. The error bars make it easier to avoid overinterpreting small differences between dimensions, although the normal approximation is less reliable near probabilities of zero or one.

My additional solver-iteration boxplot (`results/figures/solver_diagnostics.png`) shows computational variation that the original recovery plot does not address. The maximum observed iteration count was 106, well below `max_iter=20000`; the maximum recorded duality gap was approximately $3.43\times10^{-6}$. These are useful diagnostics, not proof that every numerical solution is exact.

The most challenging part was translating the paper's notation into consistent simulation and evaluation code, then investigating why the curves differed despite matching the stated Lasso objective. I used separate modules for data generation, fitting, and evaluation; saved replication-level outputs; and checked same-seed reproducibility in a pilot run.

All 10 automated tests passed, and the full output contains 25,200 rows, 126 scenarios, and no missing values. These checks make me confident in the workflow and the overall threshold pattern, but less confident in exact numerical agreement with Figure 2.

## 5. Conclusion

The replication captures the main qualitative finding of Wainwright (2009): exact signed-support recovery improves rapidly as the rescaled sample size crosses the threshold region. It does not reproduce the precise transition locations, particularly in the sublinear and fractional-power cases.

Rather than tuning the model after seeing the original figure, I retain the documented settings and report the remaining discrepancy as a limitation of the replication.

## Reference

Wainwright, M. J. (2009). Sharp thresholds for high-dimensional and noisy sparsity recovery using $\ell_1$-constrained quadratic programming (Lasso). *IEEE Transactions on Information Theory, 55*(5), 2183–2202.
