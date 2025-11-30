# Reverse-Engineering CPS Selective Enrollment Admissions

## Recovering Hidden Population Parameters from Truncated Distributions via Maximum Likelihood Estimation

**Author:** Andrew Hood

**Date:** November 2025

**Data Source:** CPS Official 2024-2025 Cutoffs (released 3/14/2025)

---

## Abstract

Chicago Public Schools (CPS) operates 11 Selective Enrollment High Schools (SEHS) where admission is determined by a composite score (grades + HSAT exam, 0-900 scale). CPS publishes cutoff scores and average scores of *admitted* students, but crucially, they do not publish the distribution of *all applicants*.

Since we only observe students above a threshold, we're dealing with **truncated distributions**—and the published statistics are systematically misleading. This project develops a Maximum Likelihood Estimation (MLE) framework to recover the hidden population parameters $(\mu, \sigma)$ from the published truncated statistics, then uses these recovered parameters to build a physics-based Monte Carlo simulation of the full admissions process.

> **Interactive Simulation:** The full analysis includes a Monte Carlo admission probability simulator. To run the interactive components, open `sehs_analysis_notebook.ipynb` locally in Jupyter or upload to Google Colab along with `sehs_data.py` and `sehs_simulation_v13.py`.

---

## Table of Contents

1. [The Truncation Problem](#1-the-truncation-problem)
2. [Mathematical Framework](#2-mathematical-framework)
3. [MLE Implementation](#3-mle-implementation)
4. [Key Results](#4-key-results)
5. [Physics-Based Simulation](#5-physics-based-simulation)
6. [Key Findings](#6-key-findings)
7. [Repository Structure](#7-repository-structure)

---

## 1. The Truncation Problem

### The CPS Tier System

CPS divides the city into four socioeconomic tiers based on census tract characteristics:

| Tier | Socioeconomic Status | Characteristics |
|------|---------------------|-----------------|
| Tier 1 | Lowest | High poverty, low median income |
| Tier 2 | Below average | Moderate poverty |
| Tier 3 | Above average | Low poverty, above-average income |
| Tier 4 | Highest | Very low poverty, high educational attainment |

Seats are allocated in two phases:
- **Phase 1 (30%):** Rank-based—highest scorers citywide, regardless of tier
- **Phase 2 (70%):** Tier-based—17.5% of seats to each tier, filled by score within tier

### The Statistical Illusion

CPS publishes the average score of **admitted** students. But this is the mean of a *truncated* distribution:

$$\bar{X}_{\text{observed}} = \mathbb{E}[X \mid X \geq c] > \mu_{\text{true}}$$

This creates a systematic upward bias. A parent seeing "average: 876 at Payton" might conclude their 850-scoring child is below average. In reality, 850 likely places them well **above** the true applicant pool mean.

**The core question:** Given only the truncated statistics, can we recover the hidden population parameters?

---

## 2. Mathematical Framework

### Truncated Normal Distribution

Let $X \sim \mathcal{N}(\mu, \sigma^2)$ represent applicant scores. CPS admits students with $X \geq c$ (the cutoff). The conditional expectation of this truncated distribution is:

$$\mathbb{E}[X \mid c \leq X \leq M] = \mu + \sigma \cdot \frac{\phi(\alpha) - \phi(\beta)}{\Phi(\beta) - \Phi(\alpha)}$$

where:
- $\alpha = \frac{c - \mu}{\sigma}$ — standardized lower bound (cutoff in z-score units)
- $\beta = \frac{M - \mu}{\sigma}$ — standardized upper bound (M = 900)
- $\phi(\cdot)$ — standard normal PDF
- $\Phi(\cdot)$ — standard normal CDF

### The Identification Problem

With one equation and two unknowns $(\mu, \sigma)$, the problem is **underidentified**. There's an infinite "banana valley" of parameter pairs that produce the same truncated mean.

**Solution:** Add a second constraint—the acceptance rate:

$$P(X \geq c) = 1 - \Phi\left(\frac{c - \mu}{\sigma}\right) = r = \frac{\text{seats}}{\text{applicants}}$$

Two equations, two unknowns → unique solution.

### Selection Bias Demonstration

Consider a hypothetical tier with:
- True population mean: $\mu = 700$
- True population SD: $\sigma = 80$
- Cutoff: $c = 800$

The cutoff is 1.25 standard deviations above the mean, admitting ~10% of applicants. But among those admitted:

$$\mathbb{E}[X \mid X \geq 800] = 841.3$$

The **selection bias** is +141.3 points—the observed mean exceeds the true mean by over 1.75 standard deviations.

---

## 3. MLE Implementation

### Loss Function

We recover $(\mu, \sigma)$ by minimizing a weighted loss:

$$\mathcal{L}(\mu, \sigma) = \underbrace{\left(\mathbb{E}[X|X \geq c; \mu, \sigma] - \bar{X}_{\text{obs}}\right)^2}_{\text{Match truncated mean}} + \lambda \underbrace{\left(P(X \geq c; \mu, \sigma) - r\right)^2}_{\text{Match acceptance rate}}$$

where $\lambda = 100$ balances the constraint scales.

### Core Functions

```python
def truncated_mean(mu: float, sigma: float, lower: float, upper: float = 900) -> float:
    """
    Compute E[X | lower <= X <= upper] for X ~ N(mu, sigma^2).

    Implements: mu + sigma * (phi(alpha) - phi(beta)) / (Phi(beta) - Phi(alpha))
    """
    alpha = (lower - mu) / sigma
    beta = (upper - mu) / sigma
    prob_mass = stats.norm.cdf(beta) - stats.norm.cdf(alpha)
    return mu + sigma * (stats.norm.pdf(alpha) - stats.norm.pdf(beta)) / prob_mass


def acceptance_prob(mu: float, sigma: float, cutoff: float) -> float:
    """
    Compute P(X >= cutoff) for X ~ N(mu, sigma^2).
    """
    return 1 - stats.norm.cdf(cutoff, mu, sigma)
```

### Optimization

We use L-BFGS-B with box constraints:
- $\mu \in [100, 890]$
- $\sigma \in [5, 200]$

Initial guesses are tier-specific (lower $\mu$ for Tier 1, higher for Tier 4).

---

## 4. Key Results

### MLE-Recovered Parameters (All 11 Schools)

| School | T1 $\hat{\mu}$ | T1 $\hat{\sigma}$ | T4 $\hat{\mu}$ | T4 $\hat{\sigma}$ | T1→T4 Gap |
|--------|----------------|-------------------|----------------|-------------------|-----------|
| **Walter Payton** | 266.7 | 200.0* | 770.0 | 81.9 | 102 pts |
| **Northside** | 448.0 | 167.4 | 887.7 | 3.4 | 186 pts |
| **Whitney Young** | 548.4 | 169.0 | 844.1 | 23.5 | 73 pts |
| **Jones** | 605.6 | 110.7 | 838.8 | 16.5 | 89 pts |
| **Lane Tech** | 545.6 | 109.7 | 830.6 | 18.7 | 113 pts |
| **Lindblom** | 490.4 | 152.5 | 671.4 | 86.5 | 89 pts |
| **Hancock** | 642.6 | 74.2 | 659.2 | 81.6 | 27 pts |
| **Brooks** | 430.0 | 200.0* | 663.1 | 73.8 | 83 pts |
| **King** | 383.9 | 200.0* | 690.3 | 60.1 | 120 pts |
| **Westinghouse** | 411.9 | 173.0 | 603.1 | 96.4 | 85 pts |
| **South Shore** | 304.6 | 196.9 | 590.1 | 87.7 | 135 pts |

*Hit optimization bounds—may indicate non-normal distribution

### Critical Finding: The Bifurcated System

The T4 $\hat{\sigma}$ values reveal **two fundamentally different competitive regimes**:

**Elite Schools** (Payton, Northside, Young, Jones, Lane):
- T4 $\bar{\sigma}$ = **28.8** — extremely tight distributions
- Northside T4 has $\hat{\sigma} = 3.4$, meaning virtually all T4 admits score 890-900
- Competition decided by 5-10 point margins

**Regional Schools** (Hancock, Lindblom, Brooks, King, Westinghouse, South Shore):
- T4 $\bar{\sigma}$ = **81.0** — much wider distributions
- Heterogeneous applicant pools spanning 200+ points
- 30-50 point buffers are common

---

## 5. Physics-Based Simulation

### Model Architecture (v13)

The MLE analysis recovers population parameters, but doesn't model **behavioral dynamics**. The physics simulation adds:

1. **Utility-based preferences:** Students rank schools by:
   $$U_{ij} = P_j - d_{ij} \cdot f(t_i, s_i) - \gamma(r_i, r_j) + \delta_j(t_i, s_i, r_i)$$
   - $P_j$ = school prestige
   - $d_{ij}$ = distance (miles)
   - $f(t_i, s_i)$ = friction coefficient (high scorers travel more)
   - $\gamma$ = cross-region penalty
   - $\delta_j$ = school-specific demand modifier

2. **Skew-normal score generation:** Conditional on region and tier:
   $$X_i \mid (R_i = r, T_i = t) \sim \text{SkewNormal}(\xi_{r,t}, \omega_{r,t}, \alpha_{r,t})$$

3. **Serial dictatorship matching:** Process students by score; each matched to highest-ranked school with available seats.

### Performance (500-trial Optuna optimization)

| Metric | Value |
|--------|-------|
| Overall MAE | 22.79 pts |
| Max Error | 84.4 pts |
| Max School MAE | 30.0 pts |

The model predicts cutoffs within ~23 points on average across all 44 school-tier combinations.

---

## 6. Key Findings

### Finding 1: The Tier System Creates 40-186 Point Advantages

| School | T1 Cutoff | T4 Cutoff | Tier Gap |
|--------|-----------|-----------|----------|
| Lane Tech | 712 | 859 | **147 pts** |
| Northside | 706.5 | 893 | **186.5 pts** |
| Payton | 796 | 898 | **102 pts** |

A Tier 1 student can gain admission to Lane Tech with a score 147 points below the Tier 4 threshold. This is the policy working as designed—but the magnitude is striking.

### Finding 2: Published Averages Overstate Competitiveness by 100-300 Points

| School | Published Avg (T1) | MLE-Recovered $\hat{\mu}$ | Selection Bias |
|--------|-------------------|---------------------------|----------------|
| Lane Tech | 758.2 | 545.6 | **+212.6 pts** |
| Whitney Young | 846.0 | 548.4 | **+297.6 pts** |
| Jones | 815.7 | 605.6 | **+210.1 pts** |

The truncated statistics create a systematic illusion of extreme competitiveness.

### Finding 3: Two Distinct School Systems Within One Policy

**Elite Schools:** T4 $\hat{\mu}$ = 830-888, $\hat{\sigma}$ = 3-82
**Regional Schools:** T4 $\hat{\mu}$ = 590-690, $\hat{\sigma}$ = 60-96

The 200+ point gap in population means reflects the fundamental bifurcation in Chicago's educational landscape.

### Finding 4: Regional Schools Show Inverted Tier Patterns

At elite schools: T4 cutoff > T3 > T2 > T1 (expected)

At some regional schools (e.g., South Shore): **T4 cutoff > T1**

This inversion occurs because high-scoring T1 students from the South Side *prefer elite schools* while high-scoring T4 students are "stuck" at regional schools due to geographic preferences.

---

## 7. Repository Structure

```
CPS_Match/
├── README.md                      # This file
├── sehs_analysis_notebook.ipynb   # Full analysis notebook
├── sehs_data.py                   # Centralized school data
└── sehs_simulation_v13.py         # Physics simulation module
```

### Running the Analysis

**Local (Recommended):**
```bash
pip install numpy pandas matplotlib seaborn scipy
jupyter notebook sehs_analysis_notebook.ipynb
```

**Google Colab:**
1. Upload all four files to Colab
2. Run all cells

---

## Technical Details

### Data Sources

- **Primary:** "Initial Offer Point Totals for Selective Enrollment High Schools 2025-2026" (CPS, released 3/14/2025)
- **Applicant estimates:** Historical enrollment data and reported application volumes

### Assumptions

1. **Normality:** Scores within each tier follow a normal distribution (relaxed to skew-normal in simulation)
2. **Uniform tier distribution:** Applicants split 25% per tier (simplification)
3. **Utility maximization:** Students rank schools by expected utility

### Limitations

- MLE assumes normal distributions; some schools hit optimization bounds suggesting non-normality
- Applicant counts are estimated, not observed
- Geographic preference model may not capture all behavioral factors (e.g., sibling attendance, program specialties)

---

## Citation

If you use this analysis, please cite:

```
Hood, A. (2025). Reverse-Engineering CPS Selective Enrollment Admissions:
An MLE Approach. https://github.com/homo-morphism/CPS_Match
```

---

## License

Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)

Copyright (c) 2025 Andrew Hood

You are free to:
  - Share: Copy and redistribute the material in any medium or format

Under the following terms:
  - Attribution: You must give appropriate credit and indicate if changes were made.
  - NonCommercial: You may not use the material for commercial purposes.
  - NoDerivatives: You may not distribute modified versions of this material.

Full license text: https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode
