# Reverse-Engineering CPS Selective Enrollment Admissions

## Recovering Hidden Score Population Parameters from Public Truncated Data Using Inferential Statistics (Constrained Maximum-Likelihood Estimation) and Building a Monte-Carlo Simulation to Predict Admission Probabilities

**Author:** Andrew Hood

**Last Updated:** November 29, 2025

**Data Source:** CPS Official 2024-2025 Cutoffs (released 3/14/2025)

---

## Abstract

Chicago Public Schools (CPS) operates 11 Selective Enrollment High Schools (SEHS) where admission is determined by a composite score (grades + HSAT exam, 0-900 scale). CPS publishes cutoff scores and average scores of *admitted* students, but crucially, they do not publish the distribution of *all applicants*.

Since we only observe students above a threshold, we're dealing with **truncated distributions**—and the published statistics are systematically misleading. This project develops a Maximum Likelihood Estimation (MLE) framework to recover the hidden population parameters $(\mu, \sigma)$ from the published truncated statistics, then uses these recovered parameters to build a physics-based Monte Carlo simulation of the full admissions process.

> **Interactive Simulation:** The full analysis includes a Monte Carlo admission probability simulator. To run the interactive components, open `sehs_analysis_notebook.ipynb` locally in Jupyter  along with `sehs_data.py` and `sehs_simulation_v13.py` or point Google Colab towards this github repo.

---

## Table of Contents
0. [On the Weight of a Number](#0-prologue)
1. [The Truncation Problem](#1-the-truncation-problem)
2. [Mathematical Framework](#2-mathematical-framework)
3. [MLE Implementation](#3-mle-implementation)
4. [Key Results](#4-key-results)
5. [Monte Carlo Simulation: Statistical Formulation](#5-monte-carlo-simulation-statistical-formulation)
6. [Key Findings](#6-key-findings)
7. [Repository Structure](#7-repository-structure)
8. [Technical Details and Model Validation](#8-technical-details)

---


# Prologue. On the Weight of a Number

## On Fairness, Equity, and the Impossible Choices of Selective Enrollment

---

There is a moment I witness every year, sometime in late October or early November, when a child learns their score. The number arrives by email, stark and final. 847. Or 791. Or 863. And in that instant, a thirteen-year-old must confront something that most adults spend their lives avoiding: the sensation of being weighed, measured, and reduced to a single data point.

I have sat with families in the aftermath of these moments. I have watched parents try to translate the number into meaning, flipping between browser tabs, cross-referencing cutoff charts, asking me questions I sometimes cannot answer. 

"Is this good? Is this enough? What does this mean for my child?"

What I have come to understand, after years of tutoring students through this process, is that the question they are really asking is not about the number at all. They are asking about fairness. They are asking whether the world their child is about to enter will recognize their effort, their potential, their worth. And they are asking me to tell them that it will.

I cannot always tell them that.

---

### What We Talk About When We Talk About Fairness

The word "fair" is a trapdoor. It sounds simple. A child's word, really; a word invoked on playgrounds and in sibling disputes. "That's not fair." But the moment you press on it, the moment you ask fair according to whom, fair measured how, fair compared to what, the ground gives way.

Consider the Tier 4 family on the North Side, Lincoln Park or Lakeview, who has watched their child work diligently for three years. Straight A's. Ninety-ninth percentile. A score of 891. And they learn that their child did not get into Northside College Prep, while a student from Englewood with an 820 did. 

Is that fair?

The parent's grievance is certainly not invented. Their child did, by any conventional academic measure, perform better. The rules were known in advance, yes, but the outcome still stings. They played the game correctly and lost to someone who, by the rules of the game as they understood it, played it less well.

Now consider the Tier 1 family on the South Side. Their child also worked diligently. Also got A's, though at a school where getting A's meant something different. A school where there was no test prep center down the block. Where the eighth-grade math teacher was a long-term substitute. Where the library closed at 3pm because there was no funding for after-school staff. This child scored 820. The same score that would be mediocre in Lincoln Park represents something closer to extraordinary in Englewood.

Is it fair to compare these two numbers as if they were the same thing?

This is the crux of it. The Tier 4 parent is asking about fairness as consistency: the same rules applied to everyone, the same bar to clear. The Tier 1 parent, if they are even aware of the comparison being made, is asking about fairness as context: the recognition that the bar was never at the same height to begin with.

Both of these are coherent definitions of fairness. Both have moral weight. And they are, in this system, mutually exclusive.

---

### The Difference Between Fair and Equitable

There is a distinction that educators and policy-makers invoke, though it often clarifies less than it obscures. Fairness versus equity.

Fairness, in the strict sense, means treating everyone the same. The same test. The same scoring rubric. The same cutoff. This is the equality of inputs. A race where everyone starts at the same line.

Equity means something different. It means adjusting inputs to achieve a more equal distribution of outcomes. If some runners started further back through no fault of their own, if their starting position was determined by their parents' zip code, by the color of their skin, by the tax base of their school district, then equity might mean moving their starting line forward. Or giving them better shoes. Or acknowledging that "the same race" was never really the same.

The CPS tier system is an equity intervention. It says: we will not pretend that a score of 820 from Englewood and a score of 820 from Lincoln Park represent the same thing. We will compare students to other students from similar circumstances. We will reserve seats for each tier, so that the selective enrollment schools are not simply mirrors of the city's existing inequality.

This is a defensible position. The data on outcomes is genuinely encouraging. Students from lower-income backgrounds who attend these schools graduate at higher rates, attend college at higher rates, earn more over their lifetimes. The intervention appears to work, in the sense that it creates pathways that would not otherwise exist.

But here is what the policy cannot do. It cannot make the Tier 4 family feel that the outcome was fair. Because by their definition of fairness, consistency, it was not.

And here is what else the policy cannot do. It cannot explain itself. The system is so opaque, so poorly communicated, that many Tier 1 families do not even know the advantage exists. They see published averages that suggest their child has no chance. They do not apply to reach schools. The policy that was designed to help them fails in its implementation, because no one told them how it works.

So we arrive at a system that is somehow equitable in design but unfair in perception; helpful to statisticians in theory but opaque to students in practice. A system that manages to leave almost everyone feeling wronged.

---

### The View from the Classroom

Teachers hate this test. I say this not as an indictment but as an observation. The educators I have spoken with, almost without exception, resent what the HSAT represents.

The reasons are pedagogical. The test is not aligned with the curriculum. Preparing students for it means diverting instructional time from the actual learning objectives of eighth grade. It creates a shadow curriculum, where some portion of every week becomes about test-taking strategies rather than genuine understanding.

But the reasons run deeper than that. Teachers enter the profession, most of them, because they believe in the transformative power of learning. They believe that education is about growth, about curiosity, about helping young people become more fully themselves. And then they are asked to participate in a system that reduces all of that potential in a young person to a number. A number that will, for some students, open doors. And for others, firmly close them.

There is something corrosive about teaching toward a high-stakes exam. It narrows the aperture of what counts as valuable. It tells students, implicitly, that the point of learning is to be measured, sorted, ranked. That the most important goals are not about understanding, but instead performance.

And yet. What is the alternative? Without some mechanism for selection, how do you allocate scarce seats? Do you lottery them, introducing randomness that is truely arbitrary? Do you assign them by neighborhood, replicating the very segregation that selective enrollment was designed to disrupt? Do you eliminate selective enrollment entirely, dismantling programs that have genuinely helped students from underserved communities?

There are no clean answers here. Every system of selection is a system of exclusion. There are only so many free parameters one can control. Every definition of merit is a choice about what to value. The test may be a blunt instrument, but the absence of any instrument is not obviously better.

---

### The View from the Kitchen Table

I think often about what it feels like to be a parent in this system.

You want, more than anything, for your child to be seen. You want the world to recognize what you recognize. The curiosity, the effort, the particular spark that makes your kid who they are. And instead, the world offers you a rubric. Four hundred fifty points for grades in four subjects. Four hundred fifty points for a test taken on a single morning in October. The sum of these numbers will determine, in large part, where your child spends the next four years.

You do not experience this as a policy question. You experience it as a parent, lying awake at 2am, wondering if you should have started test prep earlier. Wondering if the B+ in science will matter. Wondering if you are putting too much pressure on your child, or not enough.

For Tier 4 parents, there is often a particular flavor of anxiety. The sense that the game is rigged against you, that no matter how hard your child works, the rules have been written to favor someone else. This anxiety is not entirely unfounded. The tier system does, by design, create different cutoffs for different neighborhoods. But it is also, in some ways, a misreading of the landscape. The 30% of seats allocated by rank remain available to everyone. The regional schools are often far more accessible than the published statistics suggest. The anxiety is real, but the hopelessness is often unwarranted.

For Tier 1 parents, the anxiety takes a different shape. It is the anxiety of navigating a system that was not designed with you in mind. Not because the policy excludes you, but because the information does. You may not know your tier. You may not understand that the cutoff for your tier is 150 points lower than the number you see in the newspaper. You may look at the published averages, 876 at Lane Tech, 894 at Northside, and conclude, reasonably but incorrectly, that your child has no chance.

The cruelest irony of this system is that its opacity harms most the families it was built to help.

---

### The View from Inside the Test

And then there is the child. Twelve or thirteen years old. Sitting in a classroom on a Tuesday morning in October, bubbling in answers that will shape the next four years of their life.

We do not often enough ask what this experience is like from the inside. We talk about the policy, the equity considerations, the admissions statistics. We do not talk about the specific terror of being thirteen and knowing that this morning matters in ways that most mornings do not.

I have worked with students who thrive under this pressure. Students who find the test clarifying, focusing, even exhilarating. But I have worked with more who do not. Students who know the material but freeze when the stakes are real. Students who are tired, or hungry, or anxious about something at home that has nothing to do with reading comprehension. Students who simply have a bad day, as all of us have bad days, and must live with the consequences for years.

We have decided, as a city, that this is acceptable. That the benefits of selective enrollment, the rigorous academics, the pathways to college, the concentration of motivated peers, justify the cost of sorting children by a single high-stakes exam. Perhaps they do. But we should be honest about what that cost is. We are asking children to bear a weight that most adults would find crushing. We are telling them, at an age when identity is still forming, that they are the kind of person who scores 847, or 791, or 863. We are teaching them that the world measures, and that the measurement is final.

Some of them will internalize this in ways that serve them well. They will learn to perform under pressure, to meet deadlines, to navigate systems that require performance on demand. These are, for better or worse, useful skills in the world as it exists.

Others will internalize it differently. They will learn that they are not good enough. They will learn that effort does not always translate to outcome. They will learn, before they have the cognitive tools to process it, that the world is not fair. And they will not always learn the right lessons from this knowledge.

---

### What Would Better Look Like?

I do not have a clean answer to the problems I have described. I am not sure anyone does. But I can gesture toward what better might look like, even if the path there is unclear.

Better would be transparent. If the tier system creates a 150-point advantage for some students, that fact should be clearly explained to every family. Not buried in a PDF. Not discoverable only through third-party websites built by civic hackers. Stated plainly, in multiple languages, in every communication about selective enrollment.

Better would be informative. Instead of publishing truncated averages that systematically overstate competitiveness, publish actual applicant distributions. Let families see what the competition really looks like. Give them probability estimates, not just cutoffs. Trust them with the truth.

Better would be humane. The single-shot, no-retake structure of the HSAT is a choice, not a necessity. Other cities allow multiple testing dates. Others weight grades more heavily. Others have moved away from standardized testing entirely. There are alternatives, each with their own tradeoffs, but alternatives nonetheless.

Better would be honest about what we are doing. We are sorting children. We are deciding, based on imperfect information, which students will have access to certain opportunities and which will not. This may be necessary. Scarcity is real. Selection is unavoidable. But we should not pretend it is anything other than what it is.

---

### The Question That Has No Answer

Can any of this ever be simple?

No. I do not think it can.

Fairness and equity pull in different directions. Individual merit and structural inequality cannot be reconciled by any formula. The interests of students, parents, teachers, and policymakers do not fully align, and no system can satisfy all of them at once.

What we can do, what I have tried to do with this analysis, is illuminate the tradeoffs. To show what the data actually says, beneath the opacity. To give families the information they need to navigate a system that was not designed to be navigable.

And perhaps, also, to sit for a moment with the weight of it. The weight of a number. The weight carried by a thirteen-year-old in a testing room, by a parent at a kitchen table, by a teacher asked to sort children into futures.

These are not simple problems. They will not be solved by better statistics, although better statistics can help. They are human problems, which is to say they are messy, contingent, and they resist optimization.

The best we can do, I think, is to see them clearly. To name what is actually happening. And to ask, with whatever honesty we can muster, whether this is the system we would choose if we were choosing, rather than inheriting. 

I am not sure it is. It is a system with many equations, and in many variables, and when we solve our complicated system, the solution often feels unfair. But I am also not sure what we would choose instead.

I'm not sure a students' achievements can ever be truly captured within a confidence interval. And perhaps that uncertainty is the most honest place to end.




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

### The Problem with Published Statistics

CPS publishes the average score of **admitted** students. But this is the mean of a *truncated* distribution:

$$\bar{X}_{\text{observed}} = \mathbb{E}[X \mid X \geq c] > \mu_{\text{true}}$$

This creates a systematic upward bias. A parent seeing "average: 876 at Payton" might conclude their 850-scoring child is below average. In reality, 850 likely places them well **above** the true applicant pool mean.

**This then motivates the core question:** Given only the truncated statistics, can we recover the hidden population parameters?

---

## 2. Mathematical Framework

We begin by setting up a formal mathematical framework for our constrained maximum-likelihood estimation inferential statistics procedure to recover the hidden parameters for each school, that is, the mean and standard deviation of their true applicant pools, not just the mean and standard deviation of admitted students. We first formalize the sample spaces, sigma-algebras, probability measures, random variables, (trunctuated) distributions, parameters, estimators, and constraints.

### Probability Space

We work on a probability space $(\Omega, \mathcal{F}, P)$ where:

- **$\Omega$** (sample space): The set of all possible realizations of applicant scores for a given school-tier combination
- **$\mathcal{F}$** (σ-algebra): The Borel σ-algebra $\mathcal{B}(\mathbb{R})$, ensuring all intervals and their complements are measurable
- **$P$** (probability measure): The measure induced by the normal distribution with parameters $(\mu, \sigma)$

### Random Variables: Formal Definitions

Let $X$ denote the composite score of a randomly selected applicant. Formally:

$$X: \Omega \to \mathbb{R}$$

is a measurable function from the sample space to the real line. We assume $X$ follows a normal distribution:

$$X \sim \mathcal{N}(\mu, \sigma^2)$$

where:
- **$\mu \in \mathbb{R}$** (population mean): The expected value $\mathbb{E}[X]$, representing the "center" of the applicant score distribution
- **$\sigma \in \mathbb{R}_{>0}$** (population standard deviation): The square root of variance, $\sqrt{\text{Var}(X)}$, measuring dispersion

The probability density function (PDF) is:

$$f_X(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

and the cumulative distribution function (CDF) is:

$$F_X(x; \mu, \sigma) = P(X \leq x) = \Phi\left(\frac{x - \mu}{\sigma}\right)$$

where $\Phi(\cdot)$ denotes the standard normal CDF.

### Truncated Normal Distribution

CPS admits students with scores at or above a cutoff $c$. This creates a **truncated random variable**:

$$X^* = (X \mid X \geq c)$$

Formally, $X^*$ is a random variable with support $[c, \infty)$ (or $[c, M]$ where $M = 900$ is the maximum possible score) whose distribution is derived from conditioning $X$ on the event $\{X \geq c\}$.

**Truncated PDF:** For $x \in [c, M]$:

$$f_{X^*}(x; \mu, \sigma, c, M) = \frac{f_X(x; \mu, \sigma)}{P(c \leq X \leq M)} = \frac{\phi\left(\frac{x-\mu}{\sigma}\right)}{\sigma\left[\Phi\left(\frac{M-\mu}{\sigma}\right) - \Phi\left(\frac{c-\mu}{\sigma}\right)\right]}$$

where $\phi(\cdot)$ is the standard normal PDF.

**Truncated Mean (Conditional Expectation):**

$$\mathbb{E}[X \mid c \leq X \leq M] = \mu + \sigma \cdot \frac{\phi(\alpha) - \phi(\beta)}{\Phi(\beta) - \Phi(\alpha)}$$

where we define the **standardized bounds**:
- $\alpha = \frac{c - \mu}{\sigma}$ — lower bound in standard units (z-score of cutoff)
- $\beta = \frac{M - \mu}{\sigma}$ — upper bound in standard units (z-score of maximum)

The ratio $\frac{\phi(\alpha) - \phi(\beta)}{\Phi(\beta) - \Phi(\alpha)}$ is known as the **inverse Mills ratio** (generalized to two-sided truncation).

### The Identification Problem

**Problem:** We observe the truncated mean $\bar{X}_{\text{obs}} = \mathbb{E}[X \mid X \geq c]$ but want to recover $(\mu, \sigma)$.

This single equation relates two unknowns:

$$\bar{X}_{\text{obs}} = g(\mu, \sigma; c, M)$$

where $g$ is the truncated mean function. The level set $\{(\mu, \sigma) : g(\mu, \sigma; c, M) = \bar{X}_{\text{obs}}\}$ forms a curve in parameter space—infinitely many solutions exist.

**Resolution:** Introduce a second constraint via the **acceptance rate**:

$$r = P(X \geq c) = 1 - \Phi\left(\frac{c - \mu}{\sigma}\right) = \frac{\text{seats}}{\text{applicants}}$$

Now we have two equations in two unknowns:

$$\begin{cases}
\mathbb{E}[X \mid X \geq c] = \bar{X}_{\text{obs}} \\
P(X \geq c) = r
\end{cases}$$

Under mild regularity conditions, this system has a unique solution $(\hat{\mu}, \hat{\sigma})$.

### Selection Bias: A Worked Example

Consider a hypothetical tier with:
- True population mean: $\mu = 700$
- True population SD: $\sigma = 80$
- Cutoff: $c = 800$

First, compute the standardized cutoff:

$$\alpha = \frac{800 - 700}{80} = 1.25$$

The acceptance rate is:

$$r = 1 - \Phi(1.25) \approx 1 - 0.894 = 0.106$$

So roughly 10.6% of applicants are admitted. The truncated mean is:

$$\mathbb{E}[X \mid X \geq 800] = 700 + 80 \cdot \frac{\phi(1.25)}{\Phi(-1.25)} = 700 + 80 \cdot \frac{0.183}{0.106} \approx 841.3$$

The **selection bias** is:

$$\text{Bias} = \mathbb{E}[X \mid X \geq c] - \mu = 841.3 - 700 = +141.3 \text{ points}$$

This exceeds 1.75 standard deviations—the observed mean dramatically overstates the true population center.

---

## 3. MLE Implementation

### Parameter Space

We next define the **feasible parameter space**:

$$\Theta = \{(\mu, \sigma) \in \mathbb{R}^2 : \mu \in [100, 890], \, \sigma \in [5, 200]\}$$

This is a compact, convex subset of $\mathbb{R}^2$. The bounds reflect:
- $\mu \in [100, 890]$: Mean must be within plausible score range
- $\sigma \in [5, 200]$: Standard deviation must be positive and bounded (ruling out degenerate or implausibly dispersed distributions)

### Loss Function

We frame parameter recovery as an optimization problem. Define the **loss function** $\mathcal{L}: \Theta \to \mathbb{R}_{\geq 0}$:

$$\mathcal{L}(\mu, \sigma) = \underbrace{\left(\mathbb{E}[X|X \geq c; \mu, \sigma] - \bar{X}_{\text{obs}}\right)^2}_{\mathcal{L}_1(\mu, \sigma): \text{ truncated mean error}} + \lambda \underbrace{\left(P(X \geq c; \mu, \sigma) - r\right)^2}_{\mathcal{L}_2(\mu, \sigma): \text{ acceptance rate error}}$$

where:
- $\bar{X}_{\text{obs}}$ — observed (published) average score of admitted students
- $r$ — estimated acceptance rate (seats / applicants)
- $\lambda = 100$ — regularization weight balancing the two constraints

**Interpretation:** $\mathcal{L}$ measures the squared discrepancy between model predictions and observed statistics. The weight $\lambda$ accounts for different scales ($\mathcal{L}_1$ is in points², $\mathcal{L}_2$ is in probability²).

### Optimization Problem

The MLE estimator is:

$$(\hat{\mu}, \hat{\sigma}) = \underset{(\mu, \sigma) \in \Theta}{\arg\min} \; \mathcal{L}(\mu, \sigma)$$

This is a **constrained nonlinear least squares** problem. We solve it using the L-BFGS-B algorithm, which handles box constraints efficiently.

<img width="1456" height="980" alt="download1" src="https://github.com/user-attachments/assets/999a8121-ada3-4a54-bdc5-1166aa5a3c5e" />

<img width="1450" height="980" alt="download2" src="https://github.com/user-attachments/assets/0434ab0e-0b81-49ec-a267-08ae6f9766f0" />

<img width="1467" height="980" alt="download3" src="https://github.com/user-attachments/assets/39931beb-cd33-4925-a711-b1921997571b" />


### Core Implementation

```python
# truncated_mean: computes E[X | lower <= X <= upper] for X ~ N(mu, sigma^2)
# uses the inverse Mills ratio formula derived above
def truncated_mean(mu: float, sigma: float, lower: float, upper: float = 900) -> float:
    alpha = (lower - mu) / sigma  # standardized lower bound
    beta = (upper - mu) / sigma   # standardized upper bound
    
    # denominator: P(lower <= X <= upper), the probability mass in truncation region
    prob_mass = stats.norm.cdf(beta) - stats.norm.cdf(alpha)
    
    # numerator: difference in PDF values at boundaries
    pdf_diff = stats.norm.pdf(alpha) - stats.norm.pdf(beta)
    
    # inverse Mills ratio adjustment
    return mu + sigma * (pdf_diff / prob_mass)


# acceptance_prob: computes P(X >= cutoff) for X ~ N(mu, sigma^2)
# this is just the survival function (complementary CDF)
def acceptance_prob(mu: float, sigma: float, cutoff: float) -> float:
    return 1 - stats.norm.cdf(cutoff, mu, sigma)


# loss_function: the objective we minimize
# returns squared errors in matching truncated mean and acceptance rate
def loss_function(params: tuple, cutoff: float, obs_mean: float, 
                  target_rate: float, lambda_weight: float = 100) -> float:
    mu, sigma = params
    
    # predicted truncated mean under current parameters
    pred_mean = truncated_mean(mu, sigma, cutoff)
    
    # predicted acceptance probability under current parameters
    pred_rate = acceptance_prob(mu, sigma, cutoff)
    
    # weighted sum of squared errors
    mean_error = (pred_mean - obs_mean) ** 2
    rate_error = (pred_rate - target_rate) ** 2
    
    return mean_error + lambda_weight * rate_error
```

### Initialization Strategy

The loss landscape can have multiple local minima (the "banana valley" problem). We use tier-specific initial guesses based on domain knowledge:

| Tier | Initial $\mu_0$ | Initial $\sigma_0$ | Rationale |
|------|-----------------|-------------------|-----------|
| Tier 1 | 500 | 120 | Lower SES → lower expected scores, high variance |
| Tier 2 | 580 | 110 | Moderate scores |
| Tier 3 | 660 | 100 | Above-average scores |
| Tier 4 | 740 | 90 | Highest scores, tighter distribution |

### Convergence and Identifiability

The optimizer terminates when:
- Gradient norm $\|\nabla \mathcal{L}\| < 10^{-5}$, or
- Function value change $|\mathcal{L}^{(k+1)} - \mathcal{L}^{(k)}| < 10^{-9}$, or
- Maximum iterations (500) reached

**Identifiability conditions:** The system is identified when:
1. Cutoff $c$ lies within the bulk of the distribution (not in extreme tails)
2. Acceptance rate $r \in (0.01, 0.99)$ (not nearly 0 or 1)
3. Observed mean exceeds cutoff: the published average satisfies $\bar{X} > c$

When parameters hit boundary constraints (e.g., $\hat{\sigma} = 200$), this suggests model misspecification—the true distribution may not be normal.

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

*Hit optimization bounds—may indicate non-normal distribution or insufficient data

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

<img width="1981" height="1530" alt="download4" src="https://github.com/user-attachments/assets/fa0cfb0d-8bc9-40b0-8e91-82f7190eca0f" />


---

## 5. Monte Carlo Simulation: Statistical Formulation

The MLE analysis recovers population parameters but doesn't model **behavioral dynamics**: how students choose which schools to rank, and how these choices interact with the matching algorithm. This section develops the mathematical framework for the Monte-Carlo simulation. We again construct the sample space formally by defining the fundamental sets involved in the simulation, our random variables, and our score distribution model (with parameter discussion). We argue that our sample space is compact, convex, and Borel-measureable (and so the probability measure can be defined and makes sense). We define per-school capacity constraints, conditional distributions, the matching algorithm, and score-ordering permutations (to enforce strict order statistics. We highlight a theorem by (Satterthwaite, 1975) about the impossibility of strategy in this process. Finally we prove that for our Monte-Carlo simulation, our estimator (the sample mean) is unbiased, consistent, and asymptotically normal. We also compute the variance and standard error.

### 5.1 Probability Space and Measure-Theoretic Foundation

We work on a probability space $(\Omega, \mathcal{F}, P)$ where:

- **$\Omega$** (sample space): The set of all possible outcomes of the random experiment. A single outcome $\omega \in \Omega$ represents one complete realization of the applicant pool—including every student's score, tier, region, preference list, and tie-breaker.

- **$\mathcal{F}$** (σ-algebra): The Borel σ-algebra on $\Omega$, generated by the product topology of the fundamental sets. This ensures all events of interest (e.g., "student $i$ is admitted to school $s$") are measurable.

- **$P$** (probability measure): The measure induced by the joint distribution of all random variables, satisfying the Kolmogorov axioms.

### 5.2 Fundamental Sets

Before defining random variables, we establish the key sets in our model:

**School Set:**
$$\mathcal{S} = \set{s_1, s_2, \ldots, s_{11}}$$

where the schools are: Payton, Northside, Young, Jones, Lane, Lindblom, Westinghouse, King, Brooks, Hancock, South Shore.

**Tier Set:**
$$\mathcal{T} = \set{1, 2, 3, 4}$$

representing the four CPS socioeconomic tiers.

**Region Set:**
$$\mathcal{R} = \set{ \text{North}, \text{Loop}, \text{West}, \text{South}}$$

representing geographic partitions of Chicago.

**Preference List Space:**
$$\mathcal{S}^{\leq 6} = \bigcup_{k=0}^{6} \mathcal{S}^{(k)}$$

where $\mathcal{S}^{(k)}$ denotes the set of all **ordered $k$-tuples of distinct schools**. Students may rank up to 6 schools; $\mathcal{S}^{(0)} = \{\emptyset\}$ represents the empty list.

**Cardinality:** $|\mathcal{S}^{(k)}| = \frac{11!}{(11-k)!} = 11 \cdot 10 \cdots (12-k)$, so:

$$|\mathcal{S}^{\leq 6}| = 1 + 11 + 110 + 990 + 7920 + 55440 + 332640 = 397112$$

### 5.3 Random Variables: Formal Definitions

A **random variable** is a measurable function from the sample space to a measurable space. For each student $i \in \{1, 2, \ldots, n\}$ (where $n$ is the applicant pool size), we define:

| Random Variable | Formal Definition | Interpretation |
|-----------------|-------------------|----------------|
| $X_i: \Omega \to [400, 900]$ | Composite score | Sum of grades component and HSAT exam score |
| $T_i: \Omega \to \mathcal{T}$ | Tier assignment | Determined by census tract of residence |
| $R_i: \Omega \to \mathcal{R}$ | Geographic region | North/Loop/West/South partition |
| $\mathbf{P}_i: \Omega \to \mathcal{S}^{\leq 6}$ | Preference list | Ordered ranking of schools |
| $U_i: \Omega \to [0, 1]$ | Tie-breaker | Uniform random variable for ordering ties |
| $M_i: \Omega \to \mathcal{S} \cup \{\emptyset\}$ | Match outcome | Assigned school (or $\emptyset$ if unmatched) |

**Measurability:** Each random variable is measurable with respect to the appropriate σ-algebra:
- $X_i$ is $(\mathcal{F}, \mathcal{B}([400,900]))$-measurable
- $T_i$ is $(\mathcal{F}, 2^{\mathcal{T}})$-measurable (discrete)
- $M_i$ is $(\mathcal{F}, 2^{\mathcal{S} \cup \{\emptyset\}})$-measurable (discrete)

**Explicit Sample Space Construction:**

For a simulation with $n$ students, the sample space decomposes as a product:

$$\Omega = \prod_{i=1}^{n} \Omega_i$$

where each student's individual outcome space is:

$$\Omega_i = [400, 900] \times \mathcal{T} \times \mathcal{R} \times \mathcal{S}^{\leq 6} \times [0,1]$$

A single element $\omega \in \Omega$ is an $n$-tuple:

$$\omega = \bigl((x_1, t_1, r_1, \mathbf{p}_1, u_1), \ldots, (x_n, t_n, r_n, \mathbf{p}_n, u_n)\bigr)$$

The random variable $X_i$ is the **coordinate projection** extracting the score: $X_i(\omega) = x_i$.

### 5.4 Score Distribution Model

Scores are generated from a **skew-normal distribution** conditional on region and tier. The skew-normal family extends the normal distribution with an asymmetry parameter.

**Definition (Skew-Normal Distribution):** A random variable $Y$ follows a skew-normal distribution with parameters $(\xi, \omega, \alpha)$, written $Y \sim \text{SN}(\xi, \omega, \alpha)$, if its PDF is:

$$f_Y(y; \xi, \omega, \alpha) = \frac{2}{\omega} \phi\left(\frac{y - \xi}{\omega}\right) \Phi\left(\alpha \cdot \frac{y - \xi}{\omega}\right)$$

where:
- $\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$ is the standard normal PDF
- $\Phi(z) = \int_{-\infty}^{z} \phi(u) \, du$ is the standard normal CDF

**Parameters:**

| Parameter | Symbol | Domain | Effect on Distribution |
|-----------|--------|--------|------------------------|
| Location | $\xi$ | $\mathbb{R}$ | Shifts distribution left/right (not the mean unless $\alpha=0$) |
| Scale | $\omega$ | $\mathbb{R}_{>0}$ | Controls spread (not SD unless $\alpha=0$) |
| Shape | $\alpha$ | $\mathbb{R}$ | Controls asymmetry: $\alpha > 0$ → right-skew, $\alpha < 0$ → left-skew, $\alpha = 0$ → normal |

**Conditional Distribution:**

For each region-tier pair $(r, t) \in \mathcal{R} \times \mathcal{T}$, we have parameters $(\xi_{r,t}, \omega_{r,t}, \alpha_{r,t})$. The conditional distribution is:

$$X_i \mid (R_i = r, T_i = t) \sim \text{SN}(\xi_{r,t}, \omega_{r,t}, \alpha_{r,t})$$

This yields a $4 \times 4 = 16$ parameter grid, reflecting empirical heterogeneity.

**Justification:** Standardized test score distributions typically exhibit ceiling effects (mass near maximum) for high-performing subpopulations and floor effects for others. The skew-normal captures this parsimoniously.

**Truncation to Valid Range:**

Raw scores are clipped to enforce domain constraints:

$$\tilde{X}_i = \text{clip}(X_i; 400, 900) \equiv \min(\max(X_i, 400), 900)$$

Formally, $\tilde{X}_i = h \circ X_i$ where $h: \mathbb{R} \to [400, 900]$ is the clipping function. Since $h$ is Borel-measurable, $\tilde{X}_i$ remains a valid random variable.

<img width="1380" height="980" alt="download6" src="https://github.com/user-attachments/assets/1b6c9f2a-b93f-4001-b269-35bba596ef3a" />

### 5.5 The Matching Mechanism

CPS uses a **serial dictatorship** mechanism—a classical algorithm from mechanism design with important theoretical properties.

#### 5.5.1 Capacity Constraints

Each school $s \in \mathcal{S}$ has total capacity $C_s \in \mathbb{Z}_{>0}$, partitioned as:

$$C_s = C_s^{\text{Rank}} + \sum_{t=1}^{4} C_s^{(t)}$$

where:

| Capacity Type | Formula | Interpretation |
|---------------|---------|----------------|
| $C_s^{\text{Rank}}$ | $\lfloor 0.30 \cdot C_s \rfloor$ | Rank-based seats (top scorers citywide) |
| $C_s^{(t)}$ | $\lfloor 0.175 \cdot C_s \rfloor$ | Tier-$t$ seats (competition within tier) |

Since $0.30 + 4(0.175) = 1.0$, this partitions all seats.

#### 5.5.2 Tie-Breaking

When multiple students share the same score, we need a strict ordering. Define:

$$U_i: \Omega \to [0, 1], \quad U_i \sim \text{Uniform}(0, 1), \quad \text{i.i.d.}$$

Since $U_i$ is continuous, $P(U_i = U_j) = 0$ for $i \neq j$, guaranteeing a strict total order with probability 1.

#### 5.5.3 Score-Ordering Permutation

**Definition (Permutation):** A permutation of $\{1, \ldots, n\}$ is a bijection $\pi: \{1, \ldots, n\} \to \{1, \ldots, n\}$. The set of all such permutations forms the **symmetric group** $S_n$ under function composition, with $|S_n| = n!$.

**Notation:**
- $\pi(k) = i$ means "the student in rank position $k$ is student $i$"
- $\pi^{-1}(i) = k$ means "student $i$ is in rank position $k$"

Define the **score-ordering permutation** $\pi \in S_n$ by:

$$\pi^{-1}(i) < \pi^{-1}(j) \iff \bigl(\tilde{X}_i > \tilde{X}_j\bigr) \lor \bigl(\tilde{X}_i = \tilde{X}_j \land U_i < U_j\bigr)$$

This establishes a strict total ordering: student $\pi(1)$ has the highest score, $\pi(2)$ has the second-highest, and so on. Writing $X_{(k)}$ for the $k$-th order statistic, we have $X_{(1)} \geq X_{(2)} \geq \cdots \geq X_{(n)}$ where $X_{(k)} = \tilde{X}_{\pi(k)}$. Ties are broken by $U_i$.

**Observation:** $\pi$ is itself a random variable, $\pi: \Omega \to S_n$, determined by the scores and tie-breakers.

#### 5.5.4 Algorithm: Two-Phase Serial Dictatorship

**Phase 1 (Rank-Based Allocation):**

Process students in order $\pi(1), \pi(2), \ldots, \pi(n)$ (highest to lowest score):

```
for k = 1 to n:
    i = π(k)                           # student with k-th highest score
    for s in P_i:                      # iterate through student's preferences
        if C_s^Rank > 0:               # rank seats available?
            M_i = s                    # assign student to school
            C_s^Rank = C_s^Rank - 1    # decrement capacity
            break
    if no assignment made:
        M_i = ∅ (pending Phase 2)
```

**Phase 2 (Tier-Based Allocation):**

For each tier $t \in \{1, 2, 3, 4\}$:

```
I_t = {i : T_i = t and M_i = ∅}        # unmatched tier-t students
sort I_t by score (descending)
for i in I_t:
    for s in P_i:
        if C_s^(t) > 0:                # tier-t seats available?
            M_i = s
            C_s^(t) = C_s^(t) - 1
            break
```

#### 5.5.5 Matching Function (Formal)

The matching function $M_i: \Omega \to \mathcal{S} \cup \{\emptyset\}$ is defined implicitly by the algorithm. For any $\omega \in \Omega$:

$$M_i(\omega) = \begin{cases}
s^* & \text{if } \exists s^* \in \mathbf{P}_i(\omega) \text{ with available capacity when } i \text{ is processed} \\
\emptyset & \text{otherwise (unmatched)}
\end{cases}$$

**Key property:** $M_i$ is a **deterministic function** of $(X_1, \ldots, X_n, T_1, \ldots, T_n, \mathbf{P}_1, \ldots, \mathbf{P}_n, U_1, \ldots, U_n)$. All randomness in $M_i$ derives from these inputs.

#### 5.5.6 Strategy-Proofness

**Theorem (Satterthwaite, 1975):** Serial dictatorship is **strategy-proof**: for any student $i$, truthfully reporting preferences $\mathbf{P}_i$ is a (weakly) dominant strategy, regardless of others' strategies.

**Implication:** We can assume all competitors submit truthful preferences (generated via utility maximization), since no strategic manipulation improves outcomes.

### 5.6 Monte Carlo Estimator

#### 5.6.1 Target Quantity

We seek to estimate:

$$p_s \equiv P(M_{\text{user}} = s \mid X_{\text{user}} = x, T_{\text{user}} = t, s \in \mathbf{P}_{\text{user}})$$

the probability that a student with score $x$ and tier $t$ is admitted to school $s$, given they ranked $s$.

#### 5.6.2 Indicator Random Variable

For each simulation $b \in \{1, \ldots, B\}$, define the **admission indicator** random variable:

$$Y_s^{(b)}: \Omega \to \{0, 1\}$$

This random variable equals 1 if the user is admitted to school $s$ in simulation $b$, and 0 otherwise. Formally:

$$Y_s^{(b)}(\omega) = \mathbb{1}\bigl[M_{\text{user}}^{(b)}(\omega) = s\bigr]$$

where $\mathbb{1}[\cdot]$ denotes the **indicator function**:

$$\mathbb{1}[A] = \begin{cases} 1 & \text{if } A \text{ is true} \\ \newline 0 & \text{if } A \text{ is false} \end{cases}$$

**Distribution:** Since each simulation draws an independent applicant pool:

$$Y_s^{(b)} \sim \text{Bernoulli}(p_s), \quad \text{i.i.d. across } b$$

#### 5.6.3 Point Estimator

The Monte Carlo estimator is the sample mean:

$$\hat{p}_s = \frac{1}{B} \sum_{b=1}^{B} Y_s^{(b)}$$

**Properties:**

1. **Unbiasedness:** The estimator is unbiased since each $Y_s^{(b)}$ has expectation $p_s$:

$$\begin{aligned}
\mathbb{E}[\hat{p}_s] &= \frac{1}{B} \sum_{b=1}^{B} \mathbb{E}\bigl[Y_s^{(b)}\bigr]
&= \frac{1}{B} \cdot B \cdot p_s
&= p_s
\end{aligned}$$

2. **Consistency:** By the Strong Law of Large Numbers:
   $$\hat{p}_s \xrightarrow{a.s.} p_s \quad \text{as } B \to \infty$$

3. **Asymptotic Normality:** By the Central Limit Theorem:
   $$\sqrt{B}(\hat{p}_s - p_s) \xrightarrow{d} \mathcal{N}(0, p_s(1-p_s))$$

#### 5.6.4 Variance and Standard Error

Since $Y_s^{(b)} \sim \text{Bernoulli}(p_s)$ with $\text{Var}(Y_s^{(b)}) = p_s(1-p_s)$:

$$\text{Var}(\hat{p}_s) = \frac{1}{B^2} \sum_{b=1}^{B} \text{Var}(Y_s^{(b)}) = \frac{p_s(1-p_s)}{B}$$

The **standard error** (estimated by plug-in):

$$\widehat{\text{SE}}(\hat{p}_s) = \sqrt{\frac{\hat{p}_s(1-\hat{p}_s)}{B}}$$

**Numerical Examples ($B = 100$):**

| $\hat{p}_s$ | $\widehat{\text{SE}}$ | 95% CI (Wald) |
|-------------|----------------------|---------------|
| 0.50 | 0.050 | [0.40, 0.60] |
| 0.80 | 0.040 | [0.72, 0.88] |
| 0.95 | 0.022 | [0.91, 0.99] |

**Scaling:** SE decreases as $O(1/\sqrt{B})$. Quadrupling simulations halves the confidence interval width.

### 5.7 Assumptions and Limitations

1. **Independence Across Simulations:** Each simulation $b$ draws an independent applicant pool from the score distribution model. This assumes stationarity of the data-generating process.

2. **Proportional Scaling:** We simulate $n \approx 2000$ students (vs. ~22,000 actual applicants). Capacities are scaled:
   $$C_s^{(\cdot)} \leftarrow \left\lfloor \frac{n}{22000} \cdot C_s^{(\cdot)} \right\rfloor$$
   This preserves seats-to-applicants ratios but may distort edge effects.

3. **Model Misspecification:** The skew-normal family may not capture:
   - Multimodality in true score distributions
   - Heavy tails or outlier behavior
   - Year-to-year variation (temporal non-stationarity)
   - Unobserved confounders correlated with scores

4. **Deterministic Competitor Preferences:** Other students' preferences are computed via a utility function depending on school prestige, distance, and region penalties. Only the user's preferences are set exogenously.

5. **Truthful Revelation:** By strategy-proofness, we assume all students report truthfully—no strategic preference manipulation.

### 5.8 Interpretation

The estimator $\hat{p}_s$ has a **frequentist interpretation**:

> "If we repeatedly drew applicant pools from this model and ran the CPS matching algorithm, the fraction of replications in which a student with profile $(x, t)$ is admitted to school $s$ converges to $\hat{p}_s$."

This is *not* a Bayesian posterior probability (which would incorporate parameter uncertainty). We treat model parameters as fixed (though estimated) and quantify uncertainty from competitor randomness only.

---

## 6. Key Findings

### Finding 1: The Tier System Creates 40-186 Point Advantages

| School | T1 Cutoff | T4 Cutoff | Tier Gap |
|--------|-----------|-----------|----------|
| Lane Tech | 712 | 859 | **147 pts** |
| Northside | 706.5 | 893 | **186.5 pts** |
| Payton | 796 | 898 | **102 pts** |

<img width="1380" height="780" alt="output1" src="https://github.com/user-attachments/assets/83caa7b8-ceca-47f5-a7ab-9e4ee0c1a6fe" />

A Tier 1 student can gain admission to Lane Tech with a score 147 points below the Tier 4 threshold. This is the policy working as designed, yet the magnitude is striking. For top-performing students, the drop of a letter grade from an A (112.5 points) to a B (75 points) is 37.5 points. This implies that for some schools, a tier 1 student with all B's (or mostly A's and a C) might have *better* prospective chances than a tier 4 student with all A's. This effect is not present as strongly across all schools, and raises questions about how effective, lentiant, and equitable the current CPS match program is. This test subjects children to the kind of stress typically reserved for young adults.


### Finding 2: Published Averages Overstate Competitiveness by 100-300 Points

| School | Published Avg (T1) | MLE-Recovered $\hat{\mu}$ | Selection Bias |
|--------|-------------------|---------------------------|----------------|
| Lane Tech | 758.2 | 545.6 | **+212.6 pts** |
| Whitney Young | 846.0 | 548.4 | **+297.6 pts** |
| Jones | 815.7 | 605.6 | **+210.1 pts** |

The truncated statistics create a systematic illusion of extreme competitiveness.

<img width="1580" height="714" alt="output3" src="https://github.com/user-attachments/assets/1f05869b-cc11-43f5-a5d4-1acd6c4fbf57" />

### Finding 3: Two Distinct School Systems Within One Policy

**Elite Schools:** T4 $\hat{\mu}$ = 830-888, $\hat{\sigma}$ = 3-82

**Regional Schools:** T4 $\hat{\mu}$ = 590-690, $\hat{\sigma}$ = 60-96

The 200+ point gap in population means reflects fundamental bifurcation in Chicago's educational landscape.

<img width="1380" height="780" alt="output4" src="https://github.com/user-attachments/assets/84b22301-50e2-460d-ae3d-7e07e9270dcd" />

### Finding 4: Regional Schools Show Inverted Tier Patterns

At elite schools: T4 cutoff > T3 > T2 > T1 (expected)

At some regional schools (e.g., South Shore): **T4 cutoff < T1 cutoff**

This inversion occurs because high-scoring T1 students from the South Side *prefer elite schools*, while high-scoring T4 students are geographically constrained to regional options.

<img width="1580" height="580" alt="download5" src="https://github.com/user-attachments/assets/5deea892-b4d5-4e59-9a22-e440e2c78deb" />


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

## 8. Technical Details

### Data Sources

- **Primary:** "Initial Offer Point Totals for Selective Enrollment High Schools 2025-2026" (CPS, released 3/14/2025)
- **Applicant estimates:** Historical enrollment data and reported application volumes

### 8. Technical Details and Simulation Validation

Model parameters $\{\xi_{r,t}, \omega_{r,t}, \alpha_{r,t}\}$ were optimized using Optuna (500 trials) with Tree-structured Parzen Estimators, minimizing MAE between simulated and historical cutoff scores.

**Performance:**
| Metric | Value |
|--------|-------|
| Overall MAE | 22.79 pts |
| Max Error | 84.4 pts |
| Max School MAE | 30.0 pts |

<img width="1575" height="1180" alt="download7" src="https://github.com/user-attachments/assets/cc396c43-e041-4f47-8505-d0ad69477eb9" />


### Limitations

- MLE assumes normal distributions; boundary-hitting suggests non-normality at some schools
- Applicant counts are estimated, not observed
- Geographic preference model may not capture sibling attendance, program specialties, etc.

---

## Citation

If you use this analysis, please cite:

```
Hood, A. (2025). Reverse-Engineering CPS Selective Enrollment Admissions.
https://github.com/homo-morphism/CPS_Match
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
