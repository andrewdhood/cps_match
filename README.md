Chicago Public Schools (CPS) operates 11 Selective Enrollment High Schools (SEHS) where admission is determined by a composite score (grades + HSAT exam). 

CPS publishes cutoff scores and average scores of *admitted* students, but crucially, they do not publish the distribution of *all applicants*. Since we only observe students above a threshold, we're dealing with **truncated distributions**. This notebook develops a Maximum Likelihood Estimation (MLE) framework to recover the hidden population parameters $(\mu, \sigma)$ from the published truncated statistics, then uses these recovered parameters to inform a physics-based Monte Carlo simulation of the full admissions process.

You can interact with the code in this notebook online here:

https://hub.2i2c.mybinder.org/user/homo-morphism-cps_match-j8vb0bcx/notebooks/sehs_analysis_notebook.ipynb
