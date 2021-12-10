Implementing Principal Components Ridge Regression on prostate data and comparing it to standard ridge regression

We see that a well-tuned ridge regression appears to outperform the “optimal” tuning.
One potential explanation is as follows: first, the data dimension (8) is relatively small compared to the sample size n, so that ridge regularization is not particularly important — as evidenced by the fact that even for τ = .1, we achieve an improvement (on average) over the “optimal” setting of λ⋆. 
Another explanation is that the optimal setting of λ⋆ is overfitting to the observed training data.