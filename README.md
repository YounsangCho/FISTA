# FISTA

We compared platforms for CUDA GPU in Python. For comparsion purposes, we implemented Fast Iterative Shrinkage-Thresholding Algorithm (FISTA; Beck and Teboulle, 2009) for LASSO. 

## FISTA (Beck and Teboulle, 2009)
Consider the minimization problem, for $x \in R^p$:
![ \hat{**x**} = argmin_{**x**} f(**x**) + g(**x**)]
where $f(**x**)$ is differentiable convex function and $g(**x**)$ is convex possibly non-differentiable.

  - Python with dynamic-link library using kernel functionsCancel changes
    First, 
