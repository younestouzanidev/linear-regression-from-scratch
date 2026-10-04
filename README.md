# Linear Regression from Scratch

A from-scratch implementation of univariate Linear Regression and Gradient Descent in Python, built without Scikit-Learn to study numerical optimization and stability.

## Features
- **Raw Calculus:** Derivation and vectorized calculation of partial derivatives w.r.t $w$ and $b$.
- **Adaptive Convergence:** Uses an $\epsilon$-tolerance check ($\|\nabla J\| < \epsilon$) instead of blind fixed iterations.
- **Safety Fuse:** Implements divergence detection and max-iteration limits.
- **Feature Scaling Awareness:** Designed to handle Hessian conditioning and avoid exploding gradients.

## Tech Stack
- Python
- NumPy

## Next Steps
- [ ] Implement Multiple Linear Regression (Vectorized $\vec{w}$ and matrix $X$)
- [ ] Implement Feature Normalization (Z-score scaling)
- [ ] Write C extension for gradient computation
