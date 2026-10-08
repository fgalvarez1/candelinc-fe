# candelinc-fe

**CANDELINC-FE** is a constrained CP (CANDECOMP/PARAFAC) tensor decomposition in which the weights of each acquisition are not free parameters but finite element functions of known variables (for example clinical variables such as age, BMI or body position).

It decomposes a third-order tensor $\mathcal{P} \in \mathbb{R}^{S \times T \times A}$ (space × time × acquisitions) as

$$
\mathcal{P} \approx \sum_{r=1}^{R} U^{(s)}_{:,r} \otimes U^{(t)}_{:,r} \otimes (D B)_{:,r},
$$

where the spatial modes $U^{(s)}$ and temporal modes $U^{(t)}$ are shared by all acquisitions, $D$ is a design matrix built from the variables, and $B$ is the coefficient matrix that is optimized. Continuous variables are represented with piecewise linear (P1) finite element basis functions whose nodes are placed at the empirical quantiles of the data; categorical variables are encoded with indicator functions; interactions between a continuous and a categorical variable are also supported.

Because $U^{(a)} = D B$, the model:

- **quantifies** the influence of each variable on the spatiotemporal field (through $B$),
- **predicts** the field for a new acquisition from its variables alone,
- and, combined with a CP decomposition of the residuals, **samples** the variability that the variables do not explain.

The method was developed for the reduced-order modeling of spatiotemporal pleural pressure distributions from dynamic MRI; see [Demos and data](#demos-and-data).

## Installation

```bash
pip install candelinc-fe
```

or, for the latest version from GitHub:

```bash
pip install git+https://github.com/fgalvarez1/candelinc-fe.git
```

Requirements: Python ≥ 3.9, `numpy`, `tensorly` and `matplotlib`.

## Quick start

```python
import numpy as np
from candelinc_fe import CP, CandelincFE, P1DesignMatrix

# --- Synthetic data -------------------------------------------------------
rng = np.random.default_rng(0)

S, T, A = 200, 32, 40                  # spatial points, time steps, acquisitions
x = np.linspace(0, 1, S)
t = np.linspace(0, 1, T)

# Variables: one value per acquisition
age    = rng.uniform(20, 70, A)
weight = rng.uniform(50, 100, A)

# Spatial (f) and temporal (g) functions of the two modes
f1 = x**2
g1 = np.sin(2 * np.pi * t)
f2 = np.sin(np.pi * x)
g2 = np.exp(-((t - 0.5) / 0.1)**2)

# Field of acquisition a:  age_a * f1(x) g1(t)  +  weight_a * f2(x) g2(t)
X = np.stack([age_a * np.outer(f1, g1) + weight_a * np.outer(f2, g2)
              for age_a, weight_a in zip(age, weight)], axis=2)        # (S, T, A)
X += 0.5 * rng.standard_normal(X.shape)                                # noise

# --- 1. CP decomposition, to initialize the spatial and temporal modes ----
R = 2                                  # rank
cp, nmse_cp = CP().get_normalized_cp_decomposition(data_tensor=X, rank=R)

# --- 2. CANDELINC-FE ------------------------------------------------------
cfe = CandelincFE(variables_names=["age", "weight"],
                  variables_dict={"age": age, "weight": weight},
                  K=5)                 # number of P1 nodes per variable

# Design matrix D: (A, total number of nodes)
D, nodes = cfe.build_global_design_matrix()

factors, nmse = cfe.fit(X_observed=X,
                        D=D,
                        U_s_init=cp["normalized_spatial_cp"],
                        U_t_init=cp["scaled_temp_cp"],
                        R=R,
                        epochs=100)
U_s, U_t, B = factors["U_s"], factors["U_t"], factors["B"]

# --- 3. Prediction for a new acquisition, from its variables only --------
age_new    = 45
weight_new = 75

# Design matrix of the new acquisition: (1, total number of nodes)
D_age_new    = P1DesignMatrix(var_nodes=nodes["age"]).build(var_values=[age_new])
D_weight_new = P1DesignMatrix(var_nodes=nodes["weight"]).build(var_values=[weight_new])
D_new        = np.concatenate([D_age_new, D_weight_new], axis=1)

X_new = np.einsum("sr,tr,r->st", U_s, U_t, (D_new @ B)[0])             # (S, T)

# Comparison with the exact (noise-free) field
X_exact = age_new * np.outer(f1, g1) + weight_new * np.outer(f2, g2)
print("Relative error:", np.linalg.norm(X_new - X_exact) / np.linalg.norm(X_exact))
```

`fit` returns the factor matrices and the normalized mean squared error of the reconstruction. The columns of $B$, plotted against the node positions, show how each variable affects each mode (see `FiguresCandelincFE`).

## Main components

| Class | Purpose |
|---|---|
| `CandelincFE` | Builds the global design matrix $D$ from the variables (`build_global_design_matrix`) and computes the decomposition with an alternating least squares algorithm (`fit`). |
| `P1DesignMatrix` | P1 finite element design matrix of a continuous variable for given nodes; also used to evaluate new variable values. |
| `CP` | Normalized CP decomposition (via TensorLy), used as baseline and for initialization. |
| `ResidualSampling` | CP decomposition of the residual tensor and a multivariate Gaussian over its acquisition weights, to generate new fields that include the unexplained variability. |
| `FiguresCandelincFE` | Plots of the temporal modes and of the coefficients of $B$ as functions of the variables. |

### Specifying the variables

`CandelincFE(variables_names, variables_dict, K)` takes a list of variable names and a dictionary with one value per acquisition for each variable. `K` is the maximum number of P1 nodes per continuous variable (the nodes are the empirical quantiles of the data, with at least 3 acquisitions per node).

| Entry in `variables_names` | Meaning |
|---|---|
| `"age"` (any other name) | continuous variable, P1 basis functions |
| `"Position"`, `"Groupe"`, `"Sex"` | categorical variable, indicator functions |
| `"age-Position"` | interaction: separate P1 basis functions of `age` for each body position |
| `"age-Position-Groupe"` | interaction with body position and group |

In the current version, the categorical variables and their integer codes follow the conventions of the pleural pressure study: `Position` (0 = supine, 1 = prone), `Groupe` (0 = healthy, 1 = asthma, 2 = COPD) and `Sex` (0 = male, 1 = female).

## Demos and data

The demos of the article use this package and reproduce its figures:

- **Demos repository:** [Pleural-Pressure-Modeling-Paper-Demos](https://github.com/fgalvarez1/Pleural-Pressure-Modeling-Paper-Demos)
- **Browsable notebooks (Jupyter Book):** [fgalvarez1.github.io/Pleural-Pressure-Modeling-Paper-Demos](https://fgalvarez1.github.io/Pleural-Pressure-Modeling-Paper-Demos/), in particular
  - [Reduced-order model of pleural pressure distribution](https://fgalvarez1.github.io/Pleural-Pressure-Modeling-Paper-Demos/demos/Fig3-11-AppC_pleural_pressure_reduced_model.html) (Figures 3–11 and Appendix C): CANDELINC-FE applied to 40 subjects in supine and prone positions,
  - [Synthetic examples](https://fgalvarez1.github.io/Pleural-Pressure-Modeling-Paper-Demos/demos/FigAppB_synthetic_example.html) (Appendix B): a small example where the influence of each variable is known.

## Notation

The code follows the notation of the article:

| Article | Code |
|---|---|
| $U^{(s)}$, $U^{(t)}$, $U^{(a)}$ (spatial, temporal, acquisition factor matrices) | `U_s`, `U_t`, `U_a` |
| $D$ (design matrix), $B$ (coefficient matrix) | `D`, `B` |
| $\tau$ (Tikhonov regularization) | `tau` |
| $\mathcal{P}'_{\mathrm{res}}$ (residual tensor) | `P_res` |

<!-- TODO: uncomment once the article is submitted, and change "see [Demos and data](#demos-and-data)" in the introduction back to "in the article listed under [Citation](#citation)"

## Citation

If you use this package, please cite:

> F. Álvarez-Barrientos, Q. G. Herszkowicz, A. Duwat, C. Fetita, X. Maître, D. Rodriguez, M. Genet. *Physics- and Data-driven reduced-order modeling of spatiotemporal pleural pressure distribution.* 2026 (submitted).

-->

## License

[MIT](https://github.com/fgalvarez1/candelinc-fe/blob/main/LICENSE)
