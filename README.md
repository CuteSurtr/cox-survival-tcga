# Cox PH Survival on TCGA

A from-scratch implementation of **Cox proportional hazards regression**,
the **Kaplan-Meier estimator**, the **log-rank test**, and **Harrell's
concordance index**, applied to TCGA clinical data pulled live from the
Genomic Data Commons (GDC) API.

> **Portfolio framing.** This project demonstrates competence in
> (a) mathematical statistics (partial likelihood, Newton-Raphson, Fisher
> information), (b) survival analysis (censoring, risk sets, ties, baseline
> hazard), and (c) reproducible cancer-genomics pipelines (GDC API,
> harmonisation of AJCC stage, end-to-end from API -> figures).

---

## First real-data result: TCGA-LUAD

Fitting Cox PH with `age + sex + AJCC stage` on 376 lung-adenocarcinoma
cases with usable overall-survival times:

| Covariate | Hazard ratio | 95% CI | p (Wald) |
|---|---|---|---|
| age (per year) | 1.009 | 0.990 - 1.028 | 0.359 |
| is_male | 0.999 | 0.696 - 1.435 | 0.997 |
| stage II (vs I) | **2.46** | 1.57 - 3.85 | **7.8e-05** |
| stage III (vs I) | **3.17** | 2.00 - 5.02 | **8.8e-07** |
| stage IV (vs I) | **4.63** | 2.38 - 9.00 | **6.2e-06** |

- **Concordance index**: 0.675 (with just stage / age / sex, this is close to what large-scale LUAD prognostic studies report)
- **Log-rank (early I+II vs late III+IV)**: chi2 = 25.3, p = 4.9e-7

The monotone stage -> hazard relationship and the non-significant age/sex
effects reproduce well-known LUAD prognostic findings. The Kaplan-Meier
curves split cleanly by stage — see `figures/TCGA-LUAD_km_by_stage.png`
and the hazard-ratio forest plot at `figures/TCGA-LUAD_forest.png`.

Reproduce with:

```bash
pip install -e .
pytest -v                       # 7 tests vs. lifelines
python scripts/run_survival.py  # full pipeline on TCGA-LUAD
```

---

## The math

Let subject `i` have observation `(t_i, delta_i, x_i)` where `t_i` is the
observed time, `delta_i` the event indicator, and `x_i` the covariate vector.

### Partial likelihood

Cox's insight (1972) was that we can eliminate the nuisance baseline hazard
`lambda_0(t)` from the likelihood of the observed event-time *sequence*. The
partial likelihood for the ranks of the events is

```
L(beta) = prod_{i : delta_i = 1} exp(beta' x_i) / sum_{j in R(t_i)} exp(beta' x_j)
```

with `R(t_i) = {j : t_j >= t_i}` the risk set just before `t_i`. The log
partial likelihood is

```
l(beta) = sum_{i : delta_i = 1} [ beta' x_i - log sum_{j in R(t_i)} exp(beta' x_j) ]
```

### Score and information

The score and observed information are expressible in terms of risk-set
exp-weighted means and covariances of `x`:

```
U(beta) = sum_i [ x_i - E_R(x) ]
I(beta) = sum_i Cov_R(x)
```

Both are computed in `src/coxtcga/coxph.py` via reverse-cumulative sums on a
time-sorted array (so the whole gradient evaluation is O(n p^2)).

### Newton-Raphson

Updates are the classical

```
beta_{k+1} = beta_k + I(beta_k)^{-1} U(beta_k)
```

with a backtracking line search to guarantee monotone increase of the log
partial likelihood. Fisher standard errors come from the diagonal of
`I(beta)^{-1}` at convergence; Wald p-values use the N(0, 1) approximation.

### Ties

We implement **Breslow's approximation** (the most common choice; also R's
default when `method = "breslow"`). Compared to Efron's approximation
(lifelines' default), Breslow diverges by O(ties^2 / n), which is negligible
for data with few ties and bounded by ~0.05 in coefficients under heavy
tying — see `tests/test_coxph.py::test_coxph_heavy_ties_stable` for an
empirical check.

### Baseline hazard

The Breslow estimator of the baseline cumulative hazard is

```
H_0(t) = sum_{i : t_i <= t, delta_i = 1}  d_i / S_0(t_i)
```

where `d_i` is the number of events at `t_i` and `S_0` is the risk-set sum
of `exp(beta' x_j)`.

### Kaplan-Meier + log-rank + c-index

Standard textbook derivations — see `src/coxtcga/km.py` and
`src/coxtcga/metrics.py` for line-by-line implementations with math
annotations in docstrings.

---

## Verification against `lifelines`

`tests/test_coxph.py` runs 7 tests comparing our from-scratch implementation
to the reference:

- Cox PH coefficients agree with `lifelines.CoxPHFitter` to 5e-3
- Standard errors (Fisher diagonal) match to rtol 1e-2
- On n=3000 simulation with known truth, MLE is within 0.1 of ground truth
- Heavy-ties stability check (Breslow vs Efron divergence bounded)
- KM survival values match `lifelines.KaplanMeierFitter` to 1e-9
- Log-rank chi-squared and p-value match `lifelines.statistics.logrank_test` to 1e-6
- Harrell c-index matches `lifelines.utils.concordance_index` to 1e-6

---

## Repo layout

```
cox-survival-tcga/
├── src/coxtcga/
│   ├── coxph.py         # partial likelihood + Newton-Raphson + Breslow H0
│   ├── km.py            # Kaplan-Meier + Greenwood CI + log-rank
│   ├── metrics.py       # Harrell concordance index
│   ├── download.py      # GDC API client for clinical + RNA-seq
│   └── pipeline.py      # orchestration: API -> Cox -> KM -> figures
├── tests/test_coxph.py  # vs lifelines
├── scripts/run_survival.py
├── data/
│   ├── raw/             # GDC responses (gitignored, reproducible)
│   └── processed/       # design matrices + Cox summaries (committed)
├── figures/             # KM + forest plots (committed as artifacts)
├── references/          # curated theses + papers + auto-generated BIBLIOGRAPHY.md
├── Dockerfile
├── .github/workflows/ci.yml
└── pyproject.toml
```

---

## References

- Cox, D. R. (1972). *Regression models and life-tables.* JRSSB 34(2): 187-220.
- Kalbfleisch & Prentice. *The Statistical Analysis of Failure Time Data*, 2e.
- Therneau & Grambsch. *Modeling Survival Data: Extending the Cox Model.*
- Harrell et al. (1982). *Evaluating the yield of medical tests.*
- See [`references/BIBLIOGRAPHY.md`](references/BIBLIOGRAPHY.md) for an
  auto-fetched open-access bibliography including modern work on
  regularised Cox, deep survival models, and cancer-genomics prognostic
  signatures.
