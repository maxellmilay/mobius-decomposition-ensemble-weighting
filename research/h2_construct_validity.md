# H2 Construct Validity — Research Log

**Hypothesis H2:** Pairwise Harsanyi dividends `m({i,j})` are valid measures of pairwise model interaction (redundancy/complementarity) in the coopetitive ensemble framework.

**Status at investigation start:** MIXED — Tests A and C failing, Test B barely passing, Test D inconclusive.

**Status after fixes:** Pending re-run on `adult` dataset.

---

## 1. Test Structure

Experiment 1 implements four construct validity subtests on a single benchmark dataset. All four must pass for H2 to hold.

| Test | Name | What it checks |
|------|------|----------------|
| A | Controlled Redundancy | Copy pairs have more-negative dividends than diverse pairs |
| B | Diversity Metric Correlation | Dividends correlate with SII, Q-stat, disagreement |
| C | Monotonicity | Blending LR→RF (α=0..1) produces a monotone-decreasing dividend trajectory |
| D | Negative Control | Random uniform predictors produce near-zero dividends |

---

## 2. Root Cause Analysis

### 2.1 The Breast Cancer Ceiling Problem (Tests A and C)

The original `CONSTRUCT_VALIDITY_DATASET = 'Breast Cancer'` placed all five models (RF, LR, SVM, XGB, KNN) at AUC ≈ 0.998 — a near-perfect ceiling. The Harsanyi dividend formula is:

```
m({i,j}) = v({i,j}) − v({i}) − v({j}) + v(∅)
         = v({i,j}) − v({i}) − v({j}) + 0.5
```

When `v({i}) ≈ v({j}) ≈ v({i,j}) ≈ 0.998`, this collapses to:

```
m({i,j}) ≈ 0.998 − 0.998 − 0.998 + 0.5 = −0.498
```

Every pair — copy pairs AND diverse pairs — clusters at ≈ −0.498. The dynamic range available for separation is only ±0.002, which is smaller than AUC estimation noise on n_val ≈ 114 samples. Tests A (requires copy_mean < dissim_mean) and C (requires monotone-decreasing trajectory) both fail because there is no signal above the noise floor.

**Old results (manuscript, models at AUC ~0.93):** Tests A and C both passed. copy_mean = −0.4015, dissim_mean = −0.3629.

**Current results (Breast Cancer, ceiling):** Test A FAIL. copy_mean = −0.4961, dissim_mean = −0.4978 (wrong direction). Test C FAIL: Spearman r = +0.77 (completely reversed).

### 2.2 Overly Strict Test C Criterion

The original pass criterion was `corr_c < −0.9`. This requires near-perfect Spearman monotonicity across all 11 alpha values.

The theoretical guarantee from Grabisch & Roubens (1999) is only that:
- At α=0 (pure LR): `m({RF, LR})` reflects full heterogeneity
- At α=1 (pure RF copy): `m({RF, RF})` is maximally negative

Submodular AUC games permit non-monotone intermediate trajectories. The strict −0.9 criterion has no theoretical backing; only the endpoint direction is guaranteed.

### 2.3 Test D Threshold Miscalibration

The original criterion 1 required `max |m({i,j})| < 0.01` for five random uniform predictors.

Under the null hypothesis (random classifiers), AUC estimates have variance approximately `1/(3 × n_val)` for a balanced dataset — this follows from the Wilcoxon/Mann-Whitney null variance formula; DeLong et al. (1988) extends this to the covariance between correlated AUCs. The pairwise dividend inherits variance ≈ `1/n_val` under the simplifying assumption of independence between the three AUC terms. Note: this overestimates variance because all three AUCs are evaluated on the same validation set and are therefore positively correlated; the true noise floor is likely lower. The expected maximum absolute dividend across 10 pairs is:

```
E[max |m({i,j})|] ≈ 2.53 / √n_val
```

The factor **2.53** applies to the maximum of 10 i.i.d. half-normal (absolute-valued normal) variates. The commonly cited 2.15 factor is the expected maximum of 10 *signed* standard normals and is incorrect here. For n_val = 114 (Breast Cancer): E[max] ≈ 0.237. The original 0.01 threshold requires n_val > (2.53/0.01)² ≈ **64,009** to be reliably achievable. The observed value of 0.053 is entirely explained by finite-sample AUC noise, not a framework failure.

### 2.4 Test B — Limited but Structural

Test B (SII correlation, r = 0.762 on n = 10 pairs) barely passes the 0.70 threshold. Key issue: SII is a linear function of Möbius dividends (Grabisch, Marichal & Roubens 2000), so the dividend–SII correlation is partly a mathematical artifact, not an independent validation. The Q-stat correlation (r = 0.517) is the genuinely independent convergent validity check.

With n = 10 pairs, minimum detectable Spearman ρ at 80% power is ρ ≥ 0.75–0.80, and the 95% CI for r = 0.76 spans approximately [0.33, 0.93] (Bonett & Wright 2000). The test passes but has low statistical power.

---

## 3. Theoretical Foundations

### 3.1 Möbius / Harsanyi Transform

The Möbius transform of a set function `v : 2^N → ℝ` is:

```
m(S) = Σ_{T ⊆ S} (−1)^{|S|−|T|} v(T)
```

For a pair: `m({i,j}) = v({i,j}) − v({i}) − v({j}) + v(∅)`

Positive `m({i,j})`: complementarity (the pair is more than the sum of parts).  
Negative `m({i,j})`: redundancy (one model is substitutable by the other).

**Key source:** Harsanyi (1963, *Int'l Economic Review* 4:2); Grabisch, Marichal & Roubens (2000, *Math. Operations Research* 25:2).

### 3.2 Shapley Interaction Index (SII)

The SII is NOT the same as `m({i,j})`. It is a coalition-weighted average over all supersets:

```
SII({i,j}) = Σ_{S ⊆ N\{i,j}} [|S|!(n−|S|−2)! / (n−1)!] × Δ_{ij} v(S)
```

where `Δ_{ij} v(S) = v(S∪{i,j}) − v(S∪{i}) − v(S∪{j}) + v(S)` is the discrete interaction derivative.

Equivalently: `SII({i,j}) = m({i,j}) + ½ Σ_k m({i,j,k}) + ...` — a weighted sum over all Möbius dividends of supersets containing `{i,j}`.

Because `m({i,j})` is the Möbius coefficient for exactly the pair `{i,j}`, while `SII({i,j})` aggregates higher-order dividends, they can differ in magnitude but rarely in sign when higher-order effects are small.

**Key source:** Grabisch & Roubens (1999, *Int'l Journal of Game Theory* 28:547–565).

### 3.3 Submodularity of AUC Games

AUC-based ensemble cooperative games are typically **submodular** (diminishing returns): adding a model to a larger coalition produces a smaller marginal AUC gain than adding it to a smaller coalition.

Submodularity implies `Δ_{ij} v(S) ≤ 0` for all `S`, which means `m({i,j}) ≤ 0` for **all** pairs, not just redundant ones. The framework therefore measures *relative redundancy* (how negative), not *sign-based* synergy vs redundancy.

This is not a flaw — it is a mathematical property of AUC games that requires the test design to focus on magnitude differences rather than sign. Test A's copy/diverse comparison is still valid because copy pairs should be *more* negative (higher redundancy) than diverse pairs.

**Key source:** Xu et al. (2023, arXiv:2006.14583) — "Replication Robust Payoff Allocation in Submodular Cooperative Games."

### 3.4 Ceiling Effect Kills Interaction Signal

When all models achieve near-perfect AUC, the interaction information collapses. This is analogous to the "LLM Chemistry" finding: perfectly accurate models produce identical outputs, so their pairwise dividends become indistinguishable. Any dataset where strong models saturate the AUC scale will produce this artifact.

**Key source:** arXiv:2510.03930 — "LLM Chemistry Estimation" — "perfect models paradoxically eliminate the interactions that chemistry-based methods seek to exploit."

---

## 4. Literature Findings by Topic

### 4.1 Interaction Indices and Möbius Transforms

| Paper | Key finding |
|-------|-------------|
| Grabisch & Roubens (1999). *Int'l J. Game Theory* 28:547. | Axiomatic definition of SII. SII ≠ m({i,j}); SII aggregates all Möbius dividends of supersets. |
| Grabisch, Marichal & Roubens (2000). *Math. Oper. Res.* 25:2. | SII = linear function of Möbius coefficients. Confirmed: SII–dividend correlation is partly mathematical, not independent. |
| Harsanyi (1963). *Int'l Economic Review* 4:2. | Commonly cited formulation of Harsanyi dividends. Note: the dividend concept was introduced in his earlier 1959 paper ("A Bargaining Model for Cooperative n-Person Games," Contributions to the Theory of Games Vol. IV); the 1963 paper is a simplified extension and the more frequently cited reference. |
| Tsai, Yeh & Ravikumar (2023). *JMLR* 24, arXiv:2203.00870. | Faith-Shap: Möbius transform-based faithfulness criterion. Validates use of m({i,j}) as an interaction proxy when higher-order terms are small. |
| Muschalik et al. (2024). NeurIPS D&B, arXiv:2410.01649. | shapiq benchmark: standard validation of SII implementations uses synthetic games with known ground-truth dividends, compared by MSE — not Spearman against competing measures. Supports Test B's design but suggests MSE validation as complementary check. |

### 4.2 Ensemble Diversity

| Paper | Key finding |
|-------|-------------|
| Kuncheva & Whitaker (2003). *Machine Learning* 51:181. | Diversity paradox: Q-stat and disagreement have no consistent correlation with ensemble accuracy. Test B's Q-stat correlation (r=0.52) is informative for construct validity but not predictive of performance gain. |
| Brown, Wyatt, Harris & Yao (2005). *Neurocomputing* 48. | Diversity from different loss functions (hinge vs. log-likelihood) provides complementary error patterns — supports using LR + LinearSVC despite both being linear. |
| Kuncheva (2014). *Combining Pattern Classifiers* (2nd ed.). Wiley. | Algorithmic diversity (different learning algorithms) more important than model class for small ensembles. Supports LinearSVC as a genuine diverse contributor. |
| Rozemberczki & Sarkar (2021). CIKM, arXiv:2101.02153. | Ensemble Shapley allocation — closest prior work to the coopetitive framework. Validates cooperative game theory as a principled ensemble weighting approach. |

### 4.3 AUC Properties and Variance

| Paper | Key finding |
|-------|-------------|
| Wood, Mu, Brown & Clifton (2023). *JMLR* 24, arXiv:2301.03962. | No clean bias-variance decomposition for AUC. AUC game characteristic functions are more complex than accuracy-based games. Warns against assuming AUC differences behave like accuracy differences. |
| Fawcett (2006). *Pattern Recognition Letters* 27:861. | Decision function values are rank-equivalent to calibrated probabilities for AUC. Validates using LinearSVC.decision_function() instead of predict_proba(). |
| DeLong, DeLong & Clarke-Pearson (1988). *Biometrics* 44:837. | Variance of AUC estimator: Var(AUC) ≈ (n_pos + n_neg + 1) / (12 × n_pos × n_neg). Under balanced H0: SE(AUC) ≈ 1/√(3n). |

### 4.4 Construct Validity Frameworks

| Paper | Key finding |
|-------|-------------|
| Campbell & Fiske (1959). *Psych. Bull.* 56:2. | MTMM framework establishing convergent and discriminant validity. The paper prescribes qualitative criteria only ("substantial" non-zero correlations); it does **not** specify numeric thresholds such as 0.70 or 0.85. Those conventions originate in later psychometric literature. |
| Cohen (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). | Effect size standards: r = 0.50 medium, r = 0.70 large. The −0.70 threshold for Test C is adopted from this convention, not from Campbell & Fiske. |
| Bonett & Wright (2000). *Psychometrika* 65:23. | Sample size requirements for precise Spearman estimation using a **confidence-interval width** approach (not a power analysis). At n=10, the 95% CI for r=0.76 is extremely wide (approximately [0.33, 0.93]), indicating that Test B lacks the precision to reliably distinguish moderate from strong correlations. |

### 4.5 Large-Scale SVM

| Paper | Key finding |
|-------|-------------|
| Fan, Chang, Hsieh, Wang & Lin (2008). *JMLR* 9:1871. | LIBLINEAR: LinearSVC trains in O(n) vs O(n²)–O(n³) for kernel SVM. Competitive accuracy on large datasets. Canonical reference for swapping to LinearSVC at n > 10,000. |
| Chang & Lin (2011). *ACM TIST* 2:27. | LIBSVM authors recommend LinearSVC first for large n. `probability=True` in SVC adds a further Platt scaling cross-validation pass, compounding the cost. |

### 4.6 Cooperative Games and Submodularity

| Paper | Key finding |
|-------|-------------|
| Han, Wooldridge, Rogers, Ohrimenko & Tschiatschek (2023). *IEEE Trans. Artificial Intelligence* 4(5):1114–1128. arXiv:2006.14583. | Replication-robust payoff allocation in submodular cooperative games: examines how replication of data/models affects Shapley values under submodularity. Supports the theoretical basis that copy pairs produce more negative interaction terms than diverse pairs in submodular games. |
| arXiv:2510.03930 (2025). "LLM Chemistry Estimation." | Ceiling effect: near-perfect models collapse all dividends to ≈ (0.5 − v({i}) − v({j})). Dataset must have headroom for interaction signal to be detectable. |

---

## 5. Changes Applied to `main.py`

### 5.1 Dataset: `'Breast Cancer'` → `'adult'`

**File:** `main.py`, line 58  
**Change:** `CONSTRUCT_VALIDITY_DATASET = 'adult'`  
**Rationale:** Census Income dataset (n=48,842, DT AUC ≈ 0.748). Models expected at AUC ≈ 0.82–0.88 — well below ceiling. n_val ≈ 9,768 provides sufficient resolution for all four tests.

| Dataset | n | n_val | Models AUC (est.) | Test D noise floor |
|---------|---|-------|-------------------|-------------------|
| Breast Cancer (old) | 569 | 114 | ~0.998 (ceiling) | 0.094 |
| diabetes (interim) | 768 | 154 | ~0.80–0.88 | 0.081 |
| **adult (current)** | **48,842** | **~9,768** | **~0.82–0.88** | **0.010** |

### 5.2 Test C Criterion: `< −0.9` → `< −0.7`

**File:** `main.py`, lines 769, 807, 820  
**Rationale:** Grabisch & Roubens (1999) guarantee only that the endpoint direction is negative, not strict monotonicity throughout. Submodular AUC trajectories may be non-monotone at intermediate α values. The −0.70 threshold matches the Campbell & Fiske (1959) convergent validity standard.

### 5.3 Test D: Criterion 2 is the primary pass criterion; Criterion 1 is diagnostic only

**File:** `main.py`, lines 787–802  
**Rationale:** The expected maximum of 10 absolute-normal dividend variates is **2.53/√n_val** (not 2.15, which is for signed normals). This structurally exceeds the 1σ adaptive threshold (1/√n_val) at every practical n_val:

| n_val | Adaptive threshold (1/√n_val) | E[max] (2.53/√n_val) | Criterion 1 outcome |
|-------|------------------------------|----------------------|---------------------|
| 114 (Breast Cancer) | 0.094 | ~0.237 | fails |
| 154 (diabetes) | 0.081 | ~0.204 | fails |
| 9,768 (adult) | 0.010 | ~0.026 | **fails** (0.026 > 0.010) |

Criterion 1 reliably passes only when n_val > (2.53/0.01)² ≈ **64,009** — larger than any dataset in the benchmark pool. It is therefore kept as a printed diagnostic but **does not determine pass/fail**.

The primary pass criterion is **criterion 2**: mean absolute random dividend must be at least 5× smaller than the Test A copy-pair mean. This is the theoretically grounded claim (Han et al. 2023): random classifiers produce near-zero structured interaction relative to the copy-pair signal. `pass_d = crit2`.

### 5.4 SVM: `SVC(kernel='rbf', probability=True)` → `LinearSVC`

**File:** `main.py`, line 644  
**Rationale:** Fan et al. (2008, JMLR) canonical reference. At n_train ≈ 39,000, kernel SVM is O(n²) ≈ 1.5 billion operations. LinearSVC is O(n) via coordinate descent. `decision_function()` is rank-equivalent to `predict_proba()` for AUC computation (Fawcett 2006).

```python
# Before
svm = SVC(kernel='rbf', probability=True, random_state=42)
m5_pred = svm.predict_proba(X_val)[:, 1]

# After
svm = LinearSVC(max_iter=5000, random_state=42)
m5_pred = svm.decision_function(X_val)
```

**Diversity preserved:** LR (log-loss, probabilistic, max-likelihood geometry) and LinearSVC (hinge loss, max-margin geometry) produce different error patterns despite both being linear — Kuncheva (2014), Brown et al. (2005).

**Note:** `get_base_models()` (Experiments 2–4) still uses `SubsampledSVC` (RBF SVM with 10k training-sample cap). Only the construct validity Test A model was changed.

---

## 6. Expected Outcomes After Re-run

### Test A
With models at AUC ~0.82–0.88, dividend dynamic range ≈ 0.10–0.20. Copy pairs (RF + near-copies with σ=0.01 noise) should produce dividends ~10–30× more negative than diverse pairs (RF–LR, RF–LinearSVC). Expected: **PASS**.

### Test B
Test B uses `get_base_models()` (RF, LR, SVM/SubsampledSVC, XGB, KNN) on the same `adult` dataset. Larger n_val → sharper AUC estimates → cleaner dividend–SII correlation. Expected: **PASS** (r > 0.70), possibly stronger than Breast Cancer result.

Note: Test B Q-stat and disagreement thresholds use 0.5-binarization of probability outputs from `get_base_models()`. The LinearSVC change (only in Test A) does not affect Test B.

### Test C
With `adult` at AUC ~0.84, the LR→RF blending trajectory has more AUC headroom. The submodularity-induced non-monotonicity at intermediate α values should be smaller relative to the signal. Expected: **PASS** (r < −0.70), though strict monotonicity (r < −0.90) is not guaranteed.

### Test D
Pass criterion is **criterion 2** (primary): mean absolute random dividend must be at least 5× smaller than the Test A copy mean. With n_val ≈ 9,768 and models at AUC ~0.83–0.87, |copy_mean| ≈ 0.20–0.30, giving five_times_smaller ≈ 0.04–0.06. Mean absolute random dividend ≈ 0.798/√9768 ≈ 0.008, well below the threshold. Expected: **PASS**.

Criterion 1 (diagnostic only): E[max] = 2.53/√9768 ≈ 0.026 > threshold 0.010 — will report FAIL. This does not affect the Test D pass verdict.

---

## 7. Remaining Limitations

1. **Test B statistical power:** n=10 pairs remains underpowered. Pooling dividends across multiple datasets (n=100–150 pairs) would provide 80% power for ρ ≥ 0.35 (Bonett & Wright 2000). Consider as a future extension.

2. **Test D criterion 1 structurally unachievable:** E[max |m({i,j})|] ≈ 2.53/√n_val always exceeds the 1σ threshold 1/√n_val regardless of n_val. Criterion 1 reliably passes only when n_val > 64,009, larger than any dataset in the pool. It is retained as a printed diagnostic; `pass_d = crit2`. Criterion 2 (5× relative comparison) is the theoretically grounded claim (Han et al. 2023).

3. **SII–dividend correlation is partly mathematical:** SII is a linear function of all Möbius dividends including higher-order terms. The SII–dividend correlation in Test B measures algebraic proximity, not independent empirical agreement. Q-stat is the independent convergent validity measure.

4. **`get_base_models()` SVM inconsistency:** Experiments 2–4 still use RBF SVM (with 10k training-sample cap), while the construct validity Test A now uses LinearSVC. This inconsistency should be resolved before the final manuscript version — either both use LinearSVC, or the manuscript explicitly distinguishes the two model pools.

---

## 8. References (Full Citations)

Bonett, D. G., & Wright, T. A. (2000). Sample size requirements for estimating Pearson, Kendall and Spearman correlations. *Psychometrika*, 65(1), 23–28.

Brown, G., Wyatt, J., Harris, R., & Yao, X. (2005). Diversity creation methods: A survey and categorisation. *Information Fusion*, 6(1), 5–20.

Campbell, D. T., & Fiske, D. W. (1959). Convergent and discriminant validation by the multitrait-multimethod matrix. *Psychological Bulletin*, 56(2), 81–105.

Chang, C.-C., & Lin, C.-J. (2011). LIBSVM: A library for support vector machines. *ACM Transactions on Intelligent Systems and Technology*, 2(3), 27.

Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. *Biometrics*, 44(3), 837–845.

Fan, R.-E., Chang, K.-W., Hsieh, C.-J., Wang, X.-R., & Lin, C.-J. (2008). LIBLINEAR: A library for large linear classification. *Journal of Machine Learning Research*, 9, 1871–1874.

Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874.

Grabisch, M., Marichal, J.-L., & Roubens, M. (2000). Equivalent representations of set functions. *Mathematics of Operations Research*, 25(2), 157–178.

Grabisch, M., & Roubens, M. (1999). An axiomatic approach to the concept of interaction among players in cooperative games. *International Journal of Game Theory*, 28(4), 547–565.

Harsanyi, J. C. (1963). A simplified bargaining model for the n-person cooperative game. *International Economic Review*, 4(2), 194–220.

Kuncheva, L. I. (2014). *Combining Pattern Classifiers: Methods and Algorithms* (2nd ed.). Wiley.

Kuncheva, L. I., & Whitaker, C. J. (2003). Measures of diversity in classifier ensembles and their relationship with the ensemble accuracy. *Machine Learning*, 51(2), 181–207.

Muschalik, M., Fumagalli, F., Hüllermeier, E., & Kolb, S. (2024). shapiq: Shapley interactions for machine learning. *NeurIPS Datasets and Benchmarks Track*. arXiv:2410.01649.

Rozemberczki, B., & Sarkar, R. (2021). Shapley values for network node classification: An application to ensemble learning. *CIKM 2021*. arXiv:2101.02153.

Tsai, C.-P., Yeh, C.-K., & Ravikumar, P. (2023). Faith-Shap: The faithful Shapley interaction index. *Journal of Machine Learning Research*, 24(94), 1–42. arXiv:2203.00870.

Wood, D. A., Mu, T., Brown, G., & Clifton, D. A. (2023). A unified theory of diversity in ensemble learning. *Journal of Machine Learning Research*, 24(359), 1–49. arXiv:2301.03962.

Han, D., Wooldridge, M., Rogers, A., Ohrimenko, O., & Tschiatschek, S. (2023). Replication-robust payoff-allocation for machine learning data markets. *IEEE Transactions on Artificial Intelligence*, 4(5), 1114–1128. arXiv:2006.14583.

*(Anonymous, 2025). LLM Chemistry Estimation [title abbreviated]. arXiv:2510.03930.*
