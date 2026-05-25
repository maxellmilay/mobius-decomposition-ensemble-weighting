Experiment 1 (Construct Validity) — "Does the measurement tool (Harsanyi Dividends) work?" → Mostly yes
Test A (redundancy detection): Trained a Random Forest, made two noisy copies of it, trained LR and SVM separately, computed all 10 pairwise dividends. Do dividends assign more negative values to the copies? ✅ Yes — copy-pairs scored more negative than dissimilar pairs; homogeneous pool (5 tree variants) was 10/10 negative
Test B (agreement with existing measures): Trained all 5 base models, computed pairwise dividends alongside the Shapley Interaction Index, Q-statistic, and disagreement measure for all 10 pairs, then computed Spearman correlations between them. Do they agree? ✅ Yes — near-perfect agreement with SII (r = 0.96); strong agreement with Q-statistic (r = −0.86) and disagreement measure (r = 0.82); no contradictory pairs,
Test C (sensitivity to gradual change): Created a blended model that smoothly transitions from pure LR (α = 0) to pure RF copy (α = 1) in 11 steps, measured the pairwise dividend with RF at each step. Does it decline smoothly? ✅ Yes — smooth monotonic decline (r = −0.99), exceeding the −0.9 threshold
Test D (false positive check): Trained 5 logistic regressions on completely separate feature subsets so their errors are independent, computed pairwise dividends using accuracy instead of AUC. Do dividends stay near zero? ❓ Inconclusive — switching from AUC (baseline 0.5) to accuracy (baseline 0.0) inflated all magnitudes, making cross-test comparison invalid
H2 result: Partially supported — three of four subtests passed; the negative control was inconclusive due to a design flaw (metric mismatch), not a failure of the dividends themselves
Diagnostics — "Do the math assumptions hold?" → Partially
Decomposition coverage asks: does our simplified version of the math (using only individual and pairwise terms) account for most of the ensemble's total value? The answer was no — the pairwise redundancy terms are so large and negative that they drag the sum way below zero. This doesn't mean the math is broken; it means we're only looking at two out of five levels of the decomposition, and the missing levels (three-way, four-way, five-way interactions) are doing a lot of heavy lifting to compensate for all the pairwise redundancy.
Monotonicity violations asks: does adding a model to a group always help? You'd hope so, but in practice, adding a redundant model to an already good group can drag performance down through averaging. The check found this happening 3%–48% of the time depending on the dataset, which means the ensemble game isn't always "more is better."
Grand coalition optimality asks: is using all five models together actually the best option? Almost always no — some smaller combination of three or four models beat the full set. This makes sense given all the pairwise redundancy: if models overlap, you're better off dropping the redundant ones.
Rank stability asks: did we cheat by using equal weights to evaluate coalitions but then applying unequal weights at the end? If the model rankings flip when you switch from equal to coopetitive weights, that's a problem. Rankings stayed perfectly stable, but with only five models, it's hard for them to shuffle in the first place, so the test doesn't tell you much.
Decomposition coverage (summed all singleton and pairwise dividends, divided by total ensemble improvement v(N) − v(∅), checked if it's close to 100%): ❌ Universally negative (−2.6 to −5.0); the 10 negative pairwise terms (≈ −3.0) overwhelm the 5 positive singleton terms (≈ +1.75); metric is misleading for truncated decompositions and should be read as a redundancy flag
Monotonicity violations (checked every possible "add one model to a coalition" step across all 32 coalitions, counted how often performance decreased): ⚠️ Not always helpful — 2.7%–47.8% across datasets; six datasets exceeded the 40% threshold; no single model consistently at fault
Grand coalition optimality (compared v(N) against all 31 other coalition values to see if the full ensemble is actually the best): ❌ Full ensemble was best in only 8% of runs; smaller subsets usually outperformed
Rank stability (recomputed coalition values using the framework's own weights instead of equal weights, then checked if the model ranking changed via Spearman correlation): ✅ Perfect 1.0 everywhere, but with only 5 models to rank, the diagnostic is too coarse to be informative
Experiment 2 (Ablation Study) — "Which parts of the formula matter?" → Both parts needed, and don't double-count
Sanity check (computed weights using λ = 0 and compared them to simple AUC-proportional weights): ✅ Identical results (AUC = 0.8958), confirming the algebraic equivalence
Interaction value (compared Coopetitive-0.5 against Individual-Only across all 15 datasets × 5 seeds): Coopetitive-0.5 beat Individual-Only on 9/15 datasets (+0.0018 AUC), best Brier score of all methods
Interaction sufficiency (weighted models using only pairwise dividends, ignoring individual quality entirely): ❌ Worst of all ten variants (AUC = 0.8707) — interaction structure without individual quality information produces poor weights
Double-counting test (computed Shapley values, then added SII interaction terms on top, compared against plain Shapley and Möbius-based variants): ✅ Confirmed — Shapley+SII (0.8879) performed worse than plain Shapley (0.8968) by 0.009 AUC, validating that double-counting degrades performance
H1 result: Not statistically significant (Wilcoxon p = 0.148), but directionally consistent (9/15 wins, r = 0.29, best Brier score)
Experiment 3 (Benchmark Comparison) — "How does it compare to standard methods?" → Competitive but not dominant
Rank comparison (ran all 6 methods on all 15 datasets × 5 seeds, ranked methods per dataset, averaged ranks, applied Friedman test): Coopetitive-CV ranked 2nd of 6 (rank 2.87), just behind Shapley (2.80)
Stacking performance (trained a logistic regression meta-learner on base model predictions, compared against game-theoretic methods): Stacking placed 4th (rank 3.50), likely overfitting to small validation sets (40–60 instances in small datasets)
Calibration comparison (computed negative Brier score for all methods to check probability quality, not just ranking quality): Coopetitive-0.5 had the best Brier score (−0.097) of all methods, meaning it produces the most trustworthy probability estimates
H3 result: Not statistically significant (Friedman p = 0.072), but suggestive of real differences and best calibration
Experiment 4 (Sensitivity Analysis) — "Can we pick one λ that works everywhere?" → No
Optimal λ range (swept λ from 0 to 2.0 in steps of 0.1 for each dataset, recorded the λ that gave the highest AUC): Ranged from 0.0 (Banknote, Sonar) to 1.5 (Australian Credit), mean 0.67 ± 0.45
Default robustness (checked what fraction of datasets achieve ≥95% of their best possible AUC at the fixed default λ = 0.5): ❌ Only 13% of datasets — the default fails most of the time
Robustness ratio (for each dataset, counted what fraction of all 21 λ values achieve ≥95% of optimal, then averaged): Only 8% — the near-optimal zone is narrow, meaning small changes in λ can matter
Dataset dependence (ran repeated-measures ANOVA with λ and dataset as factors, tested whether the optimal λ varies by dataset): ✅ Confirmed — significant λ × dataset interaction (F = 12.5, p ≈ 0); easy datasets prefer low λ, difficult datasets prefer high λ
H4 result: Not supported — no universal default exists; cross-validated λ selection is essential

Shapley computes each model's average marginal contribution across all 32 possible coalitions, then uses those Shapley values as weights. It accounts for interactions implicitly but entangles them with individual quality into one number.
Coopetitive-CV computes singleton dividends (individual quality) and pairwise dividends (interaction effects) separately via the Möbius decomposition, combines them using the formula s_i = m({i}) + λ × ½Σm({i,j}), and selects the best λ through 5-fold cross-validation on the validation set.
Performance-Weighted assigns each model a weight proportional to its standalone AUC minus 0.5 (the random baseline). Better individual performers get higher weights; interactions are completely ignored.
Stacking trains a logistic regression meta-learner on top of the base models' predicted probabilities using cross-validation. The meta-learner's learned coefficients become the effective ensemble weights. It can implicitly capture interactions but is opaque and needs enough data to fit the meta-learner well.
Best Single Model picks whichever individual model scored highest on the validation set and uses only that one. No combination at all.
Equal gives every model the same weight (1/5). No information about quality or interactions is used.


Workshop Notes

4.1
Purely narrative
Give context of the reader to a summary of the method
“The study developed {insert method}. To measure this performance, {insert metrics}”
Active citations when comparing with benchmarks

In the discussion part, we do not mention any figures anymore
Zoom in graphs to emphasize differences

Give some context on each section before going into the results


Last few paragraphs include
Contribution
Limitations


Concepts to Note
Probability calibration
