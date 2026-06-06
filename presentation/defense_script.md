# Defense Presentation — Speaker Script

**Non-Overlapping Ensemble Weighting via Möbius Decomposition**
Maxell G. Milay · UP Cebu · June 2026

> **Total speaking time: ≈ 14 min 55 sec** (target ≤ 15:00). Pacing assumes ~140 words/minute. Times are per slide; the running total is shown in brackets. Pause briefly on each chart — the audience needs a beat to read it.

---

## Slide 1 — Title — **0:20** *[0:20]*

Good [morning/afternoon], everyone. My project is titled *Non-Overlapping Ensemble Weighting via Möbius Decomposition* — a coopetitive framework for diagnosing and leveraging the interactions between models in an ensemble. I'll walk you through the problem, the method, what the experiments showed, and where it goes next.

---

## Slide 2 — The weighting problem in ensembles — **1:20** *[1:40]*

Ensembles work by combining many models, but they raise an immediate question: how much weight should each model get? The most principled answer is the Shapley value from cooperative game theory. But it has a structural blind spot, and that blind spot is really three gaps.

First, **entanglement**. The Shapley value blends a model's standalone quality with its interaction effects into a single number. So a strong-but-redundant model and a modest-but-complementary one can end up with the same weight — and you can't tell which is which.

Second, **double-counting**. The natural fix is to add an interaction term — the Shapley Interaction Index. But that re-adds interaction information already baked into the Shapley value, counting the same effect twice and distorting the weights.

Third, **interpretability**. No standard method gives you a pairwise map showing which models are complementary, redundant, or independent.

The key point: no existing method addresses all three.

---

## Slide 3 — Our solution: the Möbius decomposition — **1:20** *[3:00]*

Our solution is the Möbius decomposition — also called Harsanyi dividends. It splits the ensemble's value into mathematically non-overlapping pieces, and that "non-overlapping" property is an algebraic identity, not something we hope holds empirically.

There are two pieces. The **singleton dividend**, m of {i}, is a model's pure standalone value. The **pairwise dividend**, m of {i,j}, captures *only* the interaction — positive means the pair is complementary, negative means redundant.

We combine them into a single **coopetitive score**: a model's singleton dividend plus lambda times one-half the sum of its pairwise dividends. A softmax turns those scores into weights.

This buys us three things. We get **interpretability** — a readable pairwise matrix of redundancy and complementarity. We get **no double-counting** — by construction. And it's **simple and tunable** — one parameter, about fifteen lines of NumPy, no meta-learner to train.

---

## Slide 4 — A four-phase validation design — **1:05** *[4:05]*

The study is a comparative experiment: the weighting strategy is what we vary, and AUC-ROC — with Brier score as a secondary check — is the outcome.

It runs in four phases, and each one gates the next. Phase one is **construct validity** — do the pairwise dividends actually measure redundancy? Phase two is the **ablation study** — does each component earn its place? Phase three is the **benchmark** against established methods. And phase four is the **lambda sensitivity analysis**.

The pipeline behind every run is the same: train the base models, evaluate all two-to-the-n coalitions on a validation set, compute the Möbius dividends, derive the weights, and test on a held-out set. For statistics I follow Demšar's standard recipe — a Friedman omnibus test, then Holm-corrected Wilcoxon comparisons.

---

## Slide 5 — Experimental setup — **1:10** *[5:15]*

The pool is five heterogeneous base models — Logistic Regression, Random Forest, SVM, XGBoost, and k-Nearest Neighbors. They span five different inductive biases — linear, bagging, kernel, boosting, and instance-based — chosen deliberately so the pairwise dividends have real structure to detect.

On the evaluation side: fifteen binary-classification datasets from OpenML and UCI, spanning medical, financial, and signal domains, from two hundred up to nearly fifty thousand samples. For each run I evaluate all thirty-two coalitions exactly — every subset of the five models — scoring each by AUC-ROC on the validation set. And I repeat everything across fifteen datasets and five random seeds, giving seventy-five runs per method, with a sixty-twenty-twenty train-validation-test split.

---

## Slide 6 — The coopetitive weighting formula — **1:15** *[6:30]*

Here's the formula itself. The coopetitive score for model i is its singleton dividend, plus lambda times one-half the sum of its pairwise dividends with every other model.

The first term, m of {i}, is the **individual** contribution. The second term is the **interaction** — the net complementarity or redundancy with the rest of the pool — and the one-half simply shares each pair fairly between its two members, following the Shapley sharing rule. Lambda is the **dial**: at zero, interactions are ignored; as it grows, the weights track interaction structure more strongly.

And there's one design choice I want to flag, because the whole study hinges on it. When lambda equals zero, the framework is *algebraically identical* to performance-weighting. So the entire case for the interaction term comes down to one question — does lambda greater than zero actually beat that baseline? That's Hypothesis H1.

---

## Slide 7 — Four experiments, four hypotheses — **1:05** *[7:35]*

That gives us four hypotheses, one per experiment.

**H1**, the pivotal one: adding interaction information beats plain performance-weighting. **H2**: the pairwise dividends correctly detect known redundancy and complementarity — this is the construct-validity gate. **H3**: the framework is competitive with established ensemble methods on the benchmark. And **H4**: a single fixed lambda of one-half works well everywhere — a practical robustness check.

H1 is make-or-break for the performance claim, and H2 gates everything else — because if the diagnostic doesn't measure what it claims, nothing downstream is trustworthy. So let's start there.

---

## Slide 8 — Construct validity: the diagnostic works — **1:25** *[9:00]*

Four controlled tests, with known ground truth. **Test A** — inject artificial redundancy by making near-copies of a model — passes: the copies are correctly flagged as more redundant. **Test B** — the dividends agree with the Shapley Interaction Index at r equals 0.96 — passes strongly. **Test C** — as I blend two models to become more similar, the dividend declines smoothly, r equals minus 0.99 — passes. **Test D**, the negative control with independent models, technically *fails*: the dividends should all be essentially zero, and on average they are — 0.017 — but the maximum reaches 0.039, about four times our strict 0.01 threshold. I attribute that excess to finite-sample noise — only about a hundred-sixty validation samples — not real interaction, since the mean is still more than five times below Test A.

So **H2 is partially supported**: three of four tests pass convincingly, and I'm reporting Test D's failure plainly rather than dressing it up.

On the right is the payoff — the pairwise dividend matrix. Every pair here is redundant; the most redundant is SVM and XGBoost, the least is KNN and LR. This matrix *is* the framework's most durable contribution.

---

## Slide 9 — Does the interaction term help? — **1:25** *[10:25]*

This is the ablation — ten weighting variants, ranked by mean AUC. Two things matter most.

First, **Coopetitive-0.5 comes out on top** — the best AUC at 0.8977 and the best calibration of any variant, winning on nine of fifteen datasets.

Second — and this is the cleanest result in the whole study — look at **Shapley plus SII**. It performs *worse* than plain Shapley, by about 0.009 AUC. That's the double-counting problem, made visible: re-adding interactions that are already in the Shapley value actively hurts. The Möbius framework avoids that by construction. Given the total spread across all variants is only 0.027, that penalty is about a third of the entire range.

Now the honest part. The formal test for H1 — Coopetitive versus performance-weighting — gives p equals 0.148. So **H1 is not statistically significant**. The direction is consistent — nine wins, a small-to-medium effect size — but it's a small effect, and we're underpowered to confirm it.

---

## Slide 10 — Benchmark ranking & λ sensitivity — **1:15** *[11:40]*

Two results here. On the left, the benchmark. **Coopetitive-CV ranks second of six methods**, essentially tied with Shapley at the top. Just behind them is performance-weighting at 3.23, then stacking, best-single, and equal weighting trailing further back. The Friedman test gives p equals 0.072 — so **H3 doesn't reach significance** either, but the method is genuinely competitive. Interestingly, stacking underperforms here, most likely because eleven of our datasets are small and its meta-learner overfits.

On the right, lambda sensitivity. There is no universal best lambda — the optimum ranges from zero all the way to 1.5 across datasets, with a mean around 0.67. A fixed lambda of one-half reaches ninety-five percent of the optimum on only thirteen percent of datasets. So **H4 is not supported** — the practical recommendation is to select lambda by cross-validation, which is essentially free since it runs on cached predictions.

---

## Slide 11 — What the evidence says — **1:15** *[12:55]*

Pulling it together. H1 — not significant. H2 — partially supported. H3 — not significant but competitive. H4 — not supported.

That looks like a lot of "not significant," so let me give the honest reading.

The **real win is interpretability** — the dividend matrix is a diagnostic no other method offers, and that value is independent of any accuracy gain. On **performance**, the framework is competitive but this isn't a breakthrough; the effects are small, and a post-hoc analysis suggests you'd need thirty to forty datasets to settle H1 conclusively. And one finding reshaped the framing: **coopetition ran one-sided** — across all seventy-five runs every dividend was negative, so in this setting the framework operated purely as a redundancy penalty. The cooperative half never appeared with these well-tuned models.

---

## Slide 12 — Conclusions & limitations — **1:10** *[14:05]*

So, conclusions. The Möbius decomposition clearly **resolves the interpretability gap** — the matrix maps redundancy and complementarity directly. It **resolves double-counting** both in theory and empirically. The entanglement and benchmark gains were directional but not significant, though calibration was the best of any method tested.

I'll be upfront about the limitations. **Truncating at pairwise terms** misses higher-order structure — that's why decomposition coverage stayed negative. There's **no fixed lambda** — it's dataset-dependent. Exact enumeration **limits the pool to about twelve models**. And the scope is binary classification, with only five models, which weakens one of the diagnostics.

The bottom line: this is a principled, interpretable diagnostic instrument for ensemble design — the kind of transparency that purely performance-focused methods don't give you.

---

## Slide 13 — Where this goes next + thank you — **0:45** *[14:50]*

Three directions forward. Add **higher-order dividends** to restore full coverage — the most direct fix. Use the matrix for **dividend-informed pruning** — dropping redundant models before combining, since the full ensemble was best in only eight percent of runs. And **broaden and scale** — to multi-class and regression, with sampling-based estimation to break the twelve-model ceiling.

Thank you — the code and data are all public on the repository shown here, and I'm happy to take your questions.

---

### Timing summary

| Section | Slides | Time |
|---|---|---|
| Introduction | 1–3 | 3:00 |
| Methods | 4–7 | 4:35 |
| Results & Discussion | 8–11 | 5:20 |
| Conclusion & Future Work | 12–13 | 1:55 |
| **Total** | **13** | **≈ 14:50** |

**Buffer tips if you're running long:** trim Slide 5 (the audience can read the setup) and Slide 7 (the hypotheses are on screen). **If running short:** expand the double-counting explanation on Slide 9 — it's your strongest result and worth dwelling on.
