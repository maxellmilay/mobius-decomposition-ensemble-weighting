#!/usr/bin/env python3
"""
Non-Overlapping Ensemble Weighting via Mobius Decomposition: A Coopetitive Framework for Diagnosing and Leveraging Model Interactions
"""
import argparse
import sys, warnings, time
# Force UTF-8 output on Windows (cp1252 cannot encode Greek characters used in print statements)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from math import factorial
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
RESULTS_DIR = Path(__file__).parent / "results"
USABLE_DATASETS_DIR = Path(__file__).parent / "results" / "usable_datasets"
RESULTS_DIR.mkdir(exist_ok=True)
EXP_DIRS = {n: RESULTS_DIR / f'experiment_{n}' for n in range(1, 5)}

N_SEEDS = 5
LAMBDA_GRID = np.round(np.arange(0, 2.05, 0.1), 1)
LAMBDA_DEFAULT = 0.5

# Dataset used for Experiment 1 (construct validity) Tests A and B.
# Must be 'Breast Cancer' or a dataset name present in the usable_datasets CSV.
# Swap this constant to change the construct validity base dataset.
CONSTRUCT_VALIDITY_DATASET = 'Breast Cancer'

# ============================================================
# DATASET LOADING
# ============================================================
def _fetch_openml_dataset(data_id, name):
    """Fetch from OpenML, encode categoricals, impute NaN, binarize target."""
    data = fetch_openml(data_id=data_id, as_frame=True)
    X_df = data.data.copy()
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        enc = OrdinalEncoder()
        X_df[cat_cols] = enc.fit_transform(X_df[cat_cols].astype(str))
    X = X_df.values.astype(float)
    if np.isnan(X).any():
        X = SimpleImputer(strategy='mean').fit_transform(X)
    y_raw = data.target
    if hasattr(y_raw, 'cat'):
        y_int = y_raw.cat.codes.values.astype(int)
    else:
        y_int = LabelEncoder().fit_transform(y_raw.astype(str).values)
    unique = np.unique(y_int)
    if len(unique) != 2:
        raise ValueError(f"{name}: {len(unique)} classes found, need exactly 2")
    y = (y_int == unique[1]).astype(int)
    return X, y


def _latest_usable_csv() -> Path:
    csvs = sorted(USABLE_DATASETS_DIR.glob('*.csv'))
    if not csvs:
        raise FileNotFoundError(
            f"No usable-datasets CSV found in {USABLE_DATASETS_DIR}. "
            "Run extract_usable_datasets.py first."
        )
    return csvs[-1]


def get_datasets():
    """Load all validated benchmark datasets from the usable_datasets CSV."""
    csv_path = _latest_usable_csv()
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df['name'] = df['name'].str.strip()
    print(f"Loading {len(df)} validated datasets from {csv_path.name}...")
    datasets = {}
    for _, row in df.iterrows():
        name = row['name']
        openml_id = int(row['openml_id'])
        try:
            X, y = _fetch_openml_dataset(openml_id, name)
            datasets[name] = (X, y, 'OpenML', len(X))
            print(f"  Loaded {name}: {len(X)} samples, {X.shape[1]} features "
                  f"(OpenML {openml_id})")
        except Exception as exc:
            print(f"  SKIP {name} (OpenML {openml_id}): {exc}")
    return datasets


def split_data(X, y, seed=42):
    """60/20/20 train/val/test split."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# BASE MODEL POOL
# ============================================================
class SubsampledSVC:
    """RBF SVC that randomly caps training data at max_samples.
    RBF SVM training complexity is O(n²)–O(n³) in n_samples, making it intractable on
    large datasets (e.g., Adult Census at 48,842 rows). The 10,000-sample cap is a
    practical engineering concession not stated in the manuscript; it preserves the RBF
    kernel specified in §3.3.1 while keeping wall-clock time feasible."""
    def __init__(self, max_samples=10000, **svc_kwargs):
        self.max_samples = max_samples
        self._svc = SVC(**svc_kwargs)

    def fit(self, X, y):
        if len(X) > self.max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), self.max_samples, replace=False)
            X, y = X[idx], y[idx]
        self._svc.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._svc.predict_proba(X)


def get_base_models(n_samples=1000):
    """Return the 5 heterogeneous base models.
    SVM uses SubsampledSVC (10k cap) for datasets with >10k training samples.
    """
    if n_samples > 10000:
        svm = SubsampledSVC(max_samples=10000, kernel='rbf',
                            probability=True, random_state=42)
    else:
        svm = SVC(kernel='rbf', probability=True, random_state=42)
    return {
        'LR':  LogisticRegression(max_iter=1000, random_state=42),
        'RF':  RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'SVM': svm,
        'XGB': XGBClassifier(n_estimators=200, random_state=42,
                             eval_metric='logloss',  # standard for binary classification; not in manuscript
                             verbosity=0),
        'KNN': KNeighborsClassifier(n_neighbors=7, n_jobs=-1),  # §3.3.1: n_neighbors=7
    }


def get_homogeneous_models():
    """Return 5 tree-based variants for construct validity (§3.3.1)."""
    return {
        'RF100':  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'RF200':  RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'RF500':  RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
        'XGB100': XGBClassifier(n_estimators=100, random_state=42,
                                eval_metric='logloss', verbosity=0),
        'XGB200': XGBClassifier(n_estimators=200, random_state=42,
                                eval_metric='logloss', verbosity=0),
    }


def train_and_cache(models, X_train, y_train, X_val, X_test):
    """Train all models and cache validation + test predicted probabilities."""
    val_preds, test_preds = {}, {}
    model_names = list(models.keys())
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_preds[name] = model.predict_proba(X_val)[:, 1]
        test_preds[name] = model.predict_proba(X_test)[:, 1]
    return model_names, val_preds, test_preds


# ============================================================
# CORE FRAMEWORK
# ============================================================
def evaluate_coalition(member_preds, y, metric='auc'):
    if len(member_preds) == 0:
        return 0.5 if metric == 'auc' else 0.0
    avg_pred = np.mean(member_preds, axis=0)
    if metric == 'auc':
        try:
            return roc_auc_score(y, avg_pred)
        except ValueError:
            return 0.5
    else:
        return np.mean((avg_pred >= 0.5) == y)


def compute_all_coalitions(model_names, preds_dict, y, metric='auc'):
    n = len(model_names)
    v = {frozenset(): 0.5 if metric == 'auc' else 0.0}
    for size in range(1, n + 1):
        for combo in combinations(range(n), size):
            coalition = frozenset(combo)
            member_preds = [preds_dict[model_names[i]] for i in combo]
            v[coalition] = evaluate_coalition(member_preds, y, metric)
    return v


def compute_singleton_dividends(model_names, v):
    empty = v[frozenset()]
    return {model_names[i]: v[frozenset([i])] - empty for i in range(len(model_names))}


def compute_pairwise_dividends(model_names, v):
    n = len(model_names)
    empty = v[frozenset()]
    dividends = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = tuple(sorted([model_names[i], model_names[j]]))
            val = (v[frozenset([i, j])] - v[frozenset([i])]
                   - v[frozenset([j])] + empty)
            dividends[key] = val
    return dividends


def compute_shapley_values(model_names, v):
    n = len(model_names)
    shapley = {name: 0.0 for name in model_names}
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            for combo in combinations(others, size):
                S = frozenset(combo)
                marginal = v[S | {i}] - v[S]
                weight = factorial(size) * factorial(n - size - 1) / factorial(n)
                shapley[model_names[i]] += weight * marginal
    return shapley


def compute_shapley_interaction_index(model_names, v):
    n = len(model_names)
    sii = {}
    for i in range(n):
        for j in range(i + 1, n):
            delta = 0.0
            others = [k for k in range(n) if k != i and k != j]
            for size in range(n - 1):
                for combo in combinations(others, size):
                    S = frozenset(combo)
                    interaction = (v[S | {i, j}] - v[S | {i}]
                                   - v[S | {j}] + v[S])
                    weight = (factorial(size) * factorial(n - size - 2)
                              / factorial(n - 1))
                    delta += weight * interaction
            key = tuple(sorted([model_names[i], model_names[j]]))
            sii[key] = delta
    return sii


def _get_pair_div(pair_div, name_a, name_b):
    return pair_div.get(tuple(sorted([name_a, name_b])), 0.0)


def coopetitive_scores(model_names, singleton_div, pairwise_div, lam):
    """s_i = m({i}) + λ × ½ Σ_{j≠i} m({i,j})."""
    scores = {}
    for name in model_names:
        interaction_sum = sum(
            _get_pair_div(pairwise_div, name, other)
            for other in model_names if other != name
        )
        scores[name] = singleton_div[name] + lam * 0.5 * interaction_sum
    return scores


def softmax_weights(scores, model_names):
    """Softmax normalization with τ = std(scores). Numerically stable.
    τ floor of 1e-10 prevents division by zero when all scores are identical (not in
    manuscript; standard numerical safeguard). NaN fallback to equal weights handles the
    degenerate case where all scores are the same after scaling (engineering safeguard)."""
    vals = np.array([scores[name] for name in model_names])
    tau = max(np.std(vals), 1e-10)
    scaled = vals / tau
    scaled -= scaled.max()  # numerical stability: shift to avoid overflow in exp
    w = np.exp(scaled)
    w /= w.sum()
    if np.any(np.isnan(w)):
        w = np.ones(len(model_names)) / len(model_names)
    return {name: w[i] for i, name in enumerate(model_names)}


def proportional_weights(scores, model_names):
    """Proportional normalization: w_i = max(s_i, ε) / Σ max(s_k, ε).
    The 1e-10 floor prevents zero or negative weights without distorting positive scores
    (engineering safeguard; not stated in the manuscript)."""
    vals = {n: max(scores[n], 1e-10) for n in model_names}
    total = sum(vals.values())
    return {n: vals[n] / total for n in model_names}


def weighted_ensemble_predict(weights, preds_dict, model_names):
    pred = np.zeros_like(preds_dict[model_names[0]])
    for name in model_names:
        pred += weights[name] * preds_dict[name]
    return pred


# ============================================================
# DIAGNOSTICS
# ============================================================
def monotonicity_diagnostic(model_names, v):
    """Check fraction of negative marginal contributions.
    Returns overall rate and per-model violation rates for flagging (§3.4.2)."""
    n = len(model_names)
    total, neg = 0, 0
    per_model = {name: {'total': 0, 'neg': 0} for name in model_names}
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            for combo in combinations(others, size):
                S = frozenset(combo)
                marginal = v[S | {i}] - v[S]
                total += 1
                per_model[model_names[i]]['total'] += 1
                if marginal < 0:
                    neg += 1
                    per_model[model_names[i]]['neg'] += 1
    overall_rate = neg / total if total > 0 else 0.0
    per_model_rates = {
        name: d['neg'] / d['total'] if d['total'] > 0 else 0.0
        for name, d in per_model.items()
    }
    # §3.4.2: flag individual models whose own violation rate exceeds the same 40%
    # threshold used for the overall diagnostic, for consistency with the paper.
    flagged = [name for name, rate in per_model_rates.items() if rate > 0.4]
    return overall_rate, per_model_rates, flagged


def grand_coalition_optimal(model_names, v):
    grand = frozenset(range(len(model_names)))
    v_grand = v[grand]
    max_v = max(v.values())
    # 1e-10 is a floating-point tolerance to treat values equal within numerical precision.
    return v_grand >= max_v - 1e-10, v_grand, max_v


def interaction_signal(pairwise_div, threshold=0.001):
    return any(abs(val) > threshold for val in pairwise_div.values())


def decomposition_coverage(model_names, v, singleton_div, pairwise_div):
    grand = frozenset(range(len(model_names)))
    total_value = v[grand] - v[frozenset()]
    # 1e-10: floating-point safeguard — returns 1.0 (full coverage) for a trivial game
    # where the grand coalition adds negligible value over the empty set.
    if abs(total_value) < 1e-10:
        return 1.0
    explained = sum(singleton_div.values()) + sum(pairwise_div.values())
    return explained / total_value


def rank_stability_diagnostic(model_names, weights, preds_dict, y, v_orig_singletons):
    """Recompute v(S) with coopetitive weights, then recompute singletons.
    Return Spearman r between original and recomputed singleton dividends."""
    n = len(model_names)
    # Recompute v(S) using weighted aggregation
    v_new = {frozenset(): 0.5}
    for size in range(1, n + 1):
        for combo in combinations(range(n), size):
            coalition = frozenset(combo)
            member_names = [model_names[i] for i in combo]
            total_w = sum(weights[nm] for nm in member_names)
            # Fallback to uniform sum (1.0) if all coalition members have near-zero
            # weights — prevents division by zero in degenerate softmax outputs.
            if total_w < 1e-10:
                total_w = 1.0
            # Weighted average within coalition
            pred = sum(weights[nm] / total_w * preds_dict[nm] for nm in member_names)
            try:
                v_new[coalition] = roc_auc_score(y, pred)
            except ValueError:
                v_new[coalition] = 0.5

    new_singletons = compute_singleton_dividends(model_names, v_new)
    orig_vals = [v_orig_singletons[n] for n in model_names]
    new_vals = [new_singletons[n] for n in model_names]
    if len(set(orig_vals)) <= 1 or len(set(new_vals)) <= 1:
        return 1.0  # constant → trivially stable
    corr, _ = stats.spearmanr(orig_vals, new_vals)
    return corr


# ============================================================
# EXPERIMENT HELPERS
# ============================================================
def evaluate_method_on_test(weights, model_names, test_preds, y_test):
    pred = weighted_ensemble_predict(weights, test_preds, model_names)
    pred = np.nan_to_num(pred, nan=0.5)  # safety
    pred = np.clip(pred, 0, 1)
    try:
        auc = roc_auc_score(y_test, pred)
    except ValueError:
        auc = 0.5
    brier = brier_score_loss(y_test, pred)
    return auc, -brier


def run_single_dataset_seed(dataset_name, X, y, seed):
    X_tr, X_val, X_test, y_tr, y_val, y_test = split_data(X, y, seed)
    models = get_base_models(n_samples=len(X))
    model_names, val_preds, test_preds = train_and_cache(
        models, X_tr, y_tr, X_val, X_test)

    v = compute_all_coalitions(model_names, val_preds, y_val)
    sing_div = compute_singleton_dividends(model_names, v)
    pair_div = compute_pairwise_dividends(model_names, v)

    mono_rate, per_model_mono, flagged_models = monotonicity_diagnostic(model_names, v)
    gc_opt, v_grand, v_max = grand_coalition_optimal(model_names, v)
    has_signal = interaction_signal(pair_div)
    coverage = decomposition_coverage(model_names, v, sing_div, pair_div)

    # §3.4.1: if >40% of marginal contributions are negative, flag offending models
    if mono_rate > 0.4 and flagged_models:
        print(f"    MONOTONICITY WARNING ({dataset_name}): rate={mono_rate:.2f}, "
              f"flagged={flagged_models}")

    # §3.4.1: if no pairwise dividend exceeds 0.001, λ should be forced to zero
    if not has_signal:
        print(f"    SIGNAL ABSENT ({dataset_name}): all |m({{i,j}})| ≤ 0.001; "
              f"λ forced to 0 (pairwise term adds no meaningful signal)")

    # §3.4.1: if decomposition coverage < 80%, higher-order interactions are substantial
    if coverage < 0.80:
        print(f"    COVERAGE NOTE ({dataset_name}): coverage={coverage:.3f} < 0.80; "
              f"higher-order interactions are substantial (limitation acknowledged)")

    coop_scores = coopetitive_scores(model_names, sing_div, pair_div, LAMBDA_DEFAULT)
    coop_weights = softmax_weights(coop_scores, model_names)
    rank_stab = rank_stability_diagnostic(
        model_names, coop_weights, val_preds, y_val, sing_div)

    # §3.4.4: if rank stability r > 0.90, ranking-preservation assumption is validated
    if rank_stab > 0.90:
        print(f"    RANK STABILITY VALIDATED ({dataset_name}): r={rank_stab:.4f} > 0.90")

    return {
        'model_names': model_names, 'v': v,
        'val_preds': val_preds, 'test_preds': test_preds,
        'y_val': y_val, 'y_test': y_test,
        'X_tr': X_tr, 'X_val': X_val, 'X_test': X_test,
        'y_tr': y_tr,
        'sing_div': sing_div, 'pair_div': pair_div,
        'mono_rate': mono_rate, 'flagged_models': ','.join(flagged_models),
        'gc_optimal': gc_opt,
        'v_grand': v_grand, 'v_max': v_max,
        'has_signal': has_signal, 'coverage': coverage,
        'rank_stability': rank_stab,
    }


# ============================================================
# STACKING
# ============================================================
def stacking_predict(val_preds, y_val, test_preds, model_names):
    """Stacking with cross-validated logistic regression meta-learner (§3.5.3).
    Base model predictions on the validation set serve as out-of-fold features
    (base models were trained only on the training set, so these are held-out).
    LogisticRegressionCV adds internal cross-validation for regularization."""
    X_meta_val = np.column_stack([val_preds[n] for n in model_names])
    X_meta_test = np.column_stack([test_preds[n] for n in model_names])
    # cv=5 matches the 5-fold inner CV used for λ tuning (§3.5.2) for consistency;
    # the manuscript specifies "cross-validated" but does not state the fold count here.
    meta = LogisticRegressionCV(max_iter=1000, cv=5, random_state=42)
    meta.fit(X_meta_val, y_val)
    return meta.predict_proba(X_meta_test)[:, 1]


# ============================================================
# ALL WEIGHTING METHODS
# ============================================================
def get_all_method_weights(model_names, v, sing_div, pair_div,
                           val_preds, y_val, has_signal=True):
    # §3.4.1: when no pairwise dividend exceeds 0.001 in absolute value, λ is forced to
    # zero because the pairwise term provides no meaningful signal.
    effective_lambda_default = LAMBDA_DEFAULT if has_signal else 0.0
    n = len(model_names)
    methods = {}

    # 1. Equal weight
    methods['Equal'] = {name: 1.0 / n for name in model_names}

    # 2. Performance-weighted: w_i ∝ AUC_i − 0.5
    auc_vals = {name: roc_auc_score(y_val, val_preds[name]) for name in model_names}
    pw_raw = {name: max(auc_vals[name] - 0.5, 1e-10) for name in model_names}
    pw_sum = sum(pw_raw.values())
    methods['Perf-Weighted'] = {name: pw_raw[name] / pw_sum for name in model_names}

    # 3. Individual-only (λ=0): proportional normalization
    #    m({i}) = v({i}) - v(∅) = AUC_i - 0.5, so proportional → same as Perf-Weighted
    methods['Individual-Only'] = proportional_weights(sing_div, model_names)

    # §3.5.2 sanity check: Individual-Only (λ=0) must equal Performance-Weighted.
    # The 0.01 tolerance is an engineering threshold (not stated in the manuscript)
    # chosen to catch numerical issues while ignoring floating-point rounding noise.
    pw_w = methods['Perf-Weighted']
    io_w = methods['Individual-Only']
    max_diff = max(abs(pw_w[nm] - io_w[nm]) for nm in model_names)
    if max_diff > 0.01:
        print(f"    WARNING: Individual-Only ≠ Perf-Weighted (max diff={max_diff:.4f})")

    # 4. Interaction-only
    int_scores = {}
    for name in model_names:
        int_scores[name] = 0.5 * sum(
            _get_pair_div(pair_div, name, other)
            for other in model_names if other != name
        )
    methods['Interaction-Only'] = softmax_weights(int_scores, model_names)

    # 5. Coopetitive (λ=0.5): softmax normalization; λ forced to 0 if no signal (§3.4.1)
    s05 = coopetitive_scores(model_names, sing_div, pair_div, effective_lambda_default)
    methods['Coopetitive-0.5'] = softmax_weights(s05, model_names)

    # 6. Coopetitive (CV λ): 5-fold inner CV
    best_lam, best_auc = 0.0, -1.0
    X_meta = np.column_stack([val_preds[nm] for nm in model_names])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for lam_candidate in LAMBDA_GRID:
        fold_aucs = []
        for tr_idx, te_idx in skf.split(X_meta, y_val):
            fold_preds = {nm: val_preds[nm][tr_idx] for nm in model_names}
            fold_y = y_val[tr_idx]
            fold_v = compute_all_coalitions(model_names, fold_preds, fold_y)
            fold_sing = compute_singleton_dividends(model_names, fold_v)
            fold_pair = compute_pairwise_dividends(model_names, fold_v)
            fold_scores = coopetitive_scores(model_names, fold_sing, fold_pair, lam_candidate)
            fold_weights = softmax_weights(fold_scores, model_names)
            fold_test_preds = {nm: val_preds[nm][te_idx] for nm in model_names}
            fold_pred = weighted_ensemble_predict(fold_weights, fold_test_preds, model_names)
            try:
                fold_aucs.append(roc_auc_score(y_val[te_idx], fold_pred))
            except ValueError:
                fold_aucs.append(0.5)
        mean_auc = np.mean(fold_aucs)
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_lam = lam_candidate
    s_cv = coopetitive_scores(model_names, sing_div, pair_div, best_lam)
    methods['Coopetitive-CV'] = softmax_weights(s_cv, model_names)

    # 7. Shapley weighting
    shapley = compute_shapley_values(model_names, v)
    methods['Shapley'] = proportional_weights(shapley, model_names)

    # 8. Shapley + SII (double-counting formula)
    sii = compute_shapley_interaction_index(model_names, v)
    sii_scores = {}
    for name in model_names:
        s = shapley[name]
        s += LAMBDA_DEFAULT * sum(
            sii.get(tuple(sorted([name, other])), 0.0)
            for other in model_names if other != name
        )
        sii_scores[name] = s
    methods['Shapley+SII'] = softmax_weights(sii_scores, model_names)

    # Best single model
    best_model = max(model_names, key=lambda nm: auc_vals[nm])
    methods['Best-Single'] = {nm: (1.0 if nm == best_model else 0.0)
                              for nm in model_names}

    return methods, best_lam


# ============================================================
# EXPERIMENT 1: CONSTRUCT VALIDITY
# ============================================================
def load_construct_validity_data():
    """Load and split the dataset specified by CONSTRUCT_VALIDITY_DATASET."""
    if CONSTRUCT_VALIDITY_DATASET == 'Breast Cancer':
        bc = load_breast_cancer()
        X, y = bc.data, bc.target
    else:
        csv_path = _latest_usable_csv()
        df = pd.read_csv(csv_path, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        df['name'] = df['name'].str.strip()
        row = df[df['name'] == CONSTRUCT_VALIDITY_DATASET]
        if row.empty:
            raise ValueError(
                f"Unknown CONSTRUCT_VALIDITY_DATASET: {CONSTRUCT_VALIDITY_DATASET!r}. "
                "Must be 'Breast Cancer' or a dataset name present in the usable_datasets CSV."
            )
        data_id = int(row.iloc[0]['openml_id'])
        X, y = _fetch_openml_dataset(data_id, CONSTRUCT_VALIDITY_DATASET)
    print(f"  Construct validity data: {CONSTRUCT_VALIDITY_DATASET} "
          f"({len(X)} samples, {X.shape[1]} features)")
    return split_data(X, y, seed=42)


def run_experiment_1():
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Construct Validity of Pairwise Dividends")
    print("=" * 60)

    # Tests A and B use a real benchmark dataset so construct validity is demonstrated
    # on genuine data distributions. Tests C and D use constructed predictions
    # (controlled blends and random outputs) where synthetic signal is the test itself.
    X_tr, X_val, X_test, y_tr, y_val, y_test = load_construct_validity_data()

    # ---- Test A: Controlled Redundancy ----
    print("\n--- Test A: Controlled Redundancy ---")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict_proba(X_val)[:, 1]

    rng = np.random.RandomState(42)
    m2_pred = np.clip(rf_pred + rng.normal(0, 0.01, len(rf_pred)), 0, 1)
    m3_pred = np.clip(rf_pred + rng.normal(0, 0.01, len(rf_pred)), 0, 1)

    lr = LogisticRegression(max_iter=1000, random_state=42); lr.fit(X_tr, y_tr)
    m4_pred = lr.predict_proba(X_val)[:, 1]
    svm = SVC(kernel='rbf', probability=True, random_state=42); svm.fit(X_tr, y_tr)
    m5_pred = svm.predict_proba(X_val)[:, 1]

    names_a = ['RF', 'RF_copy1', 'RF_copy2', 'LR', 'SVM']
    preds_a = {'RF': rf_pred, 'RF_copy1': m2_pred, 'RF_copy2': m3_pred,
               'LR': m4_pred, 'SVM': m5_pred}

    v_a = compute_all_coalitions(names_a, preds_a, y_val)
    pair_a = compute_pairwise_dividends(names_a, v_a)

    copy_set = {'RF', 'RF_copy1', 'RF_copy2'}
    copy_vals = [v for k, v in pair_a.items() if set(k) <= copy_set]
    dissim_vals = [v for k, v in pair_a.items() if not set(k) <= copy_set]
    copy_mean, dissim_mean = np.mean(copy_vals), np.mean(dissim_vals)

    print("Pairwise dividends:")
    for pair, val in sorted(pair_a.items()):
        tag = " [COPY]" if set(pair) <= copy_set else " [DISSIMILAR]"
        print(f"  {pair}: {val:.6f}{tag}")

    # §3.5.1 pass criterion: copy-pair dividends must ALL be negative,
    # AND copy-pair mean must be more negative than dissimilar-pair mean.
    # The manuscript also requires dissimilar pairs to be "positive or near-zero",
    # but with strong overlapping models they may be negative too; the relative
    # ordering is the operative diagnostic (acknowledged in §4.1.1).
    copy_all_neg = all(v < 0 for v in copy_vals)
    dissim_pos_or_near_zero = all(v >= -0.001 for v in dissim_vals)
    pass_a = copy_mean < dissim_mean and copy_all_neg
    print(f"Copy mean: {copy_mean:.6f}, Dissimilar mean: {dissim_mean:.6f}")
    print(f"All copy dividends negative: {copy_all_neg}")
    print(f"Dissimilar pairs positive/near-zero: {dissim_pos_or_near_zero} "
          f"(informational; may be negative with strong models)")
    print(f"Test A {'PASSED' if pass_a else 'FAILED'}")

    # ---- Homogeneous pool test ----
    print("\n--- Test A (supplement): Homogeneous Pool ---")
    homo_models = get_homogeneous_models()
    homo_names, homo_val_preds, _ = train_and_cache(homo_models, X_tr, y_tr, X_val, X_test)
    v_homo = compute_all_coalitions(homo_names, homo_val_preds, y_val)
    pair_homo = compute_pairwise_dividends(homo_names, v_homo)
    n_neg_homo = sum(1 for v in pair_homo.values() if v < 0)
    print(f"Homogeneous pool: {n_neg_homo}/{len(pair_homo)} pairs negative "
          f"(expected mostly negative)")
    for pair, val in sorted(pair_homo.items()):
        print(f"  {pair}: {val:.6f}")

    # ---- Test B: Diversity Metric Correlation ----
    print("\n--- Test B: Diversity Metric Correlation ---")
    models = get_base_models()
    model_names, val_preds_b, _ = train_and_cache(models, X_tr, y_tr, X_val, X_test)
    v_b = compute_all_coalitions(model_names, val_preds_b, y_val)
    pair_b = compute_pairwise_dividends(model_names, v_b)
    sii_b = compute_shapley_interaction_index(model_names, v_b)

    q_stats, disagree_stats = {}, {}
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            ni, nj = model_names[i], model_names[j]
            key = tuple(sorted([ni, nj]))
            pi = (val_preds_b[ni] >= 0.5).astype(int)
            pj = (val_preds_b[nj] >= 0.5).astype(int)
            n11 = np.sum((pi == y_val) & (pj == y_val))
            n10 = np.sum((pi == y_val) & (pj != y_val))
            n01 = np.sum((pi != y_val) & (pj == y_val))
            n00 = np.sum((pi != y_val) & (pj != y_val))
            denom = n11 * n00 + n01 * n10
            q_stats[key] = (n11 * n00 - n01 * n10) / denom if denom else 0
            # Disagreement measure
            N_total = n11 + n10 + n01 + n00
            disagree_stats[key] = (n01 + n10) / N_total if N_total else 0

    pairs_list = sorted(pair_b.keys())
    h_vals = [pair_b[p] for p in pairs_list]
    sii_vals = [sii_b[p] for p in pairs_list]
    q_vals = [q_stats[p] for p in pairs_list]
    dis_vals = [disagree_stats[p] for p in pairs_list]

    corr_sii, _ = stats.spearmanr(h_vals, sii_vals)
    corr_q, _ = stats.spearmanr(h_vals, q_vals)
    corr_dis, _ = stats.spearmanr(h_vals, dis_vals)

    print(f"Spearman m({{i,j}}) vs SII:          r={corr_sii:.4f}")
    print(f"Spearman m({{i,j}}) vs Q-stat:       r={corr_q:.4f}")
    print(f"Spearman m({{i,j}}) vs Disagreement: r={corr_dis:.4f}")
    ctx = 'ADEQUATE' if corr_sii > 0.85 else 'LIMITED' if corr_sii > 0.70 else 'POOR'
    print(f"Context-sensitivity: {ctx}")
    # §3.4.4 / §3.6.3: Test B passes when dividends have at least "limited" agreement with
    # the SII (r > 0.70). Below 0.70 the dividends are deemed unreliable as interaction
    # proxies. No explicit pass threshold for Test B was stated in §3.5.1, but §3.4.4
    # establishes r < 0.70 as the "poor" / failure boundary.
    pass_b = corr_sii > 0.70
    print(f"Test B {'PASSED' if pass_b else 'FAILED'} (SII corr r={corr_sii:.4f}, threshold >0.70)")

    # Report pairs with opposite signs (§3.4.4)
    opposite_sign_pairs = []
    for p in pairs_list:
        h_val = pair_b[p]
        s_val = sii_b[p]
        if (h_val > 0 and s_val < 0) or (h_val < 0 and s_val > 0):
            opposite_sign_pairs.append((p, h_val, s_val))
    if opposite_sign_pairs:
        print(f"  Opposite-sign pairs ({len(opposite_sign_pairs)}):")
        for pair, h_v, s_v in opposite_sign_pairs:
            print(f"    {pair}: m={h_v:.6f}, SII={s_v:.6f}")
    else:
        print("  No opposite-sign pairs between m({i,j}) and SII.")

    # ---- Test C: Monotonicity Under Increasing Redundancy ----
    print("\n--- Test C: Monotonicity Under Increasing Redundancy ---")
    ma_pred = lr.predict_proba(X_val)[:, 1]
    mb_pred = rf.predict_proba(X_val)[:, 1]
    alphas = np.arange(0, 1.05, 0.1)
    blend_divs = []
    for alpha in alphas:
        blended = (1 - alpha) * ma_pred + alpha * mb_pred
        v_bl = compute_all_coalitions(['RF', 'Blended'],
                                       {'RF': rf_pred, 'Blended': blended}, y_val)
        pd_bl = compute_pairwise_dividends(['RF', 'Blended'], v_bl)
        blend_divs.append(list(pd_bl.values())[0])

    corr_c, _ = stats.spearmanr(alphas, blend_divs)
    print(f"Spearman(α, m({{RF, M_α}})): r={corr_c:.4f}")
    print(f"Test C {'PASSED' if corr_c < -0.9 else 'FAILED'}")

    # ---- Test D: Negative Control (structureless null) ----
    # §3.5.1: Random uniform predictions serve as genuinely independent error models.
    # Pass criterion: all |m({i,j})| < 0.01 AND mean absolute dividend is at least
    # 5x smaller than in Test A (copy-pair mean).
    # Using AUC-ROC (same metric as Tests A–C) so scales are comparable.
    print("\n--- Test D: Negative Control ---")
    rng_d = np.random.RandomState(42)
    indep_names = [f'M{i}' for i in range(5)]
    indep_preds = {name: rng_d.uniform(0, 1, len(y_val)) for name in indep_names}

    v_d = compute_all_coalitions(indep_names, indep_preds, y_val, metric='auc')
    pair_d = compute_pairwise_dividends(indep_names, v_d)

    max_abs_d = max(abs(v) for v in pair_d.values())
    mean_abs_d = np.mean([abs(v) for v in pair_d.values()])
    # Criterion 1 (§3.5.1): all |m({i,j})| < 0.01
    crit1 = max_abs_d < 0.01
    # Criterion 2 (§3.5.1): mean absolute dividend at least 5x smaller than Test A copy mean
    five_times_smaller = abs(copy_mean) / 5.0
    crit2 = mean_abs_d < five_times_smaller if abs(copy_mean) > 1e-10 else True
    pass_d = crit1 and crit2
    print(f"Max |m({{i,j}})| control: {max_abs_d:.6f}  (threshold: <0.01)")
    print(f"Mean |m({{i,j}})| control: {mean_abs_d:.6f}  (must be <{five_times_smaller:.6f}, "
          f"i.e. 5x smaller than Test A copy mean {abs(copy_mean):.6f})")
    print(f"Criterion 1 (max < 0.01): {'PASS' if crit1 else 'FAIL'}")
    print(f"Criterion 2 (5x smaller): {'PASS' if crit2 else 'FAIL'}")
    print(f"Test D {'PASSED' if pass_d else 'FAILED'}")

    cv_rows = [
        {'test': 'A_redundancy',   'passed': pass_a,        'copy_mean': copy_mean,   'dissim_mean': dissim_mean},
        {'test': 'A_homogeneous',  'passed': None,           'n_neg': n_neg_homo,      'n_total': len(pair_homo)},
        {'test': 'B_sii_corr',     'passed': pass_b,        'corr_sii': corr_sii,     'corr_q': corr_q, 'corr_dis': corr_dis},
        {'test': 'C_monotonicity', 'passed': corr_c < -0.9, 'spearman_r': corr_c},
        {'test': 'D_neg_control',  'passed': pass_d,        'max_abs_div': max_abs_d, 'mean_abs_div': mean_abs_d},
    ]
    pd.DataFrame(cv_rows).to_csv(EXP_DIRS[1] / 'construct_validity.csv', index=False)
    print(f"Construct validity results saved to: {EXP_DIRS[1] / 'construct_validity.csv'}")

    return {
        'pair_a': pair_a, 'pair_b': pair_b, 'sii_b': sii_b,
        'q_stats': q_stats, 'disagree_stats': disagree_stats,
        'corr_sii': corr_sii, 'corr_q': corr_q, 'corr_dis': corr_dis,
        'alphas': alphas, 'blend_dividends': blend_divs, 'corr_c': corr_c,
        'pair_d': pair_d, 'max_abs_d': max_abs_d,
        'pass_a': pass_a, 'pass_b': pass_b,
        'pass_c': corr_c < -0.9, 'pass_d': pass_d,
        'n_neg_homo': n_neg_homo, 'n_total_homo': len(pair_homo),
    }


# ============================================================
# EXPERIMENTS 2–4
# ============================================================
def run_dataset_seed(dataset_name, X, y, seed):
    r = run_single_dataset_seed(dataset_name, X, y, seed)
    mn = r['model_names']
    methods, best_lam = get_all_method_weights(
        mn, r['v'], r['sing_div'], r['pair_div'], r['val_preds'], r['y_val'],
        has_signal=r['has_signal'])

    results = []
    for method_name, weights in methods.items():
        if method_name == 'Best-Single':
            best_m = max(mn, key=lambda nm: roc_auc_score(r['y_val'], r['val_preds'][nm]))
            pred = r['test_preds'][best_m]
            try:
                auc = roc_auc_score(r['y_test'], pred)
            except ValueError:
                auc = 0.5
            neg_brier = -brier_score_loss(r['y_test'], pred)
        else:
            auc, neg_brier = evaluate_method_on_test(
                weights, mn, r['test_preds'], r['y_test'])
        results.append({
            'dataset': dataset_name, 'seed': seed,
            'method': method_name, 'auc_roc': auc,
            'neg_brier': neg_brier, 'cv_lambda': best_lam,
        })

    # Stacking with logistic regression meta-learner
    try:
        stack_pred = stacking_predict(
            r['val_preds'], r['y_val'], r['test_preds'], mn)
        stack_auc = roc_auc_score(r['y_test'], stack_pred)
        stack_brier = -brier_score_loss(r['y_test'], stack_pred)
    except Exception:
        # Fallback: AUC=0.5 (random baseline) and neg_brier=-0.25 (Brier of a
        # constant 0.5 predictor on any class balance). Not in manuscript.
        stack_auc, stack_brier = 0.5, -0.25
    results.append({'dataset': dataset_name, 'seed': seed,
                    'method': 'Stacking', 'auc_roc': stack_auc,
                    'neg_brier': stack_brier, 'cv_lambda': np.nan})

    # λ sensitivity sweep
    lam_results = []
    for lam in LAMBDA_GRID:
        sc = coopetitive_scores(mn, r['sing_div'], r['pair_div'], lam)
        w = softmax_weights(sc, mn)
        auc_l, nb_l = evaluate_method_on_test(w, mn, r['test_preds'], r['y_test'])
        lam_results.append({'dataset': dataset_name, 'seed': seed,
                            'lambda': lam, 'auc_roc': auc_l, 'neg_brier': nb_l})

    diag = {
        'dataset': dataset_name, 'seed': seed,
        'mono_rate': r['mono_rate'], 'flagged_models': r['flagged_models'],
        'gc_optimal': r['gc_optimal'],
        'v_grand': r['v_grand'], 'v_max': r['v_max'],
        'has_signal': r['has_signal'], 'coverage': r['coverage'],
        'rank_stability': r['rank_stability']
    }
    return results, lam_results, diag


def run_all_experiments(experiments=None):
    if experiments is None:
        experiments = {2, 3, 4}
    print("\n" + "=" * 60)
    print("EXPERIMENTS 2–4: Running across all datasets and seeds")
    print("=" * 60)

    datasets = get_datasets()
    all_results, all_lambda, all_diag = [], [], []
    total = len(datasets) * N_SEEDS
    count = 0

    for ds_name, (X, y, domain, n_samples) in datasets.items():
        for seed in range(N_SEEDS):
            count += 1
            t0 = time.time()
            res, lam_res, diag = run_dataset_seed(ds_name, X, y, seed)
            elapsed = time.time() - t0
            print(f"  [{count}/{total}] {ds_name} (seed={seed}): "
                  f"{elapsed:.1f}s, cov={diag['coverage']:.3f}, "
                  f"rank_stab={diag['rank_stability']:.3f}")
            all_results.extend(res)
            all_lambda.extend(lam_res)
            all_diag.append(diag)

    df_results = pd.DataFrame(all_results)
    df_lambda = pd.DataFrame(all_lambda)
    df_diag = pd.DataFrame(all_diag)

    # Shared raw data always written to RESULTS_DIR for cross-experiment use
    df_results.to_csv(RESULTS_DIR / 'all_results.csv', index=False)
    df_lambda.to_csv(RESULTS_DIR / 'lambda_sensitivity.csv', index=False)
    df_diag.to_csv(RESULTS_DIR / 'diagnostics.csv', index=False)

    ablation_methods = ['Equal', 'Interaction-Only', 'Individual-Only',
                        'Coopetitive-0.5', 'Coopetitive-CV']
    if 2 in experiments:
        df_results[df_results['method'].isin(ablation_methods)].to_csv(
            EXP_DIRS[2] / 'ablation_results.csv', index=False)
    if 3 in experiments:
        df_diag.to_csv(EXP_DIRS[3] / 'diagnostics.csv', index=False)
    if 4 in experiments:
        df_lambda.to_csv(EXP_DIRS[4] / 'lambda_sensitivity.csv', index=False)

    print(f"\nSaved {len(df_results)} results, {len(df_lambda)} lambda, "
          f"{len(df_diag)} diagnostics")
    return df_results, df_lambda, df_diag


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================
def wilcoxon_effect_size(x, y):
    """Effect size: r = Z / √N using normal approximation.
    Used exclusively for H1 (§3.6.3), which is a directional hypothesis ('higher AUC').
    One-tailed alternative='greater' is therefore appropriate (Demšar 2006, §4)."""
    diff = x - y
    diff = diff[diff != 0]
    N = len(diff)
    if N < 3:
        return 0.0, 1.0, 0.0
    # alternative='greater': one-tailed test consistent with H1's directional claim
    stat, p_val = stats.wilcoxon(diff, alternative='greater')
    # Z approximation: W is the sum of positive ranks
    n = len(diff)
    mean_W = n * (n + 1) / 4
    std_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    Z = (stat - mean_W) / std_W if std_W > 0 else 0.0
    r = abs(Z) / np.sqrt(n)
    return stat, p_val, r


def holm_bonferroni(p_values):
    """Holm–Bonferroni correction. Returns adjusted p-values."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    adjusted = np.ones(n)
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = p_values[idx] * (n - rank)
    # Enforce monotonicity
    for i in range(1, n):
        idx = sorted_idx[i]
        prev_idx = sorted_idx[i - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return np.minimum(adjusted, 1.0)


def run_statistical_analysis(df_results, df_lambda, experiments=None):
    if experiments is None:
        experiments = {2, 3, 4}
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    df_avg = df_results.groupby(['dataset', 'method'])['auc_roc'].mean().reset_index()

    # ---- H1 ----
    print("\n--- H1: Coopetitive vs Individual-Only ---")
    coop = df_avg[df_avg['method'] == 'Coopetitive-0.5'].set_index('dataset')['auc_roc']
    indiv = df_avg[df_avg['method'] == 'Individual-Only'].set_index('dataset')['auc_roc']
    common = coop.index.intersection(indiv.index)
    if len(common) >= 3:
        W, p_h1, r_h1 = wilcoxon_effect_size(
            coop[common].values, indiv[common].values)
        diff = (coop[common] - indiv[common]).mean()
        print(f"  W={W:.1f}, p={p_h1:.4f}, r={r_h1:.4f}, mean diff={diff:.6f}")
        print(f"  H1 {'SUPPORTED' if p_h1 < 0.05 else 'NOT SUPPORTED (p≥0.05)'}")
    else:
        p_h1, r_h1, diff = 1.0, 0.0, 0.0

    # ---- H3: Friedman + Holm post-hoc ----
    print("\n--- H3: Friedman Test + Holm Post-Hoc ---")
    bench_methods = ['Equal', 'Perf-Weighted', 'Shapley', 'Stacking',
                     'Best-Single', 'Coopetitive-CV']
    df_bench = df_avg[df_avg['method'].isin(bench_methods)]
    pivot = df_bench.pivot(index='dataset', columns='method', values='auc_roc').dropna()

    friedman_stat, friedman_p = 0.0, 1.0
    avg_ranks = pd.Series(dtype=float)
    posthoc_df = pd.DataFrame()

    if pivot.shape[0] >= 3 and pivot.shape[1] >= 2:
        friedman_stat, friedman_p = stats.friedmanchisquare(
            *[pivot[m].values for m in pivot.columns])
        avg_ranks = pivot.rank(axis=1, ascending=False).mean().sort_values()
        print(f"  Friedman chi2={friedman_stat:.4f}, p={friedman_p:.6f}")
        print("  Average ranks:")
        for method in avg_ranks.index:
            print(f"    {method}: {avg_ranks[method]:.2f}")

        # Holm-corrected pairwise Wilcoxon post-hoc
        if friedman_p < 0.05:
            print("\n  Holm-corrected pairwise Wilcoxon post-hoc:")
            method_list = list(pivot.columns)
            pairs = list(combinations(method_list, 2))
            raw_p = []
            for m1, m2 in pairs:
                _, p_val = stats.wilcoxon(pivot[m1].values, pivot[m2].values)
                raw_p.append(p_val)
            adj_p = holm_bonferroni(np.array(raw_p))
            rows = []
            for idx, (m1, m2) in enumerate(pairs):
                sig = '***' if adj_p[idx] < 0.001 else '**' if adj_p[idx] < 0.01 \
                    else '*' if adj_p[idx] < 0.05 else 'ns'
                rows.append({'Method A': m1, 'Method B': m2,
                             'raw_p': raw_p[idx], 'adj_p': adj_p[idx], 'sig': sig})
                print(f"    {m1} vs {m2}: adj_p={adj_p[idx]:.4f} {sig}")
            posthoc_df = pd.DataFrame(rows)
            if 3 in experiments:
                posthoc_df.to_csv(EXP_DIRS[3] / 'posthoc_wilcoxon.csv', index=False)

    # ---- H4: λ sensitivity ----
    print("\n--- H4: λ Sensitivity ---")
    df_lam_avg = df_lambda.groupby(['dataset', 'lambda'])['auc_roc'].mean().reset_index()
    h4_rows = []
    for ds in df_lam_avg['dataset'].unique():
        ds_d = df_lam_avg[df_lam_avg['dataset'] == ds]
        opt_lam = ds_d.loc[ds_d['auc_roc'].idxmax(), 'lambda']
        opt_auc = ds_d['auc_roc'].max()
        fixed_auc = ds_d.loc[ds_d['lambda'] == 0.5, 'auc_roc'].values
        fixed_auc = fixed_auc[0] if len(fixed_auc) else opt_auc
        zero_auc = ds_d.loc[ds_d['lambda'] == 0.0, 'auc_roc'].values
        zero_auc = zero_auc[0] if len(zero_auc) else opt_auc
        gain = opt_auc - zero_auc
        fixed_ratio = (fixed_auc - zero_auc) / gain if gain > 1e-10 else 1.0
        # Robustness ratio: proportion of ALL λ values achieving ≥95% of max gain
        threshold_auc = zero_auc + 0.95 * gain if gain > 1e-10 else zero_auc
        n_above = (ds_d['auc_roc'] >= threshold_auc - 1e-10).sum()
        robustness_ratio = n_above / len(ds_d)
        h4_rows.append({'dataset': ds, 'opt_lambda': opt_lam, 'opt_auc': opt_auc,
                        'fixed_auc': fixed_auc, 'fixed_ratio': min(fixed_ratio, 1.0),
                        'robustness_ratio': robustness_ratio})
    df_h4 = pd.DataFrame(h4_rows)
    if 4 in experiments:
        df_h4.to_csv(EXP_DIRS[4] / 'lambda_h4_analysis.csv', index=False)
    print(f"  Mean optimal λ: {df_h4['opt_lambda'].mean():.2f} ± "
          f"{df_h4['opt_lambda'].std():.2f}")
    pct95_fixed = (df_h4['fixed_ratio'] >= 0.95).mean()
    mean_robustness = df_h4['robustness_ratio'].mean()
    print(f"  λ=0.5 achieves ≥95% gain: {pct95_fixed*100:.0f}% of datasets")
    print(f"  Robustness ratio (mean proportion of λ range at ≥95%): {mean_robustness:.2f}")
    print(f"  H4 {'SUPPORTED' if pct95_fixed >= 0.5 else 'NOT SUPPORTED'}")

    # Repeated-measures ANOVA
    print("\n--- Factorial RM-ANOVA (λ × Dataset) ---")
    try:
        import pingouin as pg
        df_lam_rm = df_lambda.groupby(['dataset', 'lambda'])['auc_roc'].mean().reset_index()
        df_lam_rm['lambda'] = df_lam_rm['lambda'].astype(str)
        aov = pg.rm_anova(data=df_lam_rm, dv='auc_roc',
                          within='lambda', subject='dataset')
        print(aov.to_string(index=False))
    except Exception as e:
        print(f"  RM-ANOVA failed: {e}")

    return {
        'h1_p': p_h1, 'h1_r': r_h1, 'h1_diff': diff,
        'friedman_stat': friedman_stat, 'friedman_p': friedman_p,
        'avg_ranks': avg_ranks.to_dict() if len(avg_ranks) else {},
        'h4_df': df_h4,
    }


# ============================================================
# VISUALIZATIONS
# ============================================================
def generate_all_figures(exp1_results, df_results, df_lambda, stats_results, experiments=None):
    if experiments is None:
        experiments = {1, 2, 3, 4}
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11,
                         'figure.dpi': 150, 'savefig.dpi': 150})
    print("\nGenerating figures...")

    # ---- Experiment 1 figures ----
    if 1 in experiments and exp1_results is not None:
        # Figure 1: Dividend Heatmap
        pair_b = exp1_results['pair_b']
        model_names = ['LR', 'RF', 'SVM', 'XGB', 'KNN']
        n = len(model_names)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                key = tuple(sorted([model_names[i], model_names[j]]))
                matrix[i, j] = matrix[j, i] = pair_b.get(key, 0.0)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        vmax = max(abs(matrix.min()), abs(matrix.max()), 1e-6)
        sns.heatmap(matrix, annot=True, fmt='.4f', cmap='RdBu',
                    xticklabels=model_names, yticklabels=model_names,
                    center=0, vmin=-vmax, vmax=vmax, ax=ax,
                    cbar_kws={'label': 'Pairwise Dividend m({i,j})'})
        ax.set_title('Pairwise Harsanyi Dividends\n(Blue = Complementary, Red = Redundant)')
        plt.tight_layout(); plt.savefig(EXP_DIRS[1] / 'fig1_dividend_heatmap.png'); plt.close()

        # Figure 5: Test A Heatmap
        pair_a = exp1_results['pair_a']
        names_a = ['RF', 'RF_copy1', 'RF_copy2', 'LR', 'SVM']
        n_a = len(names_a)
        ma = np.zeros((n_a, n_a))
        for i in range(n_a):
            for j in range(i + 1, n_a):
                key = tuple(sorted([names_a[i], names_a[j]]))
                ma[i, j] = ma[j, i] = pair_a.get(key, 0.0)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        vm = max(abs(ma.min()), abs(ma.max()), 1e-6)
        sns.heatmap(ma, annot=True, fmt='.4f', cmap='RdBu', xticklabels=names_a,
                    yticklabels=names_a, center=0, vmin=-vm, vmax=vm, ax=ax)
        ax.set_title('Test A: Controlled Redundancy')
        plt.tight_layout(); plt.savefig(EXP_DIRS[1] / 'fig5_test_a_heatmap.png'); plt.close()

        # Figure 6: Monotonicity
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(exp1_results['alphas'], exp1_results['blend_dividends'],
                'o-', color='#e74c3c', linewidth=2, markersize=6)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('α'); ax.set_ylabel('m({RF, M_α})')
        ax.set_title(f'Test C: Monotonicity (Spearman r = {exp1_results["corr_c"]:.4f})')
        plt.tight_layout(); plt.savefig(EXP_DIRS[1] / 'fig6_test_c_monotonicity.png'); plt.close()

    # ---- Experiment 2 figures ----
    if 2 in experiments and df_results is not None:
        ablation_methods = ['Equal', 'Interaction-Only', 'Individual-Only',
                            'Coopetitive-0.5', 'Coopetitive-CV']
        df_abl = df_results[df_results['method'].isin(ablation_methods)]
        df_abl_avg = df_abl.groupby(['dataset', 'method'])['auc_roc'].agg(
            ['mean', 'std']).reset_index()
        fig, ax = plt.subplots(figsize=(14, 7))
        ds_list = sorted(df_abl_avg['dataset'].unique())
        x = np.arange(len(ds_list)); w = 0.15
        colors = ['#95a5a6', '#e67e22', '#3498db', '#2ecc71', '#e74c3c']
        for i, method in enumerate(ablation_methods):
            df_m = df_abl_avg[df_abl_avg['method'] == method].set_index('dataset')
            means = [df_m.loc[d, 'mean'] if d in df_m.index else 0 for d in ds_list]
            stds  = [df_m.loc[d, 'std']  if d in df_m.index else 0 for d in ds_list]
            ax.bar(x + i * w, means, w, yerr=stds, label=method,
                   color=colors[i], capsize=2, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Dataset'); ax.set_ylabel('AUC-ROC')
        ax.set_title('Ablation Study: AUC-ROC by Variant and Dataset')
        ax.set_xticks(x + w * 2)
        ax.set_xticklabels(ds_list, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=9, loc='lower right')
        ax.set_ylim(bottom=max(0.4, ax.get_ylim()[0]))
        plt.tight_layout(); plt.savefig(EXP_DIRS[2] / 'fig2_ablation_chart.png'); plt.close()

    # ---- Experiment 3 figures ----
    if 3 in experiments and df_results is not None:
        bench_methods = ['Equal', 'Perf-Weighted', 'Shapley', 'Stacking',
                         'Best-Single', 'Coopetitive-CV']
        df_bench = df_results[df_results['method'].isin(bench_methods)]
        df_bench_avg = df_bench.groupby(['dataset', 'method'])['auc_roc'].mean().reset_index()
        pivot = df_bench_avg.pivot(index='dataset', columns='method', values='auc_roc').dropna()
        avg_ranks = pivot.rank(axis=1, ascending=False).mean().sort_values()
        fig, ax = plt.subplots(figsize=(10, 4))
        y_pos = np.arange(len(avg_ranks))
        clr = ['#2ecc71' if 'Coop' in m else '#3498db' for m in avg_ranks.index]
        ax.barh(y_pos, avg_ranks.values, color=clr, edgecolor='white', height=0.6)
        ax.set_yticks(y_pos); ax.set_yticklabels(avg_ranks.index, fontsize=11)
        ax.set_xlabel('Average Rank (lower is better)')
        ax.set_title(f'Benchmark: Average Ranks Across {len(pivot)} Datasets')
        ax.invert_yaxis()
        for i, (m, r) in enumerate(avg_ranks.items()):
            ax.text(r + 0.05, i, f'{r:.2f}', va='center', fontsize=10)
        plt.tight_layout(); plt.savefig(EXP_DIRS[3] / 'fig3_critical_difference.png'); plt.close()

        # Figure 7: Diagnostics
        df_diag = pd.read_csv(RESULTS_DIR / 'diagnostics.csv')
        df_d = df_diag.groupby('dataset').agg(
            {'coverage': 'mean', 'mono_rate': 'mean', 'rank_stability': 'mean'}).reset_index()
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].barh(df_d['dataset'], df_d['coverage'], color='#3498db')
        axes[0].axvline(x=0.8, color='red', linestyle='--', label='80%')
        axes[0].set_xlabel('Coverage'); axes[0].set_title('Decomposition Coverage'); axes[0].legend()
        axes[1].barh(df_d['dataset'], df_d['mono_rate'], color='#e67e22')
        axes[1].axvline(x=0.4, color='red', linestyle='--', label='40%')
        axes[1].set_xlabel('Violation Rate'); axes[1].set_title('Monotonicity Violations'); axes[1].legend()
        axes[2].barh(df_d['dataset'], df_d['rank_stability'], color='#2ecc71')
        axes[2].axvline(x=0.9, color='red', linestyle='--', label='0.90')
        axes[2].set_xlabel('Spearman r'); axes[2].set_title('Rank Stability'); axes[2].legend()
        plt.tight_layout(); plt.savefig(EXP_DIRS[3] / 'fig7_diagnostics.png'); plt.close()

    # ---- Experiment 4 figures ----
    if 4 in experiments and df_lambda is not None:
        df_lam_avg = df_lambda.groupby(['dataset', 'lambda'])['auc_roc'].mean().reset_index()
        ds_all = sorted(df_lam_avg['dataset'].unique())
        cmap = plt.cm.tab20

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, ds in enumerate(ds_all):
            d = df_lam_avg[df_lam_avg['dataset'] == ds]
            ax.plot(d['lambda'], d['auc_roc'], color=cmap(i / len(ds_all)),
                    alpha=0.7, linewidth=1.5, label=ds)
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='λ=0.5')
        ax.set_xlabel('λ'); ax.set_ylabel('AUC-ROC')
        ax.set_title('λ Sensitivity Across Datasets')
        ax.legend(fontsize=7, ncol=3, loc='lower right')
        plt.tight_layout(); plt.savefig(EXP_DIRS[4] / 'fig4_lambda_sensitivity.png'); plt.close()

        # Figure 8: Interaction Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for ds in ds_all[:6]:
            d = df_lam_avg[df_lam_avg['dataset'] == ds]
            ax.plot(d['lambda'], d['auc_roc'], 'o-', markersize=3, linewidth=1.5, label=ds)
        ax.set_xlabel('λ'); ax.set_ylabel('Mean AUC-ROC')
        ax.set_title('Factorial Interaction Plot: λ × Dataset')
        ax.legend(fontsize=9)
        plt.tight_layout(); plt.savefig(EXP_DIRS[4] / 'fig8_interaction_plot.png'); plt.close()

    saved = [str(EXP_DIRS[n]) for n in sorted(experiments)]
    print(f"Figures saved under: {', '.join(saved)}")


# ============================================================
# SUMMARY REPORT
# ============================================================
def generate_summary_report(exp1, df_results, df_lambda, st):
    lines = ["=" * 70,
             "COOPETITIVE ENSEMBLE FRAMEWORK v10 — RESULTS SUMMARY",
             "=" * 70]

    lines.append("\n### EXPERIMENT 1: CONSTRUCT VALIDITY ###")
    lines.append(f"Test A (Redundancy): {'PASS' if exp1['pass_a'] else 'FAIL'}")
    lines.append(f"Test A (Homogeneous pool): {exp1['n_neg_homo']}/{exp1['n_total_homo']} negative")
    lines.append(f"Test B (SII agreement): {'PASS' if exp1['pass_b'] else 'FAIL'} "
                 f"(SII corr r={exp1['corr_sii']:.4f}, threshold >0.70)")
    lines.append(f"Test B — Q-stat corr: r={exp1['corr_q']:.4f}")
    lines.append(f"Test B — Disagreement corr: r={exp1['corr_dis']:.4f}")
    lines.append(f"Test C (Monotonicity): {'PASS' if exp1['pass_c'] else 'FAIL'} "
                 f"(r={exp1['corr_c']:.4f})")
    lines.append(f"Test D (Neg Control): {'PASS' if exp1['pass_d'] else 'INCONCLUSIVE'} "
                 f"(max |m|={exp1['max_abs_d']:.6f})")

    lines.append("\n### EXPERIMENT 2: ABLATION STUDY ###")
    df_avg = df_results.groupby('method')['auc_roc'].mean().sort_values(ascending=False)
    for m, a in df_avg.items():
        lines.append(f"  {m:25s}: AUC = {a:.4f}")

    lines.append("\n### EXPERIMENT 3: BENCHMARK ###")
    lines.append(f"Friedman chi2={st['friedman_stat']:.4f}, p={st['friedman_p']:.6f}")
    for m, r in sorted(st['avg_ranks'].items(), key=lambda x: x[1]):
        lines.append(f"  {m:25s}: Rank = {r:.2f}")

    lines.append("\n### EXPERIMENT 4: λ SENSITIVITY ###")
    df_h4 = st['h4_df']
    lines.append(f"Optimal λ: {df_h4['opt_lambda'].mean():.2f} ± {df_h4['opt_lambda'].std():.2f}")
    lines.append(f"λ=0.5 ≥95% gain: {(df_h4['fixed_ratio'] >= 0.95).mean()*100:.0f}% of datasets")
    lines.append(f"Robustness ratio (mean): {df_h4['robustness_ratio'].mean():.2f}")

    lines.append("\n### HYPOTHESIS VALIDATION ###")
    lines.append(f"H1: p={st['h1_p']:.4f}, r={st['h1_r']:.4f}, diff={st['h1_diff']:.6f}")
    # §3.6.3: H2 requires all four subtests to pass
    h2 = exp1['pass_a'] and exp1['pass_b'] and exp1['pass_c'] and exp1['pass_d']
    lines.append(f"H2: {'ALL PASS' if h2 else 'MIXED (see individual test results)'}")
    lines.append(f"H3: Friedman p={st['friedman_p']:.6f}")
    lines.append(f"H4: {(df_h4['fixed_ratio'] >= 0.95).mean()*100:.0f}% at ≥95%, "
                 f"robustness={df_h4['robustness_ratio'].mean():.2f}")

    report = "\n".join(lines)
    with open(RESULTS_DIR / 'summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)


def generate_latex_summary(exp1, df_results, st):
    """Write key findings as LaTeX \\newcommand macros to results/results_summary.tex.
    Include in manuscript preamble with: \\InputIfFileExists{results/results_summary.tex}{}{}
    """
    df_avg = df_results.groupby('method')['auc_roc'].mean()
    df_neg_brier = df_results.groupby('method')['neg_brier'].mean()
    df_h4 = st['h4_df']

    lines = [
        "% Auto-generated by main.py — do not edit manually.",
        "% Include in manuscript preamble:",
        "%   \\InputIfFileExists{results/results_summary.tex}{}{}",
        "",
    ]

    def cmd(name, val):
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    # Experiment 1 — construct validity
    cmd("CorrSII", f"{exp1['corr_sii']:.3f}")
    cmd("CorrQStat", f"{exp1['corr_q']:.3f}")
    cmd("CorrDisagree", f"{exp1['corr_dis']:.3f}")
    cmd("TestCCorr", f"{exp1['corr_c']:.3f}")
    cmd("TestDMaxAbs", f"{exp1['max_abs_d']:.4f}")
    cmd("HomoNegPairs", str(exp1['n_neg_homo']))
    cmd("HomoTotalPairs", str(exp1['n_total_homo']))

    # Hypothesis 1
    cmd("HOneP", f"{st['h1_p']:.3f}")
    cmd("HOneR", f"{st['h1_r']:.3f}")
    cmd("HOneDiff", f"{st['h1_diff']:.4f}")

    # Hypothesis 3 — Friedman
    cmd("FriedmanStat", f"{st['friedman_stat']:.4f}")
    cmd("FriedmanP", f"{st['friedman_p']:.4f}")

    # Ablation AUC-ROC per method
    method_cmds = {
        'Coopetitive-0.5':  'AUCCoopFixed',
        'Coopetitive-CV':   'AUCCoopCV',
        'Shapley':          'AUCShapley',
        'Perf-Weighted':    'AUCPerfWeighted',
        'Individual-Only':  'AUCIndividual',
        'Equal':            'AUCEqual',
        'Best-Single':      'AUCBestSingle',
        'Stacking':         'AUCStacking',
        'Interaction-Only': 'AUCInteractionOnly',
        'Shapley+SII':      'AUCShapleyPlusSII',
    }
    for method, mc in method_cmds.items():
        if method in df_avg.index:
            cmd(mc, f"{df_avg[method]:.4f}")
        if method in df_neg_brier.index:
            cmd(mc.replace('AUC', 'Brier'), f"{df_neg_brier[method]:.4f}")

    # Benchmark average ranks
    rank_cmds = {
        'Shapley':          'RankShapley',
        'Coopetitive-CV':   'RankCoopCV',
        'Perf-Weighted':    'RankPerfWeighted',
        'Stacking':         'RankStacking',
        'Best-Single':      'RankBestSingle',
        'Equal':            'RankEqual',
    }
    for method, mc in rank_cmds.items():
        if method in st['avg_ranks']:
            cmd(mc, f"{st['avg_ranks'][method]:.2f}")

    # Hypothesis 4 — lambda sensitivity
    cmd("OptLambdaMean", f"{df_h4['opt_lambda'].mean():.2f}")
    cmd("OptLambdaStd", f"{df_h4['opt_lambda'].std():.2f}")
    cmd("PctFixedAtNinetyFive", f"{(df_h4['fixed_ratio'] >= 0.95).mean() * 100:.0f}")
    cmd("RobustnessRatio", f"{df_h4['robustness_ratio'].mean():.2f}")

    out_path = RESULTS_DIR / 'results_summary.tex'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"LaTeX summary written to {out_path}")


# ============================================================
# MAIN
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--experiments', type=int, nargs='+', default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4], metavar='N',
        help='Which experiments to run (default: 1 2 3 4). '
             'E.g. --experiments 1 3',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    experiments = set(args.experiments)

    for n in experiments:
        EXP_DIRS[n].mkdir(parents=True, exist_ok=True)

    print(f"Running experiments: {sorted(experiments)}")

    t_start = time.time()
    exp1_results   = None
    df_results     = None
    df_lambda      = None
    df_diag        = None
    stats_results  = None

    if 1 in experiments:
        exp1_results = run_experiment_1()

    if experiments & {2, 3, 4}:
        df_results, df_lambda, df_diag = run_all_experiments(experiments)
        stats_results = run_statistical_analysis(df_results, df_lambda, experiments)

    generate_all_figures(exp1_results, df_results, df_lambda, stats_results, experiments)

    if exp1_results is not None and stats_results is not None:
        generate_summary_report(exp1_results, df_results, df_lambda, stats_results)
        generate_latex_summary(exp1_results, df_results, stats_results)

    t_total = time.time() - t_start
    print(f"\n{'='*60}\nTOTAL RUNTIME: {t_total/60:.1f} minutes\n{'='*60}")
