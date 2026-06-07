#!/usr/bin/env python3
"""
validate_datasets.py

Queries OpenML for all active binary classification datasets and validates
them against the study's inclusion criteria. No catalog required — candidates
are discovered live from OpenML metadata.

Pre-filters (applied to OpenML metadata, no data download needed):
  NumberOfClasses    == 2
  NumberOfInstances  in [MIN_SAMPLES, MAX_SAMPLES]
  NumberOfFeatures   <= MAX_FEATURES
  status             == 'active'

Post-filters (requires fetching actual data):
  minority class fraction >= MIN_MINORITY_FRACTION
  DecisionTree AUC        <= MAX_TRIVIAL_AUC (excludes trivially separable datasets)

Requires: pip install openml

Outputs
-------
  Console : pass/fail table with stats and failure reasons
  File    : results/dataset_validation/<YYYYMMDD_HHMMSS>.csv

Usage
-----
  python validate_datasets.py
  python validate_datasets.py --min-samples 300 --max-samples 200000
  python validate_datasets.py --limit 50       # cap candidates for a quick test
"""
from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

# ============================================================
# THRESHOLDS — edit here to change inclusion criteria
# ============================================================
MIN_SAMPLES           = 500
# Stricter than OpenML-CC18's 0.05 — AUC-ROC becomes unreliable at extreme
# imbalance, and this study uses AUC as its primary evaluation metric.
MIN_MINORITY_FRACTION = 0.15

# Pre-filter bounds applied to OpenML metadata before any data is downloaded.
# Narrow these to reduce runtime; widen to consider more candidates.
MAX_SAMPLES           = 100_000
MAX_FEATURES          = 5_000   # aligns with OpenML-CC18 standard

# Datasets where a default DecisionTreeClassifier exceeds this AUC are flagged
# as trivially separable — ensemble weighting methods produce indistinguishable
# results on such datasets, making them uninformative for this study.
MAX_TRIVIAL_AUC       = 0.99

# Seconds to wait for a single dataset fetch before marking it as LOAD ERROR.
FETCH_TIMEOUT         = 30

OUTPUT_DIR = Path(__file__).parent / 'results' / 'dataset_validation'


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class DatasetSpec:
    name: str
    openml_id: int
    n_instances_meta: int
    n_features_meta: int


@dataclass
class ValidationResult:
    spec: DatasetSpec
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    n_classes: Optional[int] = None
    minority_fraction: Optional[float] = None
    dt_auc: Optional[float] = None
    status: str = 'PENDING'
    failures: list[str] = field(default_factory=list)
    error: str = ''

    @property
    def passed(self) -> bool:
        return self.status == 'PASS'


# ============================================================
# OPENML QUERY
# ============================================================
def query_candidates(
    min_samples: int,
    max_samples: int,
    max_features: int,
    limit: Optional[int],
) -> list[DatasetSpec]:
    """Fetch OpenML metadata and return pre-filtered dataset specs."""
    try:
        import openml
    except ImportError:
        raise ImportError(
            "The 'openml' package is required.\n"
            "Install it with: pip install openml"
        )

    print("Fetching OpenML dataset metadata...", end=' ', flush=True)
    try:
        meta = openml.datasets.list_datasets(output_format='dataframe', status='active')
    except Exception as exc:
        raise RuntimeError(f"OpenML metadata query failed: {exc}") from exc
    print(f"{len(meta)} entries retrieved.")

    meta = meta.dropna(subset=['NumberOfInstances', 'NumberOfClasses', 'NumberOfFeatures'])
    meta['NumberOfInstances'] = meta['NumberOfInstances'].astype(int)
    meta['NumberOfClasses']   = meta['NumberOfClasses'].astype(int)
    meta['NumberOfFeatures']  = meta['NumberOfFeatures'].astype(int)

    filtered = meta[
        (meta['NumberOfClasses']   == 2) &
        (meta['NumberOfInstances'] >= min_samples) &
        (meta['NumberOfInstances'] <= max_samples) &
        (meta['NumberOfFeatures']  <= max_features)
    ].copy()

    # Keep the latest version of each dataset name to avoid duplicates.
    if 'version' in filtered.columns:
        filtered = filtered.sort_values('version', ascending=False)
    filtered = (filtered
                .drop_duplicates(subset='name')
                .sort_values('NumberOfInstances')
                .reset_index(drop=True))

    if limit is not None:
        filtered = filtered.head(limit)

    print(f"Pre-filtered to {len(filtered)} binary candidates "
          f"(n in [{min_samples}, {max_samples}], features <= {max_features}).\n")

    return [
        DatasetSpec(
            name=row['name'],
            openml_id=int(row['did']),
            n_instances_meta=int(row['NumberOfInstances']),
            n_features_meta=int(row['NumberOfFeatures']),
        )
        for _, row in filtered.iterrows()
    ]


# ============================================================
# LOADING
# ============================================================
def _fetch(openml_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Download and preprocess a dataset. Runs inside a timed thread."""
    data = fetch_openml(data_id=openml_id, as_frame=True, parser='auto')
    X_df = data.data.copy()
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_df[cat_cols] = enc.fit_transform(X_df[cat_cols].astype(str))
    X = X_df.values.astype(float)
    if np.isnan(X).any():
        X = SimpleImputer(strategy='mean').fit_transform(X)
    y_raw = data.target
    if hasattr(y_raw, 'cat'):
        y_int = y_raw.cat.codes.values.astype(int)
    else:
        y_int = LabelEncoder().fit_transform(y_raw.astype(str).values)
    return X, y_int


def load_dataset(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    """Fetch a dataset with a hard timeout to avoid hanging on slow or broken entries."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_fetch, spec.openml_id)
    try:
        return future.result(timeout=FETCH_TIMEOUT)
    except FuturesTimeoutError:
        raise TimeoutError(f"fetch timed out after {FETCH_TIMEOUT}s")
    finally:
        # wait=False lets the main thread move on immediately; the background
        # thread finishes on its own without blocking the validation loop.
        executor.shutdown(wait=False)


# ============================================================
# VALIDATION
# ============================================================
def validate(spec: DatasetSpec, min_minority: float, max_trivial_auc: float) -> ValidationResult:
    result = ValidationResult(spec=spec)
    try:
        X, y = load_dataset(spec)
    except Exception as exc:
        result.status = 'LOAD ERROR'
        result.error  = str(exc)[:150]
        return result

    unique, counts = np.unique(y, return_counts=True)
    result.n_samples         = len(X)
    result.n_features        = X.shape[1]
    result.n_classes         = len(unique)
    result.minority_fraction = float(counts.min() / counts.sum())

    failures: list[str] = []
    if result.n_classes != 2:
        failures.append(f'not binary ({result.n_classes} classes after load)')
    if result.minority_fraction < min_minority:
        failures.append(f'minority={result.minority_fraction:.3f} < {min_minority}')

    # Difficulty check — only meaningful for binary datasets.
    # A default DecisionTree is a weak baseline; if it already exceeds
    # max_trivial_auc the dataset is too easy to differentiate methods.
    if result.n_classes == 2:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42)
            dt = DecisionTreeClassifier(random_state=42)
            dt.fit(X_tr, y_tr)
            result.dt_auc = float(roc_auc_score(y_te, dt.predict_proba(X_te)[:, 1]))
            if result.dt_auc > max_trivial_auc:
                failures.append(
                    f'trivially separable (DT AUC={result.dt_auc:.3f} > {max_trivial_auc})')
        except Exception as exc:
            # Non-fatal — record in error field so it's visible in the CSV
            # but does not affect pass/fail status.
            result.dt_auc = None
            result.error  = f'dt_check_skipped: {str(exc)[:80]}'

    result.failures = failures
    result.status   = 'PASS' if not failures else 'FAIL'
    return result


# ============================================================
# REPORTING
# ============================================================
def print_report(results: list[ValidationResult], min_minority: float) -> None:
    W = 100
    print(f"\nValidation Report   (minority fraction >= {min_minority}, binary)")
    print("=" * W)
    print(f"{'Dataset':<32} {'ID':>6} {'n':>7} {'feats':>5} {'minority':>8} {'classes':>7}  Status")
    print("-" * W)

    for r in results:
        if r.status == 'LOAD ERROR':
            print(f"{r.spec.name:<32} {r.spec.openml_id:>6} "
                  f"{'?':>7} {'?':>5} {'?':>8} {'?':>7}  LOAD ERROR: {r.error[:38]}")
        else:
            reason = f"  [{'; '.join(r.failures)}]" if r.failures else ''
            print(f"{r.spec.name:<32} {r.spec.openml_id:>6} "
                  f"{r.n_samples:>7} {r.n_features:>5} {r.minority_fraction:>8.3f} "
                  f"{r.n_classes:>7}  {r.status}{reason}")

    passed  = [r for r in results if r.passed]
    failed  = [r for r in results if r.status == 'FAIL']
    errored = [r for r in results if r.status == 'LOAD ERROR']

    print("=" * W)
    print(f"\nSummary: {len(passed)} PASS  |  {len(failed)} FAIL  |  {len(errored)} LOAD ERROR"
          f"  (out of {len(results)} candidates)\n")

    if passed:
        print("VALID DATASETS (PASS):")
        for r in passed:
            dt_str = f"  dt_auc={r.dt_auc:.3f}" if r.dt_auc is not None else ''
            print(f"  {r.spec.name:<32}  id={r.spec.openml_id:<6}  "
                  f"n={r.n_samples:<7}  minority={r.minority_fraction:.3f}"
                  f"  feats={r.n_features}{dt_str}")


def save_csv(results: list[ValidationResult], path: Path) -> None:
    rows = [
        {
            'name':              r.spec.name,
            'openml_id':         r.spec.openml_id,
            'n_instances_meta':  r.spec.n_instances_meta,
            'n_features_meta':   r.spec.n_features_meta,
            'n_samples':         r.n_samples,
            'n_features':        r.n_features,
            'n_classes':         r.n_classes,
            'minority_fraction': r.minority_fraction,
            'dt_auc':            r.dt_auc,
            'status':            r.status,
            'failures':          '; '.join(r.failures),
            'error':             r.error,
        }
        for r in results
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Report saved to: {path}")


# ============================================================
# ENTRY POINT
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--min-samples',  type=int,   default=MIN_SAMPLES,
                   help=f'Minimum n_samples (default: {MIN_SAMPLES})')
    p.add_argument('--max-samples',  type=int,   default=MAX_SAMPLES,
                   help=f'Maximum n_samples (default: {MAX_SAMPLES})')
    p.add_argument('--max-features', type=int,   default=MAX_FEATURES,
                   help=f'Maximum n_features (default: {MAX_FEATURES})')
    p.add_argument('--min-minority',    type=float, default=MIN_MINORITY_FRACTION,
                   help=f'Minimum minority class fraction (default: {MIN_MINORITY_FRACTION})')
    p.add_argument('--max-trivial-auc', type=float, default=MAX_TRIVIAL_AUC,
                   help=f'DT AUC above which a dataset is trivially separable (default: {MAX_TRIVIAL_AUC})')
    p.add_argument('--limit',           type=int,   default=None,
                   help='Cap number of candidates validated (useful for quick tests)')
    return p.parse_args()


def main() -> None:
    start_time = datetime.now()
    timestamp  = start_time.strftime('%Y%m%d_%H%M%S')
    args       = parse_args()

    candidates = query_candidates(
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        max_features=args.max_features,
        limit=args.limit,
    )

    print(f"Validating {len(candidates)} candidates...")
    results: list[ValidationResult] = []
    for i, spec in enumerate(candidates, 1):
        print(f"  [{i:3d}/{len(candidates)}] {spec.name} (id={spec.openml_id})...",
              end=' ', flush=True)
        r = validate(spec, args.min_minority, args.max_trivial_auc)
        print(r.status)
        results.append(r)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f'{timestamp}.csv'

    print_report(results, args.min_minority)
    save_csv(results, output_path)


if __name__ == '__main__':
    main()
