#!/usr/bin/env python3
"""
extract_usable_datasets.py

Reads a dataset validation CSV produced by validate_datasets.py and writes a
new CSV containing only the rows with status == 'PASS'.

Usage
-----
  python extract_usable_datasets.py                          # uses latest CSV in results/dataset_validation/
  python extract_usable_datasets.py path/to/validation.csv  # explicit input
  python extract_usable_datasets.py --out usable.csv        # custom output path
"""
import argparse
from pathlib import Path

import pandas as pd
from datetime import datetime

start_time = datetime.now()
timestamp  = start_time.strftime('%Y%m%d_%H%M%S')

DEFAULT_INPUT_DIR = Path(__file__).parent / 'results' / 'dataset_validation'
DEFAULT_OUTPUT    = Path(__file__).parent / 'results' / 'usable_datasets' / f'{timestamp}.csv'


def latest_csv(directory: Path) -> Path:
    csvs = sorted(directory.glob('*.csv'))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return csvs[-1]


def extract(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, skipinitialspace=True, quotechar='"', engine='python')
    df.columns = df.columns.str.strip()
    df['status'] = df['status'].str.strip()

    usable = df[df['status'] == 'PASS'].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    usable.to_csv(output_path, index=False)

    print(f"Input  : {input_path}  ({len(df)} rows)")
    print(f"Usable : {len(usable)} PASS datasets")
    print(f"Output : {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', nargs='?', type=Path,
                   help='Validation CSV to read (default: latest in results/dataset_validation/)')
    p.add_argument('--out', type=Path, default=DEFAULT_OUTPUT,
                   help=f'Output path (default: {DEFAULT_OUTPUT})')
    args = p.parse_args()

    input_path = args.input if args.input else latest_csv(DEFAULT_INPUT_DIR)
    extract(input_path, args.out)


if __name__ == '__main__':
    main()
