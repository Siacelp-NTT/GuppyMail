"""
Task 1.1 — Download & Explore Enron Email Dataset

Downloads the Enron email dataset from HuggingFace, explores its structure,
filters out noise, and saves a diverse sample for Task 1.
"""

import argparse
import os
import json
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent


def download_enron():
    """Download Enron emails from HuggingFace."""
    print("Loading Enron email dataset...")
    try:
        ds = load_dataset("jacquelinehe/enron-emails")
        print(f"Loaded dataset: {ds}")
    except Exception as e:
        print(f"jacquelinehe/enron-emails failed: {e}")
        print("Trying SetFit/enron_spam...")
        ds = load_dataset("SetFit/enron_spam")
        print(f"Loaded dataset: {ds}")
    return ds


def explore_dataset(ds):
    """Explore the dataset structure and quality."""
    df = pd.DataFrame(ds['train'])
    print("\n" + "=" * 60)
    print("DATASET EXPLORATION")
    print("=" * 60)
    print(f"\nColumns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst row keys:")
    for col in df.columns:
        print(f"  {col}: {type(df[col].iloc[0])}")
    print(f"\nMissing values:\n{df.isnull().sum()}")

    # Determine text column
    text_col = None
    for candidate in ['message', 'text', 'body', 'content']:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        text_col = df.columns[0]
        print(f"\nWarning: using first column '{text_col}' as text")
    else:
        print(f"\nUsing column '{text_col}' as email text")

    # Length distribution
    df['length'] = df[text_col].astype(str).str.len()
    print(f"\nLength distribution:\n{df['length'].describe()}")

    return df, text_col


def filter_and_sample(df, text_col, target_n=25000, min_len=200, max_len=10000, seed=42):
    """Filter emails by length, remove duplicates, and stratified sample."""
    print("\n" + "=" * 60)
    print("FILTERING & SAMPLING")
    print("=" * 60)

    # Convert to string and drop missing
    df = df[df[text_col].notna()].copy()
    df[text_col] = df[text_col].astype(str)

    # Remove very short and very long
    initial = len(df)
    df = df[(df['length'] >= min_len) & (df['length'] <= max_len)]
    print(f"Length filter ({min_len}-{max_len}): {initial} → {len(df)}")

    # Remove duplicates
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[text_col], keep='first')
    print(f"Duplicate removal: {before_dedup} → {len(df)}")

    # Remove emails that are just headers/metadata
    df = df[df[text_col].str.contains(r'\b[a-zA-Z]{3,}\b', regex=True)]
    print(f"After removing empty/header-only: {len(df)}")

    # Stratified sampling by length bins
    df['length_bin'] = pd.cut(
        df['length'],
        bins=[0, 500, 1500, 3000, 6000, float('inf')],
        labels=['very_short', 'short', 'medium', 'long', 'very_long']
    )

    n_bins = df['length_bin'].nunique()
    per_bin = target_n // n_bins
    sampled = (
        df.groupby('length_bin', group_keys=False)
        .apply(lambda x: x.sample(n=min(per_bin, len(x)), random_state=seed))
    )

    # If we don't have enough, sample more from the largest bins
    if len(sampled) < target_n:
        remaining = target_n - len(sampled)
        pool = df[~df.index.isin(sampled.index)]
        extra = pool.sample(n=min(remaining, len(pool)), random_state=seed)
        sampled = pd.concat([sampled, extra])

    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"\nFinal sample: {len(sampled)} emails")
    print(f"Length distribution:\n{sampled['length'].describe()}")
    print(f"Bin counts:\n{sampled['length_bin'].value_counts().sort_index()}")

    return sampled, text_col


def save_sample(df, text_col, out_dir='data/raw', prefix='enron_sample'):
    """Save sampled emails to parquet and jsonl."""
    os.makedirs(out_dir, exist_ok=True)

    # Parquet with metadata
    parquet_path = os.path.join(out_dir, f'{prefix}.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved parquet: {parquet_path}")

    # JSONL with just the text for easy processing
    jsonl_path = os.path.join(out_dir, f'{prefix}.jsonl')
    with open(jsonl_path, 'w') as f:
        for _, row in df.iterrows():
            record = {'text': row[text_col], 'message': row[text_col]}
            # Include other useful fields if they exist
            for field in ['subject', 'from', 'date', 'to']:
                if field in row and pd.notna(row[field]):
                    record[field] = str(row[field])
            f.write(json.dumps(record) + '\n')
    print(f"Saved JSONL: {jsonl_path}")

    # Stats
    stats = {
        'total_emails': len(df),
        'avg_length': float(df['length'].mean()),
        'median_length': float(df['length'].median()),
        'min_length': int(df['length'].min()),
        'max_length': int(df['length'].max()),
        'length_bins': df['length_bin'].value_counts().to_dict(),
        'timestamp': datetime.now().isoformat(),
        'source': 'jacquelinehe/enron-emails',
    }
    stats_path = os.path.join(out_dir, f'{prefix}_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Saved stats: {stats_path}")
    if prefix == "enron_sample":
        compat_stats_path = os.path.join(out_dir, "enron_stats.json")
        with open(compat_stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"Saved dashboard stats: {compat_stats_path}")

    return parquet_path, jsonl_path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download and sample Enron emails for guppyemail.")
    parser.add_argument("--sample-size", type=int, default=int(os.environ.get("SAMPLE_SIZE", 25000)))
    parser.add_argument("--min-len", type=int, default=200)
    parser.add_argument("--max-len", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(BASE_DIR / "data" / "raw"))
    parser.add_argument("--prefix", default="enron_sample")
    return parser.parse_args()

def main():
    """Run the command-line entry point."""
    args = parse_args()
    os.chdir(BASE_DIR)
    print("Working directory:", os.getcwd())

    target_n = int(args.sample_size)
    print(f"Target sample size: {target_n}")

    ds = download_enron()
    df, text_col = explore_dataset(ds)
    sampled, text_col = filter_and_sample(
        df,
        text_col,
        target_n=target_n,
        min_len=args.min_len,
        max_len=args.max_len,
        seed=args.seed,
    )
    save_sample(sampled, text_col, out_dir=args.output_dir, prefix=args.prefix)

    print("\n" + "=" * 60)
    print("TASK 1.1 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
