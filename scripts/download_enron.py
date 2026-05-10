"""
Task 1.1 — Download & Explore Enron Email Dataset

Downloads the Enron email dataset from HuggingFace, explores its structure,
filters out noise, and saves a diverse sample of ~2,000 emails.
"""

import os
import json
import random
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm


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


def filter_and_sample(df, text_col, target_n=2000, min_len=200, max_len=10000):
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
        .apply(lambda x: x.sample(n=min(per_bin, len(x)), random_state=42))
    )

    # If we don't have enough, sample more from the largest bins
    if len(sampled) < target_n:
        remaining = target_n - len(sampled)
        pool = df[~df.index.isin(sampled.index)]
        extra = pool.sample(n=min(remaining, len(pool)), random_state=42)
        sampled = pd.concat([sampled, extra])

    sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nFinal sample: {len(sampled)} emails")
    print(f"Length distribution:\n{sampled['length'].describe()}")
    print(f"Bin counts:\n{sampled['length_bin'].value_counts().sort_index()}")

    return sampled, text_col


def save_sample(df, text_col, out_dir='data/raw'):
    """Save sampled emails to parquet and jsonl."""
    os.makedirs(out_dir, exist_ok=True)

    # Parquet with metadata
    parquet_path = os.path.join(out_dir, 'enron_sample.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved parquet: {parquet_path}")

    # JSONL with just the text for easy processing
    jsonl_path = os.path.join(out_dir, 'enron_sample.jsonl')
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
    }
    stats_path = os.path.join(out_dir, 'enron_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Saved stats: {stats_path}")

    return parquet_path, jsonl_path


def main():
    os.chdir('/mnt/d/documents/year-2/lab/project/email-summarizer')
    print("Working directory:", os.getcwd())

    target_n = int(os.environ.get("SAMPLE_SIZE", 25000))
    print(f"Target sample size: {target_n}")

    ds = download_enron()
    df, text_col = explore_dataset(ds)
    sampled, text_col = filter_and_sample(df, text_col, target_n=target_n)
    save_sample(sampled, text_col)

    print("\n" + "=" * 60)
    print("TASK 1.1 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
