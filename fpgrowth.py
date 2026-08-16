"""
FP-Growth Market Basket Analysis with mlxtend (fully local, VS Code / venv)

Dataset : eCommerce behavior data from multi-category store - 2019-Nov.csv
Source  : https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store

Setup:
    python3 -m venv venv
    source venv/bin/activate        # (Windows: venv\\Scripts\\activate)
    pip install -r requirements.txt
    python local_fpgrowth.py

Run from the terminal (no notebook cells / no Colab dependencies).
"""

import os
import time

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

try:
    import resource

    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False
    import psutil

# ---------------------------------------------------------------------------
# Config - adjust these for your machine
# ---------------------------------------------------------------------------
DATA_PATH = "2019-Nov.csv"     
OUTPUT_DIR = "./output"              
USECOLS = ["event_type", "product_id", "user_session"]
CHUNKSIZE = 2_000_000
SAMPLE_FRAC = 1.0                   
MIN_SUPPORT = 0.001
MIN_CONFIDENCE = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main() -> None:
    # 1. Load data (chunked, NO sampling here - sample baskets later instead)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Download 2019-Nov.csv from Kaggle and "
            f"place it there, or update DATA_PATH."
        )

    t0 = time.time()
    chunks = []
    for chunk in pd.read_csv(DATA_PATH, usecols=USECOLS, chunksize=CHUNKSIZE):
        purchase_rows = chunk[chunk["event_type"] == "purchase"]
        chunks.append(purchase_rows)

    purchases_df = pd.concat(chunks, ignore_index=True)
    load_time = time.time() - t0
    print(f"Purchase rows (full, unsampled): {len(purchases_df)}")
    print(f"Data load time: {load_time:.2f}s")

    # 2. Build FULL baskets first, THEN sample at the basket level.
    t0 = time.time()
    baskets_series = purchases_df.groupby("user_session")["product_id"].apply(
        lambda ids: list(set(ids))
    )
    baskets_series = baskets_series[baskets_series.apply(len) >= 2]
    print(f"Total multi-item baskets (before sampling): {len(baskets_series)}")

    if SAMPLE_FRAC < 1.0:
        baskets_series = baskets_series.sample(frac=SAMPLE_FRAC, random_state=42)

    transactions = baskets_series.tolist()
    preprocess_time = time.time() - t0
    print(f"Number of multi-item baskets (after sampling): {len(transactions)}")
    print(f"Preprocessing time: {preprocess_time:.2f}s")

    if len(transactions) < 20:
        print(
            "WARNING: fewer than 20 baskets - results will be statistically weak. "
            "Consider raising SAMPLE_FRAC."
        )

    # 3. One-hot encode transactions
    t0 = time.time()
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions, sparse=True)
    onehot_df = pd.DataFrame.sparse.from_spmatrix(
        te_array, columns=[str(c) for c in te.columns_]
    )
    encode_time = time.time() - t0
    print(f"Number of unique items (columns): {len(te.columns_)}")
    print(f"One-hot matrix shape: {onehot_df.shape}")
    print(f"One-hot encoding time: {encode_time:.2f}s")

    
    # 4. Run FP-Growth
    t0 = time.time()
    frequent_itemsets = fpgrowth(onehot_df, min_support=MIN_SUPPORT, use_colnames=True)
    fpgrowth_time = time.time() - t0
    frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)
    print(f"FP-Growth mining time: {fpgrowth_time:.2f}s")
    print(frequent_itemsets.head(20))


    # 5. Association rules
    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE
    )
    rules = rules.sort_values("confidence", ascending=False)
    print(rules.head(20))

    # 6. Save results locally
    freq_path = os.path.join(OUTPUT_DIR, "freq_itemsets_mlxtend.csv")
    rules_path = os.path.join(OUTPUT_DIR, "rules_mlxtend.csv")
    frequent_itemsets.to_csv(freq_path, index=False)
    rules.to_csv(rules_path, index=False)
    print(f"Saved: {freq_path}")
    print(f"Saved: {rules_path}")

    # 7. Timing summary
    total = load_time + preprocess_time + encode_time + fpgrowth_time
    print("\n=== Local Python (pandas + mlxtend) Timing Summary ===")
    print(f"Data load:        {load_time:.2f}s")
    print(f"Preprocessing:    {preprocess_time:.2f}s")
    print(f"One-hot encoding: {encode_time:.2f}s")
    print(f"FP-Growth mining: {fpgrowth_time:.2f}s")
    print(f"Total:            {total:.2f}s")

    # 8. Row-count summary across every table in the pipeline
    print("\n=== Row Counts by Table ===")
    print(f"purchases_df (raw purchase rows, full/unsampled): {len(purchases_df):,}")
    print(f"transactions (multi-item baskets, after basket sampling): {len(transactions):,}")
    print(f"onehot_df (baskets x unique products):    {onehot_df.shape[0]:,} rows, {onehot_df.shape[1]:,} cols")
    print(f"frequent_itemsets:                        {len(frequent_itemsets):,}")
    print(f"rules:                                    {len(rules):,}")

    # 9. Peak memory usage for the whole run
    if HAS_RESOURCE:
        import sys

        peak_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = peak_raw / 1024 if sys.platform != "darwin" else peak_raw / (1024 * 1024)
    else:
        process = psutil.Process(os.getpid())
        peak_mb = process.memory_info().rss / (1024 ** 2)

    print(f"\nPeak memory usage: {peak_mb:.2f} MB")


if __name__ == "__main__":
    main()
