# IST3134 — FP-Growth Market Basket Analysis (Local / VS Code)

## Dataset
**eCommerce Behavior Data from a Multi-Category Store (November 2019 file)**
https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store?select=2019-Nov.csv

Not included in this repo (several GB, exceeds GitHub's 100MB limit). Download
it from the link above and place it in the same folder as `fpgrowth.py`
(or update `DATA_PATH` in the script).

## What this does
Runs FP-Growth market basket analysis on the full, unsampled 'purchase' events
from the dataset:
1. Loads all purchase rows (no sampling).
2. Groups them into baskets by `user_session`.
3. One-hot encodes baskets with mlxtend's `TransactionEncoder`.
4. Mines frequent itemsets and association rules with mlxtend's `fpgrowth()`.
5. Saves results to `results/` and prints timing + memory stats.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python fpgrowth.py
```

## Output
Results are written to `output/freq_itemsets_mlxtend.csv` and
`output/rules_mlxtend.csv` after a successful run (matches OUTPUT_DIR in the script).

## Team
- Member 1: [name] — [student ID]
- Member 2: [name] — [student ID]
