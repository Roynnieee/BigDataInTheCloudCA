"""
Runs FP-Growth on the FULL dataset

A short warm-up pass is run first (untimed) to absorb Spark's one-time
JVM/code-generation cost, so the measured runtime reflects only the
FP-Growth fit itself.

Dataset : eCommerce behavior data from multi-category store - 2019-Nov.csv
Source  : https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store

BEFORE RUNNING:
  Confirm your exact S3 paths with:
    aws s3 ls s3://aws-logs-597717962342-us-east-1/Assignmentt/

RUN WITH (on the EMR primary node):
  spark-submit aws_spark_fpgrowth_final.py 2>&1 | tee spark_run_log.txt
"""

import time
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# 1. Spark session
# ---------------------------------------------------------------------------
spark = (
    SparkSession.builder.appName("EcommerceFPGrowth")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)

# ---------------------------------------------------------------------------
# 2. CONFIG — update these to match your own S3 bucket/paths
# ---------------------------------------------------------------------------
DATA_PATH = "s3://aws-logs-597717962342-us-east-1/Assignmentt/2019-Nov.csv"
S3_OUTPUT = "s3://aws-logs-597717962342-us-east-1/results"

MIN_SUPPORT = 0.001
MIN_CONFIDENCE = 0.1
RANDOM_SEED = 42   # used only for the warm-up sample

# ---------------------------------------------------------------------------
# 3. Load data (TIMED — Data load time)
# ---------------------------------------------------------------------------
load_start = time.time()
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
df.printSchema()
total_row_count = df.count()
load_time = time.time() - load_start
print("Total row count:", total_row_count)
print(f"Data load time: {load_time:.2f}s")

# ---------------------------------------------------------------------------
# 4. Build baskets: distinct product_ids purchased in the same user_session
#    (TIMED — Preprocessing time)
# ---------------------------------------------------------------------------
prep_start = time.time()
purchases = (
    df.filter(F.col("event_type") == "purchase")
    .select("user_session", "product_id")
    .dropna()
)

baskets = purchases.groupBy("user_session").agg(
    F.collect_set("product_id").alias("items")
)
baskets = baskets.filter(F.size("items") >= 2).cache()

total_baskets = baskets.count()
prep_time = time.time() - prep_start
print("Number of multi-item baskets:", total_baskets)
print(f"Preprocessing time: {prep_time:.2f}s")

# ---------------------------------------------------------------------------
# 4.5. WARM-UP: absorb JVM/codegen warm-up cost before timing anything
# ---------------------------------------------------------------------------
print("\nRunning warm-up pass (not timed)...")
warmup_baskets = baskets.sample(fraction=0.01, seed=RANDOM_SEED)
fp_warmup = FPGrowth(itemsCol="items", minSupport=MIN_SUPPORT, minConfidence=MIN_CONFIDENCE)
_ = fp_warmup.fit(warmup_baskets)
print("Warm-up complete.\n")

# ---------------------------------------------------------------------------
# 5. Run FP-Growth on the full dataset, timed (FP-Growth mining time)
# ---------------------------------------------------------------------------
start = time.time()
fp_growth = FPGrowth(itemsCol="items", minSupport=MIN_SUPPORT, minConfidence=MIN_CONFIDENCE)
model = fp_growth.fit(baskets)
elapsed = time.time() - start
print(f"FP-Growth mining time: {elapsed:.2f}s")

# ---------------------------------------------------------------------------
# 6. Inspect the output
# ---------------------------------------------------------------------------
freq_itemsets = model.freqItemsets.orderBy(F.desc("freq"))
rules = model.associationRules.orderBy(F.desc("confidence"))

n_itemsets = freq_itemsets.count()
n_rules = rules.count()
print("frequent_itemsets:", n_itemsets)
print("rules:", n_rules)

print("Top frequent itemsets:")
freq_itemsets.show(20, truncate=80)
print("Top association rules:")
rules.show(20, truncate=80)

# ---------------------------------------------------------------------------
# 7. Persist results to S3
#    Parquet keeps array columns as-is. CSV cannot store array columns,
#    so those are flattened to comma-separated strings before writing CSV.
# ---------------------------------------------------------------------------
freq_itemsets.write.mode("overwrite").parquet(f"{S3_OUTPUT}/freq_itemsets")
rules.write.mode("overwrite").parquet(f"{S3_OUTPUT}/rules")

freq_itemsets_csv_ready = freq_itemsets.withColumn("items", F.concat_ws(",", "items"))
freq_itemsets_csv_ready.coalesce(1).write.mode("overwrite").option("header", True).csv(
    f"{S3_OUTPUT}/freq_itemsets_csv"
)

rules_csv_ready = (
    rules.withColumn("antecedent", F.concat_ws(",", "antecedent"))
         .withColumn("consequent", F.concat_ws(",", "consequent"))
)
rules_csv_ready.coalesce(1).write.mode("overwrite").option("header", True).csv(
    f"{S3_OUTPUT}/rules_csv"
)

# ---------------------------------------------------------------------------
# 8. Summary (mirrors the format used in the local VS Code comparison run)
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print("Purchase rows (full, unsampled):", total_row_count)
print(f"Data load time: {load_time:.2f}s")
print("Total multi-item baskets:", total_baskets)
print(f"Preprocessing time: {prep_time:.2f}s")
print(f"FP-Growth mining time: {elapsed:.2f}s")
print("frequent_itemsets:", n_itemsets)
print("rules:", n_rules)

print(f"\nDone. Results written to {S3_OUTPUT}/")

spark.stop()
