# Building an Operational Data Store with MongoDB, Redis, and GCS

---

## Learning Objectives

By the end of this lesson you will be able to:

1. Explain why an Operational Data Store uses multiple storage layers
2. Load data into MongoDB and run a simple aggregation query
3. Cache a query result in Redis and retrieve it
4. Upload a data file to Google Cloud Storage

**Estimated time:** 2 to 3 hours

**Environment:** Make sure you have selected the `bde` kernel before running any cells.

---

## The Problem

You are a data engineer at **ShopStream**, a growing e-commerce company.

The business has been running for 5 years. All orders are stored in **PostgreSQL**, over 10 million records and counting. The operations team needs a live dashboard showing today's revenue, active orders, and top-selling products so they can respond to problems in real time.

The naive solution is to query PostgreSQL directly. But this causes two problems:

1. **It's slow.** Scanning 10 million rows for today's orders takes several seconds.
2. **It's risky.** Heavy analytics queries compete with the application's own reads and writes, which can slow down the checkout flow for real customers.

---

## The Solution: An Operational Data Store

Instead of hitting PostgreSQL directly, we introduce dedicated layers, each one optimised for a specific job.

```mermaid
flowchart TD
    PG["PostgreSQL<br/>10M+ orders<br/>5 years of history<br/> source of truth "]
    MONGO["MongoDB  ← you are here<br/>Recent orders<br/>Fast operational queries"]
    REDIS["Redis<br/>Cached query results<br/>Lives minutes to hours"]
    GCS["Google Cloud Storage<br/>Archived data files<br/>Cheap, long-term"]
    BQ["BigQuery<br/>Historical analysis<br/>3+ years<br/>(future lesson)"]

    PG -->|"sync recent data / ETL / CDC job"| MONGO
    MONGO -->|"cache dashboard results"| REDIS
    MONGO -->|"archive older data"| GCS
    PG -->|"bulk export for analysis"| BQ
```

**Why MongoDB for recent orders?**

- Flexible document model: easy to store denormalised order data (customer + items in one document)
- Fast aggregations over tens of thousands of records without touching PostgreSQL
- Older data can be archived to GCS and removed from MongoDB, keeping it lean and fast

**Why Redis on top of MongoDB?**

- The operations team refreshes the dashboard constantly
- Running the same MongoDB aggregation 50 times per minute is wasteful
- Redis holds the computed result for a few minutes and the dashboard reads from Redis in under 1 ms instead of waiting 50 ms for MongoDB

**What about historical analysis?**

Long-term trends (revenue over 3 years, year-on-year comparisons) need the full dataset. That work belongs in a data warehouse like BigQuery, not in MongoDB. We will cover that in a future lesson.

---

## MongoDB as an Operational Data Store (ODS)

This is a good moment to introduce a pattern you will see repeatedly in real data engineering work.

Most people's first instinct with MongoDB is to use it as a **temporary staging area** - dump data in, process it, delete it. That's fine for ETL pipelines, but there's a more powerful pattern: the **Operational Data Store (ODS)**.

> An **ODS** is a database that holds a _current, queryable snapshot_ of operational data. It doesn't replace the source-of-truth database - it serves a specific audience (operations, support, dashboards) who need fast answers to "what's happening right now?" questions.

MongoDB calls this pattern **Operational Analytics**. You can read their overview here: [Operational Analytics with MongoDB](https://www.mongodb.com/use-cases/analytics/operational-analytics).

The key characteristics of an ODS:

|              | Transactional DB (OLTP) | Operational Data Store (ODS)   | Data Warehouse (DWH)   |
| ------------ | ----------------------- | ------------------------------ | ---------------------- |
| **Example**  | PostgreSQL              | MongoDB                        | BigQuery               |
| **Data age** | All history             | Recent window (hours - months) | All history            |
| **Updated**  | Every transaction       | Near real-time sync            | Daily / hourly batch   |
| **Queries**  | Row lookups, writes     | Aggregations, filters          | Complex analytics      |
| **Audience** | Application code        | Operations, dashboards         | Analysts, data science |

MongoDB is a natural fit for an ODS because:

- Its document model handles **denormalised data** well - one order document can embed customer info, line items, and status history, making queries fast without joins
- It supports **[TTL indexes](https://www.mongodb.com/docs/manual/core/index-ttl/)** - you can configure MongoDB to automatically expire and delete old documents, keeping only the data you need
- Its aggregation pipeline handles the kind of group-by-and-sum queries that operational dashboards need

### Real-World Examples

These companies all use a variant of this pattern in production.

**Uber: Driver/Rider Operations**

```mermaid
flowchart LR
    PG["PostgreSQL<br/>All trip history"]
    MONGO["MongoDB ODS<br/>Last 24 hours of trips"]
    REDIS["Redis<br/>Active rides cache"]
    DASH["Dashboard<br/>'Drivers available<br/>in SF right now'"]

    PG -->|"sync recent trips"| MONGO
    MONGO -->|"cache live counts"| REDIS
    REDIS --> DASH
```

Why MongoDB here? Uber needs sub-second answers to "how many drivers are within 2km of this rider right now?" Querying years of trip history in PostgreSQL for that is overkill. A rolling 24-hour window in MongoDB is fast, small, and purpose-built for live ops.

---

**Shopify: Merchant Dashboards**

```mermaid
flowchart LR
    PG["PostgreSQL<br/>All orders, all merchants"]
    MONGO["MongoDB ODS<br/>Last 30 days per merchant"]
    REDIS["Redis<br/>Today's totals cache"]
    DASH["Merchant Dashboard<br/>'Your sales today'"]

    PG -->|"sync per merchant"| MONGO
    MONGO -->|"cache daily KPIs"| REDIS
    REDIS --> DASH
```

When a Shopify merchant opens their admin dashboard, they expect the page to load in under a second. Running a `SUM` across every order they've ever received would be far too slow. MongoDB holds the recent window; Redis holds the pre-computed answer.

---

## What This Lesson Covers

In this lab we are focused on the **MongoDB → Redis → GCS** part of the pipeline. We skip the PostgreSQL → MongoDB sync and instead generate synthetic data representing the latest 35 days of orders - simulating what an ETL job would load into MongoDB from the source system.

```mermaid
flowchart LR
    GEN["Synthetic data<br/>np.random"]:::highlight -->|"insert_many"| MONGO["MongoDB<br/>recent orders"]:::highlight
    MONGO -->|"aggregate + cache"| REDIS["Redis<br/>dashboard cache"]:::highlight
    MONGO -->|"export + upload"| GCS["GCS<br/>archive"]:::highlight
    classDef highlight fill:#d4edda,stroke:#28a745
```

| Step   | What you will do                           | Technology             |
| ------ | ------------------------------------------ | ---------------------- |
| Part 3 | Generate orders and load into MongoDB      | `insert_many`          |
| Part 4 | Query revenue by category                  | `aggregate`            |
| Part 5 | Cache the result and test cache-miss / hit | Redis `set` / `get`    |
| Part 6 | Export orders and upload to GCS            | `upload_from_filename` |

---

## Part 0: Setup

Create a Jupyter notebook in `notebooks` folder and follow along. The code snippets in the next sections are meant to be run in separate cells.

### Cell 1: Install packages (optional, should be in `bde` environment already)

```python
%pip install pymongo[srv] redis python-dotenv --quiet
```

### Cell 2: Imports

```python
import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pymongo
from pymongo import MongoClient
import redis
from dotenv import load_dotenv

print("Imports OK")
```

---

## Part 1: Credentials

We store passwords in a `.env` file so they are never visible in the notebook.

### Cell 3: Create the `.env` file

Run this cell once. Then open the file and replace the placeholder values with your real credentials.

```python
env_path = Path("../.env")

template = """\
MONGODB_URI=mongodb+srv://YOUR_USER:YOUR_PASSWORD@YOUR_CLUSTER.mongodb.net/
REDIS_HOST=YOUR_REDIS_HOST
REDIS_PORT=YOUR_REDIS_PORT
REDIS_PASSWORD=YOUR_REDIS_PASSWORD
GCS_BUCKET_NAME=YOUR_BUCKET_NAME
"""

if not env_path.exists():
    env_path.write_text(template)
    print("Created .env. Open it and fill in your credentials.")
else:
    print(".env already exists.")
```

> **Stop here.** Open the `.env` file and replace every `YOUR_...` value before continuing.

### Cell 4: Load credentials

```python
load_dotenv(Path("../.env"))

MONGODB_URI = os.getenv("MONGODB_URI")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# (Optional) Print values to verify they loaded (don't print passwords in real code!)
# print("MongoDB URI:", MONGODB_URI)
# print("Redis Host:", REDIS_HOST)
# print("Redis Port:", REDIS_PORT)

```

---

## Part 2: Connect to MongoDB and Redis

### Cell 5: Connect to MongoDB

```python
mongo_client = MongoClient(MONGODB_URI)
mongo_client.admin.command("ping")    # raises an error if connection fails
print("Connected to MongoDB.")

# Create the "shopstream" database and "orders" collection references. They won't actually be created until we insert data, but this makes it clear what we will be using later on.
db = mongo_client["shopstream"]
orders_col = db["orders"]
```

### Cell 6: Connect to Redis

```python
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True,   # returns strings, not bytes
)

r.ping()
print("Connected to Redis.")
```

---

## Part 3: Load Data into MongoDB

In a real system the latest orders would arrive from the OLTP database via an ETL job. Here we use `numpy` to generate synthetic data representing the last 35 days of orders and insert them directly into MongoDB, simulating what that ETL job would deliver.

> **How would this work in production?**
>
> You would not reload 35 days of data every day. That would be slow and wasteful. The 35 days is the **retention window** (how far back MongoDB keeps data), not the batch size.
>
> In practice, the ETL job runs on a schedule (e.g. nightly) and only pulls **new or changed records since the last run**, typically the past 24 hours. MongoDB accumulates records over time, and a separate cleanup job removes anything older than 35 days to keep the collection lean.
>
> This is called an **incremental load**. The alternative, which is dropping and reloading the full 35-day window each time is called a **full refresh**. Full refreshes are simpler but expensive at scale; they are only used when the dataset is small or when records can be retroactively updated in ways that are hard to track incrementally.
>
> For near real-time use cases, teams replace the scheduled ETL job with **Change Data Capture (CDC)**. Tools like Debezium watch the PostgreSQL transaction log and stream every insert, update, and delete to MongoDB as it happens, with no scheduled batch needed.

### Cell 7: Generate synthetic products and customers

```python
np.random.seed(42)

# Products
CATEGORIES = ["footwear", "apparel", "accessories", "electronics"]

products_data = []

# Generate 20 products with random categories and prices
for i in range(1, 21):
    category = np.random.choice(CATEGORIES)
    products_data.append({
        "product_id": f"PROD_{i:03d}",
        "name": f"{category.title()} Item {i}",
        "category": category,
        "price": round(np.random.uniform(20, 300), 2),
    })

products_df = pd.DataFrame(products_data)
print(f"Generated {len(products_df)} products")
products_df.head()
```

### Cell 8: Generate synthetic orders

```python
REGIONS = ["North", "South", "East", "West"]
STATUSES = ["pending", "shipped", "delivered", "returned"]

orders_data = []
# Create a start date 35 days ago. This will be the earliest order date in our dataset
start_date = datetime.now() - timedelta(days=35)

# Create 500 orders with random products, quantities, dates, and statuses
for i in range(1, 501):
    product = np.random.choice(products_data)
    quantity = np.random.randint(1, 4)
    order_date = start_date + timedelta(days=int(np.random.randint(0, 35)),
                                        hours=int(np.random.randint(0, 24)))
    orders_data.append({
        "order_id": f"ORD_{i:05d}",
        "customer_id": f"CUST_{np.random.randint(1, 101):04d}",
        "region": np.random.choice(REGIONS),
        "product_id": product["product_id"],
        "category": product["category"],
        "unit_price": product["price"],
        "quantity": quantity,
        "total": round(product["price"] * quantity, 2),
        "status": np.random.choice(STATUSES, p=[0.05, 0.15, 0.75, 0.05]),
        "created_at": order_date,
    })

orders_df = pd.DataFrame(orders_data)
print(f"Generated {len(orders_df)} orders")
print(f"Date range: {orders_df['created_at'].min()} → {orders_df['created_at'].max()}")
print(f"Total revenue (delivered): ${orders_df[orders_df['status']=='delivered']['total'].sum():,.2f}")
orders_df.head()
```

**Read the output.** Notice that `created_at` is a Python `datetime` object. MongoDB stores it natively as a date, which is what lets us filter by time range later.

### Cell 9: Insert into MongoDB

```python
orders_col.drop() # start fresh on each run
result = orders_col.insert_many(orders_data)
print(f"Inserted {len(result.inserted_ids)} orders.")
```

### Cell 10: Verify

```python
total = orders_col.count_documents({})
print("Total orders:", total)

sample = orders_col.find_one({"status": "delivered"})
print("\nOne delivered order:")
print(json.dumps(sample, indent=2, default=str))
```

> **Exercise 3.1** *(optional)*: How many orders have status `"returned"`? Use `count_documents()` with a filter.

```python
# Exercise 3.1: your code here
```

<details>
<summary>💡 Solution</summary>

```python
count = orders_col.count_documents({"status": "returned"})
print(f"Returned orders: {count}")
```

</details>

> **Exercise 3.2** *(optional)*: Find the 5 most recent orders. Use `.sort("created_at", pymongo.DESCENDING).limit(5)` chained onto `find()`. Print the `order_id` and `total` for each.

```python
# Exercise 3.2: your code here
```

<details>
<summary>💡 Solution</summary>

```python
recent = orders_col.find().sort("created_at", pymongo.DESCENDING).limit(5)
for order in recent:
    print(f"{order['order_id']}  ${order['total']}")
```

</details>

---

## Part 4: Query MongoDB: A Simple Aggregation

Before we can cache anything, we need to compute a metric worth caching.

We want to answer: **"What is the total revenue per category over the last 30 days?"**

### Cell 11: Write the aggregation pipeline

Read through each stage before running the cell.

```python
now = datetime.now()
cutoff = now - timedelta(days=30)

pipeline = [
    # Stage 1: keep only orders from the last 30 days, exclude returns
    {"$match": {
        "created_at": {"$gte": cutoff},
        "status":     {"$ne": "returned"},
    }},
    # Stage 2: group by category, sum the revenue, and count the orders
    {"$group": {
        "_id":     "$category",
        "revenue": {"$sum": "$total"},
        "orders":  {"$sum": 1},
    }},
    # Stage 3: sort by revenue, highest first
    {"$sort": {"revenue": pymongo.DESCENDING}},
]

# Run the aggregation and convert the cursor to a list
results = list(orders_col.aggregate(pipeline))

print("Revenue by category (last 30 days):")
for row in results:
    print(f"  {row['_id']:<15}  ${row['revenue']:>8,.2f}   ({row['orders']} orders)")
```

**What each stage does:**

```mermaid
flowchart LR
    A["All orders<br/>in collection"] -->|"$match<br/>last 30 days<br/>not returned"| B["Filtered<br/>orders"]
    B -->|"$group<br/>by category<br/>sum totals"| C["One row<br/>per category"]
    C -->|"$sort<br/>by revenue desc"| D["Final<br/>sorted result"]
```

> **Exercise 4.1** *(optional)*: Modify the pipeline to also compute the average order value per category. Add `"avg_order": {"$avg": "$total"}` inside the `$group` stage and print it.

```python
# Exercise 4.1: your code here
```

<details>
<summary>💡 Solution</summary>

```python
pipeline_ex = [
    {"$match": {
        "created_at": {"$gte": cutoff},
        "status":     {"$ne": "returned"},
    }},
    {"$group": {
        "_id":       "$category",
        "revenue":   {"$sum": "$total"},
        "orders":    {"$sum": 1},
        "avg_order": {"$avg": "$total"},
    }},
    {"$sort": {"revenue": pymongo.DESCENDING}},
]

for row in orders_col.aggregate(pipeline_ex):
    print(f"  {row['_id']:<15}  revenue=${row['revenue']:>8,.2f}   avg=${row['avg_order']:>7,.2f}")
```

</details>

---

## Part 5: Cache the Result in Redis

Running this aggregation every time the dashboard loads is wasteful. We'll cache the result in Redis and only re-query MongoDB when the cache expires.

### The cache-miss / cache-hit pattern

```mermaid
flowchart TD
    A["Dashboard requests<br/>category revenue"] --> B{"Check Redis<br/>cache"}
    B -->|"Key exists - CACHE HIT"| C["Return cached result<br/>less than 1 ms"]
    B -->|"Key missing or expired - CACHE MISS"| D["Query MongoDB<br/>~50 ms"]
    D --> E["Store result<br/>in Redis with TTL"]
    E --> F["Return result"]
```

### Cell 12: Store the result in Redis

```python
CACHE_KEY = "dashboard:category_revenue"
CACHE_TTL = 300  # seconds (5 minutes)

# Serialise the list to a JSON string - Redis stores strings
r.set(CACHE_KEY, json.dumps(results), ex=CACHE_TTL)

print(f"Stored result at key:  {CACHE_KEY}")
print(f"Time-to-live:          {r.ttl(CACHE_KEY)} seconds")
```

> **Why store a JSON string instead of a Redis Hash?**
>
> Redis does have a Hash type (`r.hset()`) that stores field-value pairs, similar to a Python dict. You might wonder why we didn't use it here.
>
> The key reason is that our aggregation result is a **list of dicts** (one row per category). A Redis Hash maps neatly to a single flat object, but not to a list. Storing a list as a Hash would require awkward workarounds, like encoding each row as a field name.
>
> More importantly, we want a **TTL on the entire cached result**. Redis TTL works at the key level — you cannot set an expiry on individual Hash fields. With `r.set()` + a JSON string, the whole result expires together automatically.
>
> Use a Redis Hash when you're caching a single flat object and need to read or update individual fields (e.g. a user profile). Use `r.set()` + JSON when you're caching a list or a nested structure that is always read and written as a whole.

### Cell 13: Read it back from the cache

```python
raw = r.get(CACHE_KEY)

if raw:
    cached_results = json.loads(raw)
    print(f"Retrieved {len(cached_results)} rows from Redis.")
    for row in cached_results:
        print(f"  {row['_id']:<15}  ${row['revenue']:>8,.2f}")
else:
    print("Key not found - cache may have expired.")
```

### Cell 14: Put it all together: a function with cache-miss fallback

This is the pattern you will use in real projects.

```python
def get_category_revenue(use_cache=True):
    """
    Returns category revenue for the last 30 days.
    Reads from Redis if available; falls back to MongoDB otherwise.
    """
    key = "dashboard:category_revenue"

    if use_cache:
        cached = r.get(key)
        if cached:
            print("[CACHE HIT]")
            return json.loads(cached)
        print("[CACHE MISS] querying MongoDB...")

    # Query MongoDB
    cutoff = datetime.now() - timedelta(days=30)
    pipeline = [
        {"$match":  {"created_at": {"$gte": cutoff}, "status": {"$ne": "returned"}}},
        {"$group":  {"_id": "$category", "revenue": {"$sum": "$total"}, "orders": {"$sum": 1}}},
        {"$sort":   {"revenue": pymongo.DESCENDING}},
    ]
    data = list(orders_col.aggregate(pipeline))

    # Populate the cache with a TTL of 5 minutes
    r.set(key, json.dumps(data), ex=300)
    return data


# First call - nothing in cache yet (we deleted the key above)
r.delete("dashboard:category_revenue")
result = get_category_revenue()

print()

# Second call should hit the cache
result = get_category_revenue()
```

### Cell 15: Measure the difference

```python
# Adjust the number of runs if you want a more stable average
RUNS = 10

# --- Time MongoDB aggregation (always queries the database) ---
mongo_times = []
for _ in range(RUNS):
    r.delete("dashboard:category_revenue")   # force a fresh query each run
    start = time.perf_counter()
    get_category_revenue(use_cache=False)
    mongo_times.append((time.perf_counter() - start) * 1000)

# --- Time Redis cache hit (read the cached key directly) ---
# Make sure the key exists first
get_category_revenue(use_cache=False)

redis_times = []
for _ in range(RUNS):
    start = time.perf_counter()
    r.get("dashboard:category_revenue")      # raw Redis read, no function overhead
    redis_times.append((time.perf_counter() - start) * 1000)

mongo_avg = sum(mongo_times) / RUNS
redis_avg = sum(redis_times) / RUNS

print(f"MongoDB aggregation (avg of {RUNS}): {mongo_avg:.1f} ms")
print(f"Redis cache hit     (avg of {RUNS}): {redis_avg:.1f} ms")
print(f"Speedup: {mongo_avg / redis_avg:.0f}x")
```

> **What to expect:**
>
> Your numbers will depend on where your Redis instance is running:
>
> | Redis location           | Typical Redis read | Typical speedup |
> | ------------------------ | ------------------ | --------------- |
> | Local (instructor demo)  | ~0.1 ms            | ~100x           |
> | Redis Cloud (your setup) | ~5 to 20 ms        | ~2 to 5x        |
>
> The cloud speedup looks unimpressive - but that's the point. **The bottleneck for cloud Redis is the network round-trip, not the computation.** On a local Redis instance, the network disappears and the true in-memory speed shows through. In production, Redis and the application server sit in the same data centre (or even on the same machine), so the real-world speedup is much closer to the local number than the cloud number you are seeing here.

> **Exercise 5.1** *(optional)*: Write a second cached function `get_daily_order_count()` that returns the number of orders per day for the last 7 days. Use the aggregation stage `$dateToString` to group by date:
>
> ```python
> {"$addFields": {"day": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}}}}
> ```
>
> Cache the result with a TTL of 60 seconds. Call it twice and confirm the second call is a cache hit.

```python
# Exercise 5.1: your code here
```

<details>
<summary>💡 Solution</summary>

```python
def get_daily_order_count(use_cache=True):
    key = "dashboard:daily_order_count"

    if use_cache:
        cached = r.get(key)
        if cached:
            print("[CACHE HIT]")
            return json.loads(cached)
        print("[CACHE MISS] querying MongoDB...")

    cutoff_7d = datetime.now() - timedelta(days=7)
    pipeline = [
        {"$match": {"created_at": {"$gte": cutoff_7d}}},
        {"$addFields": {"day": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}}}},
        {"$group": {"_id": "$day", "count": {"$sum": 1}}},
        {"$sort": {"_id": pymongo.ASCENDING}},
    ]
    data = list(orders_col.aggregate(pipeline))
    r.set(key, json.dumps(data), ex=60)
    return data

# First call cache miss
result = get_daily_order_count()
for row in result:
    print(f"  {row['_id']}  {row['count']} orders")

print()

# Second call cache hit
get_daily_order_count()
```

</details>

> **Exercise 5.2** *(optional)*: What happens when the TTL expires? Set the TTL on your key to 5 seconds, wait, then try to read it back. What does `r.get()` return when a key does not exist?

```python
# Exercise 5.2: your code here
```

<details>
<summary>💡 Solution</summary>

```python
r.set("test:ttl", "hello", ex=5)
print("Stored key. TTL:", r.ttl("test:ttl"), "seconds")

print("Waiting 6 seconds...")
time.sleep(6)

result = r.get("test:ttl")
print("Value after expiry:", result)   # prints: None
```

`r.get()` returns `None` when a key does not exist or has expired. This is why the cache-miss check in `get_category_revenue()` uses `if cached:` — `None` is falsy in Python.

</details>

---

## Part 6: Archive to Google Cloud Storage

This part of the lesson is primarily here to teach you how to work with Google Cloud Storage and the `.jsonl` format, both of which appear frequently in real data pipelines.

Whether you actually need a GCS export step in production depends on your architecture. If PostgreSQL is already being synced to BigQuery via a tool like Fivetran or Airbyte, then a separate GCS archive may be redundant. That said, there are situations where exporting to GCS is genuinely useful:

- **Compliance and audit requirements**: some regulated industries require immutable snapshots of transaction data in object storage, separate from the operational database
- **Controlled staging before BigQuery**: teams sometimes export to GCS first to validate or transform data before loading into BigQuery, rather than loading directly from the source
- **No direct connector available**: if a source system can't be connected to BigQuery directly, exporting to GCS as `.jsonl` is a common workaround

We will export this month's orders as a JSON Lines file and upload it to GCS.

### Cell 16: Connect to GCS

```python
from google.cloud import storage

GCS_BUCKET = os.getenv("GCS_BUCKET_NAME")

gcs = storage.Client()
bucket = gcs.bucket(GCS_BUCKET)
print(f"Connected to bucket: {GCS_BUCKET}")
```

> **Before running:** Make sure `GCS_BUCKET_NAME` is in your `.env` file and the bucket exists in your GCP console.

### Cell 17: Export orders to a local JSON Lines file

**JSON Lines** (`.jsonl`) means one JSON object per line. It is the standard format for exporting document data because it is streamable and BigQuery can ingest it directly.

```python
export_path = Path("./orders_export.jsonl")

now = datetime.now()
month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

month_orders = list(orders_col.find(
    {"created_at": {"$gte": month_start}},
    {"_id": 0}     # exclude the MongoDB ObjectId
))

with open(export_path, "w") as f:
    for order in month_orders:
        f.write(json.dumps(order, default=str) + "\n")

print(f"Exported {len(month_orders)} orders to {export_path}")
print("First line:")
print(open(export_path).readline().strip())
```

### Cell 18: Upload to GCS

```python
date_prefix = now.strftime("%Y/%m/%d")
gcs_path = f"shopstream/orders/{date_prefix}/orders_export.jsonl"

blob = bucket.blob(gcs_path)
blob.upload_from_filename(str(export_path))

print(f"Uploaded to: gs://{GCS_BUCKET}/{gcs_path}")
```

### Cell 19: Verify the upload

```python
blobs = list(gcs.list_blobs(GCS_BUCKET, prefix="shopstream/"))

print(f"Files in gs://{GCS_BUCKET}/shopstream/:")
for b in blobs:
    print(f"  {b.name}  ({b.size} bytes)")
```

### Cell 20: Download and check the file

```python
download_blob = bucket.blob(gcs_path)
download_blob.download_to_filename("orders_downloaded.jsonl")

with open("orders_downloaded.jsonl") as f:
    lines = f.readlines()

print(f"Downloaded {len(lines)} records.")
print("Sample record:")
print(json.dumps(json.loads(lines[0]), indent=2))
```

> **Exercise 6.1** *(optional)*: Upload a small metadata file to GCS at the path `shopstream/metadata/run_info.json` containing today's date and the number of orders exported. Use `blob.upload_from_string()`. You do not need to create a local file.
>
> ```python
> blob = bucket.blob("shopstream/metadata/run_info.json")
> blob.upload_from_string(json.dumps({...}), content_type="application/json")
> ```

```python
# Exercise 6.1: your code here
```

<details>
<summary>💡 Solution</summary>

```python
run_info = {
    "date": datetime.now().strftime("%Y-%m-%d"),
    "orders_exported": orders_col.count_documents({}),
}

blob = bucket.blob("shopstream/metadata/run_info.json")
blob.upload_from_string(json.dumps(run_info), content_type="application/json")
print("Uploaded metadata:", run_info)
```

</details>

---

## Part 7: Clean Up

### Cell 21: Clear Redis

```python
r.flushdb()
print("Redis database cleared.")
```

### Cell 22: Drop the MongoDB collection

```python
orders_col.drop()
print("Dropped shopstream.orders.")
```

### Cell 23: Remove local files

```python
for fname in ["orders_export.jsonl", "orders_downloaded.jsonl"]:
    if os.path.exists(fname):
        os.remove(fname)
        print(f"Removed {fname}")
```

---

END
