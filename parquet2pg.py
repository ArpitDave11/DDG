Below is a **single‑run initialisation script** (`initial_setup.py`) that you can execute once to bring your environment to a clean, production‑ready state.

### What the script does —in this exact order ✔️

1. **Connects** to PostgreSQL with `psycopg2`.
2. **Creates “raw” (main) tables** ‑ one per dataset ‑ from the `table_structures` dictionary.
3. **Bulk‑loads *all* historical files** (every `data_as_of_date=*` folder) into the raw tables.
4. **Indexes the raw tables** (single‑column index on `date_column`; adjust if you need composite or unique indexes).
5. **Creates a *parent* partitioned table for each dataset** (same logical schema, but without data).
6. **Identifies the two most‑recent distinct dates present in the raw table**, creates *child* partitions for those dates, and **copies the corresponding rows** from the raw table into the partitioned table.
7. **Creates the same index on the partitioned parent**; PostgreSQL automatically propagates it to every existing and future child partition ([postgresql.org][1], [postgresql.org][2]).
8. Cleans up cursors / connections.

> **Why two physical tables?** ‑ You asked for a “regular” (raw) table that keeps the entire history and a second, date‑range partitioned table that holds only the last two data days for fast operational queries. Inserts go into the raw table; the daily refresh job will keep the partitioned table current.

---

## `initial_setup.py`

```python
"""
One‑time initialisation:
1. Create raw (main) tables.
2. Load ALL historical data into raw tables.
3. Add indexes on raw tables.
4. Create parent partitioned tables.
5. Create child partitions for the two most‑recent dates.
6. Copy those two days into the partitioned tables.
   (Subsequent daily_refresh.py will keep them current.)
"""

import os
import glob
from datetime import datetime, timedelta
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text

# ──────────────────────────────────────────────
# 1. Configuration ─ adjust once
# ──────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "mydatabase")
DB_USER = os.getenv("DB_USER", "myuser")
DB_PASS = os.getenv("DB_PASS", "mypassword")

# Mapping: dataset‑key  ➜ data directory holding date folders
TARGET_DIRS = {
    # 'table_key': '/mnt/asis/eod/rcat0300',
    # 'table_key2': '/mnt/asis/eod/edgt0100',
}

# Dict containing the column definitions for each dataset
table_structures = {
    # "table_key": "id INT, name TEXT, date_column DATE",
}

DATE_COLUMN = "date_column"          # same name in all tables
RAW_SUFFIX  = "_raw"                 # raw table name = f"{key}{RAW_SUFFIX}"

# Index definition you want on both raw & partitioned tables
IDX_NAME    = "idx_{table}_{col}"    # format string

# ──────────────────────────────────────────────
# 2. Helpers
# ──────────────────────────────────────────────
def pg_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS
    )

def ddl_raw(table_key, struct):
    """Return CREATE TABLE for the raw table (no partitions)."""
    cols = struct.strip().lstrip("(").rstrip(")")
    return (
        f"CREATE TABLE IF NOT EXISTS {table_key}{RAW_SUFFIX} (\n    {cols}\n);"
    )

def ddl_parent_partition(table_key, struct):
    """Return CREATE TABLE for the *parent* partitioned table."""
    cols = struct.strip().lstrip("(").rstrip(")")
    return (
        f"CREATE TABLE IF NOT EXISTS {table_key} (\n    {cols}\n)"
        f" PARTITION BY RANGE ({DATE_COLUMN});"
    )

def create_child_partition(cur, table_key, date_val):
    """
    Create one‑day range partition: [date_val, date_val+1d)
    e.g. table_20250617
    """
    start = date_val.strftime("%Y-%m-%d")
    next_day = (date_val + timedelta(days=1)).strftime("%Y-%m-%d")
    child = f"{table_key}_{date_val.strftime('%Y%m%d')}"
    cur.execute(
        f"""CREATE TABLE IF NOT EXISTS {child}
            PARTITION OF {table_key}
            FOR VALUES FROM ('{start}') TO ('{next_day}');"""
    )

def bulk_load_dataframe(df, dest_table, engine):
    """Append a DataFrame to PostgreSQL quickly."""
    df.to_sql(dest_table, engine, if_exists="append",
              index=False, method="multi", chunksize=10000)

# ──────────────────────────────────────────────
# 3. Main driver
# ──────────────────────────────────────────────
if __name__ == "__main__":
    conn = cur = engine = None
    try:
        # ─ DB connections
        conn = pg_conn()
        conn.autocommit = True
        cur = conn.cursor()
        engine = create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )

        # ─ For each dataset (table_key)
        for table_key, base_path in TARGET_DIRS.items():
            struct = table_structures[table_key]

            # 3.1 Create raw table
            cur.execute(ddl_raw(table_key, struct))

            # 3.2 Scan ALL date folders and load data into raw table
            pattern = os.path.join(base_path, "data_as_of_date=*")
            folders = sorted(
                [f for f in glob.glob(pattern) if os.path.isdir(f)]
            )
            if not folders:
                print(f"No data folders for {table_key} → skipped.")
                continue

            all_dates = []
            for folder in folders:
                date_str = folder.split("=", 1)[1]
                try:
                    d = datetime.strptime(date_str, "%Y-%m-%d").date()
                except ValueError:
                    continue
                all_dates.append(d)

                files = [
                    os.path.join(folder, f)
                    for f in os.listdir(folder)
                    if os.path.isfile(os.path.join(folder, f))
                ]
                if not files:
                    continue

                dfs = []
                for fp in files:
                    if fp.endswith(".csv"):
                        dfs.append(pd.read_csv(fp))
                    elif fp.endswith(".parquet"):
                        dfs.append(pd.read_parquet(fp))
                if not dfs:
                    continue

                df_concat = pd.concat(dfs, ignore_index=True)
                if DATE_COLUMN not in df_concat.columns:
                    df_concat[DATE_COLUMN] = d  # inject folder date
                bulk_load_dataframe(df_concat, f"{table_key}{RAW_SUFFIX}", engine)

            # 3.3 Create index on raw table AFTER load
            idx_raw = IDX_NAME.format(table=f"{table_key}{RAW_SUFFIX}", col=DATE_COLUMN)
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {idx_raw} "
                f"ON {table_key}{RAW_SUFFIX} ({DATE_COLUMN});"
            )

            # 3.4 Create parent partitioned table
            cur.execute(ddl_parent_partition(table_key, struct))

            # 3.5 Determine the two most‑recent dates
            recent_dates = sorted(set(all_dates))[-2:]
            for d in recent_dates:
                create_child_partition(cur, table_key, d)

            # 3.6 Copy the two dates from raw into partition parent
            date_list = ", ".join([f"'{d}'" for d in recent_dates])
            cur.execute(
                f"INSERT INTO {table_key}\n"
                f"SELECT * FROM {table_key}{RAW_SUFFIX}\n"
                f"WHERE {DATE_COLUMN} IN ({date_list});"
            )

            # 3.7 Create the same index on the parent (propagates automatically)
            idx_part = IDX_NAME.format(table=table_key, col=DATE_COLUMN)
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {idx_part} "
                f"ON {table_key} ({DATE_COLUMN});"
            )

            print(f"Initialisation done for {table_key} (raw + partitions + indexes).")

    except Exception as err:
        print(f"Initial‑setup failure: {err}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        if engine:
            engine.dispose()
```

### Where to customise

| Section            | What to edit                                                                                                          | Notes |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- | ----- |
| `TARGET_DIRS`      | Map each *logical* dataset/table key to its parent directory containing `data_as_of_date=` sub‑folders.               |       |
| `table_structures` | Exact column definitions for every dataset. Use the same dictionary you already feed into `generate_postgres_ddls_*`. |       |
| `DATE_COLUMN`      | Column that holds the partition key; **must** exist (or be injected) in every file.                                   |       |
| `IDX_NAME`         | Supply any composite keys you need; current default is a simple B‑tree on the date column.                            |       |

### Why indexes are created **after** loading raw data

Creating the index once—after bulk insert—avoids row‑by‑row index maintenance during load and can be **orders of magnitude faster**.
Creating the same index on the *parent* partitioned table causes PostgreSQL to build matching indexes on every partition and to attach any existing compatible indexes ([postgresql.org][1], [postgresql.org][2]).

Once you run `initial_setup.py`:

* **Raw tables** contain the complete history with an index.
* **Partitioned tables** exist with child partitions and indexes for the two newest dates and hold the corresponding data.
* The **daily refresh job** will simply append tomorrow’s data to the raw table, ensure new child partitions exist, and copy the latest two days—exactly as designed.

Happy loading!

[1]: https://www.postgresql.org/docs/current/ddl-partitioning.html?utm_source=chatgpt.com "Documentation: 17: 5.12. Table Partitioning - PostgreSQL"
[2]: https://www.postgresql.org/docs/current/sql-createindex.html?utm_source=chatgpt.com "Documentation: 17: CREATE INDEX - PostgreSQL"
