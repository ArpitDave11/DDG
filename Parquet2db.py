
import re

def generate_postgres_partitioned_ddls(table_structures, partition_config):
    """
    Given:
      table_structures: dict {tablename: multi-line string col-defs}
      partition_config: dict {
        tablename: {
          'type': 'RANGE'|'LIST'|'HASH',
          'columns': str or [str,...]
        }, ...
      }
    Returns:
      list of CREATE TABLE statements with PARTITION BY clauses.
    """
    table_ddls = {}

    for table, struct in table_structures.items():
        struct = struct.strip()
        # strip wrapping parentheses if present
        if struct.startswith('(') and struct.endswith(')'):
            struct = struct[1:-1]

        # split columns on commas that aren’t inside parens
        cols = re.split(r',\s*(?![^()]*\))', struct)
        col_defs = []
        for col in cols:
            col = col.strip()
            if not col:
                continue
            # strip out SQL comments
            if '--' in col:
                col = col.split('--', 1)[0].strip()
            if col:
                col_defs.append(col)

        # find partition info for this table
        cfg = partition_config.get(table)
        if not cfg:
            raise ValueError(f"No partition config for table {table!r}")

        ptype = cfg['type'].upper()
        # ensure valid
        if ptype not in ('RANGE','LIST','HASH'):
            raise ValueError(f"Invalid partition type {ptype!r} for table {table!r}")

        cols_part = cfg['columns']
        if isinstance(cols_part, str):
            cols_part = [cols_part]
        cols_part = ", ".join(cols_part)

        ddl = (
            f"CREATE TABLE public.{table} (\n"
            f"    " + ",\n    ".join(col_defs) + "\n"
            f") PARTITION BY {ptype} ({cols_part});"
        )

        table_ddls[table] = ddl

    return list(table_ddls.values())



#############################################################################################################

Thanks! I’ll prepare a complete Python script that:

* Reads only the specified parquet files (A, B, C, D) from the Azure-mounted path `/mnt/data`.
* Infers DDL (including primary keys where possible using heuristics).
* Creates PostgreSQL tables with the best indexing strategy for banking/analytical workloads.
* Offers the option to overwrite or append to tables.

I’ll get back to you shortly with the full code.


# Parquet to PostgreSQL Data Loader Script

In this solution, we create a **production-grade Python script** that reads Parquet files from an Azure Storage mount and loads them into PostgreSQL. The script infers each file’s schema (using **PyArrow** to avoid reading full data), generates a `CREATE TABLE` DDL with appropriate data types, infers a primary key by naming heuristics, and then inserts the data using **pandas** and **SQLAlchemy**. We also add **B-tree indexes** (including a composite index on an ID plus date column, if present) to optimize analytical queries. The PostgreSQL connection parameters are left as placeholders for security. Logging and error handling are included for clarity and robustness.

## Key Features and Approach

* **Schema Inference with PyArrow:** We use `pyarrow.parquet.read_schema` to get each Parquet file’s schema without reading the entire file. Each Parquet data type is mapped to an equivalent PostgreSQL column type (e.g. Arrow `int64` → `BIGINT`, `string` → `TEXT`, `timestamp` → `TIMESTAMP`). Decimal types are mapped to `NUMERIC(precision, scale)`. Any complex types (e.g. lists) are mapped to PostgreSQL arrays of the corresponding element type, following Arrow’s standard mapping.

* **Primary Key Heuristics:** The script attempts to infer a primary key for each table. If a column named `"id"` exists, it is chosen as the primary key. Otherwise, a column matching `<table_name>_id` or any single column ending in `_id` is used. If multiple such `_id` columns exist (suggesting a composite key, such as a junction table), all of them are designated as a composite primary key. These conventions align with common database naming practices (e.g. `product_id` in a product table, or composite keys in linking tables).

* **Table Creation and Load Mode:** For each Parquet file, the script executes a `CREATE TABLE IF NOT EXISTS` statement with the inferred schema and primary key. A script parameter `--mode` controls whether to **overwrite** existing tables or **append** to them. In **overwrite** mode, any existing table is dropped and recreated (so the schema can be updated), whereas in **append** mode the table is created only if it doesn’t exist. Data insertion is done with `pandas.DataFrame.to_sql` via SQLAlchemy, using `if_exists='append'` to add records to the table. This approach leverages SQLAlchemy for database compatibility and uses batch inserts (`method='multi'` with a chunk size) for efficiency. All operations for a file (table creation, insert, indexing) are executed within a single database transaction – ensuring that if any step fails, the table is not left partially loaded.

* **Indexing for Analytical Queries:** After loading data, the script creates additional **indexes** to accelerate common query patterns in banking analytics. By default, it indexes any foreign key-like columns (columns ending in `_id`) to speed up joins or filters on those fields. It also indexes date/time columns to optimize time-range queries. Furthermore, if a table has both an ID (e.g. customer or account ID) and a date column, a **composite index** on `(ID, date)` is created. This composite B-tree index helps queries that filter on an identifier and a date range simultaneously – for example, retrieving all transactions for a customer in a given month can use an index on `(customer_id, transaction_date)` to significantly improve performance. (PostgreSQL can use the leading column of a composite index for queries that filter by that column alone as well.) All index creation uses `IF NOT EXISTS` to avoid errors if the index already exists, making the script safe to rerun.

Below is the complete Python script, organized into functions with detailed comments for clarity:

```python
import os
import logging
import argparse
import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# PostgreSQL connection placeholders (to be filled with actual credentials/host)
DB_HOST = "<PG_HOST>"
DB_PORT = "<PG_PORT>"
DB_NAME = "<PG_DATABASE>"
DB_USER = "<PG_USER>"
DB_PASS = "<PG_PASSWORD>"

# Configure logging to include timestamp and level.
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

def get_postgres_type(pa_type):
    """Map a pyarrow DataType to an appropriate PostgreSQL column type (as string)."""
    import pyarrow.types as pa_types  # Import inside function for clarity
    # Numeric types (integers and floating point)
    if pa_types.is_integer(pa_type):
        # Map integer by bit width (8->SMALLINT, 16->SMALLINT, 32->INT, 64->BIGINT)
        bit_width = pa_type.bit_width
        if bit_width <= 16:
            return "SMALLINT"
        elif bit_width <= 32:
            return "INTEGER"
        else:
            return "BIGINT"
    if pa_types.is_floating(pa_type):
        # Map float32 -> REAL, float64 -> DOUBLE PRECISION
        bit_width = getattr(pa_type, "bit_width", 64)  # float32 has 32, float64 has 64
        if bit_width == 32:
            return "REAL"
        else:
            return "DOUBLE PRECISION"
    # Boolean type
    if pa_types.is_boolean(pa_type):
        return "BOOLEAN"
    # Decimal type (fixed precision)
    if pa_types.is_decimal(pa_type):
        # Extract precision and scale for numeric type
        precision = pa_type.precision
        scale = pa_type.scale
        return f"NUMERIC({precision},{scale})"
    # Date and time types
    if pa_types.is_date(pa_type):
        return "DATE"
    if pa_types.is_time(pa_type):
        return "TIME"
    if pa_types.is_timestamp(pa_type):
        # Use TIMESTAMP WITH TIME ZONE if tz info is present, else without time zone
        tz = getattr(pa_type, "tz", None)
        return "TIMESTAMP WITH TIME ZONE" if tz is not None else "TIMESTAMP"
    if pa_types.is_duration(pa_type) or pa_types.is_interval(pa_type):
        return "INTERVAL"
    # String and binary types
    if pa_types.is_string(pa_type) or pa_types.is_large_string(pa_type):
        return "TEXT"
    if pa_types.is_binary(pa_type) or pa_types.is_large_binary(pa_type):
        return "BYTEA"
    # Dictionary (categorical) type: map to the underlying value type
    if pa_types.is_dictionary(pa_type):
        return get_postgres_type(pa_type.value_type)
    # List type (map to array of underlying type in PostgreSQL)
    if pa_types.is_list(pa_type) or pa_types.is_large_list(pa_type):
        value_type = pa_type.value_type
        base_sql_type = get_postgres_type(value_type)
        return base_sql_type + "[]"  # PostgreSQL array type
    # Fallback for any other types (map to text)
    return "TEXT"

def infer_table_schema(file_path):
    """
    Infers the PostgreSQL table schema from a Parquet file.
    Returns tuple: (table_name, columns_dict, primary_key_cols).
    - table_name: derived from file name (lowercased).
    - columns_dict: Ordered dict of {column_name: postgres_type_str}.
    - primary_key_cols: list of column names chosen as primary key.
    """
    # Use pyarrow to read schema without loading data:contentReference[oaicite:12]{index=12}
    schema = pq.read_schema(file_path)
    columns = {}  # preserve insertion order (Python 3.7+ dict is ordered)
    for field in schema:
        col_name = field.name
        col_type = get_postgres_type(field.type)
        columns[col_name] = col_type

    # Infer table name from file name (without extension), use lowercase for consistency
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    table_name = base_name.lower()

    # Infer primary key using naming heuristics:contentReference[oaicite:13]{index=13}
    pk_columns = []
    # Rule 1: If an "id" column exists, use that as primary key
    if "id" in columns:
        pk_columns = ["id"]
    else:
        # Collect all columns that look like they could be primary key candidates (ending in '_id')
        candidate_cols = [col for col in columns.keys() if col.lower().endswith("_id")]
        if len(candidate_cols) == 1:
            # Single candidate *_id -> use as primary key
            pk_columns = [candidate_cols[0]]
        elif len(candidate_cols) > 1:
            # Multiple *_id columns -> assume composite primary key (e.g., linking table)
            # We will use all candidate columns as the composite key.
            pk_columns = candidate_cols.copy()
            # Optionally, ensure a deterministic order (e.g., as they appear in schema)
            # pk_columns.sort()  # uncomment if alphabetical order is desired
    return table_name, columns, pk_columns

def quote_identifier(name):
    """Return a safely quoted identifier if needed (handles uppercase or special chars)."""
    import re
    # Unquoted identifiers in PostgreSQL must start with a letter or underscore, and contain only lowercase letters, digits, or underscore
    if re.match(r'^[a-z_][a-z0-9_]*$', name):
        return name  # safe to use without quotes
    else:
        # Double-quote the identifier, escape any internal quotes by doubling them
        return '"' + name.replace('"', '""') + '"'

def generate_create_table_ddl(table_name, columns, pk_columns):
    """Generate a CREATE TABLE IF NOT EXISTS DDL statement for given table schema."""
    # Quote table name as needed
    tbl = quote_identifier(table_name)
    col_defs = []
    for col_name, col_type in columns.items():
        col_def = f"{quote_identifier(col_name)} {col_type}"
        col_defs.append(col_def)
    # Add primary key clause if we have inferred any
    pk_clause = ""
    if pk_columns:
        # Quote each PK column name and join
        pk_cols_quoted = ", ".join(quote_identifier(col) for col in pk_columns)
        pk_clause = f", PRIMARY KEY ({pk_cols_quoted})"
    ddl = f"CREATE TABLE IF NOT EXISTS {tbl} (\n    " + ",\n    ".join(col_defs) + pk_clause + "\n);"
    return ddl

def create_table_if_needed(conn, ddl, table_name, mode):
    """
    Create the PostgreSQL table using the given DDL.
    In 'overwrite' mode, drop the table if it exists before creating.
    """
    tbl = quote_identifier(table_name)
    try:
        if mode == "overwrite":
            # Drop the table if it already exists (to overwrite with new schema)
            conn.execute(text(f"DROP TABLE IF EXISTS {tbl} CASCADE;"))
            logging.info(f"Dropped existing table {table_name}")
        # Create table (if not exists to avoid error in append mode if already created)
        conn.execute(text(ddl))
        logging.info(f"Table '{table_name}' is ready (created or already exists).")
    except SQLAlchemyError as e:
        logging.error(f"Error creating table {table_name}: {e}")
        raise

def load_data_to_table(conn, table_name, file_path):
    """Read the Parquet file into a DataFrame and load it into the SQL table."""
    try:
        # Read parquet into pandas DataFrame. We use pyarrow engine for compatibility with schema.
        df = pd.read_parquet(file_path, engine="pyarrow")
    except Exception as e:
        logging.error(f"Failed to read Parquet file {file_path}: {e}")
        raise

    # Insert data using pandas to_sql (in the context of an open transaction):contentReference[oaicite:14]{index=14}
    try:
        # Use pandas to_sql to append data to the table. 
        # The connection `conn` is a SQLAlchemy connection in a transaction.
        df.to_sql(name=table_name, con=conn, if_exists="append", index=False, 
                  method="multi", chunksize=1000)
        # Log number of rows inserted
        logging.info(f"Inserted {len(df)} records into '{table_name}'.")
    except SQLAlchemyError as e:
        logging.error(f"Error inserting data into {table_name}: {e}")
        raise

def create_indexes(conn, table_name, columns, pk_columns):
    """
    Create additional indexes on the table for performance:
    - Single-column indexes on likely query filters (foreign key IDs, date columns).
    - Composite index on first foreign key ID + date (if both exist), for combined filtering.
    """
    tbl = quote_identifier(table_name)
    index_statements = []

    # Identify candidate columns for single-column indexing
    # We skip the primary key if it is a single column (already indexed), 
    # but we will index parts of composite keys as needed.
    single_pk = pk_columns[0] if len(pk_columns) == 1 else None
    # If composite PK, determine which parts to index separately (typically all except the first, 
    # because first part is covered by the composite index usage for that prefix:contentReference[oaicite:15]{index=15}).
    composite_pk_parts = pk_columns if len(pk_columns) > 1 else []
    # Gather columns that appear to be foreign keys or identifiers
    for col_name, col_type in columns.items():
        if col_name.lower().endswith("_id") or col_name.lower() == "id":
            # If it's the single primary key, skip (already indexed).
            if col_name == single_pk:
                continue
            # If it's part of composite PK, decide based on position
            if col_name in composite_pk_parts:
                if col_name == composite_pk_parts[0]:
                    # Skip the first part of composite PK (composite index covers it)
                    continue
                # Else, include other parts of composite PK for indexing
            # Otherwise, it's a non-PK id column.
            index_statements.append((
                f"idx_{table_name}_{col_name}", 
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{col_name} ON {tbl} ({quote_identifier(col_name)});"
            ))
    # Gather columns that appear to be date or timestamp for indexing (e.g. for time-range queries)
    date_cols = [col for col, col_type in columns.items() 
                 if col_type.upper().startswith("DATE") or col_type.upper().startswith("TIMESTAMP")]
    for col in date_cols:
        index_statements.append((
            f"idx_{table_name}_{col}", 
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{col} ON {tbl} ({quote_identifier(col)});"
        ))
    # Composite index: if we have at least one foreign key (not primary id) and one date, index on both
    if index_statements and date_cols:
        # Choose first foreign-key-like column for composite index (if any exist in index_statements list)
        composite_id_col = None
        for idx_name, stmt in index_statements:
            # A simple way: pick the first index that was for an _id column (we added those first)
            if "_id" in idx_name and f" ON {tbl} " in stmt:
                composite_id_col = idx_name.replace(f"idx_{table_name}_", "")
                # Ensure this id is not the surrogate PK named "id"
                if composite_id_col.lower() != "id":
                    break
                composite_id_col = None  # if it was "id", ignore and continue
        if composite_id_col and date_cols:
            comp_idx_name = f"idx_{table_name}_{composite_id_col}_{date_cols[0]}"
            comp_idx_stmt = (f"CREATE INDEX IF NOT EXISTS {comp_idx_name} "
                             f"ON {tbl} ({quote_identifier(composite_id_col)}, {quote_identifier(date_cols[0])});")
            index_statements.append((comp_idx_name, comp_idx_stmt))
    # Execute index creations
    for idx_name, idx_sql in index_statements:
        try:
            conn.execute(text(idx_sql))
            logging.info(f"Index '{idx_name}' created on table '{table_name}'.")
        except SQLAlchemyError as e:
            logging.error(f"Error creating index {idx_name} on {table_name}: {e}")
            # Non-fatal: continue trying other indexes

def process_file(conn, file_path, mode):
    """Process a single Parquet file: infer schema, create table, load data, and add indexes."""
    table_name, columns, pk_cols = infer_table_schema(file_path)
    logging.info(f"Processing file '{file_path}' for table '{table_name}'...")
    # Generate DDL and create table
    ddl = generate_create_table_ddl(table_name, columns, pk_cols)
    create_table_if_needed(conn, ddl, table_name, mode)
    # Load data into table
    load_data_to_table(conn, table_name, file_path)
    # Create additional indexes for performance
    create_indexes(conn, table_name, columns, pk_cols)
    logging.info(f"Completed processing for table '{table_name}'.")

def main():
    # Parse command-line arguments for mode (append or overwrite)
    parser = argparse.ArgumentParser(description="Load Parquet files into PostgreSQL.")
    parser.add_argument("--mode", choices=["append", "overwrite"], default="append",
                        help="Append to existing tables or overwrite them (default: append).")
    parser.add_argument("--data-dir", default="/mnt/data",
                        help="Directory containing Parquet files (default: /mnt/data).")
    args = parser.parse_args()
    mode = args.mode
    data_dir = args.data_dir

    # Define the target files to process (only A, B, C, D as specified)
    target_files = ["A.parquet", "B.parquet", "C.parquet", "D.parquet"]
    # Construct full paths
    file_paths = [os.path.join(data_dir, f) for f in target_files]

    # Create a SQLAlchemy engine for PostgreSQL
    engine_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(engine_url, future=True)  # 'future=True' to use 2.x style engine
    overall_success = True

    # Process each file within its own transaction
    for file_path in file_paths:
        # Skip files that do not exist
        if not os.path.isfile(file_path):
            logging.warning(f"File {file_path} not found, skipping.")
            continue
        try:
            # Use a transaction for each file's operations
            with engine.begin() as conn:  # this will commit or rollback automatically
                process_file(conn, file_path, mode)
        except Exception as e:
            overall_success = False
            logging.error(f"Failed to load file {file_path}: {e}")
            # Continue to next file without exiting, to attempt loading others

    engine.dispose()  # Close the database connection pool
    if not overall_success:
        logging.error("One or more files failed to process. Check logs for details.")
        # Exit with a non-zero status to indicate failure (optional in script context)
        exit(1)
    else:
        logging.info("All files processed successfully.")

if __name__ == "__main__":
    main()
```

**Usage:** Save this script to a file (e.g., `load_parquet_to_pg.py`) and run it with Python, providing the desired mode if needed. For example, to overwrite existing tables:

```bash
python load_parquet_to_pg.py --mode overwrite
```

Make sure to replace the placeholder database connection details with real credentials. The script will log progress and any errors to the console. By following this approach, we efficiently load Parquet data into PostgreSQL with proper schema, primary keys, and indexes for performant analytical queries.

**Sources:** The solution uses PyArrow to read Parquet schemas without full data scan. Data type mappings follow Apache Arrow to PostgreSQL conventions. Primary key inference is based on common naming conventions. Indexing recommendations (including composite indexes on `(ID, date)`) draw from SQL performance best practices, which help optimize query performance in analytical workloads.
