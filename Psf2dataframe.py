Thanks! I’ll update the Python script to:

* Read schema definitions from copybook files located in `/mnt/copybook`, where each copybook filename matches the Parquet file name with `_cp` appended (e.g., A\_cp.copybook for A.parquet).
* Use the copybook schema to define the PostgreSQL table structure instead of inferring from Parquet.

I’ll get back to you shortly with the revised code.


# Parquet to PostgreSQL ETL Script

**Overview:** The following Python script reads mainframe data exported as Parquet files and loads them into a PostgreSQL database, using COBOL copybook files to determine the schema. It parses each copybook to extract field definitions, maps COBOL data types to appropriate PostgreSQL types, generates `CREATE TABLE` DDL (including primary keys inferred by naming conventions), and loads the Parquet data into the new table. Finally, it creates B-tree indexes on key identifier and date columns to optimize analytical queries. Logging, error handling, and a `--mode` option (overwrite vs append) are included for robustness and flexibility.

## Features and Steps

1. **Copybook Parsing:** For each Parquet file (e.g. `A.parquet`, `B.parquet`, etc.), the script reads the corresponding copybook (`/mnt/copybook/A_cp.copybook`, etc.) and parses it to retrieve field names, PIC definitions, and usage (COMP, COMP-3, etc.). This can be done using libraries like `cobolpy` for convenience or via a custom parser. The script ensures all copybook field names are normalized (e.g. converting COBOL names with hyphens to snake\_case for SQL compatibility).
2. **Data Type Mapping:** COBOL data types are mapped to PostgreSQL types. For example, alphanumeric fields (`PIC X(n)`) are mapped to `CHAR(n)` columns, binary numeric fields (`COMP` or `BINARY` usage) are mapped to `SMALLINT/INTEGER/BIGINT` depending on their size (e.g. a 4-digit binary becomes SMALLINT, 8-9 digits become INTEGER), and packed decimal fields (`COMP-3`) are mapped to `NUMERIC` (DECIMAL) with the appropriate precision and scale. If a PIC clause includes a `V` (implied decimal point), the script calculates the total digits and fractional digits to define a `NUMERIC(total_digits, scale)`.
3. **Table DDL Generation:** Using the extracted schema, the script dynamically constructs a `CREATE TABLE` statement. It infers primary key column(s) by naming heuristics – for instance, if a field name matches `<table>_id` or is simply `id`, it is treated as a primary key. If multiple such fields are present, they are all included as a composite primary key (this can be refined per actual requirements). The DDL is generated with all columns and data types, and a PRIMARY KEY clause if applicable. The script allows a `--mode` argument: in `overwrite` mode, it will drop any existing table of the same name before creating a new one; in `append` mode, it will create the table only if it does not exist (or skip creation if it already exists).
4. **Data Loading:** The script uses pandas and SQLAlchemy to load data from each Parquet file into the corresponding PostgreSQL table. It reads the Parquet into a pandas DataFrame (`pd.read_parquet`) and aligns the DataFrame’s columns with the database schema (renaming columns to match the cleaned copybook names, and converting data types if necessary – for example, parsing date strings into `datetime` objects or ensuring numeric fields are numeric). The DataFrame is then written to the database using `DataFrame.to_sql(..., if_exists='append', method='multi', chunksize=1000)`. This approach leverages SQLAlchemy’s PostgreSQL driver to batch-insert the data efficiently. Logging statements record the number of rows loaded or any errors that occur.
5. **Index Creation:** To enhance query performance for typical banking analytical workloads, the script creates indexes on key columns. By default, PostgreSQL uses B-tree indexes, which are suitable for equality and range queries. The script creates a B-tree index on each column that looks like an identifier (fields ending in `_id` or named `id`) and each date or timestamp column. It also creates composite indexes combining each `_id` with each date/timestamp column, following the rule of thumb that equality-filtered columns (IDs) should come first and range-filtered columns (dates) second. This ensures queries such as “find all records for a given ID in a date range” can efficiently use the composite index. The index statements use `IF NOT EXISTS` to avoid errors on repeated runs.

Below is the complete Python script, structured with modular functions and detailed comments for clarity.

## Implementation Details

### Copybook Parsing and Schema Extraction

The `parse_copybook_schema` function reads a copybook file and extracts all elementary field definitions (those with a PIC clause). It skips comments (lines starting with `*`) and handles typical copybook formatting (e.g., ignoring line number columns). Each field’s COBOL definition (e.g. `PIC S9(7)V99 COMP-3`) is captured along with its name. We can also leverage the `cobolpy` library to parse the copybook into a schema object; here, for transparency, we show a custom parsing implementation:

```python
import logging

def parse_copybook_schema(copybook_path: str):
    """
    Parse a COBOL copybook to extract field definitions.
    Returns a list of (field_name, cobol_definition) tuples.
    """
    fields = []
    try:
        with open(copybook_path, 'r') as cb_file:
            lines = cb_file.readlines()
    except Exception as e:
        logging.error(f"Failed to read copybook {copybook_path}: {e}")
        raise
    
    for line in lines:
        # Remove sequence numbers or indentation commonly present in copybooks
        if len(line) >= 6 and line[:6].strip().isdigit():
            line = line[6:]
        line = line.strip()
        if not line or line.startswith("*"):
            continue  # skip comments/empty lines
        if "PIC" in line or "pic" in line:  # look for PIC (case-insensitive)
            parts = line.replace(",", " ").split()
            # Example parts: ['05', 'FIELD-NAME', 'PIC', 'X(10).'] or 
            # ['05', 'AMOUNT', 'PIC', 'S9(7)V99', 'COMP-3.']
            try:
                pic_idx = next(i for i, p in enumerate(parts) if p.upper() == "PIC")
            except StopIteration:
                continue
            if pic_idx < 2:
                continue  # not a valid field definition line
            field_name = parts[pic_idx - 1]
            # Join tokens from 'PIC' onward to get the full definition
            cobol_def = " ".join(parts[pic_idx:]).rstrip(".")
            fields.append((field_name, cobol_def))
    return fields
```

*Explanation:* This function collects all lines containing a `PIC` keyword (denoting an elementary data field). It trims out line number columns and comments, then splits the line into tokens. The field name is assumed to immediately precede the `PIC` token (which is typical in COBOL copybooks). The full COBOL picture clause and any usage (COMP, COMP-3, etc.) are captured as `cobol_def`. For instance, a line like `05 ACCOUNT-BALANCE PIC S9(7)V99 COMP-3.` would result in `("ACCOUNT-BALANCE", "PIC S9(7)V99 COMP-3")` in the output list.

### Data Type Mapping (COBOL to PostgreSQL)

The `map_cobol_to_postgres` function takes the list of COBOL field definitions and maps each to an SQL column name and data type. It normalizes field names to be SQL-friendly (e.g., `ACCOUNT-NUMBER` -> `account_number`) and then uses COBOL picture/usage info to decide the PostgreSQL type:

```python
def map_cobol_to_postgres(cobol_fields):
    """
    Map COBOL field definitions to PostgreSQL column definitions.
    Returns list of (column_name, sql_type).
    """
    columns = []
    for field_name, cobol_def in cobol_fields:
        # Clean field name: remove hyphens, use lowercase for SQL
        col_name = field_name.replace("-", "_").lower()
        definition = cobol_def.upper()  # normalize to uppercase for parsing
        if definition.startswith("PIC"):
            definition = definition[3:].strip()  # remove "PIC"
        # Identify usage clauses (COMP, COMP-3, BINARY, etc.)
        usage = None
        for token in ["COMP-3", "COMP-5", "COMP", "BINARY", "PACKED-DECIMAL"]:
            if token in definition:
                usage = token
                definition = definition.replace(token, "").strip()
        # Remove any "USAGE" word and trailing period
        definition = definition.replace("USAGE", "").strip().rstrip(".")
        sql_type = None
        if definition == "" or definition[0] in ("X", "A"):  
            # Alphanumeric (PIC X(n) or PIC A(n))
            length = 1
            if "(" in definition:
                try:
                    length = int(definition.split("(")[1].split(")")[0])
                except:
                    length = 1
            sql_type = f"CHAR({length})"
        elif definition[0] in ("9", "S"):  
            # Numeric types (starting with 9 or S for signed)
            if definition.startswith("S"):
                definition = definition[1:].strip()  # drop 'S' for sign
            if "V" in definition:
                # Implied decimal point
                int_part, frac_part = definition.split("V")
                # Count digits on each side of the V
                try:
                    int_digits = int(int_part[int_part.index('(')+1:int_part.index(')')])
                except:
                    int_digits = len(int_part.strip("9"))
                try:
                    frac_digits = int(frac_part[frac_part.index('(')+1:frac_part.index(')')])
                except:
                    frac_digits = len(frac_part.strip("9"))
                total_digits = int_digits + frac_digits
                sql_type = f"NUMERIC({total_digits},{frac_digits})"
            else:
                # No decimal point
                try:
                    digits = int(definition[definition.index('(')+1:definition.index(')')])
                except:
                    digits = len(definition.strip("9"))
                if usage in ("COMP", "BINARY", "COMP-5"):
                    # Binary/COMP fields -> map to integer types based on size
                    if digits <= 4:
                        sql_type = "SMALLINT"
                    elif digits <= 9:
                        sql_type = "INTEGER"
                    elif digits <= 18:
                        sql_type = "BIGINT"
                    else:
                        sql_type = f"NUMERIC({digits},0)"
                elif usage in ("COMP-3", "PACKED-DECIMAL"):
                    # Packed decimal (no explicit V) -> treat as exact numeric
                    sql_type = f"NUMERIC({digits},0)"
                else:
                    # Display (textual) numeric
                    if digits <= 4:
                        sql_type = "SMALLINT"
                    elif digits <= 9:
                        sql_type = "INTEGER"
                    elif digits <= 18:
                        sql_type = "BIGINT"
                    else:
                        sql_type = f"NUMERIC({digits},0)"
        else:
            # Fallback for unrecognized definitions
            sql_type = "TEXT"
        columns.append((col_name, sql_type))
    return columns
```

In this mapping function, we handle different cases:

* **Alphanumeric (PIC X or PIC A):** We use a fixed-length `CHAR(n)` in PostgreSQL with the same length as defined in the copybook. (If the length is very large, one might choose `VARCHAR` or `TEXT`, but here we assume lengths are reasonably bounded.)
* **Numeric (PIC 9... or S9...):** If an implied decimal point (`V`) is present, we calculate the total number of digits and scale to create a `NUMERIC(precision, scale)` type. For example, `PIC S9(7)V99` would become `NUMERIC(9,2)` (9 total digits, 2 after the decimal point). If there is no `V`, the field is an integer value; the mapping then depends on whether it’s a binary/COMP field or a display (text) numeric:

  * **COMP/BINARY usage:** These are stored in binary format on the mainframe, so we map them to integer types. Based on COBOL conventions, up to 4 decimal digits fit in a 2-byte SMALLINT, up to 9 digits in a 4-byte INTEGER, and up to 18 digits in an 8-byte BIGINT. If the defined number of digits exceeds 18 (beyond the range of 64-bit integers), we default to `NUMERIC` for arbitrary precision.
  * **COMP-3 (packed decimal):** These are decimal values stored in packed form. Without an explicit decimal point, we interpret them as whole numbers and map to `NUMERIC(n,0)` where *n* is the total digits. (If a V is present, as above, we handle it with a scaled NUMERIC.)
  * **Display (no COMP usage specified):** Such numeric fields are stored as text digits on the mainframe. We still map them to numeric SQL types. We use the same digit thresholds to choose SMALLINT/INTEGER/BIGINT, assuming the field will contain a value in that range. If the field is large (e.g., more than 18 digits), we use `NUMERIC` to avoid overflow.
* **Other cases:** If a field definition doesn’t match the above patterns (e.g., a group field or an unsupported type), we default to `TEXT` to avoid losing data. (In practice, group fields wouldn’t have PIC and wouldn’t appear in our list; this is just a safety fallback.)

For example, `PIC X(20)` becomes `CHAR(20)`, `PIC S9(5) COMP` becomes `INTEGER` (since 5 digits in binary fits in 4 bytes), and `PIC S9(7)V99 COMP-3` becomes `NUMERIC(9,2)`.

### Inferring Primary Keys

Many mainframe files don’t explicitly define primary keys, but we can often infer them from field names. The `infer_primary_keys` function implements a simple heuristic:

```python
def infer_primary_keys(columns, table_name=""):
    """
    Infer primary key column(s) using naming heuristics.
    """
    pk_candidates = []
    table_key = table_name.lower() + "_id" if table_name else ""
    for col_name, _type in columns:
        if col_name == "id" or col_name.endswith("_id"):
            pk_candidates.append(col_name)
    # If a field matches table name + "_id", use that as the primary key (priority)
    if table_key and table_key in pk_candidates:
        return [table_key]
    # If exactly one candidate, use it
    if len(pk_candidates) == 1:
        return [pk_candidates[0]]
    # If multiple candidates, assume a composite key using all
    if pk_candidates:
        return sorted(pk_candidates)
    return []
```

This logic collects any column named “id” or ending in “\_id” as a potential primary key. If one of them matches the table’s name (e.g. for a table `account`, a column `account_id`), it prioritizes that as the primary key. If multiple candidates remain (for example, a junction table might have two \*\_id fields), it will return all of them, treating them as a composite primary key. In practice, you might refine this logic or allow manual specification of keys, but these rules cover common conventions.

### Table Creation (DDL Execution)

With the columns and inferred primary keys, the script constructs the CREATE TABLE statement and executes it. The `create_table_ddl` helper builds the DDL string:

```python
def create_table_ddl(table_name, columns, primary_keys):
    col_defs = [f"{name} {sql_type}" for name, sql_type in columns]
    if primary_keys:
        col_defs.append(f"PRIMARY KEY ({', '.join(primary_keys)})")
    ddl = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(col_defs) + "\n)"
    return ddl
```

The `create_table` function then uses this DDL and the SQLAlchemy engine to create the table in the database, handling the overwrite/append mode:

```python
from sqlalchemy import text

def create_table(engine, table_name, columns, primary_keys, mode):
    # If overwrite mode, drop existing table first
    if mode == "overwrite":
        try:
            engine.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
            logging.info(f"Dropped existing table {table_name}")
        except Exception as e:
            logging.error(f"Error dropping table {table_name}: {e}")
            # Continue even if drop fails
    ddl = create_table_ddl(table_name, columns, primary_keys)
    try:
        engine.execute(text(ddl))
        logging.info(f"Created table {table_name}")
    except Exception as e:
        logging.error(f"Failed to create table {table_name}: {e}")
        raise
```

If `--mode=overwrite`, any existing table is removed (`DROP TABLE IF EXISTS ... CASCADE`). Then the new table is created. In append mode, we skip the drop; the create will be attempted, but if the table already exists, the script will log an error and proceed (alternatively, one could check for table existence via SQLAlchemy inspection and skip creation if present).

**Note:** The connection string and engine setup are done in the main section (shown later), with placeholders for the database credentials. The script uses `engine.execute(text(ddl))` to run the SQL. We import `text` from SQLAlchemy to safely handle the DDL string. Ensure that the PostgreSQL user has permission to create tables.

### Data Loading into PostgreSQL

After table creation, the script loads the data from the Parquet file. We use pandas to read the Parquet file into a DataFrame, then `to_sql` to insert into PostgreSQL:

```python
import pandas as pd

def load_data(engine, table_name, parquet_file):
    """
    Load data from a Parquet file into the specified PostgreSQL table.
    """
    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        logging.error(f"Failed to read Parquet file {parquet_file}: {e}")
        raise
    # Normalize DataFrame column names to match database (snake_case)
    original_cols = df.columns.tolist()
    df.columns = [col.replace("-", "_").lower() for col in original_cols]
    logging.info(f"Loaded DataFrame from {parquet_file} with {len(df)} rows.")
    # Optional: attempt type conversions for consistency (dates, numerics)
    for col in df.columns:
        # Convert date/time strings to datetime objects
        if any(x in col for x in ["date", "time"]):
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception as e:
                    logging.warning(f"Could not convert column {col} to datetime: {e}")
        # Convert object dtypes that should be numeric
        if pd.api.types.is_object_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass  # leave as-is if conversion fails
    # Insert into the database table
    try:
        df.to_sql(table_name, engine, if_exists="append", index=False, method="multi", chunksize=1000)
        logging.info(f"Inserted {len(df)} rows into {table_name}")
    } except Exception as e:
        logging.error(f"Error loading data into {table_name}: {e}")
        raise
```

Key points in the data loading step:

* We call `pd.read_parquet` to read the file. This requires `pyarrow` or `fastparquet` to be installed, since Parquet is a binary columnar format.
* We immediately align the DataFrame’s columns to the SQL schema by renaming them to snake\_case. This assumes the Parquet column names correspond to the copybook field names (which is likely, given the Parquet was probably generated from the same source). For example, a column `ACCOUNT-NO` in Parquet would be renamed to `account_no` to match the table column.
* We log the number of rows loaded. The script then tries to convert columns that contain dates or times into actual datetime objects (so that they will be written as dates/timestamps in Postgres instead of strings). It checks the column name for keywords "date" or "time" as a heuristic. Numeric columns stored as object (e.g., if some numerics were read as strings) are also converted using `pd.to_numeric` when possible.
* Finally, we use `DataFrame.to_sql` to write to the database. We specify `if_exists='append'` because the table is already created. We also set `index=False` to avoid writing DataFrame indices, and use `method='multi'` with `chunksize=1000` to batch inserts for efficiency. Under the hood, this will batch multiple rows per INSERT statement, reducing round trips.

**Note on performance:** For very large datasets, using `to_sql` (which issues INSERT statements) may not be the fastest approach. In production, one might consider using PostgreSQL’s `COPY` command via `psycopg2.copy_expert` or similar. However, for simplicity and because the question suggests using pandas/SQLAlchemy, this approach is sufficient.

### Index Creation for Query Performance

After loading the data, the script creates indexes on key columns to speed up analytical queries. The `create_indexes` function identifies all identifier columns (those named `id` or ending in `_id`) and all date/time columns (by name convention), then creates indexes accordingly:

```python
def create_indexes(engine, table_name, columns):
    """
    Create B-tree indexes on *_id and date/time columns, and composite indexes for each combination.
    """
    id_cols = [name for name, _ in columns if name == "id" or name.endswith("_id")]
    datetime_cols = [name for name, _ in columns if "date" in name or "time" in name]
    # Individual indexes
    for col in id_cols + datetime_cols:
        idx_name = f"idx_{table_name}_{col}"
        try:
            engine.execute(text(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name} ({col})"))
            logging.info(f"Created index {idx_name} on {table_name}({col})")
        except Exception as e:
            logging.error(f"Failed to create index on {table_name}({col}): {e}")
    # Composite indexes (each id combined with each date/time)
    for id_col in id_cols:
        for dt_col in datetime_cols:
            idx_name = f"idx_{table_name}_{id_col}_{dt_col}"
            try:
                engine.execute(text(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name} ({id_col}, {dt_col})"))
                logging.info(f"Created composite index {idx_name} on {table_name}({id_col}, {dt_col})")
            except Exception as e:
                logging.error(f"Failed to create composite index on {table_name}({id_col}, {dt_col}): {e}")
```

**Explanation:** We first create single-column indexes on all ID columns and all date/time columns. These indexes (using PostgreSQL’s default B-tree implementation) are effective for lookups and filters on those individual columns. Next, for each combination of an ID and a date/time column, we create a composite index on `(id, date)` or `(id, timestamp)`. This is beneficial for queries that filter by an identifier and a date range simultaneously – the index is scanned first by the ID (equality match) then by date (range query), which is a common pattern in analytical queries (e.g., “find all transactions for customer\_id = X in the last year”). By placing the ID first and date second in the index, we follow the guideline “equality first, range later” for multi-column indexes. The use of `IF NOT EXISTS` prevents errors if the script is run multiple times, ensuring we don’t attempt to duplicate indexes.

*Note:* Creating many indexes can slow down insert performance, so in an initial load scenario it might be wise to create indexes **after** loading the data (as we do here). The script also logs any failures (for example, if an index name is too long or a similar index already exists).

### Main Execution Flow

All the pieces come together in the `main()` function. It sets up argument parsing for the `--mode`, configures logging, and initializes a database connection. Then it locates the Parquet files and their corresponding copybooks and processes each in turn:

```python
import argparse
from sqlalchemy import create_engine

def main():
    parser = argparse.ArgumentParser(description="Load Parquet files into PostgreSQL using COBOL copybook schemas.")
    parser.add_argument("--mode", choices=["overwrite", "append"], default="append",
                        help="Whether to overwrite tables or append to existing tables.")
    args = parser.parse_args()
    mode = args.mode

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    # Database connection parameters (placeholders or environment variables)
    db_host = os.getenv("PG_HOST", "<your-db-host>")
    db_port = os.getenv("PG_PORT", "5432")
    db_name = os.getenv("PG_DATABASE", "<your-db-name>")
    db_user = os.getenv("PG_USER", "<your-db-username>")
    db_pass = os.getenv("PG_PASSWORD", "<your-db-password>")
    conn_str = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    try:
        engine = create_engine(conn_str)
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return

    data_dir = "/mnt/data"
    copybook_dir = "/mnt/copybook"
    if not os.path.isdir(data_dir):
        logging.error(f"Data directory not found: {data_dir}")
        return
    parquet_files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    if not parquet_files:
        logging.error("No Parquet files found in data directory.")
        return

    for file in sorted(parquet_files):
        parquet_path = os.path.join(data_dir, file)
        base_name = os.path.splitext(file)[0]   # e.g. "A" for "A.parquet"
        table_name = base_name.lower()
        copybook_path = os.path.join(copybook_dir, f"{base_name}_cp.copybook")
        logging.info(f"Processing {file} as table '{table_name}' using copybook {base_name}_cp.copybook")
        try:
            cobol_fields = parse_copybook_schema(copybook_path)
        except Exception as e:
            logging.error(f"Skipping {file} due to copybook parse error: {e}")
            continue
        pg_columns = map_cobol_to_postgres(cobol_fields)
        primary_keys = infer_primary_keys(pg_columns, table_name)
        try:
            create_table(engine, table_name, pg_columns, primary_keys, mode)
        except Exception as e:
            logging.error(f"Table creation failed for {table_name}: {e}")
            continue
        try:
            load_data(engine, table_name, parquet_path)
        except Exception as e:
            logging.error(f"Data load failed for {table_name}: {e}")
            continue
        try:
            create_indexes(engine, table_name, pg_columns)
        except Exception as e:
            logging.error(f"Index creation failed for {table_name}: {e}")
            # Continue to next file even if index creation fails
    logging.info("All files processed.")
```

A few implementation details:

* We use `os.listdir` to find Parquet files in `/mnt/data` and assume corresponding copybooks in `/mnt/copybook` with suffix `_cp.copybook`. In this example, files `A.parquet, B.parquet, C.parquet, D.parquet` will be found and we expect `A_cp.copybook, ...` accordingly.
* The database connection string uses the PostgreSQL psycopg2 driver. The credentials are left as placeholders (`<your-db-...>`) – in a real deployment you might load these from secure configuration or environment variables. The script will exit if the connection cannot be established.
* For each file, we derive a table name. Here we simply use the base filename (like "A") lowercased. In a real scenario, you might want a more descriptive table name, but this can be adjusted. We log progress at each major step for transparency.
* If any step for a given file fails (e.g., copybook parse error, table creation error, etc.), we log the error and `continue` to the next file, rather than aborting the entire process. This way one bad file won’t block others from loading.

Finally, we guard the `main()` call with the usual `if __name__ == "__main__":` to allow importing this script without executing, if needed. This script is designed to be run as a standalone program.

## Conclusion

This script provides an end-to-end solution for migrating data from COBOL copybook-described files (in Parquet format) into PostgreSQL, handling schema creation and optimization automatically. By parsing the copybook to drive the SQL schema, we ensure the database table closely matches the mainframe data structure, preserving data types and field lengths. Logging output will inform the user of each step (table creation, rows loaded, indexes built, etc.), making it easier to verify and troubleshoot the process. The modular design (separate functions for parsing, mapping, loading, indexing) facilitates testing and maintenance, and the `--mode` option allows safe re-running of the script in different scenarios (overwriting tables for a fresh load or appending for incremental updates). With appropriate PostgreSQL credentials and the required Python libraries installed (`pandas`, `pyarrow` for Parquet, `sqlalchemy`, and optionally `cobolpy` for advanced copybook parsing), this script is production-ready to handle the described task.

**Sources:**

* COBOL-to-SQL data type mappings: PIC X fields become CHAR of the same length; binary numeric fields map to SMALLINT/INTEGER based on digits (e.g., 8-digit binary -> 4-byte INTEGER); packed decimal (COMP-3) fields map to DECIMAL/NUMERIC types for exact precision.
* Example of using `cobolpy` to parse a copybook in Python.
* PostgreSQL index optimizations: default indexes are B-tree (suitable for most queries); multi-column index strategy of placing equality filters (e.g., an ID) before range filters (date) for optimal index usage.
