Perfect. I’ll now generate complete PySpark-compatible code for the following four scripts:

1. A script that reads table names from the config file, parses the JSON schema files, and generates DDL + index SQLs into `.sql` files per table.
2. A one-time script that reads and executes the generated `.sql` files to create tables and indexes in PostgreSQL.
3. A setup script that builds a JSON registry file containing column rename and drop mappings for each table, based on the `_TEXT` pattern.
4. A daily load script that reads the latest data per table, uses the control registry to apply column renames and drops, and loads the data into PostgreSQL partitioned child tables with rollback on failure.

I'll let you know when the code for all four scripts is ready.


# PySpark Pipeline Scripts for Data Loading

Below are four fully-developed PySpark-compatible Python scripts addressing the pipeline requirements. Each script is organized with clear steps and commentary for readability. All scripts assume a Databricks environment (with DBFS mounts and a Spark session available) and use the PostgreSQL `share` schema as specified. Please replace placeholder values (like host, user, password) with actual configuration (e.g. via Databricks secrets or environment variables) before running.

## Script 1: Generate SQL DDL Files

**Steps:**

1. **Load Table List:** Read the list of table names from the config file `/dbfs/mnt/dataops/config/dhub_gload_config.txt`. (In Databricks, use the `/dbfs` prefix to access DBFS files with standard Python file I/O.)
2. **Process Each Table:** For each table name:

   * Open the JSON schema file located at `/dbfs/mnt/registry/schema/db2/{table}.json`.
   * Extract the DDL string from the JSON (the JSON contains a `ddl` key with the table’s DB2 DDL).
   * Convert the DDL to PostgreSQL syntax:

     * Add `IF NOT EXISTS` and the `share.` schema qualifier to the `CREATE TABLE` statement.
     * Remove or ignore DB2-specific clauses (e.g. tablespace info, compression, logging) not applicable to Postgres.
     * Append a `PARTITION BY LIST (data_as_of_date)` clause to designate the table as partitioned by the date field.
   * If index definitions are present in the JSON (assumed under an `indexes` or `table_indexes` key), generate corresponding `CREATE INDEX` statements (with `IF NOT EXISTS`) for each index. Use the provided index name and column list, or derive them if necessary.
3. **Write SQL Files:** Save the transformed DDL and index statements into a `.sql` file for each table under `/dbfs/mnt/registry/schema/pgf/{table}.sql`. Each file contains the `CREATE TABLE` statement and any `CREATE INDEX` statements, ready for execution.

```python
# Script 1: Generate SQL DDL Files for PostgreSQL

import json, os, re

# 1. Read table names from the configuration file
config_path = "/dbfs/mnt/dataops/config/dhub_gload_config.txt"
with open(config_path, "r") as f:
    tables = [line.strip() for line in f if line.strip()]

# Ensure output directory exists
output_dir = "/dbfs/mnt/registry/schema/pgf"
os.makedirs(output_dir, exist_ok=True)

for table in tables:
    schema_path = f"/dbfs/mnt/registry/schema/db2/{table}.json"
    if not os.path.exists(schema_path):
        print(f"Schema file not found for table: {table}")
        continue

    # Load the JSON schema for the table
    with open(schema_path, "r") as jf:
        schema = json.load(jf)
    ddl_str = schema.get("ddl", "")
    if not ddl_str:
        print(f"No DDL found in schema for table: {table}")
        continue

    ddl_str = ddl_str.strip().rstrip(";")  # remove trailing semicolon if present

    # 2. Convert DDL to PostgreSQL format
    # Find the table name in the DDL (optionally with schema) and the column definitions
    ddl_upper = ddl_str.upper()
    create_idx = ddl_upper.find("CREATE TABLE")
    if create_idx == -1:
        print(f"Invalid DDL for table {table}, skipping...")
        continue

    # Identify the start of column definitions (first '(' after "CREATE TABLE ...")
    open_paren_idx = ddl_str.find("(", create_idx)
    if open_paren_idx == -1:
        print(f"No column definitions found in DDL for table {table}")
        continue

    # Determine the table name without any existing schema prefix
    name_segment = ddl_str[create_idx + len("CREATE TABLE"): open_paren_idx].strip()
    if "." in name_segment:
        # Remove existing schema qualifier if present (e.g. "SCHEMA.TABLE")
        table_name_only = name_segment.split(".", 1)[1].strip()
    else:
        table_name_only = name_segment

    # Remove any DB2-specific clauses after the column definitions.
    # Find the matching closing parenthesis for the opening parenthesis of the columns.
    close_paren_idx = None
    depth = 0
    for i in range(open_paren_idx, len(ddl_str)):
        ch = ddl_str[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                close_paren_idx = i
                break
    if close_paren_idx is None:
        print(f"Could not find end of column definitions for {table}")
        continue

    columns_def = ddl_str[open_paren_idx:close_paren_idx+1]  # include the parentheses
    # Remove known DB2-specific table properties that might follow the columns list
    after_cols = ddl_str[close_paren_idx+1:]
    # (Examples: "IN <tablespace>", "COMPRESS YES", "DATA CAPTURE", etc. are ignored)
    # Strip out partitioning in DB2 DDL if present (we will add PG partitioning)
    if "PARTITION BY" in after_cols.upper():
        # Truncate anything from "PARTITION BY" onward
        part_index = after_cols.upper().find("PARTITION BY")
        after_cols = after_cols[:part_index]
    # We won't include any content in `after_cols` for PG DDL (skip tablespace or storage clauses)

    # Construct the PostgreSQL DDL
    pg_table_name = f"share.{table_name_only}"
    pg_ddl = f"CREATE TABLE IF NOT EXISTS {pg_table_name} {columns_def} PARTITION BY LIST (data_as_of_date);"

    # 3. Append index creation statements if defined in schema
    index_statements = []
    idx_info = None
    if "table_indexes" in schema:   # check both possible keys
        idx_info = schema["table_indexes"]
    elif "indexes" in schema:
        idx_info = schema["indexes"]
    if idx_info:
        if isinstance(idx_info, list):
            for idx in idx_info:
                if isinstance(idx, str):
                    idx_sql = idx.strip().rstrip(";")
                    # Ensure the index SQL targets the 'share' schema and uses IF NOT EXISTS
                    # Replace schema in index SQL if present:
                    idx_sql_upper = idx_sql.upper()
                    if "CREATE INDEX" in idx_sql_upper:
                        # If an index name is provided in the SQL, ensure proper schema qualification for table
                        # (Assume index SQL in JSON might be like "CREATE INDEX X ON SCHEMA.TABLE...")
                        idx_sql = re.sub(r"ON\s+[^.]+\.([^ (]+)", f"ON {pg_table_name}", idx_sql, flags=re.IGNORECASE)
                        # Add IF NOT EXISTS if not present
                        if "IF NOT EXISTS" not in idx_sql_upper:
                            idx_sql = idx_sql.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS")
                    index_statements.append(idx_sql + ";")
                elif isinstance(idx, dict):
                    # JSON object with index details
                    idx_name = idx.get("name")
                    cols = idx.get("columns") or idx.get("cols")  # assume 'columns' list
                    unique_flag = idx.get("unique", False)
                    if cols:
                        col_list = ", ".join(cols)
                    else:
                        continue  # no columns, skip
                    if not idx_name:
                        # Generate an index name if not provided
                        # e.g., table_col1_col2_idx (limited to reasonable length if needed)
                        idx_name = f"{table_name_only}_{cols[0]}_idx"
                    uniq = "UNIQUE " if unique_flag else ""
                    idx_sql = (f"CREATE {uniq}INDEX IF NOT EXISTS {idx_name} "
                              f"ON {pg_table_name} ({col_list});")
                    index_statements.append(idx_sql)
        else:
            # If idx_info is a single dict or single string (not in a list)
            # handle similar to above
            if isinstance(idx_info, str):
                idx_sql = idx_info.strip().rstrip(";")
                if "CREATE INDEX" in idx_sql.upper():
                    idx_sql = re.sub(r"ON\s+[^.]+\.([^ (]+)", f"ON {pg_table_name}", idx_sql, flags=re.IGNORECASE)
                    if "IF NOT EXISTS" not in idx_sql.upper():
                        idx_sql = idx_sql.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS")
                index_statements.append(idx_sql + ";")
            elif isinstance(idx_info, dict):
                idx_name = idx_info.get("name")
                cols = idx_info.get("columns") or idx_info.get("cols")
                unique_flag = idx_info.get("unique", False)
                if cols:
                    col_list = ", ".join(cols)
                else:
                    col_list = ""
                if not idx_name:
                    idx_name = f"{table_name_only}_{cols[0]}_idx" if cols else f"{table_name_only}_idx"
                uniq = "UNIQUE " if unique_flag else ""
                idx_sql = (f"CREATE {uniq}INDEX IF NOT EXISTS {idx_name} "
                          f"ON {pg_table_name} ({col_list});")
                index_statements.append(idx_sql)

    # Combine table DDL and index statements
    full_sql = pg_ddl + "\n"
    for idx_sql in index_statements:
        full_sql += idx_sql + "\n"

    # 4. Write the combined SQL to the output .sql file
    out_path = f"/dbfs/mnt/registry/schema/pgf/{table}.sql"
    with open(out_path, "w") as outfile:
        outfile.write(full_sql)

    print(f"Generated DDL for table {table}: {out_path}")
```

**Explanation:** The script above reads each table’s DB2 DDL from JSON and outputs a PostgreSQL-compatible DDL. It ensures the `share` schema is used and that the table is created as partitioned by `data_as_of_date`. Any index information in the JSON is also converted into `CREATE INDEX IF NOT EXISTS` statements appended to the SQL file. The use of the `/dbfs` prefix allows standard Python file operations on DBFS-mounted files, making it straightforward to read the config and schema JSON files and to write out the .sql files.

## Script 2: Execute DDLs in PostgreSQL

**Steps:**

1. **Initialize Database Connection:** Establish a JDBC/ODBC connection to the PostgreSQL database using credentials from configuration (host, port, database name, user, password). In Databricks, you might retrieve these from a secret scope or environment variables. This script uses the `psycopg2` driver to execute DDL statements.
2. **Iterate Tables:** Read the same table list from `dhub_gload_config.txt`. For each table:

   * Open the corresponding SQL file from `/dbfs/mnt/registry/schema/pgf/{table}.sql` (generated by Script 1).
   * Read the full SQL script (which includes `CREATE TABLE` and index creation statements).
   * Split the script into individual SQL statements (on semicolons) and execute each statement via the PostgreSQL cursor.
   * Use `IF NOT EXISTS` in the SQL to avoid errors if the table or index already exists (thus effectively skipping creation if already done).
3. **Commit and Close:** Commit each table’s DDL transaction (ensuring the table is created) and proceed. Finally, close the database connection.

```python
# Script 2: Execute DDL statements in PostgreSQL

import psycopg2

# Database connection parameters (to be filled with actual config or secrets)
DB_HOST = "<your_postgres_host>"
DB_PORT = 5432
DB_NAME = "<your_database_name>"
DB_USER = "<your_username>"
DB_PASS = "<your_password>"

# Connect to PostgreSQL
conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
conn.autocommit = False  # Use manual commit to control transaction
cur = conn.cursor()

# Load table list from config
config_path = "/dbfs/mnt/dataops/config/dhub_gload_config.txt"
with open(config_path, "r") as f:
    tables = [line.strip() for line in f if line.strip()]

for table in tables:
    sql_file_path = f"/dbfs/mnt/registry/schema/pgf/{table}.sql"
    if not os.path.exists(sql_file_path):
        print(f"SQL file for {table} not found, skipping.")
        continue

    # Read the SQL script file
    with open(sql_file_path, "r") as sf:
        sql_script = sf.read()
    if not sql_script.strip():
        print(f"No SQL commands in file for {table}, skipping.")
        continue

    # Split the script into individual statements by semicolon
    # Filter out any empty statements (after stripping whitespace)
    statements = [stmt.strip() for stmt in sql_script.split(";") if stmt.strip()]
    try:
        for stmt in statements:
            cur.execute(stmt)
        conn.commit()  # commit after executing all statements for the table
        print(f"Executed DDL for table {table}")
    except Exception as e:
        # If any DDL execution fails, roll back and log the error for this table
        conn.rollback()
        print(f"Failed to execute DDL for {table}: {e}")

# Clean up database connection
cur.close()
conn.close()
```

**Explanation:** This script connects to PostgreSQL and runs the DDL SQL files created in the previous step. It uses `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` to safely attempt creation without errors if the objects already exist. Each table’s DDL statements are executed in sequence; if a table already exists, the `IF NOT EXISTS` clause ensures the script skips creating it (and similarly for indexes). The script commits each table’s creation to the database before moving on, and any errors cause a rollback for that table’s transaction (without stopping the whole script, so other tables can still be processed). This approach allows idempotent schema setup: running it multiple times will simply confirm the objects exist, without duplication or error.

## Script 3: Build Column Rename/Drop Registry

**Steps:**

1. **Load Table Schemas:** For each table in the config list, load the same JSON schema file used in Script 1 (`/dbfs/mnt/registry/schema/db2/{table}.json`).
2. **Identify `_TEXT` Columns:** Parse the schema to get the list of column names. If the JSON contains a structured list of columns (e.g. under a key like `"columns"`), use that. Otherwise, parse the DDL string to extract column names. Then, find any columns whose names end with `_TEXT`.
3. **Decide Rename or Drop:** For each column ending in `_TEXT`:

   * If the corresponding base column (the name without `_TEXT`) **also exists** in the table, mark the `_TEXT` column to be **dropped** (since the data is duplicated in another column).
   * If the base name **does not exist** in the table schema, plan to **rename** the `_TEXT` column to the base name (since it’s the only representation of that field, we want to remove the suffix).
4. **Build Registry Mapping:** Aggregate the results into a single JSON structure with two top-level keys: `"rename"` and `"drop"`. Under `"rename"`, map each table name to a dictionary of `{ "old_column_name": "new_column_name", ... }` for columns to rename. Under `"drop"`, map each table name to a list of column names to drop.
5. **Write Registry File:** Save this mapping as a JSON file at `/dbfs/mnt/dataops/config/column_rename_drop.json`. This file will be used in the next script to apply the transformations.

```python
# Script 3: Build JSON registry for column renames and drops

import json, os, re

# Load table list from config file
config_path = "/dbfs/mnt/dataops/config/dhub_gload_config.txt"
with open(config_path, "r") as f:
    tables = [line.strip() for line in f if line.strip()]

rename_registry = {}
drop_registry = {}

for table in tables:
    schema_path = f"/dbfs/mnt/registry/schema/db2/{table}.json"
    if not os.path.exists(schema_path):
        print(f"Schema file not found for table: {table}")
        continue
    with open(schema_path, "r") as jf:
        schema = json.load(jf)

    # 2. Get list of all column names from the schema
    columns = []
    if "columns" in schema and isinstance(schema["columns"], list):
        # If schema JSON has a "columns" list of column definitions
        for col in schema["columns"]:
            # Assuming each column entry has a 'name' field
            col_name = col.get("name") or col.get("column_name")
            if col_name:
                columns.append(col_name)
    else:
        # If no explicit column list, parse the DDL string to extract column names
        ddl_str = schema.get("ddl", "")
        ddl_str = ddl_str.strip()
        # Find the columns section inside the CREATE TABLE statement
        open_paren_idx = ddl_str.find("(")
        if open_paren_idx != -1:
            # Find matching closing parenthesis for the column list
            depth = 0
            close_paren_idx = None
            for i, ch in enumerate(ddl_str[open_paren_idx:], start=open_paren_idx):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        close_paren_idx = i
                        break
            if close_paren_idx:
                cols_part = ddl_str[open_paren_idx+1:close_paren_idx]
            else:
                cols_part = ddl_str[open_paren_idx+1:]
            # Split by commas at top level (not inside type definitions)
            cols_def_list = []
            depth = 0
            current = ""
            for ch in cols_part:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    if depth > 0:
                        depth -= 1
                if ch == ',' and depth == 0:
                    cols_def_list.append(current.strip())
                    current = ""
                else:
                    current += ch
            if current:
                cols_def_list.append(current.strip())
            # Extract column name from each column definition
            for col_def in cols_def_list:
                if not col_def:
                    continue
                col_def = col_def.strip()
                # Skip constraints or keys definitions within the column list, if any
                if col_def.upper().startswith("PRIMARY KEY") or col_def.upper().startswith("UNIQUE") \
                   or col_def.upper().startswith("CONSTRAINT") or col_def.upper().startswith("KEY "):
                    continue
                # Column definition format: [column name] [type] ...
                parts = col_def.split()
                if not parts:
                    continue
                col_name = parts[0].strip().strip('"')
                columns.append(col_name)

    # 3. Identify _TEXT columns and determine drops/renames
    columns_set = set(columns)
    to_drop = []
    to_rename = {}
    for col in columns:
        if col.endswith("_TEXT"):
            base_col = col[:-5]  # remove the suffix "_TEXT"
            if base_col in columns_set:
                # Base column exists, mark this _TEXT column to drop
                to_drop.append(col)
            else:
                # Base column does not exist, we will rename this column to the base name
                to_rename[col] = base_col

    # Add to registry dictionaries if any entries exist
    if to_drop:
        drop_registry[table] = to_drop
    if to_rename:
        rename_registry[table] = to_rename

# 4. Combine rename and drop mappings into one registry structure
registry = {
    "rename": rename_registry,
    "drop": drop_registry
}

# 5. Write the registry JSON to file
out_path = "/dbfs/mnt/dataops/config/column_rename_drop.json"
with open(out_path, "w") as out_f:
    json.dump(registry, out_f, indent=4)

print(f"Column rename/drop registry written to {out_path}")
```

**Explanation:** This script inspects each table’s schema to find columns that should be renamed or dropped. Columns ending in `_TEXT` are handled according to whether their base name exists:

* If the base column exists (e.g. both `CODE` and `CODE_TEXT` are present), then `CODE_TEXT` is marked for dropping, as it’s presumably a duplicate textual representation.
* If the base column does not exist (only `CODE_TEXT` is present), then `CODE_TEXT` is slated to be renamed to `CODE` (removing the suffix).

The output JSON file has separate sections for `rename` and `drop`. For example, it might look like:

```json
{
  "rename": {
    "table1": { "ADDRESS_TEXT": "ADDRESS" },
    "table2": { "EMP_NAME_TEXT": "EMP_NAME" }
  },
  "drop": {
    "table1": ["STATUS_TEXT", "ZIP_TEXT"],
    "table3": ["CATEGORY_TEXT"]
  }
}
```

This consolidated registry will be read by the data load script to apply the appropriate DataFrame transformations.

## Script 4: Daily Data Load Script

**Steps:**

1. **Load Config and Registry:** Read the table list from `dhub_gload_config.txt` and load the JSON registry file `column_rename_drop.json` produced by Script 3. This registry provides the column rename and drop mappings for each table.
2. **For Each Table:** Perform the following:

   * **Find Latest Data Folder:** Look under `/mnt/asis/eod/{table}/` for the latest `data_as_of_date=YYYYMMDD` folder. The script uses the naming convention to pick the folder with the greatest date (e.g., by lexicographic max of folder names, which works for zero-padded dates). (One could also use modification timestamps to find the newest folder, but using the date in the path is straightforward here.)
   * **Read Parquet Data:** Once the latest folder path is determined, read the Parquet file(s) for that date into a Spark DataFrame.
   * **Apply Renames/Drops:** Using the registry mapping, rename any columns that need to be renamed and drop any columns marked for dropping, using DataFrame transformations. This ensures the DataFrame’s schema matches the desired target schema as closely as possible.
   * **Cast Columns to DDL Types:** Load the expected schema (from the JSON schema file) and cast each column in the DataFrame to the target data type defined in the PostgreSQL DDL. For example, if a column should be INTEGER but is read as string, cast it to integer. If it should be a decimal, cast to the appropriate precision/scale DecimalType, etc. This step ensures that the data types align with the Postgres table definition when inserting.
   * **Add Partition Column:** Add a column `data_as_of_date` to the DataFrame (as an integer) with the value of the partition date. This will be used as the partition key in the Postgres table.
   * **Create Partition Table:** Form the SQL to create the new child partition in Postgres for this date (e.g., `CREATE TABLE IF NOT EXISTS share.{table}_{YYYYMMDD} PARTITION OF share.{table} FOR VALUES IN (YYYYMMDD)`). Connect to Postgres (using `psycopg2` or similar) and execute this statement. Commit the creation so the partition exists for loading.
   * **Load Data via JDBC:** Use Spark’s JDBC DataFrame writer to load the transformed DataFrame into the newly created partition table. The mode is "append" since we are inserting new data. The JDBC connection URL and properties (user, password, driver) must be specified. (Spark’s `df.write.jdbc` will handle batching the insert behind the scenes.)
   * **Error Handling:** If the load fails (e.g., due to a data issue or database error), catch the exception and perform cleanup: drop the partition table that was just created to roll back the partial load. This keeps the database consistent (no empty partition table should remain if the load didn’t succeed). Log the error for further investigation.
   * **Repeat for all tables in the list.**

```python
# Script 4: Daily data load to PostgreSQL partitioned tables

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType
import json, os
import psycopg2

# Initialize Spark session (if not already available)
spark = SparkSession.builder.getOrCreate()

# 1. Load table list and column rename/drop registry
config_path = "/dbfs/mnt/dataops/config/dhub_gload_config.txt"
with open(config_path, "r") as f:
    tables = [line.strip() for line in f if line.strip()]

registry_path = "/dbfs/mnt/dataops/config/column_rename_drop.json"
with open(registry_path, "r") as rf:
    registry = json.load(rf)
rename_registry = registry.get("rename", {})
drop_registry = registry.get("drop", {})

# Database connection info for JDBC and psycopg2 (fill in actual credentials or use secrets)
DB_HOST = "<your_postgres_host>"
DB_PORT = 5432
DB_NAME = "<your_database_name>"
DB_USER = "<your_username>"
DB_PASS = "<your_password>"
jdbc_url = f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}"
jdbc_properties = {
    "user": DB_USER,
    "password": DB_PASS,
    "driver": "org.postgresql.Driver"
}

for table in tables:
    print(f"Processing table: {table}")
    base_path = f"dbfs:/mnt/asis/eod/{table}/"
    # 2. Find the latest data folder by date
    try:
        entries = dbutils.fs.ls(base_path)
    except Exception as e:
        print(f"Error accessing {base_path}: {e}")
        continue
    # Filter for directories named data_as_of_date=YYYYMMDD
    date_dirs = [entry.name.rstrip("/") for entry in entries if entry.name.startswith("data_as_of_date=")]
    if not date_dirs:
        print(f"No data_as_of_date folders found for {table}, skipping.")
        continue
    # Determine the latest date folder (assuming YYYYMMDD format, lexicographic max works for dates)
    latest_folder_name = max(date_dirs)
    date_str = latest_folder_name.split("=")[1]  # extract the YYYYMMDD part
    data_date_int = int(date_str)
    data_path = base_path + latest_folder_name  # full path to the latest folder
    print(f"Latest data folder for {table}: {latest_folder_name}")

    # 3. Read the Parquet data for that date into a Spark DataFrame
    try:
        df = spark.read.parquet(data_path)
    except Exception as e:
        print(f"Failed to read parquet data for {table} at {data_path}: {e}")
        continue

    original_columns = df.columns

    # 4. Apply column renames and drops based on the registry
    cols_to_drop = drop_registry.get(table, [])
    rename_map = rename_registry.get(table, {})
    # Rename columns first
    for old_col, new_col in rename_map.items():
        if old_col in df.columns:
            df = df.withColumnRenamed(old_col, new_col)
            print(f"Renamed column {old_col} -> {new_col} in {table} DataFrame")
    # Drop columns marked for drop
    if cols_to_drop:
        df = df.drop(*[c for c in cols_to_drop if c in df.columns])
        if cols_to_drop:
            print(f"Dropped columns {cols_to_drop} from {table} DataFrame")

    # 5. Cast columns to match the target schema data types
    # Load the schema JSON to get column types (same JSON used earlier)
    schema_path = f"/dbfs/mnt/registry/schema/db2/{table}.json"
    col_types = {}
    if os.path.exists(schema_path):
        with open(schema_path, "r") as jf:
            schema = json.load(jf)
        # If structured column info is available
        if "columns" in schema and isinstance(schema["columns"], list):
            for col in schema["columns"]:
                col_name = col.get("name") or col.get("column_name")
                col_type = col.get("type") or col.get("data_type") or col.get("datatype")
                if not col_name or not col_type:
                    continue
                ct = col_type.upper()
                if ct.startswith("DECIMAL") or ct.startswith("NUMERIC"):
                    # e.g., "DECIMAL(10,2)" -> use DecimalType(10,2)
                    match = re.match(r'(?:DECIMAL|NUMERIC)\s*\((\d+)\s*,\s*(\d+)\)', ct)
                    if match:
                        prec = int(match.group(1)); scale = int(match.group(2))
                        col_types[col_name] = DecimalType(prec, scale)
                    else:
                        # No precision/scale specified, default to a general DecimalType
                        col_types[col_name] = DecimalType(38, 10)  # default precision/scale
                elif ct in ("INTEGER", "INT", "SMALLINT"):
                    # use IntegerType (or ShortType for SMALLINT if needed)
                    col_types[col_name] = "int"
                elif ct == "BIGINT":
                    col_types[col_name] = "long"
                elif ct in ("FLOAT", "DOUBLE", "DOUBLE PRECISION", "REAL"):
                    col_types[col_name] = "double"
                elif ct.startswith("VARCHAR") or ct.startswith("CHAR") or ct == "TEXT" or "CLOB" in ct:
                    col_types[col_name] = "string"
                elif ct == "DATE":
                    col_types[col_name] = "date"
                elif ct.startswith("TIMESTAMP"):
                    col_types[col_name] = "timestamp"
                elif ct in ("BOOLEAN", "BIT"):
                    col_types[col_name] = "boolean"
                else:
                    # default to string for any unrecognized types
                    col_types[col_name] = "string"
        else:
            # If no structured columns list, parse DDL for columns and types
            ddl_str = schema.get("ddl", "")
            ddl_str = ddl_str.strip()
            open_idx = ddl_str.find("(")
            if open_idx != -1:
                # extract column definitions substring from DDL
                depth = 0
                col_defs = []
                current = ""
                for ch in ddl_str[open_idx+1:]:
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        if depth == 0:
                            break  # end of columns section
                        depth -= 1
                    if ch == ',' and depth == 0:
                        col_defs.append(current.strip())
                        current = ""
                    else:
                        current += ch
                if current:
                    col_defs.append(current.strip())
                for col_def in col_defs:
                    if not col_def or col_def.upper().startswith(("PRIMARY KEY", "UNIQUE", "CONSTRAINT", "KEY ")):
                        continue
                    parts = col_def.strip().split()
                    if not parts:
                        continue
                    col_name = parts[0].strip().strip('"')
                    if len(parts) < 2:
                        continue  # no type found
                    col_type = parts[1]
                    # If the type part is like DECIMAL(10,2), it will still be in parts[1] (no space before '(')
                    # If type is two words like "DOUBLE PRECISION", parts[1] might be "DOUBLE" and parts[2] = "PRECISION"
                    if col_type.upper() == "DOUBLE" and len(parts) > 2 and parts[2].upper() == "PRECISION":
                        col_type = "DOUBLE PRECISION"
                    ct = col_type.upper()
                    if ct.startswith("DECIMAL") or ct.startswith("NUMERIC"):
                        match = re.match(r'(?:DECIMAL|NUMERIC)\s*\((\d+)\s*,\s*(\d+)\)', col_def.upper())
                        if match:
                            prec = int(match.group(1)); scale = int(match.group(2))
                            col_types[col_name] = DecimalType(prec, scale)
                        else:
                            col_types[col_name] = DecimalType(38, 10)
                    elif ct in ("INTEGER", "INT", "SMALLINT"):
                        col_types[col_name] = "int"
                    elif ct == "BIGINT":
                        col_types[col_name] = "long"
                    elif ct in ("FLOAT", "DOUBLE PRECISION", "DOUBLE", "REAL"):
                        col_types[col_name] = "double"
                    elif ct.startswith("VARCHAR") or ct.startswith("CHAR") or ct == "TEXT" or "CLOB" in ct:
                        col_types[col_name] = "string"
                    elif ct == "DATE":
                        col_types[col_name] = "date"
                    elif ct.startswith("TIMESTAMP"):
                        col_types[col_name] = "timestamp"
                    elif ct in ("BOOLEAN", "BIT"):
                        col_types[col_name] = "boolean"
                    else:
                        col_types[col_name] = "string"
    else:
        print(f"Schema JSON not found for {table}, skipping type casts.")
        col_types = {}

    # Now cast DataFrame columns to the target types
    for col_name, target_type in col_types.items():
        if col_name in df.columns:
            try:
                if isinstance(target_type, str):
                    df = df.withColumn(col_name, df[col_name].cast(target_type))
                else:
                    # target_type could be a DecimalType or other Spark type object
                    df = df.withColumn(col_name, df[col_name].cast(target_type))
            except Exception as e:
                print(f"Warning: Failed to cast column {col_name} to {target_type} for table {table}: {e}")

    # 6. Add the partition key column (data_as_of_date)
    df = df.withColumn("data_as_of_date", F.lit(data_date_int).cast("int"))
    # Ensure the partition column is the last column (optional, for clarity)
    # Reorder columns if necessary (Spark will preserve the new column at end by default)
    # df = df.select(*[c for c in df.columns if c != "data_as_of_date"], "data_as_of_date")

    # 7. Create the child partition table in PostgreSQL if not exists
    child_table = f"{table}_{date_str}"  # child partition table name, e.g., SALES_20250101
    partition_sql = (f"CREATE TABLE IF NOT EXISTS share.{child_table} "
                     f"PARTITION OF share.{table} FOR VALUES IN ({data_date_int});")
    try:
        pg_conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
        pg_cur = pg_conn.cursor()
        pg_cur.execute(partition_sql)
        pg_conn.commit()
        pg_cur.close()
        pg_conn.close()
        print(f"Created partition table share.{child_table} (if not exists).")
    except Exception as e:
        print(f"Error creating partition table for {table} on {date_str}: {e}")
        continue  # skip loading this table if partition creation failed

    # 8. Load the DataFrame into the PostgreSQL child partition using JDBC
    try:
        df.write.jdbc(url=jdbc_url, table=f"share.{child_table}", mode="append", properties=jdbc_properties)
        print(f"Successfully loaded data for {table} date {date_str} into share.{child_table}.")
    except Exception as e:
        print(f"Error loading data for {table} date {date_str}: {e}")
        # 9. On error, drop the partition table to roll back
        try:
            rollback_conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
            rollback_cur = rollback_conn.cursor()
            rollback_cur.execute(f"DROP TABLE IF EXISTS share.{child_table}")
            rollback_conn.commit()
            rollback_cur.close()
            rollback_conn.close()
            print(f"Dropped partition table share.{child_table} due to load failure.")
        except Exception as drop_err:
            print(f"Warning: Failed to drop partition table share.{child_table}: {drop_err}")
```

**Explanation:** This final script orchestrates the daily load of data into the partitioned PostgreSQL tables. It uses Spark to read and transform the data and then writes to Postgres. Key points to note:

* We find the latest folder by comparing date strings. This leverages the folder naming convention. (In Databricks, one could also use file modification times to find the newest file, but in this case using the date in the path is reliable.)
* Column renaming and dropping are applied to the DataFrame as specified by the registry from Script 3. This ensures we don’t insert redundant `_TEXT` columns into the database.
* Before writing, we cast each column to the expected type defined in the schema. This helps avoid type mismatches when Spark writes to Postgres. For example, if a column is numeric in Postgres but came in as string, we cast it to a numeric type. Spark’s DataFrame `withColumn().cast()` is used for this conversion. We map common SQL types to Spark SQL types or use `DecimalType` for precise numeric types.
* We explicitly add the `data_as_of_date` column with the partition value (as an integer) so that each row carries the partition key.
* We then create a new partition child table in Postgres for that date. The naming convention here is `share.<table>_<YYYYMMDD>`. We use `CREATE TABLE ... PARTITION OF ... FOR VALUES IN (...)` to create the partition. This is executed via psycopg2 immediately before loading the data, so that the partition exists.
* Data loading is done with `df.write.jdbc(...)` which uses the JDBC driver to insert the Spark DataFrame into the Postgres table. We use `mode="append"` because we are adding new data. The connection properties include the JDBC URL, user, password, and driver. (Spark will handle batching the inserts; you can tune batch size or parallelism if needed, but those details are omitted for clarity. By default it will map Spark types to PostgreSQL types appropriately.)
* **Error handling:** If the Spark write throws an exception (for example, a database constraint violation or connectivity issue), the script catches it and uses a psycopg2 connection to drop the newly created child partition table. This clean-up step ensures that a failed load doesn’t leave an empty partition table in the database. After dropping, the script continues with the next table.

Finally, one consideration: when dropping columns from the DataFrame that still exist in the target table, the Postgres table columns will simply be unfilled by the insert. It’s assumed those extra columns are nullable or have defaults so that inserting without them does not cause errors. If they are NOT NULL with no default, the insert would fail. In such cases, you might need to modify the insertion to include all columns (providing nulls for dropped columns). For example, you could add those columns to the DataFrame with `lit(None)` so the schema matches exactly. In this pipeline, we assume the dropped columns are either removed from the schema or made nullable so that omitting them is acceptable.

With this script, each day’s new data (for each table) will be loaded into its corresponding partition in the `share` schema, and any issues will be logged and handled without compromising the rest of the load process.
