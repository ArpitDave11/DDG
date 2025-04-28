Thanks for confirming! I'll update the full code to use the mounted path `/mnt/dataops/autosys/` instead of the abfss:// path.

I'll keep everything else (Delta table creation, log insertion, flag update, SSH command generation) the same and fully compatible with Databricks mount.

I'll share the ready-to-use version shortly!

# Logging Pipeline with Delta Table on Databricks (Using Mount)

**Scenario:** We need to log file processing events to a Delta Lake table stored on Azure Data Lake, and then trigger external **Autosys** jobs via SSH depending on success or failure. We will use the **mounted** path (`/mnt/dataops/autosys/`) instead of raw `abfss://` URIs for simplicity. (Databricks mount points allow cloud storage to be accessed through normal file paths ([Mounting cloud object storage on Azure Databricks - Azure Databricks | Microsoft Learn](https://learn.microsoft.com/en-us/azure/databricks/dbfs/mounts#:~:text=Azure%20Databricks%20mounts%20create%20a,that%20stores%20the%20following%20information)).)

We'll build a PySpark solution in a Databricks notebook with the following steps:

1. **Create a Delta table** at `/mnt/dataops/autosys/` if it does not already exist.  
2. **Insert log entries** into the Delta table (fields: `file_name`, `execution_timestamp`, `status`, and `autosys_update_flag`). New entries default `autosys_update_flag` to `"No"`.  
3. **Update the autosys flag** to `"Yes"` for a given `file_name` once Autosys has been notified.  
4. **Generate SSH command strings** dynamically for success or failure events (to notify the Autosys scheduler or an external system).  
5. **Demonstrate the functions** with clear example calls in a Databricks notebook.

Let's go through each part with clean, well-documented code.

## 1. Setup: Define Paths and Schema

First, define the mount path and the schema for our Delta log table. We use a StructType for clarity. The mount point (`/mnt/dataops`) is assumed to be pre-configured on the cluster (pointing to the storage container). All reads/writes will use this path.

```python
# Path to the Delta table on the mounted storage
log_table_path = "/mnt/dataops/autosys/"  # Using mount path instead of abfss:// URL

# Define schema for the Delta table
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

log_schema = StructType([
    StructField("file_name", StringType(), True),
    StructField("execution_timestamp", TimestampType(), True),
    StructField("status", StringType(), True),
    StructField("autosys_update_flag", StringType(), True)
])
```

**Note:** Using the mount path means our code accesses storage via a DBFS mount. This abstracts the Azure Data Lake Gen2 path behind a simple `/mnt/...` path, which is often easier to work with and share in notebooks ([Mounting cloud object storage on Azure Databricks - Azure Databricks | Microsoft Learn](https://learn.microsoft.com/en-us/azure/databricks/dbfs/mounts#:~:text=Azure%20Databricks%20mounts%20create%20a,that%20stores%20the%20following%20information)). The Delta table will physically reside in that storage location.

## 2. Creating the Delta Table (if not exists)

Before logging events, ensure the Delta table exists at the specified path. We can check if a Delta table is already present at that location using Delta Lake's API. If not, we create an empty Delta table with the defined schema.

```python
from delta.tables import DeltaTable

# Function to create the Delta table if it doesn't exist
def create_log_table():
    # Create an empty DataFrame with the schema and write as Delta to the path
    empty_df = spark.createDataFrame([], schema=log_schema)
    empty_df.write.format("delta").mode("overwrite").save(log_table_path)
    print(f"Created Delta table at {log_table_path}")

# Check if the delta table exists at the path; create if it doesn't
if not DeltaTable.isDeltaTable(spark, log_table_path):
    create_log_table()
else:
    print("Delta table already exists at the path.")
```

Here we used `DeltaTable.isDeltaTable(spark, path)` to detect existence ([scala - check if delta table exists on a path or not in databricks - Stack Overflow](https://stackoverflow.com/questions/64230204/check-if-delta-table-exists-on-a-path-or-not-in-databricks#:~:text=According%20to%20the%20DeltaTable%27s%20Javadoc%2C,path%20with%20the%20following%20command)). If the path is not a Delta table (or is empty), we write an empty DataFrame with the correct schema to initialize it. This effectively does a **"create if not exists"**. (Alternatively, one could use a SQL DDL command or `CREATE TABLE` syntax in Spark SQL.)

## 3. Inserting Log Entries 

Now define a function to insert a new log entry (a single row) into the Delta table. Each log entry will include the file name, a timestamp, the status ("success" or "fail"), and default the `autosys_update_flag` to "No" (meaning not yet updated to Autosys). We capture the current timestamp for the log entry.

```python
import datetime

def log_entry(file_name: str, status: str):
    """Append a new log entry to the Delta log table with autosys_update_flag = 'No'."""
    # Normalize status to lower-case just for consistency
    status = status.lower()
    # Ensure status is recorded as 'success' or 'fail'
    if status not in ("success", "fail"):
        raise ValueError("Status must be 'success' or 'fail'")
    # Prepare a single-row DataFrame with the log entry
    now_ts = datetime.datetime.now()               # current execution timestamp
    new_entry = [(file_name, now_ts, status, "No")]  # autosys_update_flag default "No"
    df = spark.createDataFrame(new_entry, schema=log_schema)
    # Append to the Delta table
    df.write.format("delta").mode("append").save(log_table_path)
    print(f"Logged entry for file '{file_name}' with status '{status}'.")
```

**How it works:** We create a one-row Spark DataFrame for the new log entry and append it to the Delta table using `mode("append")`. We use Python's `datetime.now()` for the timestamp. (In a real job, you might use `pyspark.sql.functions.current_timestamp()` to let Spark handle timestamps, but using Python datetime is straightforward here.)

After calling `log_entry("some_file.csv", "success")`, the Delta table will get a new row like:

| file_name       | execution_timestamp       | status   | autosys_update_flag |
|-----------------|--------------------------|----------|---------------------|
| some_file.csv   | 2025-04-28 13:16:00.123   | success  | No                  |

*(The timestamp will reflect the actual current time.)*

## 4. Updating the Autosys Flag to "Yes"

When Autosys has been notified about a file's processing (or when we are ready to mark it), we need to update that log entry's `autosys_update_flag` from "No" to "Yes". Delta Lake allows in-place updates on tables, which is one reason we chose Delta for logging (it supports ACID transactions and updates, unlike plain CSV/Parquet). We will use the DeltaTable utility to perform an update on the specific file's record.

```python
def update_flag(file_name: str):
    """Update autosys_update_flag to 'Yes' for the given file_name in the Delta table."""
    delta_tbl = DeltaTable.forPath(spark, log_table_path)
    # Use DeltaTable.update to set autosys_update_flag = 'Yes' where file_name matches
    delta_tbl.update(
        condition = f"file_name = '{file_name}'",
        set = { "autosys_update_flag": "'Yes'" }
    )
    print(f"Updated autosys_update_flag to 'Yes' for file '{file_name}'.")
```

In the above, `DeltaTable.forPath` loads the Delta table and we call `update` with a SQL-formatted condition string. We set the `autosys_update_flag` to `'Yes'` for any row matching the given `file_name`. (Delta Lake’s Python API allows using a condition string or Spark `col` expressions for updates ([Table deletes, updates, and merges — Delta Lake Documentation](https://docs.delta.io/latest/delta-update.html#:~:text=,%7D)). We chose the SQL string for brevity.)

After calling `update_flag("some_file.csv")` on the earlier example, that row in the Delta table becomes:

| file_name       | execution_timestamp       | status   | autosys_update_flag |
|-----------------|--------------------------|----------|---------------------|
| some_file.csv   | 2025-04-28 13:16:00.123   | success  | Yes                 |

Now the log indicates that the Autosys system has been (or will be) updated for that file.

## 5. Generating SSH Command Strings for Notifications

For external notifications, we create a function to build the SSH command string dynamically based on status. Typically, an Autosys or external scheduler might be updated by running a remote command or script via SSH. Our function will produce the appropriate SSH command given a file name and status:

```python
def get_ssh_command(file_name: str, status: str, user: str, host: str) -> str:
    """
    Generate an SSH command to notify an external system (e.g., Autosys) 
    about the processing result of a file.
    """
    status = status.lower()
    if status == "success":
        # Example: notify success
        return f"ssh {user}@{host} 'echo \"Processing of {file_name} completed SUCCESSFULLY\"'"
    else:
        # Handle "fail" or any other status as failure
        return f"ssh {user}@{host} 'echo \"Processing of {file_name} FAILED\"'"
```

**Explanation:** This function returns a string that could be executed in a shell. We insert the `file_name` and a message reflecting success or failure. In a real scenario, instead of a simple `echo`, this could call an Autosys CLI command or trigger a script. For example, you might replace the echo with an Autosys `sendevent` command or a custom notification script. The key point is that the command includes the dynamic file name and status.

## 6. Example Usage in a Databricks Notebook

Finally, let's demonstrate how these functions would be used in order, as if running in a Databricks notebook. Each step would typically be in its own cell:

**Step 1: Create the Delta table (if not exists)**  
Run the setup and creation code (from Steps 1-2 above) to ensure the Delta table is ready:

```python
# Ensure Delta table is initialized
if not DeltaTable.isDeltaTable(spark, log_table_path):
    create_log_table()
```

**Step 2: Log a file processing event** – for example, a successful processing of `customer_data.csv`:

```python
log_entry("customer_data.csv", "success")
# Output: Logged entry for file 'customer_data.csv' with status 'success'.
```

This appends a new row to the Delta table with `autosys_update_flag = 'No'`. 

**Step 3: Update the Autosys flag** after notifying Autosys that `customer_data.csv` was processed:

```python
update_flag("customer_data.csv")
# Output: Updated autosys_update_flag to 'Yes' for file 'customer_data.csv'.
```

Now the log entry for `customer_data.csv` is marked as updated (`Yes`). (Under the hood, this used a Delta Lake transaction to update the specific row.)

**Step 4: Generate an SSH command for the success case** of `customer_data.csv` to illustrate what would be run:

```python
ssh_cmd_success = get_ssh_command("customer_data.csv", "success", user="dataops_user", host="autosys-server.mycompany.com")
print(ssh_cmd_success)
# Output: ssh dataops_user@autosys-server.mycompany.com 'echo "Processing of customer_data.csv completed SUCCESSFULLY"'
```

You could take this string and execute it (e.g., via a `%sh` cell in Databricks or using Python’s `os.system`/`subprocess` module) to perform the actual SSH call. In practice, replace the `echo` with the real command or script that informs Autosys of the success.

**Step 5: (Optional) Log a failure scenario** – e.g., processing of `orders.csv` failed – and generate the SSH command for failure:

```python
log_entry("orders.csv", "fail")
update_flag("orders.csv")  # If we also mark it (this depends on workflow, possibly skip for failures)
ssh_cmd_fail = get_ssh_command("orders.csv", "fail", user="dataops_user", host="autosys-server.mycompany.com")
print(ssh_cmd_fail)
# Output: ssh dataops_user@autosys-server.mycompany.com 'echo "Processing of orders.csv FAILED"'
```

*(In a real failure case, you might not call `update_flag` immediately, or at all, depending on whether Autosys should only be flagged on success. Here we show it for completeness — in practice, the logic could differ.)*

---

By following these steps in order, we have:

- Created a **Delta Lake table** on the mounted storage path for logging.
- Inserted entries to log file processing events with **timestamp and status**.
- Performed an **update** on a Delta table entry to mark it as handled (leveraging Delta Lake's update capabilities).
- Constructed dynamic **SSH command strings** to notify external systems of success or failure.
- Demonstrated the usage with example function calls in a Databricks notebook format.

Each code block is self-contained and uses best practices (like using the mount for storage access, checking for table existence with Delta APIs, and using parameterized functions for reuse). This approach ensures an **end-to-end logging and notification pipeline** that is easy to maintain and integrate with both Spark processes and external job schedulers.

