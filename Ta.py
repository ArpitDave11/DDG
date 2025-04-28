Thank you! I'll now update the full code so it:
- Uses Azure default credentials to access ADLS Gen2.
- Points the Delta path to: `abfss://dataops@adlseus2devdhubeis1.dfs.core.windows.net/autosys/`
- Works cleanly in Azure Databricks with managed identity.
- Updates all functions accordingly.

I’ll share the updated, ready-to-run version shortly!

# Updated PySpark Code for Azure Databricks (ADLS Gen2 with Default Credentials)

**Assumptions:** The Azure Databricks workspace/cluster is already configured to use **Azure default credentials** (e.g. managed identity or passthrough) for ADLS Gen2 access. This means you can directly use the `abfss://` path without setting any credentials in the code. The cluster's identity has proper permissions on the storage. 

Below we provide functions to handle the required operations on a Delta table stored at the ADLS path `abfss://dataops@adlseus2devdhubeis1.dfs.core.windows.net/autosys/`. The code is organized into tasks:

- **Create Delta Table** (if it doesn't exist).
- **Insert Log Entry** (with `file_name`, `execution_timestamp`, `status`).
- **Update `autosys_update_flag`** (set to "Yes" for a given `file_name`).
- **Generate SSH Command** strings (different for success vs failure).
- **Example usage** in a Databricks notebook.

## 1. Setup and Table Creation

First, import necessary libraries and define a function to create the Delta table if it does not exist. The table schema includes `file_name`, `execution_timestamp`, `status`, and `autosys_update_flag`. We assume the flag is a string field that will be "Yes" or "No". 

```python
from delta.tables import DeltaTable
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from pyspark.sql import functions as F

# Define the ADLS Gen2 path for the Delta table
delta_path = "abfss://dataops@adlseus2devdhubeis1.dfs.core.windows.net/autosys/"

def create_autosys_table_if_not_exists():
    """Create the Delta table at delta_path if it does not already exist."""
    # Check if a Delta table already exists at the path
    if not DeltaTable.isDeltaTable(spark, delta_path):
        # Define the schema for the table
        schema = StructType([
            StructField("file_name", StringType(), True),
            StructField("execution_timestamp", TimestampType(), True),
            StructField("status", StringType(), True),
            StructField("autosys_update_flag", StringType(), True)
        ])
        # Create an empty DataFrame with the schema
        empty_df = spark.createDataFrame([], schema)
        # Write the empty dataframe to initialize the Delta table
        empty_df.write.format("delta").mode("overwrite").save(delta_path)
        print(f"Created Delta table at {delta_path}")
    else:
        print(f"Delta table already exists at {delta_path}")
```

**Note:** Because we use Azure default credentials, we do **not** set any Spark configs for keys or tokens. The `spark` session will use the cluster's identity to authorize access to the `abfss` path.

## 2. Inserting a Log Entry

Next, we create a function to insert a new log entry (a single row) into the Delta table. It takes `file_name`, `execution_timestamp`, and `status` ("success" or "fail") as parameters. On insert, it will set `autosys_update_flag` to `"No"` by default (since the update flag will be turned "Yes" later when appropriate).

```python
def insert_log(file_name: str, execution_timestamp, status: str):
    """
    Insert a new log entry into the Delta table.
    Parameters:
      file_name (str): Name of the file.
      execution_timestamp (datetime or str): Execution timestamp.
      status (str): "success" or "fail".
    """
    # Ensure the table exists before inserting
    create_autosys_table_if_not_exists()
    # Create a DataFrame for the new log entry
    new_data = [(file_name, execution_timestamp, status, "No")]
    columns = ["file_name", "execution_timestamp", "status", "autosys_update_flag"]
    new_df = spark.createDataFrame(new_data, schema=None).toDF(*columns)
    # Append the new row to the Delta table
    new_df.write.format("delta").mode("append").save(delta_path)
    print(f"Inserted log for file '{file_name}' with status='{status}'.")
```

**Details:** We use `mode("append")` to add the new row to the Delta table. The `execution_timestamp` can be a Python `datetime` object or a timestamp string – Spark will handle converting it to a timestamp if the schema is specified. If the table doesn't exist yet, we call the creation function to initialize it.

## 3. Updating the `autosys_update_flag`

We provide a function to update the `autosys_update_flag` to "Yes" for a given `file_name` after the corresponding process is completed. It uses Delta Lake's upsert capabilities to update the specific row.

```python
def update_autosys_flag(file_name: str):
    """
    Update the autosys_update_flag to "Yes" for the given file_name.
    """
    # Reference the Delta table
    delta_table = DeltaTable.forPath(spark, delta_path)
    # Perform the update where file_name matches
    delta_table.update(
        condition = F.col("file_name") == file_name,
        set = { "autosys_update_flag": F.lit("Yes") }
    )
    print(f"Updated autosys_update_flag to 'Yes' for file '{file_name}'.")
```

**How it works:** We load the Delta table from the path using `DeltaTable.forPath`. Then we call `.update()` with a condition to match the specific `file_name` and set the new value of `autosys_update_flag` to `"Yes"`. This will only affect rows where the `file_name` matches the input.

## 4. Generating SSH Command Strings for Success/Failure

Finally, we create a function to generate an SSH command string based on the status. This can be used to trigger external processes (e.g., notify an Autosys job scheduler on a remote machine) after logging. The command is constructed dynamically: if the status is "success", it might call a different remote script or command than if the status is "fail".

```python
def generate_ssh_command(file_name: str, status: str) -> str:
    """
    Generate an SSH command string based on status.
    Returns a command to run on a remote server for "success" or "fail".
    """
    remote_user = "dataops_user"      # placeholder remote username
    remote_host = "autosys.server.com"  # placeholder remote host
    if status.lower() == "success":
        # Command for success scenario (e.g., call success script with file name)
        cmd = f"ssh {remote_user}@{remote_host} 'bash /path/to/success_script.sh {file_name}'"
    else:
        # Command for failure scenario (e.g., call failure script with file name)
        cmd = f"ssh {remote_user}@{remote_host} 'bash /path/to/failure_script.sh {file_name}'"
    return cmd
```

**Note:** Adjust `remote_user`, `remote_host`, and the script paths to your actual environment. In practice, you might include more details in the SSH command (for example, passing the execution timestamp or other context). Here we keep it simple: a placeholder command that invokes either a success or failure script on a remote server with the `file_name` as an argument.

## 5. Example Usage in a Databricks Notebook

The following illustrates how you might use these functions in a Databricks notebook. We use a sample file name and current timestamp for demonstration:

```python
from datetime import datetime

# 1. Create the Delta table if it doesn't exist
create_autosys_table_if_not_exists()

# 2. Insert some log entries
insert_log("file_A.csv", datetime.now(), "success")
insert_log("file_B.csv", datetime.now(), "fail")

# 3. Update the autosys_update_flag for a specific file after processing
update_autosys_flag("file_A.csv")

# 4. Generate SSH command strings for each scenario and print them
cmd_success = generate_ssh_command("file_A.csv", "success")
cmd_fail = generate_ssh_command("file_B.csv", "fail")
print("SSH command on success:", cmd_success)
print("SSH command on failure:", cmd_fail)
```

When you run the above in Databricks:

- The first call will ensure the Delta table is ready at the ADLS path.
- Two log entries are appended to the Delta table (one marked as success, one as fail, both with `autosys_update_flag="No"` initially).
- We then mark `file_A.csv` as updated by setting its flag to "Yes".
- Finally, we output the SSH command strings that you could use to trigger external notifications for the success and failure cases.

Each function prints a confirmation message for clarity. In a production setting, you might replace these prints with proper logging (e.g., using the `logging` module or Databricks notebook logging). 

