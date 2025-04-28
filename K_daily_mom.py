Thanks for confirming! I’ll get started creating the required code:

- A function to create the Delta table in ADLS if it doesn't exist.
- A function to insert a log record based on status (success or failure).
- A function to update the `autosys_update_flag`.
- A function to generate SSH command strings dynamically for success or failure.
- Example code to show how you would call each function inside Databricks.

I’ll prepare clean, ready-to-run PySpark code for you now!

# Delta Lake Logging and SSH Command Utilities in Databricks

## 1. Creating the Delta Table (if not exists)

First, we need to ensure a Delta Lake table exists at the path `/mnt/ingestion/autosys/es` to store our log entries. The table will have the following schema:

- **file_name** – string (name of the file being processed, provided as input)
- **execution_timestamp** – timestamp (when the file was processed, provided as input)
- **data_as_of_date** – date (the current date, which we set at runtime)
- **status** – string (processing status, e.g., "success" or "fail", provided as input)
- **autosys_update_flag** – string (flag to indicate if AutoSys was updated, default "No")

In Databricks, the Spark session (`spark`) is usually readily available. We use the Delta Lake API to check if a Delta table already exists at the given path, and if not, create one with the specified schema. Here’s how to do it: 

```python
from delta.tables import DeltaTable
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DateType

# Define the Delta table path
delta_path = "/mnt/ingestion/autosys/es"

# Define the schema for the Delta table
schema = StructType([
    StructField("file_name", StringType(), True),
    StructField("execution_timestamp", TimestampType(), True),
    StructField("data_as_of_date", DateType(), True),
    StructField("status", StringType(), True),
    StructField("autosys_update_flag", StringType(), True)
])

# Check if a Delta table already exists at the path; if not, create an empty table
if not DeltaTable.isDeltaTable(spark, delta_path):
    # Create an empty DataFrame with the schema and write it as a Delta table
    empty_df = spark.createDataFrame([], schema)
    empty_df.write.format("delta").mode("overwrite").save(delta_path)
    # (Optionally, register the table in the Hive metastore if needed)
```

The code above will create a Delta table at the specified path if it doesn’t exist. We use `DeltaTable.isDeltaTable(spark, path)` to check for existence, and if false, we write an empty DataFrame with the defined schema to initialize the table.

## 2. Function to Insert a Log Entry

Next, we create a function to insert a new log entry (row) into the Delta table. This function will take the **file_name**, **execution_timestamp**, and **status** as parameters and append a new row with those values. The **data_as_of_date** will be set to the current date (in UTC), and **autosys_update_flag** will default to "No" for new entries.

Below is the function definition. It uses the schema and path defined above, and leverages PySpark to append a new row to the Delta table:

```python
from datetime import datetime, timezone

def insert_log(file_name: str, execution_timestamp, status: str) -> None:
    """
    Insert a log entry into the Delta table with the given file name, execution timestamp, and status.
    The data_as_of_date is set to today's date (UTC), and autosys_update_flag defaults to "No".
    """
    # Get current date (UTC) for data_as_of_date
    current_date = datetime.now(timezone.utc).date()
    # Prepare the new row of data as a list of tuples, matching the schema order
    new_row = [(file_name, execution_timestamp, current_date, status, "No")]
    # Create DataFrame for the new row with the same schema
    df = spark.createDataFrame(new_row, schema=schema)
    # Append the new row to the Delta table
    df.write.format("delta").mode("append").save(delta_path)
```

**How this works:** The function converts the input into a single-row DataFrame and then writes it in **append** mode to the Delta table at `delta_path`. We use `datetime.now(timezone.utc).date()` to capture the current date for the `data_as_of_date` field. The `execution_timestamp` can be provided as a Python `datetime` object or a timestamp string (ensure it’s in a format that Spark can interpret as a timestamp if using a string). The schema is reused to ensure the DataFrame’s columns and types match the Delta table.

## 3. Function to Update the `autosys_update_flag`

After processing a file and (optionally) notifying AutoSys, we may need to update the `autosys_update_flag` to "Yes" for that file. This indicates that the external AutoSys scheduler has been updated for the given file’s job. We will create a function that, given a `file_name`, updates the corresponding row in the Delta table by setting its `autosys_update_flag` to "Yes`.

Delta Lake allows updates to tables via the `DeltaTable` utility. We can use the `update` method to set a new value on rows that meet a condition (in this case, matching the file_name). Here is the function:

```python
from pyspark.sql.functions import col, lit

def mark_autosys_updated(file_name: str) -> None:
    """
    Update the autosys_update_flag to "Yes" for the specified file_name in the Delta table.
    """
    # Create a DeltaTable object for the table at delta_path
    delta_table = DeltaTable.forPath(spark, delta_path)
    # Perform an in-place update on the Delta table:
    delta_table.update(
        condition = col("file_name") == lit(file_name),            # find rows with matching file_name
        set = { "autosys_update_flag": lit("Yes") }                # set autosys_update_flag = "Yes"
    )
```

In this function, `DeltaTable.forPath` loads the Delta table for in-place operations. The `update` method applies the condition and sets the new value. We use Spark SQL functions `col` and `lit` to build the condition and assignment. After calling `mark_autosys_updated("some_file")`, any log entry with `file_name == "some_file"` will have its flag updated to "Yes". (Typically, `file_name` entries are unique per ingestion run, so this should affect at most one row.)

## 4. Function to Generate an SSH Command String

Sometimes we need to trigger an AutoSys job or notify a remote service about the success or failure of a job. We can do this by constructing an SSH command string that can be executed to send the notification. The function below generates an SSH command based on three inputs: **status** (either "success" or "fail"), **job_name** (the AutoSys job or process name we are notifying about), and **load_balancer_id** (the host or identifier of the remote system to SSH into, e.g., a load balancer or server address).

The SSH command string will incorporate these parameters. For example, if a job succeeded, we might send a success event; if it failed, a failure event. Here’s a possible implementation:

```python
def generate_ssh_command(status: str, job_name: str, load_balancer_id: str) -> str:
    """
    Generate an SSH command string to notify about job status.
    - status: "success" or "fail"
    - job_name: name of the job to include in the notification
    - load_balancer_id: the host (or load balancer) to SSH into
    """
    status_lower = status.strip().lower()
    if status_lower not in ("success", "fail"):
        raise ValueError("status must be 'success' or 'fail'")
    # Construct the remote command part (could be an AutoSys CLI command or script invocation)
    if status_lower == "success":
        remote_cmd = f"sendevent -E JOB_SUCCESS -J {job_name}"
    else:
        remote_cmd = f"sendevent -E JOB_FAILURE -J {job_name}"
    # Combine into a full SSH command string (using a placeholder user 'autosys_user')
    ssh_command = f"ssh autosys_user@{load_balancer_id} \"{remote_cmd}\""
    return ssh_command
```

In this example, we assume an AutoSys environment where we can use the `sendevent` command to mark a job's status. The function builds the appropriate remote command based on the `status` and then wraps it in an `ssh` call to the given host. We used a hypothetical user `autosys_user` for SSH; in a real scenario, replace this with an actual username or include it as another parameter. The output of `generate_ssh_command` is a string, for example:

- `generate_ssh_command("success", "daily_job_01", "lb-123.mycompany.com")`  
  returns something like:  
  **`ssh autosys_user@lb-123.mycompany.com "sendevent -E JOB_SUCCESS -J daily_job_01"`**

- `generate_ssh_command("fail", "daily_job_01", "lb-123.mycompany.com")`  
  returns:  
  **`ssh autosys_user@lb-123.mycompany.com "sendevent -E JOB_FAILURE -J daily_job_01"`**

You can adjust the remote command template as needed for your environment.

## 5. Example: Inserting Success/Failure Logs in Databricks

Now that we have our utility functions, let's demonstrate how to use them in a Databricks notebook. Below are examples of inserting both a "success" log and a "fail" log for different files, and then updating the AutoSys flag for one of them.

```python
from datetime import datetime, timezone

# Example: Insert a successful processing log
insert_log("datafile1.csv", datetime.now(timezone.utc), "success")

# Example: Insert a failed processing log
insert_log("datafile2.csv", datetime.now(timezone.utc), "fail")

# (After some time, if AutoSys has been updated for datafile1.csv, update the flag)
mark_autosys_updated("datafile1.csv")
```

In the example above:
- We call `insert_log` with a file name (e.g., `"datafile1.csv"`), a timestamp (here we use the current time), and a status. This will add a new row to the Delta table.
- We do this for a success case and a failure case. In practice, you would call this function at the end of your job’s execution, recording whether it succeeded or failed.
- Finally, we demonstrate calling `mark_autosys_updated("datafile1.csv")`. This would flip the `autosys_update_flag` to "Yes" for the entry with `file_name = "datafile1.csv"`, presumably after an AutoSys notification is successfully sent for that file's job.

You can verify the table contents using Spark to ensure the rows are inserted correctly:

```python
# Reading the Delta table to show contents (for verification/debugging)
spark.read.format("delta").load(delta_path).show()
```

This should display the log entries with their statuses and flags.

## 6. Executing SSH Commands from Databricks

Finally, let's discuss how to execute the SSH command string we generated. In a Databricks environment, executing an SSH command directly from a notebook is not straightforward, because the cluster might not allow direct SSH out without proper configuration. However, you have a couple of options:

- **Using the Databricks `%sh` magic**: In a notebook cell, you can use `%sh` to run shell commands. For example, you could do `%sh ssh autosys_user@lb-123.mycompany.com "sendevent -E JOB_SUCCESS -J daily_job_01"`. This requires that the cluster’s driver node has network access and proper SSH keys or credentials configured for the target host.

- **Using a Python SSH library (e.g., `paramiko`)**: This is a more programmatic way to open an SSH connection and run commands. You would need to install `paramiko` (or another library) on the cluster and use it to connect to the remote host.

Below is an example using `paramiko` to run the SSH command. This assumes that SSH key authentication is set up (or you provide a password), and that the `paramiko` library is installed in your environment:

```python
# Example of executing the SSH command using paramiko
import paramiko

ssh_cmd_str = generate_ssh_command("success", "daily_job_01", "lb-123.mycompany.com")
print(f"SSH command to run: {ssh_cmd_str}")

# Parse the generated command to extract user, host, and remote command
# ssh_cmd_str format is: ssh autosys_user@<host> "<remote_cmd>"
user_host, remote_command = ssh_cmd_str.split(" ", 2)[1:]  # split and get user@host and the remote command part
# Remove quotes around the remote command
remote_command = remote_command.strip().strip('"')

# Set up SSH client and connect to the host
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# You may need to use your own credentials/key here
client.connect(hostname="lb-123.mycompany.com", username="autosys_user", key_filename="/path/to/your/private/key")

# Execute the remote command on the host
stdin, stdout, stderr = client.exec_command(remote_command)
print(stdout.read().decode())   # print the output of the remote command
print(stderr.read().decode())   # print any error output (if any)

client.close()
```

In this code, we take the command string from `generate_ssh_command` and break it into the target (`user@host`) and the remote command. We then use `paramiko` to connect to the host and run the remote command directly. (When using `paramiko`, you actually don't need to prefix the command with `ssh user@host` because you establish the connection in code. We included parsing logic to extract the actual remote command to run.)

**Note:** If you don't want to use an external library, and you have SSH access set up on the Databricks driver node, you could try using Python's `subprocess` module or the `%sh` magic. For example:

```python
import subprocess
cmd = generate_ssh_command("fail", "daily_job_01", "lb-123.mycompany.com")
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(f"SSH command failed with error: {result.stderr}")
```

This would attempt to run the SSH command string in a subshell on the driver. However, this approach requires that the driver can directly SSH to the target (with keys or credentials set up). In many cases, using a library like `paramiko` (or the Databricks `%sh` magic for simple scenarios) is more flexible.

---

**Summary:** We created a Delta Lake table to log file processing events, wrote utility functions to insert log entries and update a status flag, and provided a way to construct and (optionally) execute SSH commands to notify external systems (like AutoSys) of job status. These code snippets can be run in a Databricks notebook to maintain an audit log of file ingestions and integrate with an AutoSys scheduling environment. Each piece is modular: you can call `insert_log` whenever a file is processed, use `mark_autosys_updated` after sending a notification, and use `generate_ssh_command` to prepare the command needed to perform that notification. The examples demonstrate how to tie these pieces together in practice.
