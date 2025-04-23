Introduction

This Databricks notebook orchestrates running a shell command on an Azure VM selected from an Azure Load Balancer's backend pool, and logs the job status to Delta. It extends the initial solution with the following enhancements:
Delta Logging with Upsert: Job status is logged to Delta files (Delta Lake format) instead of a traditional table, partitioned by date, with upsert logic to avoid duplicate entries.
Unity Catalog Integration: If the Delta tables do not exist, they are created in the Unity Catalog (e.g. under finance_catalog) at specified storage locations. This ensures the logs are queryable as managed tables.
Logging Schema: The log entries include file_name, job_id, status, autosys_flag, data_as_update (today’s date partition), and timestamp (log entry time).
Notifications Table: A separate Delta table (and underlying file storage) for notifications includes all the above fields plus a notification_flag to track notification status.
Immediate Notification Log Upsert: After writing each status log entry, the same entry is upserted into the notifications Delta table with notification_flag = false (pending notification).
Kafka Consumer Pseudocode: Outline of a Kafka consumer (or streaming job) that:
Polls the notifications table for entries where autosys_flag = true and notification_flag = false.
Sends out a notification for each such entry (e.g. via a Kafka producer or other notification mechanism).

Updates each processed entry’s notification_flag to true to mark it as notified.
Scheduled Execution: This notebook is intended to be triggered by an Azure Data Factory (or Synapse) pipeline every 15 minutes, as well as once daily at 03:00. The daily 3:00 AM run corresponds to an "Autosys" job (signified by autosys_flag = true for special handling and notification), while the 15-minute runs are regular incremental checks (autosys_flag = false by default).
With these features, the notebook can run in a Unity Catalog-enabled workspace and maintain comprehensive logs with notifications. Below, we proceed step-by-step through the implementation.
1. Configuration and Parameter Setup

First, we retrieve input parameters for the load balancer and the shell command to execute on the VM. We also define paths for Delta log storage and ensure the Unity Catalog is set to the correct catalog/schema. If needed, we determine the Autosys flag (true for the daily run at 3:00 AM, false otherwise). We also prepare or retrieve identifiers like job_id and file_name for logging purposes.
# Databricks notebook widgets for input parameters (provided by ADF pipeline or manually set)
dbutils.widgets.text("lb_id", "")            # Azure Load Balancer resource ID or name
dbutils.widgets.text("shell_command", "")    # Shell command to run on the VM
dbutils.widgets.text("job_id", "")           # Optional job identifier (if not provided, will generate)

# Retrieve parameter values
lb_id = dbutils.widgets.get("lb_id")
shell_command = dbutils.widgets.get("shell_command")
job_id = dbutils.widgets.get("job_id")

# Generate a job_id if not provided
import uuid, datetime
if not job_id:
    job_id = str(uuid.uuid4())  # Use a UUID or timestamp-based ID for the job run

# Derive a file_name for logging (e.g., from the shell command or known context)
# Here we take the last part of the command if it looks like a file path, otherwise use the command itself.
if "/" in shell_command:
    file_name = shell_command.rsplit("/", 1)[-1].split()[0]
else:
    file_name = shell_command.split()[0]
    
# Determine autosys_flag: mark True for the daily scheduled run (around 3 AM), else False for regular runs.
current_hour = datetime.datetime.now().hour
autosys_flag = True if current_hour == 3 else False

print(f"Parameters - LB: {lb_id}, Command: '{shell_command}', Job ID: {job_id}, File: {file_name}, Autosys flag: {autosys_flag}")
Explanation: We use Databricks widgets to supply the Load Balancer ID and shell command (these would be set by the ADF pipeline triggers). We also include an optional job_id widget for an external job identifier (if the scheduler provides one; otherwise we generate a UUID for uniqueness). The file_name is inferred from the shell command (this could represent the script or process being run). The autosys_flag is set to True for the special daily run (here simplistically determined by hour == 3; in practice, the pipeline could pass a parameter or use a separate pipeline for the daily job). These values will be used in logging.
2. Identify Active VM from Load Balancer Backend

Next, we connect to Azure to retrieve the list of backend VM IPs from the given Azure Load Balancer, then ping each to find the first responsive VM. This uses Azure SDK or REST API calls (with appropriate authentication set up) and network checks:
# 1. Retrieve backend pool IP addresses of the given Azure Load Balancer
# (Assuming a helper function or Azure SDK call is available for this)
lb_backend_ips = get_load_balancer_backend_ips(lb_id)
print(f"Found {len(lb_backend_ips)} backend IPs in load balancer '{lb_id}': {lb_backend_ips}")

# 2. Ping each IP to find the first responsive VM
active_ip = None
for ip in lb_backend_ips:
    if is_ping_successful(ip):  # is_ping_successful is a utility that pings the IP and returns True if reachable
        active_ip = ip
        break

if not active_ip:
    raise Exception(f"No backend VM responded to ping for Load Balancer {lb_id}")
print(f"Active VM IP found: {active_ip}")
Explanation: We assume get_load_balancer_backend_ips retrieves all VM IP addresses in the load balancer's backend pool (using Azure APIs). We then iterate through these IPs, using a helper is_ping_successful to test connectivity (for example, by sending an ICMP ping or similar). The loop breaks at the first responsive IP, which we consider the active VM to target. If none respond, the notebook raises an exception since no VM is available.
3. Execute Shell Command on the Identified VM

Once we have the active VM's IP, we need its Azure resource information (like NIC or VM ID) to run the command. We find the VM's network interface and resource ID, then use Azure's Run Command feature to execute the given shell command on that VM:
# 3. Map the IP to the VM's network interface and resource ID
vm_nic, vm_id = get_vm_nic_and_id_by_ip(active_ip)
print(f"Resolved VM NIC: {vm_nic}, VM Resource ID: {vm_id} for IP: {active_ip}")

# 4. Execute the provided shell command on the VM via Azure RunCommand
try:
    run_command_response = execute_vm_run_command(vm_id, shell_command)
    command_status = "Success"
    print("RunCommand executed successfully.")
except Exception as e:
    run_command_response = str(e)
    command_status = "Failed"
    print(f"RunCommand failed: {e}")
Explanation: Using the active IP, we call a helper (or Azure SDK) get_vm_nic_and_id_by_ip to retrieve the VM's NIC name and the VM resource ID. This typically involves querying Azure for the NIC resource associated with that IP, then the VM attached to that NIC. With the VM's resource ID, we use Azure Compute's Run Command API to execute the shell_command on the VM (execute_vm_run_command encapsulates this Azure API call). We wrap this in a try/except to capture success or failure. On success, command_status is "Success"; if an exception occurs, we capture it and mark command_status as "Failed". The run_command_response could contain the output or error message from the execution, which we print or could log if needed. At this point, the shell command has been run on the chosen VM, and we have a status for the job.
4. Delta Lake Paths and Table Initialization (Unity Catalog)

Before logging, we set up the Delta Lake storage locations and ensure the corresponding Delta tables exist in Unity Catalog. We will log data to these paths in Delta format. If the tables are not already created, we create them (unmanaged Delta tables with specified schema, partitioned by date). Partitioning by data_as_update (the date) is used to organize log data by run date, which is a recommended practice for log data to optimize queries and upsert ranges​
docs.databricks.com
.
# Define Delta file storage paths for logs and notifications (these should be configured to an accessible location, e.g., ADLS mount)
process_logs_path = "<path_to_process_logs_delta_files>"         # e.g., "dbfs:/mnt/logs/process_logs"
notifications_path = "<path_to_notifications_delta_files>"       # e.g., "dbfs:/mnt/logs/notifications"

# Ensure the Unity Catalog is set (if using Unity Catalog)
# For example, use the finance_catalog and schema (if applicable)
# spark.sql("USE CATALOG finance_catalog")
# spark.sql("USE SCHEMA <your_schema>")  # use appropriate schema if needed

# Create Delta tables in Unity Catalog if they do not exist
spark.sql(f"""
CREATE TABLE IF NOT EXISTS finance_catalog.process_logs (
    file_name STRING,
    job_id STRING,
    status STRING,
    autosys_flag BOOLEAN,
    data_as_update DATE,
    timestamp TIMESTAMP
)
USING DELTA
PARTITIONED BY (data_as_update)
LOCATION '{process_logs_path}'
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS finance_catalog.process_notifications (
    file_name STRING,
    job_id STRING,
    status STRING,
    autosys_flag BOOLEAN,
    data_as_update DATE,
    timestamp TIMESTAMP,
    notification_flag BOOLEAN
)
USING DELTA
PARTITIONED BY (data_as_update)
LOCATION '{notifications_path}'
""")
Explanation: We specify the storage paths for the Delta files (the user should replace <path_to_...> with actual paths configured in their environment, such as an ADLS Gen2 location or DBFS mount). We then use Spark SQL DDL to create the tables if not exists. These tables are external (unmanaged) Delta tables, as we explicitly provide the LOCATION. They are partitioned by the data_as_update column (date), aligning with our data. The schema is defined to include all required fields, and for the notifications table we add the notification_flag. By doing CREATE TABLE IF NOT EXISTS, if the table is already created in the Unity Catalog, the command will not overwrite it (and if the Delta files already exist with data, the table will register that data). If the table doesn't exist, this will create a new entry in Unity Catalog pointing to the Delta files location (inheriting our specified schema).
Note: Using Unity Catalog means the table name finance_catalog.process_logs is a fully qualified identifier (CatalogName.SchemaName.TableName). In this example, finance_catalog might represent the catalog (and possibly a default schema). Ensure to include the proper schema name if required (e.g., finance_catalog.logging.process_logs if "logging" is the schema). The above assumes a default schema or a schema name implicitly included. Adjust the naming to your UC setup. The storage locations must be within a Unity Catalog volume or an allowed external location.
5. Log Job Status to Delta (Process Logs Table)

Now we create a DataFrame for the current job's status and write it to the Delta logs. We use an upsert (merge) operation to insert the new log entry, or update an existing entry if one with the same job_id already exists. This upsert behavior is crucial for idempotency – if the same job is run again or the log is retried, we won't duplicate entries but update the existing record.
from pyspark.sql.functions import lit, current_date, current_timestamp
from delta.tables import DeltaTable

# Prepare the log entry as a Spark DataFrame
log_df = spark.createDataFrame([
    (file_name, job_id, command_status, autosys_flag)
], schema=["file_name", "job_id", "status", "autosys_flag"]
).withColumn("data_as_update", current_date()
).withColumn("timestamp", current_timestamp())

# Upsert (MERGE) the log entry into the process_logs Delta table
# Identify matching record by job_id (assuming job_id is a unique identifier for the job run)
delta_process_logs = DeltaTable.forPath(spark, process_logs_path)
delta_process_logs.alias("t").merge(
    log_df.alias("s"),
    "t.job_id = s.job_id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()
print("Upserted job status into process_logs Delta table.")
Explanation: We construct a single-row DataFrame log_df with the job information: file_name, job_id, status (success or failed), autosys_flag, and we add the current date and timestamp. The schema matches our Delta table. We then obtain a DeltaTable object for the process logs (using the path to the Delta files). Using Delta Lake's merge operation, we merge this new DataFrame into the Delta table on the key job_id. The whenMatchedUpdateAll() and whenNotMatchedInsertAll() clauses ensure that if an entry with the same job_id already exists, it will be updated with the latest values; if it does not exist, it will be inserted​
learn.microsoft.com
. This pattern effectively implements an UPSERT (update or insert) in one operation. Delta Lake handles this in an atomic, ACID transaction. The partitioning by date (data_as_update) is handled automatically since the DataFrame contains that column; new partitions will be created as needed (for each date). Partitioning helps optimize merges by allowing us to scope operations to the relevant date range when needed​
docs.databricks.com
.
6. Log to Notifications Table and Upsert Notification Entry

After logging to the main process log, we also insert/update the entry in the notifications Delta table. The notifications table has an extra notification_flag field. We will write the same log entry with notification_flag = false (meaning not yet notified). Like before, we use an upsert merge on job_id. This duplication into a separate table allows a decoupled process to monitor pending notifications without interfering with the main log (and could have different retention or security settings if needed).
# Prepare notification log entry by adding notification_flag = false
notification_df = log_df.withColumn("notification_flag", lit(False))

# Upsert the entry into the process_notifications Delta table
delta_notifications = DeltaTable.forPath(spark, notifications_path)
delta_notifications.alias("t").merge(
    notification_df.alias("s"),
    "t.job_id = s.job_id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()
print("Upserted job status into notifications Delta table (notification_flag=false).")
Explanation: We take the log_df DataFrame and add a column notification_flag set to False. This DataFrame notification_df has schema matching the notifications table. We then perform a similar merge/upsert into the process_notifications Delta table on job_id. If the job entry already exists in notifications (perhaps from a previous run attempt), it will be updated; otherwise, inserted anew. At this point, the process_notifications table has an entry for this job with autosys_flag (possibly true for the daily job) and notification_flag = false. This flags it as a pending notification for our Kafka/consumer process to pick up. By logging to the notifications table immediately, we ensure no delay between job completion and the notification queue. Each run of this notebook will append (or update) exactly one row in both the main log and the notifications log.
7. Kafka Notification Consumer (Pseudocode)

Finally, we outline a separate process (outside this notebook) that will handle sending notifications for completed Autosys jobs. In practice, this could be a continuously running Spark Structured Streaming job, a Kafka Connect job, or an external microservice. Here is pseudocode for a Kafka-based consumer that processes the notifications:
# PSEUDOCODE: Notification Consumer Logic (to be implemented as a separate service or job)
# This consumer checks the notifications Delta table for any entries that need notifications sent.

# Initialize Kafka producer or notification system (not fully implemented here)
# e.g., producer = KafkaProducer(bootstrap_servers=[...]) or an email/SMS API client

while True:
    # 1. Read pending notifications from Delta (those with autosys_flag = true and notification_flag = false)
    pending_df = spark.read.format("delta").load(notifications_path) \
        .filter("autosys_flag = true AND notification_flag = false")
    pending_notifications = pending_df.collect()  # collect to driver (for small numbers of notifications)

    # 2. Send notifications for each pending entry
    for row in pending_notifications:
        # Construct notification message (could be an email, Kafka message, etc.)
        message = f"Job {row.job_id} ({row.file_name}) completed with status {row.status} at {row.timestamp}."
        if row.status != "Success":
            message += " (Attention: Job did not succeed.)"

        # Send the notification (for example, send to a Kafka topic or call an alerting service)
        # producer.send("autosys_job_notifications", value=message.encode('utf-8'))
        send_notification(message)  # placeholder for actual send logic

        # 3. Mark this notification as sent by updating notification_flag to true
        DeltaTable.forPath(spark, notifications_path).alias("t") \
            .merge(
                spark.createDataFrame([row.asDict()]).withColumn("notification_flag", lit(True)).alias("s"),
                "t.job_id = s.job_id"
            ) \
            .whenMatchedUpdate(set = {"notification_flag": "true"}) \
            .execute()
        print(f"Notification sent for job {row.job_id}, marked as notified.")
    
    # 4. Sleep or wait for the next poll interval
    time.sleep(60)  # wait 1 minute before checking again (or use structured streaming for real-time)
Explanation: The above pseudocode represents how a notification service might work:
It queries the process_notifications Delta table for any rows where autosys_flag is true (meaning it’s a job that requires notification) and notification_flag is false (meaning not yet notified). These are "pending" notifications.
For each pending record, it composes a notification message. This could be an email, a message to a Kafka topic, a Slack alert, etc. In a Kafka scenario, one might use a Kafka producer to send the message to a notifications topic. In other contexts, this could directly call an API or send an email.
After sending the notification successfully, it updates that record’s notification_flag to true in the Delta table. Here we show a Delta merge that finds the same job_id and updates the flag to true. This prevents sending duplicate notifications on the next poll.
The loop then waits for a short interval and repeats. In a real implementation, this could be a continuous streaming job instead of a manual loop and sleep – for example, using Spark Structured Streaming on the Delta table or a Kafka consumer that is triggered by new messages (if we integrate Delta with Kafka Connect).
Note: The actual implementation of the consumer depends on the infrastructure:
If using Spark Structured Streaming, one could set up a streaming query on the Delta table that triggers a function (foreachBatch or foreach) for new entries.
If using Kafka, another approach is to produce to a Kafka topic at the time of logging (instead of or in addition to writing to Delta), then have a standalone Kafka consumer service handle it. In that case, the Delta notifications table could be updated after producing to Kafka. The pseudocode above consolidates the logic in one place for clarity.
The pseudocode uses spark.read in a loop for simplicity. A production solution should be careful with performance (possibly caching the table or using structured streaming) and with exactly-once semantics of notifications.
Conclusion

This notebook provides a complete solution to execute remote shell commands on Azure VMs behind a load balancer and log the outcome with Delta Lake. By leveraging Delta's ACID transactions and merge/upsert capabilities, we maintain reliable log tables that can be queried and upserted efficiently​
learn.microsoft.com
. The logging is partitioned by date for manageability and performance. A dedicated notifications table and a Kafka consumer process ensure that important job completions (such as the daily Autosys job run) trigger notifications promptly, without manual intervention. The scheduled ADF/Synapse pipeline triggers this notebook every 15 minutes and once daily at 03:00, automating the entire process end-to-end.
