# Part 1: Round-Robin VM Selection

from datetime import datetime
import subprocess
from delta.tables import DeltaTable
from pyspark.sql import functions as F

# Example list of VMs behind the load balancer (replace with actual hostnames or IPs)
vm_list = ["vm1.mycompany.net", "vm2.mycompany.net", "vm3.mycompany.net"]

# Initialize the Delta table for round-robin index if it doesn't exist.
# The table will have a single row storing the last used index (starting at -1 so the first VM used will be index 0).
if not spark.catalog.tableExists("default.vm_round_robin"):
    (spark.createDataFrame([(-1,)], ["last_index"])
         .write.format("delta").mode("overwrite").saveAsTable("default.vm_round_robin"))

def is_vm_reachable(host: str) -> bool:
    """Ping the given host once to check if it's reachable. Returns True if reachable, False if not."""
    try:
        # '-c 1' sends a single ping, '-W 3' sets a 3-second timeout for the ping reply
        subprocess.check_output(["ping", "-c", "1", "-W", "3", host])
        return True
    except subprocess.CalledProcessError:
        return False

def select_next_reachable_vm(vm_list: list) -> str:
    """
    Selects the next VM in round-robin order from vm_list that is reachable.
    Updates the Delta table with the chosen index and returns the VM host as a string.
    Throws an exception if no VM is reachable.
    """
    # Fetch the last used index from the Delta table
    last_index = spark.table("default.vm_round_robin").select("last_index").head()[0]
    if last_index is None:
        last_index = -1
    # Compute the next index in round-robin fashion
    start_index = (last_index + 1) % len(vm_list)
    
    chosen_index = None
    # Try each VM in the list (up to length of list attempts) to find a reachable one
    for offset in range(len(vm_list)):
        idx = (start_index + offset) % len(vm_list)
        vm_host = vm_list[idx]
        # Ping the VM to check reachability
        if is_vm_reachable(vm_host):
            chosen_index = idx
            break
    if chosen_index is None:
        # No VM responded to ping
        raise Exception("No reachable VM found in the list.")
    
    # Update the Delta table with the new last-used index (chosen_index)
    vm_index_dt = DeltaTable.forName(spark, "default.vm_round_robin")
    vm_index_dt.update(condition="true", set={"last_index": F.lit(chosen_index)})
    return vm_list[chosen_index]





Part 2: Autosys Job Status Logging

This part introduces a Delta table autosys_job_status to log each job's status:
The Delta table has columns: file_name, job_id, execution_timestamp, status (e.g., "success"/"fail"), and autosys_update_flag (boolean).
Each job run (identified by a unique combination of job_id and execution_timestamp) appends a new record to this table.
Uses an upsert (Delta Lake merge) so that if the same job execution is logged more than once (e.g., updated status or flag), it updates the existing record instead of duplicating or overwriting the whole table.
Does not overwrite or delete previous job records — each execution retains its history. Upsert keys (job_id + execution_timestamp) ensure late-arriving updates for a specific run update that run's record only.
# Part 2: Autosys Job Status Logging

# Create the Delta table for job status if not exists (empty table with specified schema).
spark.sql("""
CREATE TABLE IF NOT EXISTS default.autosys_job_status (
  file_name STRING,
  job_id STRING,
  execution_timestamp TIMESTAMP,
  status STRING,
  autosys_update_flag BOOLEAN
) USING DELTA
""")

from delta.tables import DeltaTable  # (If not already imported above)

def log_job_status(file_name: str, job_id: str, status: str):
    """
    Logs the status of a job run to the autosys_job_status Delta table.
    - file_name: Name of the file or job input associated with the run.
    - job_id: Unique identifier of the job.
    - status: "success" or "fail".
    This function uses an upsert (merge) to add a new record or update an existing one with the same job_id and execution_timestamp.
    """
    execution_ts = datetime.now()
    autosys_flag = False  # Initially, assume Autosys has not yet processed/updated this job status
    # Create a DataFrame for the new log entry
    new_record_df = spark.createDataFrame(
        [(file_name, job_id, execution_ts, status, autosys_flag)],
        schema=["file_name", "job_id", "execution_timestamp", "status", "autosys_update_flag"]
    )
    # Perform upsert (merge) into the Delta table on (job_id, execution_timestamp)
    job_status_dt = DeltaTable.forName(spark, "default.autosys_job_status")
    job_status_dt.alias("t").merge(
        new_record_df.alias("s"),
        "t.job_id = s.job_id AND t.execution_timestamp = s.execution_timestamp"
    ).whenMatchedUpdateAll() \
     .whenNotMatchedInsertAll() \
     .execute()
Part 3: Notification Logging and Updates

This part replaces the previous JSON-based notification logging with a Delta table job_notifications:
The Delta table stores notification events with columns: job_id, notification_message, timestamp, and status.
New notifications (e.g., when a job completes or fails and needs a notification) are inserted into this table with status = "pending" (or similar).
The Autosys updater (a separate process or job) is expected to read pending notifications and send them out. After processing a notification, it will update the corresponding record's status to "notified" (to avoid duplicate processing).
We provide functions to log a notification and to mark it as notified. The updater would typically use these (or directly use Delta merges/updates) to manage notification statuses.
# Part 3: Notification Logging and Updates

# Create the Delta table for notifications if not exists.
spark.sql("""
CREATE TABLE IF NOT EXISTS default.job_notifications (
  job_id STRING,
  notification_message STRING,
  timestamp TIMESTAMP,
  status STRING
) USING DELTA
""")

def log_notification(job_id: str, message: str):
    """
    Inserts a new notification entry into the job_notifications table with status 'pending'.
    Returns the timestamp of the logged notification for reference.
    """
    notif_ts = datetime.now()
    status = "pending"
    new_notif_df = spark.createDataFrame(
        [(job_id, message, notif_ts, status)],
        schema=["job_id", "notification_message", "timestamp", "status"]
    )
    # Insert the new notification (merge used for safety to avoid duplicates on rare key collision)
    notif_dt = DeltaTable.forName(spark, "default.job_notifications")
    notif_dt.alias("t").merge(
        new_notif_df.alias("s"),
        "t.job_id = s.job_id AND t.timestamp = s.timestamp"
    ).whenNotMatchedInsertAll() \
     .execute()
    return notif_ts

def mark_notification_as_notified(job_id: str, notif_ts):
    """
    Marks a notification as 'notified' (processed) in the job_notifications table.
    Should be called by the Autosys updater after sending the notification.
    Parameters:
      job_id   - the job ID of the notification to update
      notif_ts - the timestamp of the notification entry to mark as notified (datetime or timestamp)
    """
    notif_dt = DeltaTable.forName(spark, "default.job_notifications")
    # Update the status to 'notified' for the specific notification (identified by job_id and timestamp)
    notif_dt.update(
        condition=(F.col("job_id") == job_id) & (F.col("timestamp") == F.lit(notif_ts)) & (F.col("status") == F.lit("pending")),
        set={"status": F.lit("notified")}
    )
Example Workflow

Finally, here's how the pieces come together in an end-to-end workflow:
Select a VM using round-robin selection (Part 1) and ensure it's reachable.
Run the job on the selected VM (e.g., via an SSH or remote command – not explicitly shown here).
Log the job's status in the autosys_job_status table (Part 2).
Log a notification entry about the job (Part 3), then have the Autosys updater process it and mark it as notified.
# Example usage of the above components:

# 1. Select the next reachable VM to run the job
target_vm = select_next_reachable_vm(vm_list)
print(f"Selected VM for job execution: {target_vm}")
# (Here you would execute the required shell command on the target VM, e.g., via SSH or an API call)

# 2. Assume the job ran on the VM and we have its outcome details:
job_id = "JOB_001"
file_name = "example_file.txt"
job_status = "success"  # or "fail" depending on outcome

# 3. Log the job status to the Delta table
log_job_status(file_name, job_id, job_status)
print(f"Job status for {job_id} logged in autosys_job_status table.")

# 4. Create a notification entry for this job completion
notification_message = f"Job {job_id} completed with status {job_status}."
notif_timestamp = log_notification(job_id, notification_message)
print(f"Notification for {job_id} logged as pending in job_notifications table.")

# ... Autosys updater process would query job_notifications for pending entries,
# send out the notifications, then mark them as notified ...

# 5. Mark the notification as notified after processing (simulating the Autosys updater action)
mark_notification_as_notified(job_id, notif_timestamp)
print(f"Notification for {job_id} marked as notified.")
Optional: You can verify the entries in the Delta tables using Spark SQL or DataFrame queries, for example:
# Verify the job status log
spark.sql("SELECT * FROM default.autosys_job_status").show()
# Verify the notification log
spark.sql("SELECT * FROM default.job_notifications").show()

Sources
You said:
Show me sample input of each function values as an example an out !! 
ChatGPT said:

Thought for 24 seconds
Here are concrete examples of calling each of the new helper functions, with sample inputs and the effects/results you’d see (assuming “now” is April 22, 2025 in America/New_York).

1. select_next_reachable_vm
# Precondition:
#   - vm_round_robin table exists and currently has last_index = -1
vm_list = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]

# Suppose:
#   ping 10.0.0.1 → fails
#   ping 10.0.0.2 → succeeds

chosen = select_next_reachable_vm(vm_list)
print(chosen)
# → "10.0.0.2"
Effect:
The Delta table default.vm_round_robin now has last_index = 1 (pointing at “10.0.0.2”).

2. log_job_status
# Inputs:
file_name = "sales_data.csv"
job_id     = "JOB_SALES_001"
status     = "success"

log_job_status(file_name, job_id, status)
Effect:
A new row is inserted (or upserted) into default.autosys_job_status:

file_name	job_id	execution_timestamp	status	autosys_update_flag
sales_data.csv	JOB_SALES_001	2025-04-22 14:05:00	success	false
(execution_timestamp is the current time when you called it.)

3. log_notification
# Input:
job_id  = "JOB_SALES_001"
message = "Job JOB_SALES_001 completed successfully."

notif_ts = log_notification(job_id, message)
print(notif_ts)
# → 2025-04-22 14:05:01
Effect:
A new row appears in default.job_notifications:

job_id	notification_message	timestamp	status
JOB_SALES_001	Job JOB_SALES_001 completed successfully.	2025-04-22 14:05:01	pending
(Returned notif_ts matches the timestamp column.)

4. mark_notification_as_notified
# Input: use the same job_id and timestamp returned above
mark_notification_as_notified("JOB_SALES_001", notif_ts)
Effect:
That same row in default.job_notifications is updated:

job_id	timestamp	status
JOB_SALES_001	2025-04-22 14:05:01	notified
