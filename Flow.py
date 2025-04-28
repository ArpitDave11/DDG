
import re

def extract_job_names(text):
    """
    Given a block of text, use a regex to find all unique
    Autosys job names of the form WMA_ESL_5481_DEV_DMSH_PRCSSNG_<ID>.
    """
    pattern = r'WMA_ESL_5481_DEV_DMSH_PRCSSNG_[A-Z0-9]+'
    jobs = re.findall(pattern, text)
    return sorted(set(jobs))


def generate_autosys_commands(job_names):
    """
    For each job name, generate a tuple of (failure_cmd, success_cmd).
    """
    cmds = []
    for job in job_names:
        failure = f"sendevent -E CHANGE_STATUS -s FAILURE -J {job};"
        success = f"sendevent -E STARTJOB           -J {job};"
        cmds.append((job, failure, success))
    return cmds


if __name__ == "__main__":
    # Example: paste your SSH snippet here as a multiline string
    raw_snippet = """
    sudo -H -u eisladmin bash -c 'source /mnt/ingestion/autosys/esl.env;
    sendevent -E STARTJOB -J WMA_ESL_5481_DEV_DMSH_PRCSSNG_EDCT0800;
    sendevent -E STARTJOB -J WMA_ESL_5481_DEV_DMSH_PRCSSNG_FCSTHIER;
    sendevent -E STARTJOB -J WMA_ESL_5481_DEV_DMSH_PRCSSNG_LSPT0300;
    sendevent -E STARTJOB -J WMA_ESL_5481_DEV_DMSH_PRCSSNG_MNFT1200;
    sendevent -E STARTJOB -J WMA_ESL_5481_DEV_DMSH_PRCSSNG_MNFT6500;
    sendevent -E STARTJOB -J WMA_ESL_5481_DEV_DMSH_PRCSSNG_PDBT931Z;
    sendevent -E STARTJOB -J WMA_ESL_5481_DEV_DMSH_PRCSSNG_RCAT5900;
    sendevent -E STARTJOB -J WMA_ESL_5481_DEV_DMSH_PRCSSNG_RGMT0600;'
    """

    # 1. Extract job names
    jobs = extract_job_names(raw_snippet)

    # 2. Generate failure & success commands
    commands = generate_autosys_commands(jobs)

    # 3. Print them out
    for job, fail_cmd, success_cmd in commands:
        print(f"# {job}")
        print(fail_cmd)
        print(success_cmd)
        print()




Perfect, I understand. I’ll write code that:

- Reads the Delta table from `/mnt/dataops/autosys/`.
- Filters for records where `autosys_update_flag = 'No'`.
- For each file:
   - If `status = 'success'`, generate a "success" SSH command.
   - If `status = 'fail'`, generate a "failure" SSH command.
- Outputs the list of generated SSH commands.
- Then updates the `autosys_update_flag` to 'Yes' for those file names.

I'll prepare clean, runnable code that covers all of this in one place!

# Generating SSH Commands and Updating Delta Table in PySpark

## Problem Context
We have a Delta table at `/mnt/dataops/autosys/` with records of file processing status. We need to:  
- Filter records with `autosys_update_flag = 'No'` (not updated yet).  
- For each such record, create an SSH command string to indicate success or failure (depending on the `status` field).  
- Display these SSH command strings in the output (e.g., as a list).  
- Mark those records as updated by setting `autosys_update_flag = 'Yes'` in the Delta table.

## Approach
To achieve this, the solution will:  
1. **Load the Delta Table:** Use `spark.read.format("delta").load(...)` to read the Delta table data into a DataFrame.  
2. **Filter Records:** Apply a filter to select rows where `autosys_update_flag` equals "No".  
3. **Generate SSH Command Strings:** Use a conditional expression (`when`/`otherwise`) or equivalent to create a new column `ssh_command`. This column will contain an SSH command string tailored to each record:  
   - If `status` is "success", the command string might be something like `ssh <user>@<host> "echo FileX succeeded"`.  
   - If `status` is "fail", the command string might be `ssh <user>@<host> "echo FileX failed"`.  
   - (Here, `<user>` and `<host>` come from provided variables `ssh_user` and `ssh_host`.)  
4. **Collect/Display the Commands:** Collect the generated command strings into a Python list or use Spark's `show()` to display them. This makes it easy to view all commands.  
5. **Update the Delta Table:** Use Delta Lake's upsert capabilities to update the original table. We set `autosys_update_flag = 'Yes'` for all records that were processed (previously 'No'). This can be done via the `DeltaTable` API for an efficient in-place update ([Table deletes, updates, and merges — Delta Lake Documentation](https://docs.delta.io/latest/delta-update.html#:~:text=deltaTable,)).

## PySpark Solution
Below is the PySpark code that implements the above steps on Databricks:

```python
from delta.tables import DeltaTable
from pyspark.sql import functions as F

# Variables for SSH (assumed to be provided or defined earlier)
ssh_user = "your_username"
ssh_host = "your.remote.host"

# 1. Load the Delta table from the given path
df = spark.read.format("delta").load("/mnt/dataops/autosys/")

# 2. Filter entries where autosys_update_flag is 'No'
df_pending = df.filter(F.col("autosys_update_flag") == "No")

# 3. Generate SSH command for each entry based on status
df_commands = df_pending.withColumn(
    "result_word", 
    F.when(F.col("status") == "success", F.lit("succeeded"))
     .otherwise(F.lit("failed"))
).withColumn(
    "ssh_command",
    F.concat(F.lit(f'ssh {ssh_user}@{ssh_host} "echo '),
             F.col("file"), F.lit(" "), F.col("result_word"),
             F.lit('"'))
)

# 4. Collect or display the generated SSH commands
ssh_commands_list = [row["ssh_command"] for row in df_commands.select("ssh_command").collect()]
for cmd in ssh_commands_list:
    print(cmd)

# (Alternatively, to display in a Databricks notebook:
# display(df_commands.select("ssh_command"))
# or simply df_commands.select("ssh_command").show(truncate=False))

# 5. Update the Delta table to set autosys_update_flag = 'Yes' for processed entries
delta_table = DeltaTable.forPath(spark, "/mnt/dataops/autosys/")
delta_table.update(
    condition = F.col("autosys_update_flag") == "No",
    set = { "autosys_update_flag": F.lit("Yes") }
)
```

**Explanation:**  
- We use `withColumn` to add two new columns: `result_word` (which is "succeeded" or "failed" based on status) and `ssh_command` (the full SSH command string). The string is constructed using `F.concat` and `F.lit` to insert constant parts and the dynamic fields (filename and result word).  
- The `collect()` is used to bring the results into a Python list so we can print them cleanly. In a Databricks environment, using `display()` or `show()` would also show the commands.  
- Finally, the DeltaTable API's `update` method is used to set the flag to "Yes" for all rows that had `autosys_update_flag = 'No'` ([Table deletes, updates, and merges — Delta Lake Documentation](https://docs.delta.io/latest/delta-update.html#:~:text=deltaTable,)). This marks those entries as processed.

## Example Input and Output
Let's say the Delta table initially contains the following records (simplified):

| file        | status   | autosys_update_flag |
|-------------|----------|---------------------|
| `data_1.csv`| success  | No                  |
| `data_2.csv`| fail     | No                  |
| `data_3.csv`| success  | Yes                 |

After running the above code, the output of the SSH command generation (step 4) would be:

```
ssh your_username@your.remote.host "echo data_1.csv succeeded"
ssh your_username@your.remote.host "echo data_2.csv failed"
```

Each line corresponds to an SSH command string for a record that was not yet updated. Additionally, the Delta table will be updated so that `autosys_update_flag` is now "Yes" for `data_1.csv` and `data_2.csv` (marking them as processed).
