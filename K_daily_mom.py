Great, I’ll now prepare a complete Python script that takes your extracted text input, uses OpenAI LLM via LangChain to process it, and generates structured JSON for each model, entity, and their attributes—strictly following your template. I’ll include full code, parsing logic, and output formatting.


# LLM-Powered Unstructured Text Parser to JSON

## Overview

In this guide, we’ll create a Python script that uses an OpenAI Large Language Model (LLM) via LangChain to parse unstructured text (extracted from a 141-page PDF) into a structured JSON format. The script will identify **models**, their **entities**, and each entity’s **attributes**, producing a JSON **array** where each element represents one model. We leverage the LLM’s understanding to interpret the text structure, rather than brittle manual parsing. LangChain will help orchestrate the LLM calls and format the responses as JSON, making integration easier.

**Target JSON Schema:** Each model object in the output array will have:

* **`MODEL_NAME`** – Name of the model (from lines like “Model : XYZ”).
* **`entities`** – A list of entity objects under that model. Each entity object contains:

  * **`ENTITY_NAME`** – Name of the entity (text following “Entity Name”).
  * **`TABLE_NAME`** – Table name of the entity (text after “Table Name”).
  * **`DEFINITION`** – Description of the entity or table (text after the names; optional, use a placeholder if missing).
  * **`attributes`** – List of attribute objects for that entity. Each attribute object has:

    * **`NAME`** – Friendly name of the attribute.
    * **`DEFINITION`** – Attribute description (possibly multi-line). If no definition is provided in text, use “No definition available” as a placeholder.
    * **`COLUMN_NAME`** – The physical column name corresponding to the attribute.
    * **`COLUMN_TYPE`** – Data type of the column.
    * **`PK`** – `"Yes"` or `"No"`, indicating if this attribute is a primary key.

This JSON schema preserves the hierarchy: models → entities → attributes. If any fields are not present in the text, the script will insert a placeholder (e.g. `"No definition available"` for missing definitions, or `"Not available"` for other missing values) as instructed by the user.

## Installation and Setup

To set up the environment, follow these steps:

1. **Install Required Libraries:** Use pip to install LangChain and OpenAI’s Python API (and any other needed packages):

   ```bash
   pip install langchain openai
   ```

   This will install LangChain and its dependencies. LangChain provides prompt management and output parsing utilities that help ensure the LLM outputs JSON in the correct format.

2. **Set OpenAI API Key:** The script uses OpenAI’s GPT model via the API, so you need to provide your API key. The simplest method is to set the environment variable `OPENAI_API_KEY` before running the script, for example:

   * On Linux/macOS: `export OPENAI_API_KEY="sk-...yourkey..."`
   * On Windows (Command Prompt): `set OPENAI_API_KEY="sk-...yourkey..."`
     *Alternatively*, you can modify the script to assign your API key to a variable (not recommended for shared code). Using an environment variable is safer.

3. **Python Environment:** Ensure you’re using a recent Python (3.8+ recommended) and that you can run the script in PyCharm or a standard IDE/terminal. No special interpreter is required beyond the installed packages.

## Approach

**Chunking the Input:** Rather than sending the entire 141-page text in one huge prompt (which would exceed token limits), the script will split the text by **Model sections**. Each section of text starting with “Model : …” and containing its entities and attributes will be processed independently. We use Python to split the text into manageable chunks by looking for the `"Model : "` pattern. This approach ensures each LLM call stays within context length limits while maintaining logical grouping by model.

**Using the LLM for Parsing:** For each model’s text chunk, we’ll prompt the LLM (via LangChain’s `ChatOpenAI` interface) to extract the required information and format it as a JSON object. The LLM is well-suited to understand the context and structure (e.g., recognizing lines like “Entity Name”, “Table Name”, etc., and grouping attributes accordingly). By providing clear instructions and a JSON schema in the prompt, we guide the LLM to output strictly formatted JSON. (Simply asking an LLM for JSON can be unreliable without such guidance – earlier approaches had only \~35% schema compliance, but with structured prompts or OpenAI’s function calling mode, we can get close to 100% well-formed output.)

**Ensuring Correct JSON Format:** We set the model’s temperature to 0 for deterministic output and explicitly instruct the model to output *only* JSON with the exact keys. LangChain’s prompt tooling and Pydantic integration can enforce structure – for example, using Pydantic models to define the schema helps validate the response. In our script, we’ll manually check or load the JSON. If the model’s response isn’t valid JSON, LangChain also offers an `OutputFixingParser` to correct minor format issues, but our goal is to have the LLM produce correct JSON on the first try by using a robust prompt.

**Combining Results:** Each model section will produce one JSON object. The script will collect these objects into a Python list. Finally, we’ll output the list as a JSON array (pretty-printed) to show the structured data for all models. This array-of-objects format matches the user’s requirements (one object per model in the top-level JSON array).

## The Parsing Script

Below is the complete Python script. It includes comments to explain each part. You can run this script in PyCharm or any Python environment. Before running, ensure you have installed the packages and set your `OPENAI_API_KEY` as described. Save the unstructured text (extracted from the PDF) to a file (e.g., `input.txt`) so the script can read it.

```python
import os
import re
import json
from langchain.chat_models import ChatOpenAI

# Load OpenAI API key from environment (ensure it's set prior to running)
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise EnvironmentError("OPENAI_API_KEY not set. Please set this environment variable with your OpenAI API key.")

# Initialize the OpenAI LLM via LangChain (using gpt-3.5-turbo-16k or gpt-4 for large context)
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, openai_api_key=api_key)
# You can switch to model_name="gpt-4" (or "gpt-4-32k" if available) for potentially better accuracy, 
# especially if sections are very large. Ensure you have access to the model you choose.

# Read the unstructured text from a file
input_file = "input.txt"  # Path to the text file containing the PDF-extracted text
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Split the text into sections, each starting with "Model :". 
# We use a regular expression to capture each model section.
model_sections = re.finditer(r"Model\s*:\s*([^\n]+)\n([\s\S]*?)(?=(?:Model\s*:|\Z))", text)

# Prepare a list to hold parsed model data
parsed_models = []

# Define a prompt template for extracting JSON from a single model section
prompt_template = (
    "You are a parser extracting structured data from text. "
    "The text describes a data model, its entities, and attributes. "
    "Output a JSON **object** with keys `MODEL_NAME`, `entities`. "
    "`MODEL_NAME` is the name of the model. `entities` is a list of entities, each an object with keys: "
    "`ENTITY_NAME`, `TABLE_NAME`, `DEFINITION`, `attributes`. "
    "Each `attributes` is a list of objects with keys: `NAME`, `DEFINITION`, `COLUMN_NAME`, `COLUMN_TYPE`, `PK`. "
    "Use the exact keys and structure as specified. Do NOT add extra explanation or text, ONLY output valid JSON.\n"
    "Instructions:\n"
    "- If a definition is missing for an entity or attribute, use \"No definition available\" as the `DEFINITION` value.\n"
    "- Fill all other fields with the text content; if something is entirely missing, use \"Not available\" as a placeholder.\n"
    "- Ensure the JSON is syntactically correct and follows the schema.\n\n"
    "{model_text}\n"
    "---\n"
    "Now extract the above information and format it as a JSON object."
)

# Loop through each found model section and parse it
for match in model_sections:
    model_name = match.group(1).strip()
    model_body = match.group(2).strip()
    # Reconstruct the text for this model (including the "Model : Name" line for completeness)
    section_text = f"Model : {model_name}\n{model_body}"
    # Format the prompt with the section text
    prompt = prompt_template.format(model_text=section_text)
    try:
        # Use the LLM to get the JSON output for this model
        response = llm.predict(prompt)
    except Exception as e:
        raise RuntimeError(f"LLM call failed for model {model_name}: {e}")
    # The response should be a JSON string (or at least JSON-like). We attempt to parse it:
    parsed_json = None
    try:
        parsed_json = json.loads(response.strip())
    except json.JSONDecodeError:
        # If JSON parsing fails, we can try minor fixes or log the response for debugging.
        print(f"Warning: Received invalid JSON for model '{model_name}'. Attempting to fix formatting.")
        # Simple fix attempt: remove any non-JSON text (e.g., apologies or extra info) by finding first brace
        if "{" in response:
            response = response[response.index("{"):]
        if response.rfind("}") != -1:
            response = response[:response.rfind("}")+1]
        # Try loading again
        try:
            parsed_json = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON for model {model_name}: {e}")
            print("Raw response was:", response)
            # Use a placeholder or skip this model if it cannot be parsed
            continue
    # At this point, parsed_json is a dict representing the model. 
    # Ensure the top-level key is the model object itself, not a list or other structure.
    if isinstance(parsed_json, dict):
        # If the LLM mistakenly returned an array with one element, grab that.
        if len(parsed_json) == 1 and isinstance(list(parsed_json.values())[0], list) and list(parsed_json.values())[0] and isinstance(list(parsed_json.values())[0][0], dict):
            # This handles if the output was like {"models": [ {...} ]} or similar wrapping.
            # We flatten it by taking the first (and only) list of model(s).
            parsed_json = list(parsed_json.values())[0][0] if len(list(parsed_json.values())[0]) == 1 else list(parsed_json.values())[0]
    elif isinstance(parsed_json, list):
        # If the LLM returned a list (array) of model objects (instead of a single object), 
        # assume it’s a list with one element and extract it.
        if len(parsed_json) >= 1:
            parsed_json = parsed_json[0]
    # Add the parsed model object to our list (with the model name ensured)
    # Ensure the MODEL_NAME field is present and correct
    parsed_json["MODEL_NAME"] = model_name
    parsed_models.append(parsed_json)

# At this point, parsed_models is a list of model dictionaries.
# Output the final JSON array. We will print it and also save to a file.
output_json = parsed_models
# Save to file (optional)
with open("output.json", "w", encoding="utf-8") as out_f:
    json.dump(output_json, out_f, indent=2)
# Print to console
print(json.dumps(output_json, indent=2))
```

Let’s break down a few key parts of this script:

* We use a **regular expression** to iterate over each “Model : …” section. This regex (`r"Model\s*:\s*([^\n]+)\n([\s\S]*?)(?=(?:Model\s*:|\Z))"`) captures the model name and the content up until the next model or end of text. Each `match` gives us `model_name` (group 1) and `model_body` (group 2).

* The **prompt template** `prompt_template` clearly instructs the LLM about the desired JSON structure. It explicitly lists the keys and hierarchy, and reminds the model not to produce extra text. We also include instructions for placeholders when data is missing. Providing such a detailed system/message prompt is crucial for reliable structured output. We then insert the actual section text into the prompt (using `format(model_text=section_text)`).

* We call `llm.predict(prompt)` to get the LLM’s response for that section. We’ve set `temperature=0` for consistency (the model will not randomness, so it’s more likely to follow format instructions exactly). If the API call fails (for instance, due to network issues or rate limits), we catch exceptions and raise a runtime error.

* The response from the model is expected to be a JSON string (text). We use Python’s `json.loads()` to parse it into a Python dict. If the parsing fails (maybe the model included extra commentary or minor format issues), we attempt a simple fix: strip any text before the first `{` or after the last `}`. LangChain offers more robust fixes via an OutputFixingParser or function calling mode to enforce JSON, but a manual fix is shown here for clarity. In a production scenario, one might use LangChain’s structured output parsing utilities to automatically retry/correct format errors.

* We ensure the parsed object is a **dictionary for a single model**. In rare cases, the LLM might return an array with one element or wrap the result in an extra key (like `"models": [ {...} ]`). The code checks for those scenarios and normalizes the result to a single model dict. We also explicitly set the `MODEL_NAME` field to the known model\_name (the LLM should output it, but this double-check ensures the field is present and exactly matches the source text).

* Each parsed model dict is appended to `parsed_models`. After looping through all model sections, `parsed_models` will be a list of all model objects. This is our final data structure matching the desired JSON schema (except it’s in Python form). We dump this list to `output.json` file for convenience, and also print the pretty-formatted JSON to the console. The output is a JSON array `[…]` containing one object per model, with all the nested details.

## Running the Script and What to Expect

Run the script with `python script_name.py`. It will process the input text and produce an `output.json` file (and print the JSON). Given the instructions, the JSON output might look like this (an **illustrative example** with dummy data):

```json
[
  {
    "MODEL_NAME": "Sales",
    "entities": [
      {
        "ENTITY_NAME": "Customer",
        "TABLE_NAME": "Customer",
        "DEFINITION": "Represents a customer in the sales system",
        "attributes": [
          {
            "NAME": "Customer ID",
            "DEFINITION": "Unique identifier for a customer",
            "COLUMN_NAME": "CUST_ID",
            "COLUMN_TYPE": "INT",
            "PK": "Yes"
          },
          {
            "NAME": "Customer Name",
            "DEFINITION": "Full name of the customer",
            "COLUMN_NAME": "CUST_NAME",
            "COLUMN_TYPE": "VARCHAR(100)",
            "PK": "No"
          }
          // ... more attributes ...
        ]
      },
      {
        "ENTITY_NAME": "Order",
        "TABLE_NAME": "Order",
        "DEFINITION": "Stores order records. No definition available",
        "attributes": [
          {
            "NAME": "Order ID",
            "DEFINITION": "No definition available",
            "COLUMN_NAME": "ORDER_ID",
            "COLUMN_TYPE": "INT",
            "PK": "Yes"
          },
          {
            "NAME": "Order Date",
            "DEFINITION": "Date when the order was placed",
            "COLUMN_NAME": "ORDER_DATE",
            "COLUMN_TYPE": "DATETIME",
            "PK": "No"
          }
        ]
      }
    ]
  },
  {
    "MODEL_NAME": "Marketing",
    "entities": [
      {
        "ENTITY_NAME": "Campaign",
        "TABLE_NAME": "Campaigns",
        "DEFINITION": "Details of marketing campaigns",
        "attributes": [
          {
            "NAME": "Campaign ID",
            "DEFINITION": "Primary key for campaigns",
            "COLUMN_NAME": "CAMPAIGN_ID",
            "COLUMN_TYPE": "INT",
            "PK": "Yes"
          },
          {
            "NAME": "Budget",
            "DEFINITION": "No definition available",
            "COLUMN_NAME": "BUDGET",
            "COLUMN_TYPE": "DECIMAL(10,2)",
            "PK": "No"
          }
        ]
      }
    ]
  }
]
```

In the above example, each model (“Sales”, “Marketing”) is an object in the array. Under “Sales”, there are two entities (“Customer” and “Order”), each with their table name, definition, and a list of attributes. Missing definitions were replaced with `"No definition available"` as instructed. The structure and key names exactly follow the schema we outlined.

## Additional Notes

* **Choosing the Model:** We used `gpt-3.5-turbo-16k` in the script for its larger context window (16k tokens) to handle sizable text chunks. If your sections are very large or if you need better accuracy on complex parsing, consider using `gpt-4-32k` (if available) as the `model_name`. Keep in mind API usage costs – parsing a 141-page text with an LLM will consume a significant number of tokens.

* **LangChain Features:** This script manually handles prompt formatting and JSON parsing. LangChain provides higher-level tools to simplify this. For instance, one could define Pydantic data models for the output schema and use `llm.with_structured_output(...)` to have the LLM directly return parsed data classes. LangChain’s structured output mechanism (leveraging OpenAI’s function calling under the hood) can enforce the JSON schema strictly. In our case, we demonstrated the logic in a clear step-by-step manner for transparency. As LangChain evolves, such tasks may become even more straightforward to implement with built-in chains.

* **Error Handling:** The script includes basic handling for JSON decoding errors. If the model output is not valid JSON on the first try, the simple truncate-and-parse strategy is applied. In practice, you might implement a retry with a more constrained prompt or use LangChain’s `OutputFixingParser` to automatically correct common issues. With temperature 0 and a well-specified prompt, the chance of error is low, but it’s good to have a fallback. Additionally, ensure that the input text is cleaned (for example, the PDF extraction might include stray line breaks or OCR errors that could confuse the model). Cleaning or preprocessing the text (removing page numbers, headers, etc.) can improve the LLM’s performance.

By following this approach, you harness the power of OpenAI’s LLM to interpret and restructure complex unstructured text into a clean JSON format. This method saves significant time over writing a custom parser for each nuance of the document. The combination of LangChain for prompt and output management and Pydantic/JSON schema for structure ensures the resulting data is reliably formatted for downstream use. Feel free to adjust the script for your specific PDF text patterns, and good luck with your data extraction task!

**Sources:**

* Harshit D., *How to Get Only JSON response from Any LLM Using LangChain*, Medium (demonstrating LangChain’s ability to format LLM outputs as JSON).
* LangChain Documentation – *Structured Outputs & Extraction Chains* (using Pydantic schemas to extract information with LLMs).
* OpenAI, *Function calling in the OpenAI API*, Aug 2024 (introducing strict JSON format compliance for outputs).







#########################
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
