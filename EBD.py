Thank you! I’ll now prepare a finalized Python script that:

1. Processes all JSON files in a folder, where each file is a JSON array of records.
2. Embeds each attribute individually, using OpenAI's text-embedding-3-small model.
3. Handles long attribute definitions with chunking and averages their embeddings.
4. Writes enriched JSON back into the same folder with `_enriched.json` suffix.
5. Appends all attribute-level embedding data into a single CSV file in the same folder.

It will also include exponential backoff logic to handle API rate limits. I’ll let you know once it’s ready.


# Python Script for Embedding Data Model Attributes

**Goal:** We will create a Python script that reads JSON files describing data models (with entities and attributes), generates vector **embeddings** for each attribute using OpenAI’s `text-embedding-3-small` model, and outputs enriched JSON files and a consolidated CSV. An *embedding* is essentially a numerical vector representation of text that captures its semantic meaning. By embedding each attribute’s name and definition, we can transform textual definitions into vectors for tasks like semantic search or clustering.

Below we outline the steps, provide explanations, and present the complete script. The script will:

* **Scan a folder for JSON files** (each containing a list of data model records).
* **Parse each JSON file** and iterate through each record’s attributes.
* **Prepare text for embedding** by combining each attribute’s name and definition. Handle cases where the text is very long by **chunking** it.
* **Generate an embedding vector** for each attribute using OpenAI’s API (model `text-embedding-3-small`), with automatic **exponential backoff** to handle rate limits.
* **Attach the embedding** (a list of floats) to the attribute data and write out a new enriched JSON file (with a `_enriched.json` suffix).
* **Collect all attributes into a CSV file** with columns for model, table, entity, attribute, etc., including the embedding as a JSON string in one column.

We will ensure the solution handles OpenAI API token limits by chunking long text inputs (the model has a limit of 8191 tokens), and be mindful of irregular JSON field names (like keys containing spaces or punctuation).

## Scanning the Folder for JSON Files

First, the script needs to locate all JSON files in a specified directory. We can use Python’s built-in libraries to achieve this. One convenient way is using the `glob` module to match all `*.json` files in the folder:

```python
import os
from glob import glob

folder_path = "/path/to/json/folder"
json_files = glob(os.path.join(folder_path, "*.json"))
```

In this snippet, `glob(os.path.join(folder_path, "*.json"))` returns a list of all filenames ending with `.json` in the given folder. We could also use `os.listdir` and filter for `.json` extension, but `glob` handles patterns and is straightforward for this use case.

*Handling no files:* If `json_files` comes up empty (no JSON files found), the script can simply end or print a message. In our script, we will proceed only if there are files to process.

## Reading JSON Data Model Files

For each JSON file found, we will read its content using Python’s `json` module. Each file is expected to be a JSON **array** of records. We’ll parse it into a Python list of dictionaries:

```python
import json

for file_path in json_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        data_records = json.load(f)  # Parse the JSON array into a Python list
    # Now data_records is a list of dicts, each representing a data model record
```

We use `encoding='utf-8'` to safely handle any special characters in the JSON files. Each `data_records` list item is a dictionary with keys like `"Model"`, `"Entity"`, `"TABLE NAME"`, `"ENTITY NAME"`, `"DEFINITION"`, and an `"Attributes"` list. For example, a single record might look like:

```json
{
  "Model": "Sales",
  "Entity": 1,
  "TABLE NAME": "Customers",
  "ENTITY NAME": "Customer",
  "DEFINITION": "Contains customer master data.",
  "Attributes": [
    {
      "NAME": "Customer ID",
      "DEFINITION": "Unique identifier for a customer.",
      "Column Name": "CUST_ID",
      "Column Data Type": "INTEGER",
      "PK?": "Y"
    },
    {
      "NAME": "Customer Name",
      "DEFINITION": "Full name of the customer.",
      "Column Name": "FULL_NAME",
      "Column Data Type": "VARCHAR",
      "PK?": "N"
    },
    ...
  ]
}
```

Our script will iterate through each record in `data_records` and then through each attribute in the record’s `"Attributes"` list. We’ll construct an embedding for each attribute as described next.

## Preparing Attribute Text for Embedding

For each attribute, we need to create the text that will be sent to the embedding model. The problem statement suggests using the **attribute name + definition** as the text. We should concatenate these in a clear way, for instance:

```python
text_to_embed = f"{attr.get('NAME', '')}: {attr.get('DEFINITION', '')}"
```

Here we use `attr.get('NAME', '')` and `attr.get('DEFINITION', '')` to safely retrieve the values (using empty string if the key is missing). We include a colon and space between the name and definition for readability, but the exact format isn't crucial as long as both pieces are present.

**Handling irregular fields:** We use `.get()` in case some attribute entries don’t have the expected keys or use slightly different key names. The structure given uses `"NAME"` and `"DEFINITION"`, which we assume are consistent. If keys like `"Column Name"` or others have spaces, using `attr.get('Column Name')` is fine (key names in dictionaries can have spaces and punctuation). Our script will be careful to use the correct keys as provided. If a certain field like `"PK?"` is missing in an attribute, we might default it to something (e.g., `False` or `N`), but in the CSV we will just write whatever is present (or empty string).

## Handling Long Text: Chunking and Averaging Embeddings

OpenAI embedding models have a maximum context length (input size) measured in tokens. The `text-embedding-3-small` model can handle up to **8191 tokens** in the input. (8191 tokens roughly corresponds to \~6,000 English words or about 24k characters, depending on the text.) If an attribute’s name + definition combined is very large, we cannot embed it in one go. The instructions suggest using a threshold of \~2000 characters as a precaution.

To be safe, our script will **split the text into chunks** if it exceeds a certain length (e.g. 2,000 characters). Each chunk will then be embedded separately, and the resulting vectors will be combined by averaging. This approach is recommended by OpenAI for long inputs: break the input into manageable chunks, embed each chunk, then either use all chunk embeddings or combine them (for example, by averaging, possibly weighted by chunk size).

**Chunking strategy:** We will implement a simple character-based chunking for simplicity. For example, we can split the text every N characters (N \~ 1500 or 2000 to stay under token limits with a margin). We will try to split on sentence boundaries or spaces if possible to avoid breaking words. A simple approach is:

```python
MAX_CHARS = 2000
text = text_to_embed
if len(text) > MAX_CHARS:
    chunks = []
    # Split by sentences or paragraphs if possible
    sentences = text.split('. ')
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 2 <= MAX_CHARS:  # +2 for the period and space
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    if current_chunk:
        chunks.append(current_chunk.strip())
else:
    chunks = [text]
```

This logic tries to accumulate sentences (separated by ". ") into chunks not exceeding the `MAX_CHARS` limit. If the text is still extremely large even after sentence splitting, one could fallback to splitting by characters. The end result is a list of one or more text chunks.

**Averaging embeddings:** Once we have multiple chunks, we request an embedding for each chunk. We then compute the *average* of the embedding vectors. A simple average treats each chunk equally, but note that longer chunks contain more information. A more precise method is to do a **weighted average by chunk length** (e.g., weighting by number of tokens in each chunk). For simplicity, we might use an unweighted average (each chunk contributes equally) unless the chunks vary drastically in length. Averaging means summing up the vectors element-wise and dividing by the number of vectors. This yields a single embedding of the same dimension that represents the full text.

*Note:* The OpenAI Cookbook demonstrates weighting by token count and even normalizing the final vector to unit length. These steps can improve the embedding representation for combined chunks, but we will stick to the basic average as requested.

## Generating Embeddings with OpenAI’s API

Once we have the text (or text chunks) for an attribute, we use OpenAI’s API to get the embedding vector. We will use the `openai` Python library (make sure to install it via `pip install openai` and set your API key). The model specified is `"text-embedding-3-small"`, which produces a 1536-dimensional vector for each text input. This model is part of OpenAI’s third-generation embedding models, offering improved accuracy at a lower cost than the older `text-embedding-ada-002`.

A basic call to get an embedding for a piece of text looks like this:

```python
import openai
# Make sure your OpenAI API key is set, e.g., via openai.api_key = "sk-...".

response = openai.Embedding.create(input=text_to_embed, model="text-embedding-3-small")
vector = response['data'][0]['embedding']  # this is the list of float values
```

OpenAI allows sending a **list of texts** in one API call as well (`input` can be a list of strings). For example, `openai.Embedding.create(input=[text1, text2], model="text-embedding-3-small")` would return two embeddings. In our case, when we have multiple chunks, we could send the list of chunks in one go to get all chunk embeddings in one API call. For clarity, we might embed one chunk at a time, but batching is an option to reduce API calls.

**Handling API credentials:** The script should retrieve the API key securely. Typically, you would set `openai.api_key = <your key>` in the code, or better, set the environment variable `OPENAI_API_KEY` and have `openai.api_key = os.getenv("OPENAI_API_KEY")`. In production, avoid hardcoding the API key in the script.

## Implementing Exponential Backoff with Tenacity

When calling the OpenAI API repeatedly (potentially thousands of times, one per attribute), we risk hitting rate limits (HTTP 429 Too Many Requests errors). To make our script robust, we will implement **automatic retries with exponential backoff** for the embedding calls. This means if a request is rate-limited or encounters a transient error, the script will wait for an increasing delay before retrying, rather than failing immediately. The [`tenacity`](https://pypi.org/project/tenacity/) library provides a convenient decorator for this purpose.

We will use `tenacity.retry` with `wait_random_exponential(min=1, max=60)` and `stop_after_attempt(6)`. This configuration will retry up to 6 times, waiting with exponential backoff starting at 1 second and capping at 60 seconds between retries. We will instruct it to retry on specific exceptions (like `openai.error.RateLimitError` or other transient OpenAI API errors). For example:

```python
import tenacity
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from openai.error import RateLimitError, APIError, Timeout, APIConnectionError

# Decorator for exponential backoff on OpenAI API rate limits and transient errors
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6),
       retry=retry_if_exception_type((RateLimitError, APIError, Timeout, APIConnectionError)))
def get_embedding_with_backoff(text, model="text-embedding-3-small"):
    response = openai.Embedding.create(input=text, model=model)
    return response['data'][0]['embedding']
```

In this function `get_embedding_with_backoff`, any call that raises one of the specified exceptions will trigger a retry after a delay. The delay increases exponentially (with some jitter/randomness) between attempts. After 6 failed attempts, it will give up. We exclude `openai.error.InvalidRequestError` (which is raised for irrecoverable issues like exceeding context length or invalid input) so that those are not retried unnecessarily.

We will use this helper function whenever we call the OpenAI API to embed text. This ensures our script is resilient to temporary rate limits or network issues.

## Writing the Enriched JSON Output

After embedding all attributes in a record, we will have added an `"embedding"` field (list of floats) to each attribute dictionary. We then save the modified records back to a new JSON file. The new filename can be the original name with `_enriched.json` appended before the extension. For example, if the original file was `data_models.json`, the output could be `data_models_enriched.json` in the same folder.

We can use `json.dump` to write the Python object back to a JSON file. It’s good practice to specify `ensure_ascii=False` and an indentation for readability:

```python
output_path = file_path.replace(".json", "_enriched.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data_records, f, ensure_ascii=False, indent=4)
```

This will create a nicely formatted JSON file with UTF-8 encoding, preserving any non-ASCII characters properly. Each record in this JSON will be identical to the input, except every attribute now has an `"embedding"` field containing the list of floats.

**File size note:** Embeddings are 1536-dimensional by default for this model, so each attribute’s embedding will be a list of 1536 numbers. This will increase the size of the JSON files. If a JSON file had N attributes total (across all records), it will gain N \* 1536 float numbers. This is something to be aware of in terms of file size and memory usage.

## Creating a Combined CSV of All Attributes

Finally, we also write a CSV file summarizing all attributes across all JSON files. Each **row** in the CSV will represent one attribute, with the following columns:

* `model_name` – the model name (from the record’s `"Model"` field)
* `table_name` – the table name (from `"TABLE NAME"` field)
* `entity_name` – the entity name (from `"ENTITY NAME"` field)
* `attribute_name` – the attribute’s name (from attribute `"NAME"` field)
* `column_name` – the physical column name (from `"Column Name"` field)
* `column_data_type` – data type of the column (from `"Column Data Type"`)
* `pk` – primary key indicator (from `"PK?"` field, e.g., `Y` or `N`)
* `embedding` – the embedding vector as a JSON string

We will open a CSV file (let’s call it `attributes_enriched.csv` in the same folder) and write a header row followed by one row per attribute. Using Python’s built-in `csv` module, it’s important to open the file with `newline=''` to avoid blank lines on some systems. For example:

```python
import csv

csv_path = os.path.join(folder_path, "attributes_enriched.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Write header:
    writer.writerow(["model_name", "table_name", "entity_name", "attribute_name",
                     "column_name", "column_data_type", "pk", "embedding"])
    # Write data rows:
    for record in all_records:             # we will accumulate all records from all files
        for attr in record["Attributes"]:
            embedding_json = json.dumps(attr.get("embedding", []))
            writer.writerow([
                record.get("Model", ""), 
                record.get("TABLE NAME", ""), 
                record.get("ENTITY NAME", ""), 
                attr.get("NAME", ""), 
                attr.get("Column Name", ""), 
                attr.get("Column Data Type", ""), 
                attr.get("PK?", ""), 
                embedding_json
            ])
```

Here we convert the embedding list to a JSON string (`json.dumps`) before writing, so that the entire list of floats appears as a single field in the CSV (enclosed in quotes). This prevents commas in the embedding from creating extra columns. The CSV will thus have an embedding column that contains a JSON array of numbers as text.

We iterate through all records and their attributes. We can gather all records from all files into a single list `all_records` for convenience, or write to CSV inside the loop as we go. Either approach works as long as we ensure the header is written once.

## Full Python Script

Below is the **complete script** incorporating all the above steps. You can adjust the `folder_path` to point to your folder containing the JSON files. Ensure you have installed the `openai` and `tenacity` packages and set your OpenAI API key before running.

```python
import os, json, csv
import openai
from glob import glob
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from openai.error import RateLimitError, APIError, Timeout, APIConnectionError, InvalidRequestError

# Configuration
folder_path = "/path/to/json/folder"          # TODO: set this to your directory path
output_csv = os.path.join(folder_path, "attributes_enriched.csv")
MAX_CHARS = 2000                              # max characters before chunking text
openai.api_key = os.getenv("OPENAI_API_KEY")  # or set your API key directly

# Prepare CSV writer
csvfile = open(output_csv, 'w', newline='', encoding='utf-8')
writer = csv.writer(csvfile)
writer.writerow(["model_name", "table_name", "entity_name", "attribute_name",
                 "column_name", "column_data_type", "pk", "embedding"])

# Tenacity retry decorator for OpenAI API calls (exponential backoff on certain errors)
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6),
       retry=retry_if_exception_type((RateLimitError, APIError, Timeout, APIConnectionError)))
def get_embedding_with_backoff(text, model="text-embedding-3-small"):
    """Call OpenAI API to get embedding, with retries on rate limit and transient errors."""
    response = openai.Embedding.create(input=text, model=model)
    return response['data'][0]['embedding']

# Process each JSON file in the folder
for file_path in glob(os.path.join(folder_path, "*.json")):  # find all JSON files:contentReference[oaicite:18]{index=18}
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Skipping {file_path}: JSON decode error: {e}")
            continue

    # Process each record in the JSON array
    for record in records:
        model_name = record.get("Model", "")
        table_name = record.get("TABLE NAME", "")
        entity_name = record.get("ENTITY NAME", "")
        # Some records might not have 'Attributes' or it might not be a list
        attributes = record.get("Attributes", [])
        if not isinstance(attributes, list):
            continue  # skip if attributes is not a list as expected

        for attr in attributes:
            # Compose text from attribute name and definition
            name = attr.get("NAME", "") 
            definition = attr.get("DEFINITION", "")
            text = f"{name}: {definition}" if name or definition else ""
            if not text:
                # If there's no text (empty name and definition), skip embedding
                attr["embedding"] = []
                continue

            # Chunk the text if it's too long for safety
            chunks = []
            if len(text) > MAX_CHARS:
                # Split by sentence (naive approach)
                sentences = text.split('. ')
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 <= MAX_CHARS:
                        current_chunk += sentence + '. '
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence + '. '
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks = [text]

            # Get embeddings for each chunk and average them if multiple
            try:
                if len(chunks) == 1:
                    # Single chunk - directly embed
                    embedding = get_embedding_with_backoff(chunks[0])
                else:
                    # Multiple chunks - embed each and then average
                    chunk_vectors = []
                    for chunk in chunks:
                        vec = get_embedding_with_backoff(chunk)
                        chunk_vectors.append(vec)
                    # Average the vectors element-wise
                    # (assuming all vectors are of equal length, as they should be)
                    avg_vector = [0] * len(chunk_vectors[0])
                    for vec in chunk_vectors:
                        for i, val in enumerate(vec):
                            avg_vector[i] += val
                    avg_vector = [val / len(chunk_vectors) for val in avg_vector]
                    embedding = avg_vector
                attr["embedding"] = embedding
            except InvalidRequestError as e:
                # This can happen if input is still too long or other issues; handle gracefully
                print(f"Embedding failed for attribute '{name}' in {file_path}: {e}")
                attr["embedding"] = []
            except Exception as e:
                # Catch-all for any other errors (should be rare with tenacity)
                print(f"Unexpected error embedding '{name}': {e}")
                attr["embedding"] = []

            # Write the attribute row to CSV
            embedding_json = json.dumps(attr.get("embedding", []))
            writer.writerow([
                model_name, 
                table_name, 
                entity_name, 
                name, 
                attr.get("Column Name", ""), 
                attr.get("Column Data Type", ""), 
                attr.get("PK?", ""), 
                embedding_json
            ])
    # Save enriched JSON for this file
    output_path = file_path.rsplit(".", 1)[0] + "_enriched.json"
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(records, out_f, ensure_ascii=False, indent=4)

# Close the CSV file
csvfile.close()
```

**Explanation of the script:**

* We import necessary modules: `os`, `json`, `csv`, `openai`, `glob`, and `tenacity` with specific functions.
* We set `folder_path` to the directory containing JSON files and define `output_csv` path. We also define `MAX_CHARS = 2000` for chunking threshold. The OpenAI API key is fetched from environment for security.
* We open the CSV file for writing (`newline=''` to avoid extra blank lines) and write the header row.
* We define `get_embedding_with_backoff` as a wrapper around `openai.Embedding.create`. The `@retry` decorator from tenacity will automatically handle retries with exponential backoff if a RateLimitError, APIError, Timeout, or APIConnectionError is raised. We stop after 6 attempts max.
* We loop over each JSON file in the folder (using `glob` to find them). For each file, we load the JSON content into `records`.
* For each `record` in `records`, we retrieve `model_name`, `table_name`, `entity_name`. We get the `attributes` list, and if it’s not present or not a list, we skip processing that record.
* For each `attr` in `attributes`, we form the text to embed (concatenating name and definition). If both name and definition are empty, we skip embedding (assign an empty list).
* If the text is longer than `MAX_CHARS`, we split it into `chunks` by sentences. The chunking logic accumulates sentences until adding another would exceed the limit, then starts a new chunk. This is a simple strategy; complex cases could use more robust methods (or token-based splitting via `tiktoken` for precision).
* We then call `get_embedding_with_backoff` on each chunk (or the single text if no chunking was needed). The tenacity decorator will retry the API call on rate limits, helping us handle heavy loads gracefully.
* If multiple chunk embeddings are produced, we compute an average vector. We initialize `avg_vector` with zeros of the correct length (length of the first chunk’s embedding) and add each chunk vector to it. Then divide each element by the number of chunks to get the average. This averaged vector is our representation for the full text.
* We attach the resulting `embedding` list to the attribute dictionary (`attr["embedding"] = embedding`). In case of an OpenAI `InvalidRequestError` (e.g., text still too long or other issues like malformed input), we catch it and set the embedding to an empty list, logging the error. We also catch any other exception to avoid crashing the script if something unexpected happens for a particular attribute.
* We write a row to the CSV for this attribute, converting the embedding list to a JSON string so it stays as one field.
* After processing all attributes of all records in a file, we dump the modified `records` list to a new JSON file with `_enriched.json` suffix. We use `ensure_ascii=False, indent=4` for pretty printing in UTF-8.
* Finally, we close the CSV file to ensure all data is flushed to disk.

## Conclusion

This script automates the embedding of data model attributes and compiles results into both JSON and CSV formats. By using OpenAI’s efficient embedding model and handling lengthy text and rate limits carefully, it ensures compatibility with API constraints (token limits and rate limits). The use of tenacity for retries and text chunking for long inputs makes the process robust. You can now run this script on your dataset to obtain embeddings for each attribute, facilitating advanced analysis like semantic search or clustering on the definitions of your data model fields.

**Sources:**

* OpenAI API – *New embedding models and API updates*
* OpenAI Cookbook – *Embedding long inputs and handling chunking*
* OpenAI Cookbook – *Using embeddings with tenacity backoff*
* Python Docs/StackOverflow – JSON and CSV handling
