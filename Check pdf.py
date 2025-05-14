Thanks for confirming. I’ll now build a chunk-aware version of the pipeline where the PDF (200+ pages) is parsed entity by entity, and each section is processed separately by GPT to avoid hitting context limits. This ensures reliability and full coverage of large files while maintaining accuracy.

I’ll update the solution to include logic that detects entity boundaries, processes them one at a time with OpenAI’s API, and assembles the final knowledge base safely.


# Generating a JSON Knowledge Base from a PDF Data Dictionary with Python and GPT

This solution outlines a **production-grade Python pipeline** to convert a large text-based *Data Dictionary* PDF (200+ pages) into a structured JSON knowledge base. We will use **PyMuPDF** for PDF text extraction and the **OpenAI GPT API** (GPT-4 or GPT-3.5) for parsing and structuring the data. The process involves splitting the PDF content by each entity (identified by lines like "**Entity Name : X**"), converting each entity's details into JSON via GPT, and then post-processing to fill in any missing attribute definitions.

**Key Steps Overview:**

1. **PDF Text Extraction:** Use PyMuPDF (`fitz`) to extract all text from the PDF.
2. **Chunking by Entity:** Split the extracted text into chunks, each starting at an "`Entity Name : ...`" line, to ensure each GPT prompt stays within context token limits.
3. **GPT Parsing to JSON:** For each entity chunk, call the OpenAI API with a prompt instructing it to output a structured JSON object (fields: `entity_name`, `table_name`, `entity_definition`, `attributes` list, etc.).
4. **Combine Results:** Collect all entity JSON objects and combine them into one JSON array (the knowledge base). Validate and write this to a JSON file.
5. **Fill Missing Definitions:** Identify any attributes with missing or `"NA"` definitions, and prompt GPT to generate definitions based on the entity context, updating the final JSON.
6. **Robustness Features:** Incorporate **entity-aware chunking** (to avoid breaking context mid-entity), use environment variables for API keys (no hard-coding), and include logging and exception handling for reliability.

We will now dive into each step in detail, with code snippets and explanations.

## 1. PDF Text Extraction with PyMuPDF

First, install and import PyMuPDF (also known as `fitz`). Then open the PDF and extract its text content. We iterate through each page and concatenate text. PyMuPDF's `page.get_text()` method returns the page's text content:

```python
import fitz  # PyMuPDF
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def extract_text_from_pdf(pdf_path):
    """Extract all text from the PDF using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        logging.info(f"Opened PDF '{pdf_path}' with {doc.page_count} pages.")
        for page in doc:
            text += page.get_text()  # Extract text from each page:contentReference[oaicite:3]{index=3}
        doc.close()
    except Exception as e:
        logging.error(f"Failed to read PDF: {e}")
        raise
    return text

# Example usage:
pdf_path = "DataDictionary.pdf"
full_text = extract_text_from_pdf(pdf_path)
logging.info(f"Extracted {len(full_text)} characters of text from the PDF.")
```

**Explanation:** We open the PDF and loop through pages to build one large text string. Logging is used to record progress (pages opened, characters extracted). In case of an error (e.g., file not found or read issue), we log it and re-raise the exception. At this point, `full_text` contains the entire content of the data dictionary.

## 2. Chunking Content by Entity

Given the PDF is a data dictionary, each *entity* (e.g., a database table or data entity) is described in a section starting with a line like `"Entity Name : X"`. We need to split the full text into chunks per entity. This **entity-aware chunking** ensures we don't exceed GPT's context window and that each prompt is self-contained. By chunking at logical boundaries, we avoid splitting important information mid-context (unlike naive splitting which could break sentences or sections).

**Chunking Strategy:** We can split the text by the marker `"Entity Name :"`. One approach is using Python string operations or regex to find these markers and slice the text. A straightforward method is to iterate through lines, and whenever a new entity header is found, start a new chunk.

```python
import re

def split_into_entities(full_text):
    """Split the full text into chunks, each corresponding to one entity section."""
    entities = []
    current_chunk = []
    for line in full_text.splitlines():
        # Identify the start of a new entity section
        if line.strip().startswith("Entity Name :"):
            # If we have collected a chunk for the previous entity, save it
            if current_chunk:
                entities.append("\n".join(current_chunk).strip())
                current_chunk = []
        current_chunk.append(line)
    # Add the last entity chunk if exists
    if current_chunk:
        entities.append("\n".join(current_chunk).strip())
    logging.info(f"Split text into {len(entities)} entity sections.")
    return entities

entity_chunks = split_into_entities(full_text)
# (Optional) Check size of chunks for context limits
max_chunk_length = max(len(chunk) for chunk in entity_chunks)
logging.info(f"Largest entity chunk length (chars): {max_chunk_length}")
```

In this code, each `entity_chunks[i]` string contains one entity's full description (from its "Entity Name" line up to just before the next entity's "Entity Name"). We also measure the largest chunk size as a sanity check for GPT context. If an entity chunk is extremely large (e.g., thousands of words), we might need to further split within that entity – but typically data dictionary entries are moderate in size. Each chunk will be fed to GPT separately.

**Why chunk by entity?** GPT-3.5 has a context limit around \~4096 tokens and GPT-4 around 8k (or more with 32k models). Sending the entire 200-page text at once would **exceed these limits**, so we divide the text. By splitting on entity boundaries, we ensure each prompt+response stays under the limit while preserving full context for that entity.

## 3. Converting Each Entity to JSON via OpenAI GPT

For each entity chunk, we use the OpenAI API to parse the text into a structured JSON format. We will call the GPT model (GPT-4 for better accuracy, or GPT-3.5 for cost/speed) with a carefully crafted prompt. The prompt will instruct the model to output a JSON object with the required fields:

* `entity_name`
* `table_name`
* `entity_definition`
* `attributes`: a list of objects, each with `attribute_name`, `column_name`, `data_type`, `is_primary_key`, `definition`.

**OpenAI API Setup:** We use the `openai` Python library. For security, load the API key from an environment variable rather than hard-coding it. For example, set `OPENAI_API_KEY` in your environment or use a `.env` file. Then initialize the API key in code:

```python
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")  # API key is read from env variable
if not openai.api_key:
    raise RuntimeError("OpenAI API key not set. Please set OPENAI_API_KEY environment variable.")
```

**Prompt Design:** We will use the Chat Completion endpoint with a system message to instruct the assistant to only output JSON, and a user message containing the entity text with instructions. We set `temperature=0` for deterministic output (avoiding creative deviations, since we need a consistent JSON structure). We also ensure to ask for *only* JSON in the answer (no explanatory text).

Here is a function to process one entity chunk:

````python
import json

def parse_entity_to_json(entity_text, model="gpt-4"):
    """
    Send an entity's text to OpenAI and get back a structured JSON object.
    Returns a Python dict parsed from the JSON.
    """
    # Define the system and user prompts for JSON conversion
    system_prompt = (
        "You are a data dictionary parsing assistant. "
        "Extract the entity information from the text and output it as a JSON object. "
        "Do not add any extra explanation, only output valid JSON."
    )
    user_prompt = (
        "Convert the following entity description into a JSON object with keys: "
        "entity_name, table_name, entity_definition, attributes. "
        "The 'attributes' should be a list of objects, each with keys: "
        "attribute_name, column_name, data_type, is_primary_key, definition.\n"
        "Text:\n```TEXT\n" + entity_text + "\n```"
    )
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=1000  # adjust based on expected size of JSON
        )
    except Exception as e:
        logging.error(f"OpenAI API call failed for an entity: {e}")
        raise

    result_text = response['choices'][0]['message']['content']
    # Basic validation: attempt to parse JSON
    try:
        result_json = json.loads(result_text)
    except json.JSONDecodeError as je:
        logging.error(f"JSON parse error: {je} – Received: {result_text[:100]}...")
        raise RuntimeError("Invalid JSON from GPT")
    return result_json
````

**Notes on the implementation:**

* We use a triple backtick or another delimiter in the prompt to clearly separate the entity text (marked as ` ```TEXT ... ``` `) to avoid confusion. This isn't strictly necessary, but helps indicate the exact text to parse.
* We instruct the model to follow the required JSON structure closely. The system prompt sets the role, and the user prompt describes the output format and provides the data.
* We parse the model's response with `json.loads`. If the response is not valid JSON (which can happen if the model adds extra text), we log an error and raise an exception. This is a safety net to ensure we only proceed with valid structured data.

**Sample Prompt for GPT (JSON Conversion):**

To clarify, here is an **example prompt** that might be sent to GPT for one entity (as assembled by the code above):

````text
System: "You are a data dictionary parsing assistant. Extract the entity information... output as JSON only."

User: "Convert the following entity description into a JSON object with keys: entity_name, table_name, entity_definition, attributes (list of objects with attribute_name, column_name, data_type, is_primary_key, definition).

Text:
```TEXT
Entity Name : Customer
Table Name  : CUSTOMER_MASTER
Entity Definition : Information about customers of the company including personal and account details.

Attribute Name: Customer ID
Column Name   : CUST_ID
Data Type     : INTEGER
Is Primary Key: Yes
Definition    : Unique identifier for each customer record.

Attribute Name: Customer Name
Column Name   : CUST_NAME
Data Type     : VARCHAR(100)
Is Primary Key: No
Definition    : Full name of the customer.

... (more attributes) ...
```"
````

The model should return only the JSON, for example:

```json
{
  "entity_name": "Customer",
  "table_name": "CUSTOMER_MASTER",
  "entity_definition": "Information about customers of the company including personal and account details.",
  "attributes": [
    {
      "attribute_name": "Customer ID",
      "column_name": "CUST_ID",
      "data_type": "INTEGER",
      "is_primary_key": true,
      "definition": "Unique identifier for each customer record."
    },
    {
      "attribute_name": "Customer Name",
      "column_name": "CUST_NAME",
      "data_type": "VARCHAR(100)",
      "is_primary_key": false,
      "definition": "Full name of the customer."
    },
    ...
  ]
}
```

*(The JSON structure and content will of course depend on the actual PDF's text.)*

We repeat this API call for each entity chunk and collect the results. The `max_tokens` parameter is set with some buffer (1000 here) to allow the model to output the entire JSON. Since our prompt is asking for a fairly direct transformation, the response length should be proportional to the input size (attributes count).

**Rate Limiting & API Errors:** In a production scenario with many entities, be mindful of API rate limits. If you have to process dozens of entities quickly, you may hit the OpenAI rate limit (error 429). A robust solution would implement retries with exponential backoff on failures. For example, catching `openai.error.RateLimitError` and waiting before retrying can help the script complete without crashing. Logging each failure and retry is also advisable for monitoring.

## 4. Combining Entity JSON Objects into a Knowledge Base

As we parse each entity, we add the resulting Python dict to a list. After processing all entities, this list represents the *knowledge base* of the data dictionary. We can then output it as a JSON file.

Here's how we might tie it together in a main function:

```python
def main(pdf_path, output_json_path, model="gpt-4"):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    # Split into entity sections
    entities = split_into_entities(text)
    knowledge_base = []  # list to hold all entity dicts
    for entity_text in entities:
        # Parse each entity section to JSON
        entity_data = parse_entity_to_json(entity_text, model=model)
        knowledge_base.append(entity_data)
        logging.info(f"Parsed entity: {entity_data.get('entity_name', 'Unknown')}")
    # After all entities processed, handle missing definitions (next step)
    fill_missing_definitions(knowledge_base, model=model)
    # Write the final knowledge base to JSON file
    try:
        with open(output_json_path, "w") as f:
            json.dump(knowledge_base, f, indent=2)
        logging.info(f"Knowledge base JSON saved to {output_json_path}")
    except Exception as e:
        logging.error(f"Failed to write output JSON: {e}")
        raise

# Run the main function (for example purposes)
if __name__ == "__main__":
    main("DataDictionary.pdf", "DataDictionary_KB.json", model="gpt-4")
```

After this, `DataDictionary_KB.json` will contain a list of JSON objects, one per entity. For instance:

```json
[
  {
    "entity_name": "Customer",
    "table_name": "CUSTOMER_MASTER",
    "entity_definition": "...",
    "attributes": [ {...}, {...}, ... ]
  },
  {
    "entity_name": "Product",
    "table_name": "PRODUCT_CATALOG",
    "entity_definition": "...",
    "attributes": [ {...}, ... ]
  },
  ...
]
```

Each entity JSON is self-contained. Storing them as a list makes it easy to iterate or load into databases or other tools. (If desired, you could also index by entity\_name in a dictionary for direct access by name, but a list preserves the original order.)

We also call `fill_missing_definitions()` in the workflow, which we will define next.

## 5. Filling Missing Attribute Definitions

If some attributes have missing definitions (perhaps marked as "N/A", "NA", or left blank in the PDF), we want to populate those with meaningful definitions. This can be done by an additional GPT call per missing definition, leveraging the context of the entity.

**Approach:** Loop over each entity and each attribute; if an attribute's definition is blank or a placeholder like "NA", use GPT to generate a definition. To provide context for GPT, we supply at least the entity name and the entity description, and possibly the attribute name itself in the prompt.

We'll define `fill_missing_definitions(knowledge_base)` to perform this in-place update:

```python
def fill_missing_definitions(knowledge_base, model="gpt-4"):
    """
    For any attribute in the knowledge_base list with missing/'NA' definition,
    call GPT to generate a suitable definition based on the entity context.
    Modifies the knowledge_base in place.
    """
    for entity in knowledge_base:
        entity_name = entity.get("entity_name", "")
        entity_def = entity.get("entity_definition", "")
        for attr in entity.get("attributes", []):
            definition = attr.get("definition", None)
            if definition is None or str(definition).strip().upper() in {"NA", "N/A", ""}:
                attr_name = attr.get("attribute_name")
                # Prepare a prompt to generate definition
                system_msg = "You are a data dictionary assistant with expertise in contextual definitions."
                user_msg = (f"The entity '{entity_name}' is described as: {entity_def}\n"
                            f"Provide a concise definition for the attribute '{attr_name}' in the context of this entity.")
                try:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg}
                        ],
                        temperature=0.3,
                        max_tokens=100
                    )
                    new_def = response['choices'][0]['message']['content'].strip().strip('"')
                    # Update the attribute's definition
                    attr['definition'] = new_def
                    logging.info(f"Filled definition for {entity_name}.{attr_name}: {new_def}")
                except Exception as e:
                    logging.error(f"Failed to generate definition for {entity_name}.{attr_name}: {e}")
```

**Prompt for missing definitions:** The prompt here provides the entity name and its definition (`entity_def`) to give context about what the entity represents, then asks for a definition of the specific attribute. This way, GPT can infer the role of the attribute within that entity. The system message simply sets the tone as an "expert assistant". We use a slightly higher `temperature=0.3` to allow the model to be a bit creative in phrasing, while still focusing on factual accuracy from context.

For example, if an entity "Customer" has an attribute "Middle Name" with definition "NA", the user prompt might look like:

```text
"The entity 'Customer' is described as: Information about customers including personal details.
Provide a concise definition for the attribute 'Middle Name' in the context of this entity."
```

GPT might respond with: *"The middle given name of the customer, if provided."* – which our code captures and inserts into the JSON.

We strip any extraneous quotes or whitespace from the response. If the API call fails (network issue, etc.), we log an error but continue without crashing the entire process.

After this function runs, all attributes should have a filled `definition`. We opted to do this in a second pass to clearly separate extraction of existing info (step 3) from creation of new info (step 5). This also means we don't distract the initial parsing step with tasks beyond extracting what's present.

## 6. Logging and Error Handling in Production

Throughout the code, we used the `logging` module for visibility into the process. For a production-grade script, consider these logging practices:

* Use `logging.info()` for high-level progress (e.g., number of entities, when a certain entity is parsed).
* Use `logging.error()` for exceptions or API failures, with enough context to debug (e.g., which entity failed).
* Optionally, use `logging.debug()` for very detailed internals if needed (in our sample, we kept it at info/error for clarity).

We've wrapped critical operations (PDF reading, OpenAI API calls, JSON parsing, file writing) in try/except blocks to catch errors and log them. For example, after getting GPT output we do `json.loads` to ensure it's valid JSON, and if not, we raise an error. This would prevent writing incomplete data.

Additionally, if running this in a long batch, you might want to catch exceptions around individual entity processing so that one bad chunk doesn't stop the entire run. For instance, you could enclose the `parse_entity_to_json` call in a try/except inside the loop, and on failure, log it and optionally continue to the next entity or retry.

**Environment and Security:** We used `os.getenv("OPENAI_API_KEY")` to load the API key. In a Databricks environment, you'd typically set this as a secret. For example, you might do `openai.api_key = dbutils.secrets.get("<scope>", "<key-name>")` if using Databricks Secrets. The key point is to avoid putting the key directly in code or notebooks. This script can run in Databricks by ensuring the PDF is accessible (e.g., placed in DBFS or accessible via a path) and that the environment variable or secret is set for the API key.

**Modularity:** We structured the solution into functions (`extract_text_from_pdf`, `split_into_entities`, `parse_entity_to_json`, `fill_missing_definitions`) to make it easier to maintain and test. Each function handles a distinct piece of logic. This modular approach is important in production for clarity and potential reuse (for example, you could reuse `parse_entity_to_json` for any similar text-to-JSON parsing task by adjusting prompts).

## 7. Sample Prompts Recap

For clarity, here are the two main prompt patterns used in this solution:

* **Parsing an Entity to JSON (Extraction Prompt):**
  *System message:* Instruct the model to act as a parsing assistant and output only JSON.
  *User message:* "Convert the following entity description into a JSON object with keys: `entity_name`, `table_name`, `entity_definition`, `attributes` (with `attribute_name`, `column_name`, `data_type`, `is_primary_key`, `definition`)." followed by the raw text of the entity.

  This prompt ensures the model knows the exact JSON structure expected and has the content to parse.

* **Generating Missing Definitions (Completion Prompt):**
  *System message:* Sets the role as an expert data dictionary assistant.
  *User message:* Provides the entity context (name and description) and asks: "Provide a concise definition for the attribute '\<attr\_name>' in the context of this entity."

  This prompt focuses the model on using the given context to fill in a single piece of information (the attribute definition).

By separating these uses, we first **extract** what exists in the PDF verbatim into JSON, and then **enhance** the data by filling gaps, which is aligned with a human-in-the-loop verification approach (you could review the generated definitions if needed).

## 8. Running in Different Environments

The solution is implemented as a standard Python script, so it can run in a local environment, on a server, or in a Databricks notebook with minimal changes. If using Databricks:

* Make sure to install `PyMuPDF` (`%pip install PyMuPDF`) and `openai` (`%pip install openai`) in your cluster.
* Store the PDF in DBFS or accessible path and use that path in `main()`.
* Handle the API key via Databricks Secrets (or set an environment variable in the cluster configuration).
* You can run the `main` function in a notebook cell or as a job.

The output JSON file (`DataDictionary_KB.json` in the example) will be saved to the working directory. In Databricks, this would likely be on the driver node storage; you might want to save to DBFS (e.g., `/dbfs/FileStore/...`) or upload to cloud storage depending on your needs.

## 9. Conclusion

In this solution, we demonstrated a complete pipeline to transform an unstructured PDF data dictionary into a structured JSON knowledge base using Python and GPT, without any model fine-tuning. The approach is **modular, robust, and scalable**:

* We carefully chunk input data to respect model limits and preserve logical sections.
* We integrate with OpenAI's API in a secure way (using env vars and proper error handling).
* We produce clean JSON output that can be directly loaded or queried.
* We handle missing data by leveraging the power of GPT in a controlled manner.

This method can significantly accelerate building documentation or metadata repositories from legacy resources. With further enhancements, one could enforce a JSON schema for even more rigid output (OpenAI's newer function calling or structured output features can help with this), but even with prompt-based parsing, the above approach should yield a reliable result set.

**References:**

* PyMuPDF usage for PDF text extraction
* Strategy for splitting large text inputs to fit GPT context windows
* Using environment variables for OpenAI API keys (avoid hard-coding secrets)
* Logging best practices in Python (basicConfig and usage)
* JSON parsing and file output handling in a similar OpenAI parsing context
* Handling rate limit errors by implementing retries with exponential backoff
