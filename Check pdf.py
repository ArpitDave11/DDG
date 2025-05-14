Great — I’ll prepare a complete Databricks notebook solution that works with Azure OpenAI using both key-based and Entra ID (Azure AD) authentication. You’ll be able to switch between the two methods easily.

The notebook will:

* Use your Azure OpenAI endpoint and deployment name
* Let you extract entities from a large PDF, send each chunk to Azure OpenAI, and structure it as a JSON knowledge base
* Fill in missing attribute definitions with GPT-4 responses
* Work end-to-end in Databricks, with clear setup and error handling

I’ll share both connection setup options and the full runnable pipeline code shortly.


# Azure OpenAI Data Dictionary Parser (Databricks Notebook)

This notebook uses **Azure OpenAI (GPT-4)** to parse a large data dictionary PDF into a structured JSON knowledge base. It supports two authentication modes (API key or Azure AD/Entra ID) and processes the PDF in chunks (by each `Entity Name:` section) to stay within GPT-4's context window. The output is a consolidated JSON file containing all entities, their attributes, and definitions.

## Overview

* **PDF Extraction:** Uses PyMuPDF to extract text from a large PDF (200+ pages).
* **Chunking by Entity:** Splits the text by `Entity Name:` sections to avoid context overflow per GPT-4 call.
* **Azure OpenAI GPT-4 Parsing:** Sends each entity section to Azure OpenAI (GPT-4) with a prompt to extract:

  * `entity_name`, `table_name`, `entity_definition`
  * `attributes`: list of attributes (`attribute_name`, `column_name`, `data_type`, `is_primary_key`, `definition`)
* **Error Handling:** Retries Azure OpenAI calls on timeouts or API errors, and ensures the response is valid JSON (re-requests if needed).
* **Filling Missing Definitions:** If any attribute `definition` is "NA" or empty, it calls GPT-4 again to generate a definition based on the entity context.
* **Output:** Combines all entities into one JSON structure and saves it to DBFS (Databricks FileStore) for easy access.

**Note:** Replace placeholder values (like API keys, endpoint URLs, deployment names, file paths) with your actual values before running. You can toggle between API Key auth and Azure AD auth by setting a variable.

## Setup - Install Dependencies

Make sure the required libraries are installed in your Databricks environment. You can use `%pip install` in a notebook cell to install them if not already available:

```bash
%pip install openai azure-identity PyMuPDF
```

This installs:

* `openai` (v1.x) for Azure OpenAI API client
* `azure-identity` for Azure AD authentication (DefaultAzureCredential)
* `PyMuPDF` (imported as `fitz`) for PDF text extraction

## Configuration and Authentication

Set up your Azure OpenAI credentials and choose the authentication method:

1. **API Key Authentication:** Use your Azure OpenAI resource's API key and endpoint.
2. **Azure AD (Entra ID) Authentication:** Use Azure AD to obtain a bearer token (requires appropriate Azure AD role or managed identity setup).

Fill in the `azure_endpoint`, `deployment_name`, and optionally `azure_api_key` if using key auth. Then set `AUTH_METHOD` to `"API_KEY"` or `"AAD"` accordingly. The `azure_endpoint` is your Azure OpenAI resource endpoint (e.g., `https://YOUR_RESOURCE.openai.azure.com`), and `deployment_name` is the name of your GPT-4 deployment on Azure.

```python
import os
import logging
from openai import AzureOpenAI

# --- User Configuration ---
azure_endpoint = "https://<YOUR_OPENAI_RESOURCE>.openai.azure.com"  # Azure OpenAI endpoint (replace with your resource URL)
deployment_name = "<YOUR_GPT4_DEPLOYMENT_NAME>"                     # Azure OpenAI deployment name for GPT-4
azure_api_key   = "<YOUR_API_KEY>"                                 # API key (if using key auth)
azure_api_version = "2023-05-15"  # API version for Azure OpenAI (ensure this matches the model deployment) 

# Choose authentication method: "API_KEY" for key-based or "AAD" for Azure AD (Entra ID)
AUTH_METHOD = "API_KEY"  # or "AAD"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataDictParser")
```

Now, initialize the Azure OpenAI client using the chosen auth method. The OpenAI SDK v1 uses the `AzureOpenAI` client for Azure endpoints. For Azure AD, we obtain a token via `DefaultAzureCredential` and `get_bearer_token_provider`:

```python
# Authentication: Initialize AzureOpenAI client
client = None
if AUTH_METHOD == "API_KEY":
    # Key-based authentication
    if not azure_api_key or "YOUR_API_KEY" in azure_api_key:
        raise ValueError("Please set azure_api_key to your Azure OpenAI API key.")
    client = AzureOpenAI(
        api_key       = azure_api_key,
        azure_endpoint= azure_endpoint,
        api_version   = azure_api_version
    )
    logger.info("Azure OpenAI client initialized with API key.")
else:
    # Azure AD authentication (Entra ID)
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAI(
        azure_endpoint         = azure_endpoint,
        azure_ad_token_provider= token_provider,
        api_version            = azure_api_version
    )
    logger.info("Azure OpenAI client initialized with Azure AD credentials.")
```

> **Note:** The Azure AD flow assumes the environment is configured for `DefaultAzureCredential` (e.g., Azure ML/Databricks with a managed identity, or environment variables for a service principal). The `get_bearer_token_provider` utility from `azure.identity` provides a token to the AzureOpenAI client.

## PDF Text Extraction

Next, read the PDF file and extract its text content. We use PyMuPDF (`fitz`) for fast PDF text extraction. Ensure the PDF is accessible in DBFS. For example, if you've uploaded the PDF to Databricks, you might have a path like `dbfs:/FileStore/<path>/data_dictionary.pdf`. We can open it via the `/dbfs` mount point.

```python
import fitz  # PyMuPDF

# PDF input path (DBFS). Replace with your actual PDF path in DBFS or local file system.
pdf_path = "/dbfs/FileStore/data_dictionary.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at {pdf_path}. Please check the path.")

# Function to extract all text from PDF
def extract_pdf_text(pdf_path: str) -> str:
    """Extracts and returns the full text from the PDF file."""
    text_content = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_content.append(page.get_text())  # Extract text from each page:contentReference[oaicite:5]{index=5}
    full_text = "\n".join(text_content)
    return full_text

logger.info("Reading PDF and extracting text...")
pdf_text = extract_pdf_text(pdf_path)
logger.info(f"Extracted {len(pdf_text)} characters of text from PDF.")
```

This will load the PDF and concatenate text from all pages. We then have the full text (which could be very large for 200+ pages).

## Splitting Text by Entity

To manage GPT-4's context limits, we split the text by `Entity Name:` sections. We assume each entity in the data dictionary starts with a line like `Entity Name: XYZ`. We will create a list of entity sections, each starting from "Entity Name:" up to just before the next "Entity Name:".

```python
import re

# Split the text by "Entity Name:" sections.
sections = re.split(r'(?=Entity Name:)', pdf_text)
# The first element may be empty or header info before the first entity.
if sections and not sections[0].strip().startswith("Entity Name:"):
    sections = sections[1:]
logger.info(f"Found {len(sections)} entity sections in the data dictionary.")
```

Now `sections` is a list of text blocks, each corresponding to one entity's documentation.

## GPT-4 Parsing Function

We define a function to parse an entity section using Azure OpenAI GPT-4. The function will send a prompt instructing GPT-4 to output a JSON with the required structure. It handles API call retries and JSON parsing.

**Prompt Design:** We provide a system message to instruct the model to act as a parser and output only JSON. The user message contains the entity text. We explicitly ask for a JSON with `entity_name`, `table_name`, `entity_definition`, and an `attributes` list (with each attribute's `attribute_name`, `column_name`, `data_type`, `is_primary_key`, `definition`). The model is asked to use the exact text provided (without adding external info) for definitions, except if none is provided (we'll handle that later).

We use `temperature=0` for consistency in formatting. We also set a reasonable `max_tokens` to allow for the JSON output (e.g., 1000 tokens). The Azure deployment name is passed via the `model` parameter (which should be your GPT-4 deployment).

````python
import json
import time
from openai.error import OpenAIError

# Define the prompt template (system and user messages)
SYSTEM_PROMPT = (
    "You are a JSON formatter. Extract the entity name, table name, entity definition, "
    "and all attributes from the provided data dictionary section. "
    "Output a JSON object with keys: entity_name, table_name, entity_definition, attributes. "
    "The attributes value should be a list of objects, each with keys: attribute_name, column_name, data_type, is_primary_key, definition. "
    "Do not include any text besides the JSON."
)
USER_PROMPT_TEMPLATE = (
    "Entity section:\n```\n{entity_text}\n```\n"
    "Parse the above entity description and output the data in the specified JSON format."
)

def parse_entity_section(entity_text: str, max_retries: int = 3) -> dict:
    """Uses Azure OpenAI GPT-4 to parse an entity section text into a structured dict."""
    # Prepare the chat messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(entity_text=entity_text.strip())}
    ]
    # Attempt the API call with retries on errors
    for attempt in range(1, max_retries+1):
        try:
            response = client.chat.completions.create(
                model       = deployment_name,  # Azure deployment name for GPT-4:contentReference[oaicite:7]{index=7}
                messages    = messages,
                temperature = 0,
                max_tokens  = 1000
            )
            # Extract the assistant's reply (the JSON as a string)
            reply_content = response.choices[0].message.content.strip()
            # Sometimes the model might include markdown or code fences, strip them
            if reply_content.startswith("```"):
                reply_content = reply_content.strip("``` \n")  # remove triple backticks and whitespace
            # Try to load JSON
            parsed = json.loads(reply_content)
            return parsed  # Return the parsed JSON as Python dict
        except OpenAIError as e:
            logger.error(f"OpenAI API error on attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)  # backoff before retry
                continue
            else:
                raise  # If max retries exceeded, propagate the error
        except json.JSONDecodeError as je:
            logger.warning(f"JSON decode error on attempt {attempt}: {je}")
            if attempt < max_retries:
                # If JSON parsing failed, refine the prompt or try again
                # We can add an instruction to output valid JSON only
                messages.insert(0, {"role": "system", "content": "Output only valid JSON strictly."})
                continue
            else:
                raise  # Give up after max retries
````

This function will:

* Call the Azure OpenAI chat completion API for the given `entity_text`.
* Handle `OpenAIError` exceptions (which cover timeouts, rate limits, etc.) by retrying with exponential backoff.
* Handle JSON decoding issues by retrying with an added instruction to the model if needed.
* Return a Python dictionary of the parsed data if successful, or raise an error if it fails after retries.

## Parsing All Entities and Refining Missing Definitions

Using the above function, we loop through all entity sections and parse them. After parsing, we check each entity's attributes for missing definitions (e.g., `"definition": "NA"` or an empty string). For any missing definitions, we will call the model again to generate a definition using the context of the entity.

We'll create another helper function to generate a definition for a missing attribute by providing the entity name, entity definition, and attribute name to the model. This function will also use the Azure OpenAI client with a suitable prompt.

```python
# Helper to generate a definition for an attribute using GPT-4
def generate_attribute_definition(entity_name: str, entity_definition: str, attribute_name: str, max_retries: int = 2) -> str:
    """Generate a definition for a given attribute based on the entity context using GPT-4."""
    prompt = (
        f"The entity '{entity_name}' is described as: {entity_definition}\n"
        f"Provide a concise definition for the attribute '{attribute_name}' in the context of this entity."
    )
    messages = [
        {"role": "system", "content": "You are an assistant that provides definitions for data dictionary attributes."},
        {"role": "user", "content": prompt}
    ]
    for attempt in range(1, max_retries+1):
        try:
            response = client.chat.completions.create(
                model       = deployment_name,
                messages    = messages,
                temperature = 0,
                max_tokens  = 200
            )
            definition_text = response.choices[0].message.content.strip().strip("\"")
            # Ensure it's a clean definition without extraneous text
            return definition_text
        except OpenAIError as e:
            logger.error(f"Error generating definition for '{attribute_name}' (attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(1 * attempt)
                continue
            else:
                return ""  # If it fails, return empty string
```

Now we process each entity section:

```python
parsed_entities = []  # list to hold all parsed entity dictionaries

for idx, section in enumerate(sections, start=1):
    logger.info(f"Processing entity {idx}/{len(sections)}")
    try:
        entity_data = parse_entity_section(section)
    except Exception as e:
        logger.error(f"Failed to parse section {idx}: {e}")
        continue  # skip this entity on failure
    
    # Check for missing attribute definitions and fill them
    entity_name = entity_data.get("entity_name") or ""
    entity_def  = entity_data.get("entity_definition") or ""
    for attr in entity_data.get("attributes", []):
        if not attr.get("definition") or str(attr.get("definition")).strip().upper() in ["", "NA", "N/A"]:
            attr_name = attr.get("attribute_name") or attr.get("column_name") or "Unknown Attribute"
            logger.info(f"Attribute '{attr_name}' in entity '{entity_name}' has missing definition. Generating...")
            generated_def = generate_attribute_definition(entity_name, entity_def, attr_name)
            if generated_def:
                attr["definition"] = generated_def
            else:
                logger.warning(f"No definition generated for '{attr_name}' in entity '{entity_name}'.")
    parsed_entities.append(entity_data)
```

This loop will:

* Parse each section into a dict `entity_data`.
* If parsing fails for a section, log the error and skip it.
* For each attribute in the parsed `entity_data`, check if `definition` is empty or "NA". If yes, call `generate_attribute_definition` to get a definition and update the attribute.
* Append the completed `entity_data` to `parsed_entities`.

## Saving the Results to JSON

Finally, we save the combined list of entity dictionaries to a JSON file in DBFS. This will allow later retrieval or use in Databricks.

```python
# Output path in DBFS for the JSON knowledge base
output_json_path = "/dbfs/FileStore/processed_data_dictionary.json"

try:
    with open(output_json_path, "w") as outfile:
        json.dump(parsed_entities, outfile, indent=2)
    logger.info(f"Saved parsed data dictionary JSON to {output_json_path}")
except Exception as e:
    logger.error(f"Failed to save JSON output: {e}")
```

The JSON file will contain an array of entities, for example:

```json
[
  {
    "entity_name": "Customer",
    "table_name": "cust_master",
    "entity_definition": "Contains customer profile information ...",
    "attributes": [
      {
        "attribute_name": "Customer ID",
        "column_name": "cust_id",
        "data_type": "INT",
        "is_primary_key": true,
        "definition": "Unique identifier for a customer"
      },
      ...
    ]
  },
  ...
]
```

You can download this JSON from Databricks or use it directly in your workflows.

## Conclusion

This notebook provides a robust pipeline to convert a large data dictionary PDF into a structured JSON knowledge base using Azure OpenAI GPT-4. It demonstrated both API key and Azure AD authentication methods, handled potential errors (with retries and logging), and filled in missing pieces of information by leveraging the context provided. The result is a machine-friendly JSON ready for further analysis or integration.

**Sources:**

* Azure OpenAI Python SDK v1 usage (AzureOpenAI client)
* Azure AD authentication with DefaultAzureCredential and token provider
* PyMuPDF text extraction example
