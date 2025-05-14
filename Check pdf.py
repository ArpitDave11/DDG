Thanks for the clarification. I’ll prepare an end-to-end script using LangChain that integrates with your Azure OpenAI deployment using the provided endpoint, subscription key, API version, and base URL. I’ll also implement token limit handling to avoid request failures.

I’ll let you know once the script is ready for use in PyCharm.


# End-to-End PDF Metadata Extraction with LangChain and Azure OpenAI

## Introduction

This guide demonstrates a comprehensive Python script that reads a PDF file and extracts structured metadata (in JSON format) using LangChain integrated with Azure OpenAI. We will use **PyPDFLoader** (a LangChain community PDF loader) to parse the PDF content into text. Next, we'll leverage **AzureChatOpenAI** (LangChain's Azure OpenAI wrapper) to call an Azure-deployed GPT model with your credentials. To enforce the output format, we define a Pydantic data model representing the desired JSON schema and use LangChain's **PydanticOutputParser**, which ensures the LLM’s output conforms to that schema. The script also handles token limits by chunking the PDF text if it's too long, so each chunk stays within the model's context window (e.g. \~4096 tokens for GPT-3.5). This chunking is done using LangChain's recommended **RecursiveCharacterTextSplitter** for splitting text into manageable sections.

**Output JSON format:** The extracted metadata will be saved as JSON with the exact structure specified in the question:

```json
{
  "Entity": {
    "TableName": "<TABLE_NAME>",
    "EntityName": "<ENTITY_NAME>",
    "Definition": "<SHORT_DESCRIPTION_OF_ENTITY>",
    "Attributes": [
      {
        "Name": "<ATTRIBUTE_FRIENDLY_NAME>",
        "Definition": "<WHAT_THIS_ATTRIBUTE_MEANS>",
        "ColumnName": "<PHYSICAL_COLUMN_NAME>",
        "ColumnType": "<DATA_TYPE>",
        "PK": "<Yes|No>"
      }
    ]
  }
}
```

## Installation

Make sure to install the required packages (all are publicly available on PyPI) before running the script:

* **LangChain** – core library for building LLM applications (`pip install langchain`)
* **LangChain Community** – community extensions, including PyPDFLoader (`pip install langchain_community`)
* **OpenAI** – OpenAI Python SDK (required for Azure OpenAI via LangChain, `pip install openai`)
* **PyPDF** – PDF parser used by PyPDFLoader (`pip install pypdf`)
* **tiktoken** – *(optional)* tokenizer for counting tokens (`pip install tiktoken`) – useful for precise chunk sizing.

Ensure you have a Python 3.8+ environment for compatibility with LangChain and Pydantic.

## Usage and Credentials Setup

Before running the script, gather your Azure OpenAI credentials:

* **Deployment Name** – the name of your deployed model (e.g., GPT-3.5-Turbo or GPT-4 deployment).
* **Subscription Key** – your Azure OpenAI API key.
* **API Version** – the API version to use (matching your deployment, e.g. `"2023-05-15"` or a newer version).
* **Base URL** – your Azure OpenAI endpoint URL (e.g. `https://<your-resource-name>.openai.azure.com`).

You can provide these credentials to the script in two ways:

1. **Environment Variables:** Set `AZURE_OPENAI_DEPLOYMENT_NAME`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, and `AZURE_OPENAI_ENDPOINT` in your environment (or in PyCharm's run configuration). The script will read from these variables.
2. **Hard-code in Script:** Alternatively, you can edit the script to directly assign these values to the variables for deployment name, key, version, and endpoint.

For security, using environment variables is recommended so you don't hard-code secrets.

## Solution Overview

The script will perform the following steps:

1. **Load PDF Content:** Use PyPDFLoader to read the PDF file and extract its text content.
2. **Initialize Azure OpenAI LLM:** Configure LangChain's AzureChatOpenAI with your Azure credentials (deployment name, endpoint, API key, and version) to create an LLM client.
3. **Define Output Schema:** Define Pydantic models for the Entity and its Attributes to represent the JSON structure. Create a PydanticOutputParser with this model to guide and validate the LLM's output.
4. **Chunk Text if Necessary:** If the document text is larger than the model's context window (\~4096 tokens for GPT-3.5, \~8192 or more for GPT-4), split the text into smaller chunks. We use LangChain’s RecursiveCharacterTextSplitter, which is recommended for general text chunking, to ensure each chunk fits within the token limit.
5. **LLM Prompting:** For each chunk (or the whole text if small enough), prompt the Azure OpenAI model to extract the metadata. The prompt will include instructions for the required JSON format (using the output parser's format instructions) and the chunk of text as context.
6. **Parse and Merge Results:** Parse the LLM's response through the PydanticOutputParser to get a structured `OutputModel` object. If multiple chunks were processed, merge their results – combining attribute lists and ensuring no duplicates – into a single final `Entity` data structure.
7. **Save to JSON:** Convert the final Pydantic model to JSON and save it to a file (e.g., `output.json`). The saved JSON will follow the exact schema required.

With this approach, even if the PDF is large or contains many attributes, the script processes it in parts and still outputs a single consolidated JSON. The use of a Pydantic schema guarantees the output format is as expected, and any deviation can be caught and handled.

## Full Python Script

Below is the full Python script implementing the above steps. You can copy this into a Python file (for example, `extract_metadata.py`) and run it in PyCharm or any environment after installing the required packages. Make sure to update the `PDF_FILE_PATH` and provide your Azure credentials as explained.

```python
import os
import json
# Install langchain and related packages before running this script.

# 1. Azure OpenAI credentials – either set these environment variables or replace the default placeholder strings.
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "<YOUR_DEPLOYMENT_NAME>")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "<YOUR_API_KEY>")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "<YOUR_API_VERSION>")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "<YOUR_BASE_URL>")  # e.g. "https://<resource>.openai.azure.com"

# Ensure all credentials are provided
if "<YOUR" in AZURE_DEPLOYMENT_NAME or "<YOUR" in AZURE_API_KEY or "<YOUR" in AZURE_API_VERSION or "<YOUR" in AZURE_ENDPOINT:
    raise ValueError("Please set your Azure OpenAI credentials in the script or as environment variables before running.")

# 2. Initialize the Azure OpenAI LLM via LangChain
from langchain.chat_models import AzureChatOpenAI
# Set up the Azure OpenAI chat model with provided credentials
llm = AzureChatOpenAI(
    deployment_name=AZURE_DEPLOYMENT_NAME,
    openai_api_base=AZURE_ENDPOINT,
    openai_api_version=AZURE_API_VERSION,
    openai_api_key=AZURE_API_KEY,
    openai_api_type="azure"   # explicitly denote Azure OpenAI
)

# 3. Load PDF and extract text using PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
PDF_FILE_PATH = "sample_entity.pdf"  # TODO: replace with your PDF file path
loader = PyPDFLoader(PDF_FILE_PATH)
# Load all pages from the PDF into documents
docs = loader.load()  # returns a list of Documents (one per page, by default)
# Combine page texts into one string (with page breaks)
full_text = "\n".join(doc.page_content for doc in docs)

# 4. Define Pydantic models for the expected JSON schema and create an output parser
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

class Attribute(BaseModel):
    Name: str = Field(..., description="Attribute friendly name")
    Definition: str = Field(..., description="What this attribute means")
    ColumnName: str = Field(..., description="Physical column name in the database")
    ColumnType: str = Field(..., description="Data type of the column")
    PK: str = Field(..., description="Primary Key indicator (Yes/No)")

class Entity(BaseModel):
    TableName: str = Field(..., description="Name of the source table")
    EntityName: str = Field(..., description="Name of the entity")
    Definition: str = Field(..., description="Short description of the entity")
    Attributes: list[Attribute] = Field(..., description="List of attributes of the entity")

class OutputModel(BaseModel):
    Entity: Entity

# Create a Pydantic output parser for the OutputModel
output_parser = PydanticOutputParser(pydantic_model=OutputModel)

# Prepare the format instructions for the LLM prompt from the parser
format_instructions = output_parser.get_format_instructions()

# 5. Split text into chunks if it is too large for the model's context window
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define chunk size and overlap (tweak these for your model's token limit; using characters as proxy for tokens)
chunk_size = 3000  # max characters per chunk (roughly ~750 tokens, assuming ~4 chars per token)
chunk_overlap = 300  # overlap between chunks to avoid cutting important info
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
# Split the full text into chunks (each chunk is a Document with page_content)
document_chunks = text_splitter.create_documents([full_text])

# 6. Process each chunk with the LLM to extract structured data
from langchain.schema import HumanMessage, SystemMessage

# System prompt to set the assistant's role and context (helps ensure formatting)
system_msg = SystemMessage(content="You are a helpful assistant that extracts structured metadata from text. "
                                   "Follow the output format instructions carefully and output only valid JSON.")

# We will collect parsed results from each chunk
parsed_chunks = []
for idx, doc in enumerate(document_chunks, start=1):
    chunk_text = doc.page_content
    # Construct the user prompt with format instructions and the chunk content
    user_prompt = (
        f"Extract the entity metadata from the following text and output it in the specified JSON format.\n"
        f"{format_instructions}\n\n"  # instructions for format
        f"TEXT:\n{chunk_text}"
    )
    # Send system and user messages to the chat model
    messages = [system_msg, HumanMessage(content=user_prompt)]
    response = llm(messages=messages)
    output_text = response.content if hasattr(response, 'content') else str(response)
    # Parse the LLM output into the Pydantic model (this will validate the JSON structure)
    try:
        parsed = output_parser.parse(output_text)
    except Exception as e:
        # If parsing fails, you might print the raw output for debugging
        raise RuntimeError(f"Failed to parse LLM output for chunk {idx}: {e}\nOutput was: {output_text}")
    parsed_chunks.append(parsed)
    print(f"Chunk {idx} processed, extracted {len(parsed.Entity.Attributes)} attributes.")

# 7. Merge results from all chunks into one final Entity
if not parsed_chunks:
    raise RuntimeError("No data was extracted from the PDF.")
# Start with the first chunk's Entity data
final_entity: Entity = parsed_chunks[0].Entity
# Merge attributes from subsequent chunks
all_attributes = list(final_entity.Attributes)
for parsed in parsed_chunks[1:]:
    # Skip adding duplicate attributes (by ColumnName or Name)
    for attr in parsed.Entity.Attributes:
        if not any(existing.ColumnName == attr.ColumnName for existing in all_attributes):
            all_attributes.append(attr)
# Update the final entity's attributes list
final_entity.Attributes = all_attributes

# 8. Save the final structured JSON to a file
output_data = {"Entity": json.loads(final_entity.json())}  # convert Pydantic model to dict
output_path = "output.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print(f"Extraction complete! Structured data saved to {output_path}")
```

**Notes:**

* The script uses a `SystemMessage` to instruct the model to only produce JSON output in the correct format. This helps prevent the AI from adding extra commentary.
* We use a conservative chunk size (`chunk_size=3000` characters) as a proxy to stay within token limits. You may adjust this based on the model's context size (for example, for GPT-4 8k context, you could use a larger chunk size). If you have the `tiktoken` library installed, you could refine this by counting tokens more precisely.
* The `RecursiveCharacterTextSplitter` is used to split the text into chunks on natural boundaries (paragraphs, sentences, spaces) without breaking words arbitrarily. We include a small overlap (`chunk_overlap=300` chars) so that if an entity attribute description is cut off at the end of a chunk, the next chunk still contains it, reducing the chance of missing or garbling information.
* We parse each chunk's LLM output with `output_parser.parse()`. This uses the Pydantic model to validate and convert the JSON text into a Python object. If the model returns invalid JSON or missing fields, an exception will be raised, alerting us to adjust the prompt or model parameters.
* Finally, we merge all attributes from each chunk. We ensure no duplicate attributes are added by checking the `ColumnName` (or you could use the `Name`) field for uniqueness. This way, if an attribute appears in two chunks due to overlap, it will only be included once in the final output.

## Running the Script

1. Put the script into a file (e.g., `extract_metadata.py`) and adjust the `PDF_FILE_PATH` to point to your PDF document.
2. Set your Azure OpenAI credentials as environment variables or fill them in the script.
3. Run the script (for example, `python extract_metadata.py`). It will connect to Azure OpenAI and process the PDF.
4. After completion, open the `output.json` file to find the extracted metadata in the specified JSON format. You can further load this JSON in Python or any tool to use the structured data.

By following this approach, you can reliably extract structured entity definitions from PDFs using Azure OpenAI. The combination of LangChain's PDF loader, Azure OpenAI's language models, and Pydantic for output validation provides a powerful end-to-end solution for parsing documents into clean JSON data.
