Understood. I will prepare a complete end-to-end Python script using OpenAI LangChain that you can run in PyCharm to extract structured entity and attribute metadata from a PDF, formatted as you specified.

I’ll include code to:

* Load and parse a PDF
* Use LangChain with an LLM to extract structured metadata
* Output the result in your required JSON format
* Include necessary installation and setup guidance

I’ll notify you once the script is ready.


# Python Script for PDF Entity Metadata Extraction using LangChain

This end-to-end script reads a PDF, uses OpenAI's LangChain with an LLM to extract structured metadata, and outputs the result as JSON. It is designed to be run in PyCharm (or any Python environment) and handles varied PDF formats without relying on a fixed layout. The output JSON will have the structure shown below, capturing an entity's table name, entity name, definition, and its attributes with details:

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

## Dependencies and Setup

Make sure to install the required packages before running the script:

* **LangChain and integrations:** `langchain`, `langchain-openai`, `langchain-community` (for PDF loader)
* **OpenAI SDK:** `openai`
* **PDF processing:** `pypdf` (PyPDF2) for reading PDFs

You can install these via pip:

```bash
pip install langchain langchain-openai langchain-community openai pypdf
```

*Note:* `langchain-community` provides the `PyPDFLoader` used for PDF reading. Also ensure you have an OpenAI API key and set it as an environment variable (`OPENAI_API_KEY`) before running the script.

## How the Script Works

1. **PDF Reading:** The script uses LangChain's `PyPDFLoader` to load the PDF. This converts the PDF into `Document` objects (one per page) that contain text content. We then extract all text into a single string for processing. (If the PDF is very large, you could chunk it, but for simplicity we use the full text.)

2. **Defining the Output Schema:** We use Pydantic models to define the expected JSON structure (Entity with its Attributes). LangChain's `PydanticOutputParser` will use this model to guide the LLM to output JSON matching the schema. This ensures the LLM’s response is well-structured JSON, not free-form text.

3. **Prompting the LLM:** We create a prompt that instructs the LLM to extract the required fields (table name, entity name, definition, attributes list). We inject the `format_instructions` from the output parser into the prompt, which tells the LLM exactly how to format the JSON output. We use OpenAI's GPT model via LangChain’s `ChatOpenAI` with a low temperature (for deterministic output).

4. **LLM Integration with LangChain:** We set up an `LLMChain` with our prompt and the chat model. The chain is executed with the PDF text as input, and it returns the LLM's response. LangChain's output parser will attempt to parse this response into our Pydantic data model. If the model strays from the format or returns invalid JSON, we handle the exception.

5. **Output Parsing and Saving:** The parsed result (a Pydantic object) is converted to a JSON string and saved to a file (e.g., `output.json`). We include error handling to catch issues like file I/O errors, API errors, or JSON parsing problems, printing informative messages in those cases.

## Complete Python Script

Below is the full Python script with all the steps integrated. You can copy this into a PyCharm project (or any IDE) and run it. Make sure to replace the placeholder PDF path and have your `OPENAI_API_KEY` set in the environment.

```python
import os
import json

# 1. Install dependencies before running:
# pip install langchain langchain-openai langchain-community openai pypdf

# 2. Ensure OpenAI API key is set (replace '...' with your key or set OPENAI_API_KEY in env).
# os.environ["OPENAI_API_KEY"] = "sk-...YOURAPIKEY..."  # It's safer to set this in your environment.

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain

# Pydantic for defining the expected JSON schema
from pydantic import BaseModel, Field
from typing import List

# Define Pydantic models for the expected JSON structure
class Attribute(BaseModel):
    Name: str = Field(description="Attribute friendly name")
    Definition: str = Field(description="What this attribute means")
    ColumnName: str = Field(description="Physical column name in the database")
    ColumnType: str = Field(description="Data type of the column")
    PK: str = Field(description="Is this attribute a primary key? Answer Yes or No")

class EntitySchema(BaseModel):
    TableName: str = Field(description="Physical table name of the entity")
    EntityName: str = Field(description="Business-friendly name of the entity")
    Definition: str = Field(description="Short description of the entity")
    Attributes: List[Attribute] = Field(description="List of attributes of this entity, with details")

# Wrap the EntitySchema in an "Entity" key as required
class EntityWrapper(BaseModel):
    Entity: EntitySchema

# Initialize the output parser with our Pydantic schema
parser = PydanticOutputParser(pydantic_object=EntityWrapper)

# Build the prompt template, including the format instructions from the parser
prompt_template = """You are an AI assistant extracting data dictionary metadata from a document.
Extract the entity information and format it as JSON with the specified schema.

{format_instructions}

Document Text:
{document}"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["document"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Initialize the OpenAI chat model (using GPT-3.5-turbo by default; adjust model_name if needed)
# Setting temperature to 0 for deterministic output
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Create the LLM Chain with our prompt and model
chain = LLMChain(llm=llm, prompt=prompt)

def extract_metadata_from_pdf(pdf_path: str, output_path: str):
    """Load a PDF file, extract entity metadata using an LLM, and save as JSON."""
    # Step 1: Load PDF and extract text
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()  # Load PDF into Document objects (one per page):contentReference[oaicite:7]{index=7}
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF: {e}")
    # Combine all page texts into one string (for simplicity; can be chunked if needed)
    document_text = " ".join([page.page_content for page in pages])
    if not document_text.strip():
        raise ValueError("PDF appears to have no text content to extract.")
    
    # Step 2: Run the LLM chain to get structured metadata
    model_output = ""
    try:
        model_output = chain.run(document=document_text)
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")
    
    # Step 3: Parse the LLM output into the structured format
    try:
        result_obj = parser.parse(model_output)  # Parse into Pydantic model
        result_json = result_obj.json(indent=2)  # Convert to JSON string (pretty-printed)
    except Exception as e:
        # If parsing fails, we handle it (e.g., the model output was not valid JSON)
        raise RuntimeError(f"Failed to parse LLM output as JSON: {e}\nRaw output: {model_output}")
    
    # Step 4: Save the JSON output to a file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result_json)
        print(f"Extraction successful! JSON saved to: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Could not write JSON to file: {e}")

# ======= Usage Example =======
if __name__ == "__main__":
    # Example usage: replace 'path/to/input.pdf' and 'path/to/output.json' as needed.
    pdf_file = "path/to/input.pdf"
    output_file = "path/to/output.json"
    try:
        extract_metadata_from_pdf(pdf_file, output_file)
    except Exception as err:
        # Print any errors that occurred
        print(f"Error: {err}")
```

**How to use this script:** Update the `pdf_file` variable to the path of your PDF (e.g., `"C:/Users/YourName/Documents/sample.pdf"` in Windows or `"/home/user/sample.pdf"` on Linux/Mac). Update `output_file` to the desired JSON output path. Then run the script. The script will print a success message and produce the JSON file if extraction is successful.

## How It Handles Different PDF Formats

We do not make any hard assumptions about the PDF layout. The entire text of the PDF is fed to the language model, which interprets it to find the required fields. Whether the PDF has tables, lists, or paragraphs describing the entity, the LLM is prompted to extract the relevant pieces. LangChain’s output parser provides a schema so the LLM knows the exact JSON structure to follow, increasing the chances of correctly structured output. This approach leverages the AI to understand context, making it flexible for various PDF formats.
