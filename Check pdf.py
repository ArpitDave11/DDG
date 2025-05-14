import os
import re
import json
import logging
import time
from typing import List, Dict, Any
import fitz  # PyMuPDF

# Configure logging for debug info and errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Set OpenAI API key (for demonstration; in practice use a secure method)
os.environ['OPENAI_API_KEY'] = 'abc'  
# Initialize OpenAI client (using v1.x SDK style)
try:
    from openai import OpenAI
except ImportError:
    import openai
    OpenAI = openai  # fallback to older usage if OpenAI class isn't available
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from the PDF at pdf_path using PyMuPDF.
    Returns the extracted text as a single string.
    """
    logging.info(f"Extracting text from PDF: {pdf_path}")
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()  # get all text on the page
            text += "\n"            # separate pages by a newline
    return text

def split_into_entities(text: str) -> List[str]:
    """
    Splits the full document text into sections by 'Entity Name:'.
    Returns a list of text chunks, each corresponding to one entity.
    """
    logging.info("Splitting text into entity sections...")
    sections = re.split(r'(?=Entity Name:)', text)  # lookahead to keep the delimiter
    sections = [s.strip() for s in sections if s.strip()]  # remove empty chunks
    logging.info(f"Found {len(sections)} entity sections in the document.")
    return sections

def parse_entity_to_json(entity_text: str) -> Dict[str, Any]:
    """
    Uses GPT-4 to parse a single entity's text into a structured JSON (as a dict).
    Returns a dictionary with keys: entity_name, table_name, entity_definition, attributes.
    """
    # System prompt to define the JSON structure
    system_prompt = (
        "You are an assistant extracting structured data from a data dictionary. "
        "The user will provide the text for one entity (including its attributes). "
        "Extract the information and output it as a JSON object with keys: "
        "entity_name, table_name, entity_definition, attributes. "
        "The 'attributes' should be a list of objects, each with keys: "
        "attribute_name, column_name, data_type, is_primary_key, definition. "
        "If a field is missing in the text, use an empty string (or false for is_primary_key if not specified). "
        "Respond with JSON only, no extra text."
    )
    user_prompt = f"Entity section:\n{entity_text}\n"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    # Attempt API call (with one retry on error or format issue)
    for attempt in range(2):
        try:
            logging.info("Calling GPT-4 API to parse an entity section...")
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0
            )
            content = response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI API call failed on attempt {attempt+1}: {e}")
            time.sleep(2)  # brief pause before retry
            continue
        # Clean up the response content (remove markdown code fences if present)
        content = content.strip()
        if content.startswith("```"):
            # Remove any triple backticks and language hints like ```json
            content = re.sub(r'^```[a-zA-Z]*\n', '', content)
            content = content.rstrip("`")
        # Try to load JSON
        try:
            data = json.loads(content)
            return data  # return parsed JSON as dict
        except json.JSONDecodeError as e:
            logging.warning("Failed to parse JSON on first attempt, will retry with stricter prompt.")
            # If parsing failed, instruct GPT-4 to correct its format
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": "The output was not valid JSON. Please correct it and output valid JSON only."
            })
            # loop continues to next attempt
    # If we exit the loop without returning, it failed both attempts
    logging.error("Could not parse entity section into JSON after retries.")
    return {}  # return empty dict (or could raise an exception)

def fill_missing_definitions(entities: List[Dict[str, Any]]):
    """
    For each entity in the list, find attributes with missing/NA definitions and 
    use GPT-4 to generate a definition based on the entity context.
    Modifies the entities list in place.
    """
    for entity in entities:
        entity_name = entity.get("entity_name", "")
        entity_def = entity.get("entity_definition", "")
        attributes = entity.get("attributes", [])
        if not isinstance(attributes, list):
            continue
        for attr in attributes:
            # Check if definition is missing or marked as NA/N/A
            def_val = str(attr.get("definition", "")).strip()
            if def_val == "" or def_val.upper() in {"NA", "N/A", "NONE"}:
                attr_name = attr.get("attribute_name") or attr.get("column_name") or "(unknown)"
                data_type = attr.get("data_type", "")
                prompt = (
                    f"The entity '{entity_name}' is described as: {entity_def}\n"
                    f"Provide a concise definition for the attribute '{attr_name}' (data type: {data_type})."
                )
                try:
                    logging.info(f"Filling missing definition for attribute '{attr_name}' in entity '{entity_name}'...")
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0
                    )
                    definition = response.choices[0].message.content.strip()
                    # Remove any formatting (if the model returned a fenced block, etc.)
                    if definition.startswith("```"):
                        definition = re.sub(r'^```[a-zA-Z]*\n', '', definition).rstrip("`").strip()
                    attr["definition"] = definition
                except Exception as e:
                    logging.error(f"Error generating definition for {entity_name}.{attr_name}: {e}")
                    # Leave definition as is (empty or NA) if API call fails
    # No return needed, list is modified in place

def main(pdf_path: str, output_json_path: str):
    """
    Orchestrates the entire extraction and transformation process.
    """
    # Step 1: Extract text from the PDF
    full_text = extract_text_from_pdf(pdf_path)
    # Step 2: Split text into entity sections
    entity_sections = split_into_entities(full_text)
    all_entities = []
    # Step 3: Parse each entity section with GPT-4
    for section in entity_sections:
        entity_data = parse_entity_to_json(section)
        if entity_data:
            all_entities.append(entity_data)
        else:
            logging.warning("Skipped an entity section due to parse issues.")
    # Step 4: Fill in missing attribute definitions
    fill_missing_definitions(all_entities)
    # Step 5: Write the results to a JSON file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_entities, f, indent=2)
    logging.info(f"Saved structured data to {output_json_path}")

# Example usage (in a Databricks notebook, replace paths as needed):
# pdf_file_path = "/dbfs/FileStore/data_dictionary.pdf"
# output_json_path = "/dbfs/FileStore/data_dictionary_structured.json"
# main(pdf_file_path, output_json_path)
