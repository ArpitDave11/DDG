import os
import fitz  # PyMuPDF for PDF text extraction
import openai
import json
import logging
from openai import OpenAIError

# Configuration and setup
PDF_PATH = "data_dictionary.pdf"
OUTPUT_JSON_PATH = "data_dictionary_knowledge_base.json"
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set!")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s")

try:
    # Step 1: Extract text from PDF
    doc = fitz.open(PDF_PATH)
    all_text = "".join([page.get_text() for page in doc])  # Read all pages:contentReference[oaicite:8]{index=8}
    doc.close()
    logging.info("Extracted text from PDF (%d characters)", len(all_text))
except Exception as e:
    logging.error("Failed to read PDF: %s", e)
    raise

try:
    # Step 2: Parse content with GPT-4 into JSON
    # Create prompt for GPT-4
    schema_description = (
        '{"entity_name": ..., "table_name": ..., "entity_definition": ..., "attributes": ['
        '{"attribute_name": ..., "column_name": ..., "data_type": ..., "is_primary_key": ..., "definition": ...}'
        ']}'
    )
    user_prompt = (
        f"Parse the following data dictionary and output a JSON array of entities with the schema: {schema_description}.\n"
        f"Only output JSON, no explanation.\n\n"
        f"Data Dictionary:\n{all_text}"
    )
    messages = [
        {"role": "system", "content": "You are a JSON formatter. Output only valid JSON."},
        {"role": "user", "content": user_prompt}
    ]
    response = openai.ChatCompletion.create(model="gpt-4", messages=messages, temperature=0)
    raw_json_output = response['choices'][0]['message']['content']
    data_dict = json.loads(raw_json_output)  # parse the JSON output
    logging.info("Parsed %d entities from GPT-4 output", len(data_dict))
except OpenAIError as e:
    logging.error("OpenAI API error during JSON parsing: %s", e)
    raise
except json.JSONDecodeError as e:
    logging.error("Failed to decode JSON output: %s", e)
    # Optionally save raw output for debugging
    with open("gpt_output_raw.txt", "w") as f:
        f.write(raw_json_output)
    raise

try:
    # Step 3: Generate definitions for missing attributes
    for entity in data_dict:
        entity_name = entity.get("entity_name", "<unknown>")
        entity_def = entity.get("entity_definition", "")
        for attr in entity.get("attributes", []):
            # Identify missing or placeholder definitions
            def_text = str(attr.get("definition", "")).strip().lower()
            if def_text == "" or def_text in {"na", "n/a", "none"}:
                attr_name = attr.get("attribute_name", attr.get("column_name", ""))
                col_name = attr.get("column_name", attr_name)
                # Prompt GPT-4 for attribute definition
                prompt = (f'Entity: "{entity_name}"\nDescription: {entity_def}\n'
                          f'Attribute: "{attr_name}" (Column: {col_name}). '
                          f'Provide a brief definition of this attribute in the context of the entity.')
                try:
                    resp = openai.ChatCompletion.create(model="gpt-4", 
                                                        messages=[{"role": "user", "content": prompt}],
                                                        max_tokens=100, temperature=0.5)
                    gen_def = resp['choices'][0]['message']['content'].strip().strip('"')
                    attr['definition'] = gen_def
                    logging.info("Filled missing definition for %s - %s", entity_name, attr_name)
                except OpenAIError as e:
                    logging.error("Failed to generate definition for %s - %s: %s", entity_name, attr_name, e)
                    # Set a fallback definition or leave it as is.
except Exception as e:
    logging.error("Error during missing definitions generation: %s", e)
    raise

try:
    # Step 4: Save the knowledge base to a JSON file
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4)
    logging.info("Saved knowledge base JSON to %s", OUTPUT_JSON_PATH)
except Exception as e:
    logging.error("Failed to write output JSON: %s", e)
    raise
