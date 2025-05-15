import json, re
from typing import Dict, List

# Optional: use tiktoken for accurate token counting (fallback to heuristic if not installed)
try:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")  # o3-mini uses similar tokenizer
    def count_tokens(text: str) -> int:
        return len(enc.encode(text))
except ImportError:
    # Approximate: 1 token ≈ 4 characters in English:contentReference[oaicite:3]{index=3}
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

# Token limits for safety
MAX_INPUT_TOKENS = 7000  # Max tokens to send in prompt (leave headroom for response)
MAX_OUTPUT_TOKENS = 1000  # Max tokens we expect in the completion (adjust as needed)

# Initialize Azure OpenAI model (assuming environment variables or credentials are set)
from langchain.chat_models import AzureChatOpenAI
llm = AzureChatOpenAI(
    deployment_name="o3-mini",
    # model="gpt-35-turbo",  # optionally specify model name if required
    temperature=0,
    max_tokens=MAX_OUTPUT_TOKENS
)

# Path to the PDF (in Databricks DBFS or local filesystem)
pdf_path = "/dbfs/mnt/genai/knowledge_base/Data_Dictionary-pdf"

# 1. Read PDF content and insert page break markers
text_content = ""
try:
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    pages_text = [page.extract_text() for page in reader.pages]
    # Join pages with a marker to allow splitting by page later
    text_content = "\n<PAGE_BREAK>\n".join(pages_text)
except Exception as e:
    # Fallback if PDF reading fails (e.g., already extracted text is available)
    with open(pdf_path, "r", encoding="utf-8") as f:
        text_content = f.read()

# 2. Split content by "Model :" sections
sections = re.split(r'(?=Model\s*:)', text_content)
sections = [sec.strip() for sec in sections if sec.strip()]

output_entities: Dict[str, Dict] = {}  # to merge results by Model|Entity key

# Prepare a system prompt for the model to ensure consistent JSON output
system_prompt = (
    "You are a data dictionary extraction assistant. Extract all entities and attributes from the given text. "
    "Return the output in **JSON** format with keys 'Model', 'Entity', and 'Attributes'. "
    "'Entity' should be an object (with at least 'Name' and 'Definition'), and 'Attributes' should be a list of objects "
    "(each with 'Name', 'Definition', 'DataType', 'Length'). "
    "If any field (definition, data type, length, etc.) is missing in the text, use 'N/A' as a placeholder. "
    "Do NOT include any explanation, only valid JSON."
)

for section in sections:
    # Extract model name (text after "Model :")
    lines = section.splitlines()
    if not lines:
        continue
    model_line = lines[0].strip()                   # e.g. "Model : Customer"
    model_name = model_line.split(":", 1)[1].strip() if ":" in model_line else model_line

    # Get section body (everything after the first line)
    section_body = section[len(model_line):].strip()
    if not section_body:
        continue

    # 3. Split by entities within this model section
    entity_parts = re.split(r'(?=Entity\s*:)', section_body)
    entity_parts = [part.strip() for part in entity_parts if part.strip()]

    # Handle model-level description before the first entity, if present
    model_intro = ""
    if entity_parts and not entity_parts[0].startswith("Entity"):
        model_intro = entity_parts[0]
        entity_parts = entity_parts[1:]  # remove intro from list

    # If no explicit "Entity:" was found, treat the whole section as one entity block
    if not entity_parts:
        entity_parts = [section_body]

    for idx, entity_text in enumerate(entity_parts):
        # Compose chunk text with Model line and (if available) model intro
        chunk_text = model_line + "\n"
        if idx == 0 and model_intro:  # attach model intro to first entity
            chunk_text += model_intro + "\n"
        # Ensure the entity_text starts with "Entity:" label for consistency
        if not entity_text.startswith("Entity"):
            # If missing, assume the model name itself is the entity name
            entity_text = f"Entity: {model_name}\n{entity_text}"
        chunk_text += entity_text

        # 4. Further split chunk_text if it exceeds token limit
        if count_tokens(chunk_text) <= MAX_INPUT_TOKENS:
            subchunks = [chunk_text]  # fits in one go
        else:
            # Split by page markers while preserving context
            pages = chunk_text.split("<PAGE_BREAK>")
            pages = [p.strip() for p in pages if p is not None]
            subchunks = []
            # Identify entity name (for adding header to continued chunks)
            ent_name_match = re.search(r'Entity\s*:\s*([^\n]+)', chunk_text)
            ent_name = ent_name_match.group(1).strip() if ent_name_match else model_name
            # Prepare header to prepend for subsequent subchunks (Model and Entity name)
            header = f"{model_line}\nEntity: {ent_name}\n"
            current_chunk = ""
            current_tokens = 0

            for page_text in pages:
                if not page_text:
                    continue
                page_tokens = count_tokens(page_text)
                if page_tokens > MAX_INPUT_TOKENS:
                    # If a single page is still too large, split by lines
                    lines = page_text.splitlines()
                    for line in lines:
                        line_tokens = count_tokens(line)
                        # If adding this line would overflow, finalize the current subchunk
                        if current_tokens + line_tokens > MAX_INPUT_TOKENS and current_chunk:
                            subchunks.append(current_chunk.strip())
                            current_chunk = ""
                            current_tokens = 0
                        # Add header for new chunk (if not the very first chunk)
                        if not current_chunk and subchunks:
                            current_chunk = header
                            current_tokens = count_tokens(header)
                        current_chunk += line + "\n"
                        current_tokens += line_tokens
                    # Flush remaining lines as a subchunk
                    if current_chunk:
                        subchunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                else:
                    # If adding the whole page stays within limit, accumulate
                    if current_tokens + page_tokens > MAX_INPUT_TOKENS and current_chunk:
                        # Close the current chunk and start a new one
                        subchunks.append(current_chunk.strip())
                        current_chunk = ""
                        current_tokens = 0
                    if not current_chunk and subchunks:
                        # Start a new chunk with header (for continued pages)
                        current_chunk = header
                        current_tokens = count_tokens(header)
                    current_chunk += page_text + "\n"
                    current_tokens += page_tokens
            # Add the last accumulated chunk
            if current_chunk:
                subchunks.append(current_chunk.strip())

        # 5. Send each subchunk to Azure OpenAI and get JSON output
        for subchunk_text in subchunks:
            if not subchunk_text:
                continue
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": subchunk_text}
            ]
            response = llm.invoke(messages)  # invoke the model with the prompt

            # The response is expected to be a JSON (string). Parse it:
            content = response if isinstance(response, str) else response.content
            try:
                parsed_output = json.loads(content)
            except json.JSONDecodeError:
                # If the model returned multiple JSON objects or extra text, try to extract JSON
                match = re.search(r'\{.*\}', content, flags=re.DOTALL)
                parsed_output = json.loads(match.group(0)) if match else None
            if parsed_output is None:
                continue  # skip if no JSON could be parsed

            # 6. Merge results into output_entities dict
            # The model might return a list of entities or a single entity object
            entities = parsed_output if isinstance(parsed_output, list) else [parsed_output]
            for ent_obj in entities:
                model_val = ent_obj.get("Model", model_name)
                ent_info = ent_obj.get("Entity", {})
                # Get entity name (assuming JSON structure is as instructed)
                entity_name = ""
                if isinstance(ent_info, dict):
                    entity_name = ent_info.get("Name", "") or ent_info.get("name", "")
                else:
                    entity_name = str(ent_info)
                    ent_obj["Entity"] = {"Name": entity_name}
                if entity_name == "":
                    entity_name = model_name  # fallback to model name if entity name missing
                    ent_obj["Entity"]["Name"] = entity_name

                key = f"{model_val}|{entity_name}"
                if key in output_entities:
                    # Merge attributes if this entity already has an entry (from a previous chunk)
                    existing = output_entities[key]
                    existing_attrs = existing.get("Attributes", [])
                    new_attrs = ent_obj.get("Attributes", [])
                    # Ensure both are lists
                    if isinstance(existing_attrs, list) and isinstance(new_attrs, list):
                        # Merge attribute lists by name to avoid duplicates
                        existing_attr_names = {attr.get("Name"): attr for attr in existing_attrs if isinstance(attr, dict)}
                        for attr in new_attrs:
                            if not isinstance(attr, dict) or "Name" not in attr:
                                continue
                            name = attr["Name"]
                            if name in existing_attr_names:
                                # Update placeholder values if the new chunk has real data
                                for k, v in attr.items():
                                    if k not in existing_attr_names[name] or existing_attr_names[name][k] == "N/A":
                                        existing_attr_names[name][k] = v
                            else:
                                # Add new attribute
                                existing_attrs.append(attr)
                    # Merge any missing Entity fields (e.g., Definition) if present
                    if "Entity" in existing and isinstance(ent_obj.get("Entity"), dict):
                        for k, v in ent_obj["Entity"].items():
                            if k not in existing["Entity"] or existing["Entity"][k] == "N/A":
                                existing["Entity"][k] = v
                    output_entities[key] = existing
                else:
                    # New entity, add to dictionary
                    output_entities[key] = ent_obj

# 7. Write results to JSON Lines file
with open("output.jsonl", "w", encoding="utf-8") as outfile:
    for ent in output_entities.values():
        outfile.write(json.dumps(ent, ensure_ascii=False) + "\n")
print(f"Extraction complete. {len(output_entities)} entities written to output.jsonl.")


##################



import os
import re
import json

# 1. Load PDF and extract full text
try:
    from PyPDF2 import PdfReader
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "PyPDF2"], check=True)
    from PyPDF2 import PdfReader

pdf_path = "/dbfs/mnt/genai/knowledge_base/Data_Dictionary-pdf"
reader = PdfReader(pdf_path)
text_content = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text_content += page_text + "\n"

# 2. Identify and separate sections by "Model :"
# Find the first occurrence of "Model :" and ignore anything before it (preface or intro)
first_model_index = text_content.find("Model :")
if first_model_index == -1:
    raise ValueError("No 'Model :' section found in the PDF text.")
if first_model_index > 0:
    text_content = text_content[first_model_index:]

# Find all indices where a model section starts
model_positions = [m.start() for m in re.finditer(r'Model\s*:\s*', text_content)]
model_positions.append(len(text_content))  # add end of text as last boundary

# Slice out each model section text
model_sections = []
for i in range(len(model_positions) - 1):
    section_text = text_content[model_positions[i]: model_positions[i+1]]
    section_text = section_text.strip()
    if not section_text:
        continue
    model_sections.append(section_text)

# Ensure we have at least one model section
if not model_sections:
    raise ValueError("No model sections could be extracted from the text.")

# 3. Set up Azure OpenAI LLM (AzureChatOpenAI) with environment variables
endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION")
if not endpoint or not deployment or not api_key or not api_version:
    raise EnvironmentError(
        "Missing one of the required Azure OpenAI environment variables: "
        "ENDPOINT_URL, DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION."
    )

# Map the provided env vars to those expected by AzureChatOpenAI
os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
os.environ["AZURE_OPENAI_API_KEY"] = api_key
# We will pass the deployment name and API version directly to the AzureChatOpenAI initializer

try:
    from langchain_openai import AzureChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "langchain-openai"], check=True)
    from langchain_openai import AzureChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage

# Initialize the AzureChatOpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=deployment,
    api_version=api_version,
    temperature=0
)

# 4. Define the system prompt for parsing instructions
system_prompt = (
    "You are an assistant that extracts structured data from a data dictionary document.\n"
    "The user will provide text for a data model, including the model name and one or more entities with attributes.\n"
    "Your task is to output a JSON **array** of objects, where each object represents one entity from the text with the following format:\n"
    "{\n"
    '  "Model": "<Model Name>",\n'
    '  "Entity": {\n'
    '    "TABLE_NAME": "<table name of the entity or \'Not available\'>",\n'
    '    "ENTITY NAME": "<entity name or \'Not available\'>",\n'
    '    "DEFINITION": "<entity definition or \'No definition available\'>"\n'
    '  },\n'
    '  "Attributes": [\n'
    '    {\n'
    '      "NAME": "<attribute name or \'Not available\'>",\n'
    '      "DEFINITION": "<attribute definition or \'No definition available\'>",\n'
    '      "Column Name": "<column name or \'Not available\'>",\n'
    '      "Column Data Type": "<column data type or \'Not available\'>",\n'
    '      "PK?": "<Yes or No or \'Not available\'>"\n'
    '    }, ...\n'
    '  ]\n'
    '}\n'
    "Include **all** entities and their attributes from the input text. Do not skip any.\n"
    "If any field value is missing in the text, use \"Not available\" (for general fields) or \"No definition available\" (for missing definitions).\n"
    "Do not add explanations or any extra text outside of the JSON. Only output the JSON data structure as specified.\n"
    "Use the exact key names and formatting shown (e.g., \"ENTITY NAME\" with a space, \"PK?\" with a question mark)."
)

# 5. Parse each model section with the LLM and collect results
all_entities = []  # list to hold all entity JSON objects
for section in model_sections:
    # Prepare messages for the LLM: system instructions + human prompt with the section text
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=section)
    ]
    # Invoke the Azure OpenAI model to get the parsed JSON output
    response = llm.invoke(messages)  # :contentReference[oaicite:5]{index=5}
    output_text = response.content

    # Attempt to load the response content as JSON
    try:
        parsed_output = json.loads(output_text)
    except json.JSONDecodeError:
        # If the model response is not directly parseable, try minor cleanup (e.g., remove code fences)
        cleaned = output_text.strip().strip("```").strip()
        try:
            parsed_output = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM output: {e}\nModel output was: {output_text}")

    # The model is instructed to return a JSON array of entities.
    # If it returned a single object (in case of one entity), wrap it into a list for consistency.
    if isinstance(parsed_output, dict):
        parsed_output = [parsed_output]
    elif not isinstance(parsed_output, list):
        raise ValueError("Unexpected JSON output format from LLM (expected list of objects).")

    # Append each entity object to the master list
    for entity_obj in parsed_output:
        # Ensure required keys exist, fill placeholders if missing (safety check, model should handle this)
        if "Model" not in entity_obj:
            entity_obj["Model"] = "Not available"
        if "Entity" not in entity_obj:
            entity_obj["Entity"] = {"TABLE_NAME": "Not available", "ENTITY NAME": "Not available", "DEFINITION": "No definition available"}
        if "Attributes" not in entity_obj:
            entity_obj["Attributes"] = []
        # Also ensure within Entity, each key exists
        ent = entity_obj["Entity"]
        if "TABLE_NAME" not in ent:
            ent["TABLE_NAME"] = "Not available"
        if "ENTITY NAME" not in ent:
            ent["ENTITY NAME"] = "Not available"
        if "DEFINITION" not in ent:
            ent["DEFINITION"] = "No definition available"
        # For each attribute in the list, ensure all keys are present
        for attr in entity_obj.get("Attributes", []):
            if "NAME" not in attr:
                attr["NAME"] = "Not available"
            if "DEFINITION" not in attr:
                attr["DEFINITION"] = "No definition available"
            if "Column Name" not in attr:
                attr["Column Name"] = "Not available"
            if "Column Data Type" not in attr:
                attr["Column Data Type"] = "Not available"
            if "PK?" not in attr:
                attr["PK?"] = "Not available"

        all_entities.append(entity_obj)

# 6. Write the results to output.jsonl (each entity as one JSON line)
output_path = "output.jsonl"
with open(output_path, "w") as outfile:
    for entity_obj in all_entities:
        json.dump(entity_obj, outfile)
        outfile.write("\n")

print(f"Parsing complete. JSONL output saved to {output_path}")
