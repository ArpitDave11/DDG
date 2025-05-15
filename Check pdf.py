import os
import re
import json
# We'll use PyPDF2 for PDF reading
from PyPDF2 import PdfReader
# LangChain Azure OpenAI chat model and message schema
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
# Tokenizer for estimating tokens (tiktoken for OpenAI encodings)
import tiktoken

# === 1. Configuration and Setup ===

# PDF file path (in DBFS or local filesystem)
pdf_path = "/dbfs/mnt/genai/knowledge_base/Data_Dictionary.pdf"  # Assuming .pdf extension

# Output file path for JSONL results
output_path = "output.jsonl"  # or use "/dbfs/mnt/genai/knowledge_base/output.jsonl" to save to DBFS

# Azure OpenAI environment variables (ensure these are set in your environment)
endpoint = os.getenv("ENDPOINT_URL")          # e.g. "https://<your-resource-name>.openai.azure.com/"
deployment_name = os.getenv("DEPLOYMENT_NAME")  # e.g. "o3-mini"
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION")   # e.g. "2023-05-15" or the appropriate API version

# Validate that required environment variables are present
if not all([endpoint, deployment_name, api_key, api_version]):
    raise EnvironmentError("Please set ENDPOINT_URL, DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY, and OPENAI_API_VERSION environment variables for Azure OpenAI.")

# Initialize the Azure OpenAI chat model via LangChain
llm = AzureChatOpenAI(
    openai_api_base=endpoint,
    deployment_name=deployment_name,
    openai_api_version=api_version,
    openai_api_key=api_key,
    # Optionally, you can set temperature=0 for deterministic output since we want structured parsing
    temperature=0
)

# Initialize the tokenizer for the model (using tiktoken with GPT-3.5/GPT-4 encoding, assuming o3-mini is similar to GPT-3.5-turbo)
token_encoder = tiktoken.get_encoding("cl100k_base")

# Set a token limit threshold (to ensure prompt + text stays within model context window, ~8192 tokens for 8k models)
TOKEN_THRESHOLD = 7000  # we will aim to keep tokens per chunk below this number

# === 2. Read PDF and extract text with page markers ===

print("Reading PDF document...")
reader = PdfReader(pdf_path)
pages_text = []
for page in reader.pages:
    text = page.extract_text()
    if text:
        # Strip to remove leading/trailing whitespace/newlines
        text = text.strip()
    else:
        text = ""
    pages_text.append(text)
# Join pages with <PAGE_BREAK> marker to denote page boundaries
full_text = "\n<PAGE_BREAK>\n".join(pages_text)
print(f"PDF contains {len(pages_text)} pages of content.")

# === 3. Extract the Model name (first occurrence of "Model : <Name>") ===

model_name = None
model_match = re.search(r"Model\s*:\s*([^\n]+)", full_text, flags=re.IGNORECASE)
if model_match:
    model_name = model_match.group(1).strip()
    print(f"Model name found: '{model_name}'")
else:
    print("Model name not found in text.")
    model_name = "Not available"

# === 4. Split text into entity-level chunks ===

# Use regex to find all entity start positions (lines starting with "Entity", "Entity Name", etc.)
entity_pattern = re.compile(r"^(?:Entity\s*Name|Entity|ENTITY)\s*:\s*", flags=re.IGNORECASE | re.MULTILINE)
entity_positions = [m.start() for m in entity_pattern.finditer(full_text)]

if not entity_positions:
    raise ValueError("No entities found in the PDF text. Check the PDF format or regex pattern.")

# Add end of text position to help define the last entity's boundary
entity_positions.append(len(full_text))

# Slice the full_text into chunks for each entity
entity_chunks = []
for i in range(len(entity_positions) - 1):
    start_idx = entity_positions[i]
    end_idx = entity_positions[i+1]
    # Get text for this entity (from start index up to just before next entity)
    chunk_text = full_text[start_idx:end_idx].strip()
    if not chunk_text:
        continue
    entity_chunks.append(chunk_text)

print(f"Found {len(entity_chunks)} entities in the model.")

# === 5. Further split large entity chunks to respect token limits ===

# Function to count tokens of a text using the tokenizer
def count_tokens(text):
    # We add a few tokens as buffer for prompt overhead (system/user formatting)
    # but here just count text tokens for simplicity.
    tokens = token_encoder.encode(text)
    return len(tokens)

# We will create a list of sub-chunks (each sub-chunk will still pertain to a single entity)
entity_subchunks = []  # list of tuples: (entity_id, subchunk_text)
for idx, chunk in enumerate(entity_chunks):
    token_count = count_tokens(chunk)
    if token_count <= TOKEN_THRESHOLD:
        # This chunk is within token limit, use as is
        entity_subchunks.append((idx, chunk))
    else:
        # If too large, split by page breaks (or by lines) to create sub-chunks
        print(f"Entity {idx} is large (approx {token_count} tokens). Splitting into sub-chunks...")
        # Split by the page markers for this chunk
        # (We include the marker itself in split output to know where pages split)
        pages_in_chunk = chunk.split("<PAGE_BREAK>")
        # Remove any empty segments and strip each
        pages_in_chunk = [p.strip() for p in pages_in_chunk if p.strip()]
        # Now combine pages incrementally until threshold reached, then start new subchunk
        current_subchunk = ""
        current_token_count = 0
        for page_text in pages_in_chunk:
            page_token_count = count_tokens(page_text)
            if current_token_count + page_token_count <= TOKEN_THRESHOLD:
                # If adding this page stays within limit, add to current subchunk
                if current_subchunk:
                    # If already some content, insert page break marker between pages in subchunk
                    current_subchunk += "\n<PAGE_BREAK>\n" + page_text
                else:
                    current_subchunk = page_text
                current_token_count += page_token_count
            else:
                # Current page would overflow the limit, so finalize the current subchunk and start a new one
                if current_subchunk:
                    entity_subchunks.append((idx, current_subchunk))
                # Start new subchunk with the current page text
                current_subchunk = page_text
                current_token_count = page_token_count
        # Append the last subchunk for this entity
        if current_subchunk:
            entity_subchunks.append((idx, current_subchunk))

# Sort subchunks by their entity index to preserve original order
# (Though we collected in order, splitting might have appended out of sequence if multiple subchunks for one entity)
entity_subchunks.sort(key=lambda x: x[0])

print(f"Total chunks to process (including sub-chunks): {len(entity_subchunks)}")

# === 6. Define the system prompt and perform LLM parsing for each chunk ===

# Define the system message prompt enforcing the JSON output format
system_prompt = f"""You are a data extraction assistant. Parse the given text into a specific JSON format without explanation.
The JSON format should be:

{{
  "Model": "{model_name}",
  "Entity": {{
    "TABLE_NAME": "<table name>",
    "ENTITY NAME": "<entity name>",
    "DEFINITION": "<entity definition>"
  }},
  "Attributes": [
    {{
      "NAME": "<attribute name>",
      "DEFINITION": "<definition or 'No definition available'>",
      "Column Name": "<column name or 'Not available'>",
      "Column Data Type": "<column type or 'Not available'>",
      "PK?": "<Yes|No|Not available>"
    }}
  ]
}}

Guidelines:
- Only output a **single JSON object** per entity. Do not list multiple entities at once.
- Use the exact keys "Model", "Entity", and "Attributes".
- Fill "Model" with the model name "{model_name}".
- Under "Entity", provide "TABLE_NAME", "ENTITY NAME", and "DEFINITION" for this entity.
- Under "Attributes", list each attribute as an object with keys "NAME", "DEFINITION", "Column Name", "Column Data Type", and "PK?".
- If any field is missing in the text, use "Not available" (or "No definition available" for a missing definition) as the value.
- "PK?" should be "Yes" or "No" if specified, otherwise "Not available".
- Output **only** valid JSON, no extra commentary.
"""

# Prepare a list to collect parsed results
parsed_entities = []  # each element will be a dict representing one entity's data

print("Parsing text chunks with Azure OpenAI...")

for (entity_idx, sub_text) in entity_subchunks:
    # Create user prompt message with the chunk text
    # We include a brief instruction and the text to parse
    user_prompt = f"Extract the entity and attributes from the following text:\n{sub_text}"
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    try:
        # Invoke the Azure OpenAI model with the system and user messages
        response = llm.invoke(messages)
        raw_output = response.content  # the assistant's raw JSON string output
    except Exception as e:
        print(f"LLM API call failed for entity index {entity_idx}: {e}")
        continue  # skip this chunk on failure

    # Clean and parse the JSON output
    output_str = raw_output.strip()
    # If the model mistakenly included markdown or code fences, remove them
    if output_str.startswith("```"):
        # Remove markdown code block markers if present
        output_str = output_str.strip('`')
        # If it started with ```json, remove that as well
        output_str = output_str.replace("json\n", "").replace("json\r\n", "")
    # Trim any content before the first '{' or after the last '}'
    if "{" in output_str:
        start_idx = output_str.index("{")
    else:
        start_idx = 0
    if "}" in output_str:
        end_idx = output_str.rfind("}")
    else:
        end_idx = len(output_str) - 1
    output_str = output_str[start_idx:end_idx+1]

    # Attempt to load JSON
    try:
        parsed = json.loads(output_str)
    except json.JSONDecodeError as je:
        # If JSON decoding fails, attempt a fallback by correcting common issues
        print(f"JSON decode error for entity index {entity_idx}: {je}. Retrying with cleaned output...")
        # Remove any trailing or leading non-JSON text one more time
        # (e.g., sometimes a model might add extra text after JSON)
        # We will extract the substring from first '{' to last '}' again (in case of extra chars outside)
        if "{" in raw_output and "}" in raw_output:
            start = raw_output.index("{")
            end = raw_output.rfind("}")
            cleaned = raw_output[start:end+1]
        else:
            cleaned = raw_output
        # Second try to parse
        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            print(f"Failed to parse JSON for entity index {entity_idx} after cleanup: {e}")
            continue  # skip if we cannot get valid JSON

    # Store the parsed JSON object along with the entity index (to help merging by entity)
    parsed_entities.append((entity_idx, parsed))

# === 7. Merge results for entities split into multiple chunks & remove duplicates ===

print("Merging parsed results...")
merged_entities = {}  # key: entity (table_name, entity_name), value: merged JSON dict

for (entity_idx, data) in parsed_entities:
    if "Entity" not in data or "Attributes" not in data:
        # Skip if the output is not in expected format
        continue
    # Extract identifying fields for the entity
    entity_info = data["Entity"]
    table_name = entity_info.get("TABLE_NAME") or "Not available"
    entity_name = entity_info.get("ENTITY NAME") or "Not available"
    # Use a tuple key (table_name, entity_name) to identify unique entities
    entity_key = (table_name, entity_name)
    # Prepare a normalized JSON object for this entity if not already present
    if entity_key not in merged_entities:
        merged_entities[entity_key] = {
            "Model": model_name if model_name else "Not available",
            "Entity": {
                "TABLE_NAME": table_name,
                "ENTITY NAME": entity_name,
                "DEFINITION": entity_info.get("DEFINITION", "Not available")
            },
            "Attributes": []
        }
    # Merge attributes: append new attributes, avoiding duplicates by name
    existing_attrs = merged_entities[entity_key]["Attributes"]
    existing_attr_names = {attr.get("NAME", "") for attr in existing_attrs}
    for attr in data.get("Attributes", []):
        attr_name = attr.get("NAME", "")
        if attr_name and attr_name in existing_attr_names:
            # If this attribute already in list, we can skip or update if missing info
            # Here we choose to skip duplicates to avoid repetition
            continue
        existing_attrs.append(attr)
        if attr_name:
            existing_attr_names.add(attr_name)

# Write merged entities to JSONL output file
with open(output_path, "w", encoding="utf-8") as f_out:
    for entity_key, entity_data in merged_entities.items():
        json_line = json.dumps(entity_data, ensure_ascii=False)
        f_out.write(json_line + "\n")

print(f"Extraction complete. {len(merged_entities)} entities written to {output_path}")
