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
