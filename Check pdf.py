Thanks for confirming. I’ll now generate a complete, updated Python script that:

* Uses Azure OpenAI via `AzureChatOpenAI` from `langchain_openai` (no deprecated code)
* Loads your credentials from environment variables
* Reads model/entity/attribute text from `input.txt`
* Parses and sends each model section to Azure OpenAI with structured prompts
* Outputs a final `output.json` file with the correct JSON schema

I’ll make sure it runs cleanly in PyCharm and includes all dependencies and error handling. I’ll notify you when it’s ready.


# Azure OpenAI Metadata Extraction Script

This Python script reads a raw text file of model specifications and uses an Azure-hosted OpenAI model (via LangChain’s AzureChatOpenAI) to extract structured metadata in JSON format. It splits the input text by each **Model** section and prompts the LLM to return the model name, its entities, and their attributes in a JSON structure. The final result is an array of JSON objects (one per model) written to `output.json` and also printed to the console. We include robust error handling for JSON parsing and API errors, and the script uses the latest LangChain interfaces (no deprecated methods).

## Installation and Setup

Make sure to install the required packages and set up your Azure OpenAI credentials:

```bash
pip install -U langchain-openai openai
```

Set the following environment variables in your system (or in PyCharm run configuration) before running the script:

* **ENDPOINT\_URL** – Your Azure OpenAI endpoint (e.g. `https://<resource-name>.openai.azure.com/`).
* **DEPLOYMENT\_NAME** – The name of your chat model deployment (e.g. `gpt-4` or `gpt-35-turbo`).
* **AZURE\_OPENAI\_API\_KEY** – Your Azure OpenAI API key.
* **OPENAI\_API\_VERSION** – The API version to use (e.g. `2024-12-01-preview`).

These correspond to the Azure OpenAI configuration parameters. For example, `AZURE_OPENAI_ENDPOINT` (here provided via `ENDPOINT_URL`) is the base URL of your Azure resource, and `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` (here `DEPLOYMENT_NAME`) is the deployment model name. The script will load these from `os.environ` and pass them to `AzureChatOpenAI` during initialization.

## Approach Outline

The script performs the following steps:

1. **Read Input File:** It opens `input.txt` (which contains the extracted text from the 141-page PDF) and reads the entire content into a string. It’s assumed that the text uses a consistent format where each model section starts with a line beginning with `"Model : "` followed by the model name.

2. **Split into Sections:** Using the marker `"Model : "` as a delimiter, the script splits the text into sections. Each section corresponds to one model and contains that model’s entities and attributes descriptions. We handle edge cases (like an empty first split result if the text starts with "Model :") by skipping empty chunks.

3. **Initialize Azure OpenAI LLM:** We instantiate an `AzureChatOpenAI` chat model with the appropriate parameters for Azure:

   * `azure_endpoint` – loaded from `ENDPOINT_URL`
   * `azure_deployment` – loaded from `DEPLOYMENT_NAME`
   * `openai_api_key` – loaded from `AZURE_OPENAI_API_KEY`
   * `openai_api_version` – loaded from `OPENAI_API_VERSION`
   * `temperature=0` for deterministic output (suitable for extraction tasks)

   The `AzureChatOpenAI` class reads these settings and prepares the API call to Azure OpenAI. We ensure all required environment variables are present, otherwise the script will alert and stop.

4. **Prepare LLM Prompt:** For each model section, the script constructs a prompt with a **system message** and a **human message**:

   * **SystemMessage:** Establishes the role of the assistant (e.g. *“You are a helpful assistant that extracts structured metadata from text...”*). This primes the model to focus on JSON extraction.
   * **HumanMessage:** Contains instructions and the raw text of the model section. We explicitly describe the JSON schema we expect (keys for model name, entities, attributes, etc.) and instruct the model to output **only JSON** with no extra commentary. We also note that if a definition is missing in the text, it should output `"No definition available"` for that field.

   Using LangChain’s message classes, we create a list like:

   ```python
   messages = [
       SystemMessage(content="<system instructions>"),
       HumanMessage(content="<prompt with model text>")
   ]
   ```

   Then we call the model with `llm.invoke(messages)` to get the completion. The `.invoke()` method sends the sequence of messages to Azure OpenAI and returns an `AIMessage` response object (which includes the model’s content in `result.content`).

5. **Parse JSON Output:** We take the `result.content` (the model’s answer) and attempt to parse it with Python’s `json.loads()`. The assistant is instructed to return valid JSON only. However, in case the model’s output is not strictly valid JSON (for example, if it included extraneous text or formatting), we have exception handling to catch `json.JSONDecodeError`. In the exception handler, the script will attempt a simple cleanup (such as stripping out any non-JSON text or Markdown code fences) and try parsing again. If it still fails, the script logs an error for that section and skips it.

6. **Collect Results:** Each parsed JSON (a Python dict) for a model is appended to a list `models_data`. The expected structure for each model’s data is:

   ```json
   {
     "MODEL_NAME": "Example Model",
     "entities": [
       {
         "ENTITY_NAME": "...",
         "TABLE_NAME": "...",
         "DEFINITION": "..."
       },
       … 
     ],
     "attributes": [
       {
         "NAME": "...",
         "DEFINITION": "...",
         "COLUMN_NAME": "...",
         "COLUMN_TYPE": "...",
         "PK": true/false
       },
       … 
     ]
   }
   ```

   If a definition is missing in the input, the model will use `"No definition available"` as the value.

7. **Output to JSON File and Console:** After processing all sections, the script serializes the `models_data` list to pretty-formatted JSON and writes it to `output.json`. It also prints the JSON to the console for immediate viewing.

## Exception Handling

The script includes comprehensive error handling:

* **Environment Errors:** If any required environment variable is missing, it prints an error message and exits, so the user knows to set up credentials properly.
* **API Call Errors:** The Azure OpenAI invocation is wrapped in a try/except block. Any exception during the API call (e.g. due to incorrect credentials, network issues, or Azure errors) will be caught, logged with the model name, and the script will continue to the next section (instead of crashing).
* **JSON Decoding Errors:** If the LLM’s response isn’t valid JSON on the first try, the script catches the `JSONDecodeError`. In the handler, we attempt to clean the output (for example, remove \`\`\`json code block markers or stray characters) and parse again. If it still fails, an error is logged and that model’s data is skipped. This way, one malformed section won’t break the entire process.

By handling these exceptions, the script can run to completion and inform you of any issues rather than silently failing.

## Full Python Script

Below is the complete Python script implementing the above logic. You can run this script in PyCharm (or any environment) after installing the packages and setting the environment variables as described. It will produce an `output.json` file with the array of model metadata.

````python
import os
import re
import json
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Ensure required environment variables are present
required_vars = ["ENDPOINT_URL", "DEPLOYMENT_NAME", "AZURE_OPENAI_API_KEY", "OPENAI_API_VERSION"]
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")

# Load Azure OpenAI configuration from environment
endpoint = os.environ["ENDPOINT_URL"]
deployment = os.environ["DEPLOYMENT_NAME"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
api_version = os.environ["OPENAI_API_VERSION"]

# Initialize the Azure OpenAI chat model (LLM) with given credentials
try:
    llm = AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        openai_api_key=api_key,
        openai_api_version=api_version,
        temperature=0  # use deterministic output for extraction
    )
except Exception as e:
    # If initialization fails (e.g., invalid credentials), exit with error
    raise RuntimeError(f"Failed to initialize AzureChatOpenAI: {e}")

# Read the entire input file
input_file = "input.txt"
try:
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()
except FileNotFoundError:
    raise FileNotFoundError(f"Input file '{input_file}' not found. Ensure the file exists in the script directory.")

# Split the content by 'Model : ' sections
sections = content.split("Model : ")
models_data = []

for idx, section in enumerate(sections):
    section = section.strip()
    if not section:
        continue  # skip empty result (e.g., before the first "Model :")
    # The first line of each section (after splitting) should be the model name
    lines = section.splitlines()
    model_name = lines[0].strip()
    # The rest of the lines correspond to entities and attributes of this model
    model_description = "\n".join(lines[1:]).strip()

    # Prepare the system and human messages for the LLM
    system_msg = SystemMessage(content=(
        "You are a helpful assistant that extracts structured metadata from text according to a specified JSON format."
    ))
    human_msg = HumanMessage(content=(
        f"Extract the model name, entities, and attributes from the text below and output them in a JSON format with the keys "
        f"MODEL_NAME, entities, and attributes as described. Each entity should include ENTITY_NAME, TABLE_NAME, and DEFINITION. "
        f"Each attribute should include NAME, DEFINITION, COLUMN_NAME, COLUMN_TYPE, and PK (use true/false for PK). "
        f"If a definition is missing in the text, use \"No definition available\" as its value. Do not include any text other than the JSON.\n\n"
        f"Text to extract from:\n```text\nModel: {model_name}\n{model_description}\n```"
    ))

    # Call the LLM to get structured metadata for this model
    try:
        result_message = llm.invoke([system_msg, human_msg])
    except Exception as e:
        print(f"Error during LLM API call for model '{model_name}': {e}")
        continue

    # Get the content of the AI's response
    response_content = result_message.content if hasattr(result_message, "content") else str(result_message)

    # Try to parse the JSON content
    try:
        model_data = json.loads(response_content)
    except json.JSONDecodeError as je:
        # Attempt to clean minor formatting issues and retry
        cleaned = response_content.strip()
        # Remove common extraneous markers like markdown code fences or language tags
        cleaned = re.sub(r'^```[\w]*\n', '', cleaned)  # remove ```json or ```text at start
        cleaned = cleaned.rstrip('`')  # remove trailing backticks if any
        cleaned = cleaned.strip()
        try:
            model_data = json.loads(cleaned)
        except json.JSONDecodeError as je2:
            print(f"JSON decoding failed for model '{model_name}': {je2}")
            continue  # skip this model if we cannot parse the output

    # Ensure the model name in output matches (or assign if missing)
    if isinstance(model_data, dict) and "MODEL_NAME" in model_data:
        # If MODEL_NAME is present but empty, fill it
        if not model_data["MODEL_NAME"]:
            model_data["MODEL_NAME"] = model_name
    else:
        # If the LLM returned a list or missing key, insert the model name
        model_data = {"MODEL_NAME": model_name, **(model_data if isinstance(model_data, dict) else {})}

    models_data.append(model_data)

# Save the results to output.json
with open("output.json", "w", encoding="utf-8") as outfile:
    json.dump(models_data, outfile, indent=2)

# Also print the JSON to console
print(json.dumps(models_data, indent=2))
````

**How it works:** The script goes through each model section, uses `AzureChatOpenAI.invoke()` with a SystemMessage and HumanMessage prompt to get a JSON snippet, then parses it into Python data. We accumulate all model data in a list and finally output it as JSON. Using LangChain’s `AzureChatOpenAI` ensures we leverage Azure’s hosted GPT model with the given deployment and API version. The use of `SystemMessage` and `HumanMessage` follows LangChain’s recommended practice for chat models, and `.invoke()` sends the messages to the model and returns the assistant’s reply. All sensitive values (keys, endpoint) are loaded from environment variables for security, and no deprecated methods (like `.predict`) are used.
