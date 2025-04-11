End-to-End  Using Azure OpenAI and RAG

The pipeline includes the following steps:

Loading Metadata from Azure Blob Storage

Building a Retrieval Index from Historical Fine-Tune Examples

Descriptions via RAG using the Fine-Tuned GPT‑4 Model

Storing Descriptions into PostgreSQL

Exposing a Real-Time API for On-Demand Generation

Triggering a Fine-Tuning Job on Azure OpenAI

Below you'll find detailed instructions to set up, run, and deploy the system.

Table of Contents
Project Structure

Prerequisites

Configuration

Dependencies



Sample Inputs and Expected Outputs

Deployment and Running the Project

Final Remarks

Project Structure

project/
├── config.yaml
├── requirements.txt
├── config.py
├── metadata_loader.py
├── embedding_store.py
├── description_generator.py
├── db_writer.py
├── main.py
├── api_service.py
└── trigger_fine_tuning.py
Prerequisites
Python 3.9+

An Azure Blob Storage account with metadata files (CSV/JSON) in a specified container.

An Azure OpenAI resource with a fine-tuned GPT‑4 model, plus an OpenAI deployment for embeddings (e.g. text-embedding-ada-002).

A PostgreSQL database with proper credentials.

Environment variables or configuration file to securely store API keys and connection strings.

Configuration
Create a file named config.yaml in the root directory and update it with your actual credentials:

File Descriptions and Detailed Steps
1. config.yaml
This file holds all configuration parameters such as connection strings, API keys, and model settings.

2. requirements.txt
Lists all the external libraries and packages required for the project.

3. config.py
Loads configuration from config.yaml and makes it available to other modules.

5. embedding_store.py
Loads historical fine-tuning examples from a JSONL file, computes embeddings (using OpenAI’s embedding API), and builds a FAISS vector index for fast similarity search.

python
Copy
# embedding_store.py
import openai, faiss, numpy as np, logging, json
from config import config

# Configure OpenAI for Azure
openai.api_type = "azure"
openai.api_base = config['azure']['openai_endpoint']
openai.api_key = config['azure']['openai_api_key']
openai.api_version = config['azure']['openai_api_version']

6. description_generator.py
Uses your fine-tuned GPT‑4 model (via Azure OpenAI) and RAG to generate attribute descriptions.
The constructed prompt includes system instructions, few-shot examples (retrieved from the vector index), and the new attribute query.


7. db_writer.py
Handles database connections and upserts generated descriptions into your PostgreSQL table.


# db_writer.py

8. main.py (Batch Generation Pipeline)
This script coordinates the entire process:

Loads metadata from Azure Blob

Loads fine-tuning examples and builds the FAISS index

Generates descriptions using RAG and your fine-tuned GPT‑4 model

Saves results into PostgreSQL

9. api_service.py (Real-time API with FastAPI)
This script provides a RESTful endpoint for on-demand description generation, reusing the same retrieval and generation logic.


10. trigger_fine_tuning.py (Fine-Tuning Job Trigger)
This script demonstrates how to trigger a fine-tuning job on Azure OpenAI using your JSONL training dataset.

python
Copy
# trigger_fine_tuning.py
import requests, json
from config import config

def trigger_fine_tuning_job():
    # Extract the short resource name from the endpoint URL
    resource_name = config['azure']['openai_endpoint'].split("//")[1].split(".")[0]
    api_key = config['azure']['openai_api_key']
    training_file_url = config['azure']['fine_tune_file_url']
    base_model = "gpt-4"  # The base model to fine-tune
    hyperparameters = {
        "n_epochs": config['settings']['fine_tuning_n_epochs'],
        "batch_size": config['settings']['fine_tuning_batch_size'],
        "learning_rate_multiplier": config['settings']['fine_tuning_learning_rate_multiplier']
    }
    payload = {
        "training_file": training_file_url,
        "model": base_model,
        "hyperparameters": hyperparameters
    }
    api_version = "2021-06-01-preview"  # Update this if needed per Azure OpenAI docs
    endpoint_url = f"https://{resource_name}.openai.azure.com/openai/fine-tunes?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    response = requests.post(endpoint_url, headers=headers, json=payload)
    if response.status_code == 202:
        print("Fine-tuning job submitted successfully!")
        print("Job details:", json.dumps(response.json(), indent=4))
    else:
        print("Error submitting fine-tuning job:", response.status_code, response.text)


Sample Inputs and Expected Outputs
Metadata Loader
Input: CSV file (e.g., CustomerAttributes.csv) with:

pgsql

    table_name,column_name,data_type,description
    Customer,CustomerID,INT,Unique identifier for the customer
    Customer,AnnualIncome,DECIMAL,Annual income of the customer
Output:

    [
        {"table": "Customer", "column": "CustomerID", "type": "INT", "existing_desc": "Unique identifier for the customer"},
        {"table": "Customer", "column": "AnnualIncome", "type": "DECIMAL", "existing_desc": "Annual income of the customer"}
    ]

Embedding Store
    Input: JSONL fine-tune dataset with prompt/completion pairs.
    
    Output: A FAISS index containing embeddings for each prompt and the list of examples.

Description Generator
Input:
      New attribute:
      Column: Customer.AnnualIncome (DECIMAL) – Provide a brief description:

Output:
      Generated description 
      "The annual income of the customer, measured in decimal format, representing their total earnings over the year."

Database Writer
      Input: Tuple(s):
      ("Customer", "AnnualIncome", "DECIMAL", <generated description>, "Annual income of the customer")

Output:
      Record inserted into the PostgreSQL table.

API Service
    Input (via HTTP POST):
      
      {
          "table": "Order",
          "column": "OrderDate",
          "data_type": "DATETIME"
      }

    Output (HTTP Response):

json

    {
        "table": "Order",
        "column": "OrderDate",
        "data_type": "DATETIME",
        "description": "The date and time when the order was placed."
    }
Tuning Trigger
    Input: JSON payload with hyperparameters submitted to Azure OpenAI’s endpoint.
    
    Output: HTTP response JSON indicating a fine-tuning job is queued.

Deployment and Running the Project
    Update Configuration:
    Modify config.yaml with your environment’s credentials and endpoints.

