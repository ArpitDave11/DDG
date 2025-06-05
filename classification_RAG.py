Thanks! I’ll now provide production-ready Python code that:

* Uses a `.env` file to securely manage all DB and Azure OpenAI credentials.
* Supports Azure OpenAI (with deployment name, API base, API key, and version).
* Includes structured error handling, logging, and modular function design.

I’ll update you shortly with the full refactored code and dotenv integration.


# Production-Ready Embedding and Classification Pipeline

To fulfill the requirements, we refactored the embedding and classification pipeline with proper environment variable management, Azure OpenAI configuration, error handling, modular structure, and CLI support. Below we outline the key changes and then provide the complete code and an example `.env` file.

## Environment Variables Management with `.env`

We use **python-dotenv** to load configuration from a `.env` file, avoiding hard-coded secrets and following 12-factor app principles. All sensitive parameters (database credentials and Azure OpenAI keys/URLs) are stored in the `.env` file and loaded via `load_dotenv`. The required keys in `.env` are:

* **PostgreSQL settings:** `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
* **Azure OpenAI settings:** `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_BASE` (the endpoint URL), `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT_NAME`

Using `load_dotenv()` will read these values and make them available via `os.getenv`. This keeps credentials secure and easily configurable.

## Azure OpenAI Embedding Configuration

The code is configured to use **Azure OpenAI**'s `text-embedding-ada-002` model for embeddings. We set `openai.api_type = "azure"` and supply the `api_base`, `api_version`, and `api_key` from environment variables. Notably, Azure OpenAI requires using the *deployment name* for the model (not just the model name) in API calls. Thus, we pass the deployment name (from `AZURE_OPENAI_DEPLOYMENT_NAME`) when requesting embeddings. The `text-embedding-ada-002` model produces 1536-dimensional vectors, so our database schema uses `VECTOR(1536)` for storage.

When generating an embedding, we replace any newline characters in the text with spaces (as recommended for better embedding quality) and call the Azure OpenAI embeddings endpoint. We handle Azure OpenAI API errors with try/except blocks to ensure the application logs the error and continues or exits gracefully, as appropriate (production code should handle such errors properly).

## PostgreSQL and pgvector Setup

We ensure the **pgvector** extension is enabled and the database table is configured with a vector column for embeddings. On startup, the code executes:

* `CREATE EXTENSION IF NOT EXISTS vector;` – enables pgvector (logged as a warning if the database user lacks permission, but we proceed if it's already installed).
* `CREATE TABLE IF NOT EXISTS embeddings (...)` – creates a table with columns for `id` (primary key), `text`, `embedding VECTOR(1536)`, and `label`. The vector dimension 1536 matches the OpenAI embedding size.

We also create a vector index on the embedding column for efficient similarity search:

```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_embedding 
ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
```

This uses the IVF Flat index with cosine distance, as cosine similarity is recommended for OpenAI embeddings. The index significantly speeds up the nearest-neighbor search on the `embedding` column.

## Modular Code Structure and Error Handling

The pipeline is structured into clear components:

* **`load_data(file_path)`** – reads input data (e.g. a CSV) and returns a list of texts and labels for ingestion.
* **`generate_embedding(text)`** – calls the Azure OpenAI API to get the embedding vector for a given text.
* **`ensure_database(conn)`** – ensures the pgvector extension, table, and index exist in the PostgreSQL database.
* **`ingest_data(conn, data_list)`** – orchestrates the ingestion process: for each record, generates embedding and inserts it into the DB.
* **`classify_text(conn, text)`** – generates an embedding for the input text and finds the closest stored vector in the database to predict the label (k-Nearest Neighbor classification with k=1).

Each function has try/except blocks to catch errors. We log meaningful messages at each step (using Python’s `logging` module). For example, database connection issues, OpenAI API errors, or insertion failures are caught and logged as errors, with the program safely terminating or skipping problematic records. This structured error handling ensures the pipeline can run reliably in production, with visibility into any issues via logs.

A basic logging configuration is set at startup (`logging.basicConfig(...)`) to include timestamps and log levels, helping with debugging and monitoring in production. Key actions (connecting to DB, successful insertion count, classification result, etc.) are logged at INFO level, while exceptions are logged at ERROR level (including the exception details).

## CLI Support with `argparse`

The script supports command-line execution for two modes: **ingestion** and **classification**. We use the `argparse` library to define sub-commands:

* `ingest`: Ingests data from a provided file (e.g. `--file data.csv`) and stores embeddings in the database.
* `classify`: Classifies a given input text (provided via `--text "some query"`) by finding the nearest embedding in the database and returning the associated label.

This allows running the script as:

```bash
# Ingest data from a CSV file
python pipeline.py ingest --file data.csv

# Classify a new text input
python pipeline.py classify --text "Your input text here"
```

The argparse configuration ensures the required arguments are present for each mode and prints usage help if needed. This makes the tool easy to use from the command line in different scenarios.

Below is the complete refactored Python code, followed by an example `.env` template:

## Refactored Pipeline Code (Python)

```python
import os
import logging
import argparse
from dotenv import load_dotenv
import openai
import psycopg2

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Read database connection info from environment
PG_HOST = os.getenv("POSTGRES_HOST")
PG_PORT = os.getenv("POSTGRES_PORT")
PG_DB   = os.getenv("POSTGRES_DB")
PG_USER = os.getenv("POSTGRES_USER")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# Read Azure OpenAI config from environment
AZURE_API_KEY       = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_BASE      = os.getenv("AZURE_OPENAI_API_BASE")      # e.g. "https://<resource>.openai.azure.com/"
AZURE_API_VERSION   = os.getenv("AZURE_OPENAI_API_VERSION")   # e.g. "2023-05-15"
AZURE_DEPLOYMENT    = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # Deployment name for text-embedding-ada-002

# Validate critical env variables
if not all([PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD, AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION, AZURE_DEPLOYMENT]):
    logging.error("Missing one or more required environment variables. Please check your .env file.")
    exit(1)

# Configure OpenAI API for Azure
openai.api_type = "azure"
openai.api_key = AZURE_API_KEY
openai.api_base = AZURE_API_BASE
openai.api_version = AZURE_API_VERSION

# Constants
TABLE_NAME = "embeddings"   # database table for storing embeddings

def ensure_database(conn):
    """Ensure the pgvector extension, embeddings table, and index exist in the database."""
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension if not already enabled
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
            except Exception as e:
                # Log a warning if extension creation fails (e.g., due to permissions)
                logging.warning("Could not enable pgvector extension (might require superuser): %s", e)
                conn.rollback()
            # Create the embeddings table if not exists
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id SERIAL PRIMARY KEY,
                text TEXT,
                embedding VECTOR(1536),
                label TEXT
            );
            """
            cur.execute(create_table_sql)
            # Create an index on the vector column for efficient nearest-neighbor search (cosine distance)
            create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_embedding
            ON {TABLE_NAME}
            USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
            """
            cur.execute(create_index_sql)
            conn.commit()
    except Exception as e:
        logging.error("Database setup failed: %s", e)
        conn.rollback()
        raise  # re-raise exception to abort if we cannot set up the database

def load_data(file_path):
    """Load data from the given file. Expecting a CSV with 'text' and 'label' columns."""
    import csv
    data = []
    try:
        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'text' in row:
                    text = row['text']
                    label = row.get('label', None)
                    data.append((text, label))
                else:
                    # If no header, assume each line is the text and label is unknown
                    line = list(row.keys())[0]  # the entire line
                    data.append((line, None))
    except Exception as e:
        logging.error("Failed to load data from file %s: %s", file_path, e)
        raise
    logging.info("Loaded %d records from %s", len(data), file_path)
    return data

def generate_embedding(text):
    """Generate embedding vector for the given text using Azure OpenAI."""
    # Replace newlines to improve embedding quality as per OpenAI guidance
    text = text.replace("\n", " ")
    try:
        response = openai.Embedding.create(input=text, engine=AZURE_DEPLOYMENT)
        embedding = response['data'][0]['embedding']
        return embedding
    except Exception as e:
        # Log the error and propagate it
        logging.error("OpenAI embedding API call failed: %s", e)
        raise

def ingest_data(conn, data_list):
    """Generate embeddings for each text in data_list and insert into the database."""
    try:
        ensure_database(conn)
    except Exception as e:
        logging.critical("Database initialization failed, aborting ingestion.")
        return

    inserted_count = 0
    # Use a cursor for insertion
    with conn.cursor() as cur:
        for text, label in data_list:
            try:
                emb = generate_embedding(text)
            except Exception as e:
                # If embedding generation fails, skip this record and continue
                logging.error("Skipping text due to embedding failure: %s", e)
                continue
            try:
                # Insert the text, embedding vector, and label into the table
                cur.execute(
                    f"INSERT INTO {TABLE_NAME} (text, embedding, label) VALUES (%s, %s, %s);",
                    (text, emb, label)
                )
                inserted_count += 1
                # Commit each insert to avoid transaction issues on error
                conn.commit()
            except Exception as e:
                logging.error("Failed to insert record into database: %s", e)
                conn.rollback()
                # continue to next record without terminating entire ingestion
                continue
    logging.info("Ingestion complete: %d records inserted.", inserted_count)

def classify_text(conn, input_text):
    """Classify the input_text by finding the nearest embedding in the database and returning its label."""
    try:
        query_emb = generate_embedding(input_text)
    except Exception as e:
        logging.error("Failed to generate embedding for input text: %s", e)
        return None
    try:
        with conn.cursor() as cur:
            # Find the closest vector in the table using cosine distance (<=> operator)
            cur.execute(
                f"SELECT label FROM {TABLE_NAME} ORDER BY embedding <=> %s LIMIT 1;",
                (query_emb,)
            )
            result = cur.fetchone()
            if result:
                return result[0]  # label of nearest neighbor
            else:
                return None
    except Exception as e:
        logging.error("Database query failed during classification: %s", e)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding & Classification Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for ingestion
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data and store embeddings in the database")
    ingest_parser.add_argument("--file", type=str, required=True, help="Path to input CSV file with 'text' and 'label' columns")

    # Subparser for classification
    classify_parser = subparsers.add_parser("classify", help="Classify an input text using nearest neighbor in the embeddings")
    classify_parser.add_argument("--text", type=str, required=True, help="Text input to classify")

    args = parser.parse_args()

    # Connect to PostgreSQL database
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD
        )
        logging.info("Connected to PostgreSQL database %s at %s:%s", PG_DB, PG_HOST, PG_PORT)
    except Exception as e:
        logging.critical("Failed to connect to the PostgreSQL database: %s", e)
        exit(1)

    # Run the chosen subcommand
    if args.command == "ingest":
        data_list = load_data(args.file)
        ingest_data(conn, data_list)
    elif args.command == "classify":
        label = classify_text(conn, args.text)
        if label is not None:
            logging.info("Input text classified as: %s", label)
            print(f"Predicted label: {label}")
        else:
            logging.info("No similar embedding found or classification failed.")
            print("No classification result available.")

    # Close the database connection
    conn.close()
```

## Example `.env` File Template

```bash
# PostgreSQL database credentials
POSTGRES_HOST=your_postgres_host
POSTGRES_PORT=5432
POSTGRES_DB=your_database_name
POSTGRES_USER=your_db_username
POSTGRES_PASSWORD=your_db_password

# Azure OpenAI Service credentials and deployment info
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_API_BASE=https://your-resource-name.openai.azure.com/  # include trailing slash
AZURE_OPENAI_API_VERSION=2023-05-15  # example API version for embeddings
AZURE_OPENAI_DEPLOYMENT_NAME=your_embedding_deployment_name
```

**Notes:** Ensure you have deployed the `text-embedding-ada-002` model on Azure and use its deployment name in the `.env` file. The PostgreSQL database must have the pgvector extension installed (the script attempts to enable it). After setting up the `.env` file, you can run the ingestion and classification as shown above. This refactored pipeline uses secure practices for configuration, proper error handling/logging, and a modular design for maintainability, making it production-ready.

**Sources:**

* Use of python-dotenv for loading environment variables
* Azure OpenAI embedding setup with `text-embedding-ada-002` and deployment name
* pgvector storage of 1536-dimension embeddings and recommended cosine similarity
* Inserting and querying vectors in PostgreSQL with psycopg2 and pgvector
* Creating an IVF flat index on vector column for efficient similarity search






####################################
Great! I’ll generate a complete Python code example that:

1. Reads your Excel file.
2. Extracts and prepares fields for embedding.
3. Uses Azure OpenAI with the `text-embedding-ada-002` model to generate embeddings.
4. Stores all metadata and embeddings into a PostgreSQL database with pgvector support.
5. Provides a method for classifying new attributes using ANN search.

I’ll include table creation, embedding logic, and query-based classification.
I’ll update you shortly.


# Python Pipeline for Embedding and Classification with pgvector

## Overview

This Python pipeline reads an Excel file containing attribute metadata, generates embeddings using Azure OpenAI's `text-embedding-ada-002` model, and stores the results in a PostgreSQL database with the `pgvector` extension. The stored embeddings enable similarity search for classifying new attributes. The code is organized into modular functions and can be used via a CLI or imported functions for embedding, storing, and classifying. We use **pandas** for Excel parsing, **OpenAI** (or LangChain) for embeddings, and **psycopg2/SQLAlchemy** for database interactions.

## 1. Reading Excel Data and Preparing Text

We first load the Excel file into a pandas DataFrame. The file is expected to have columns such as **CID status**, **CID category**, **Attributes Sub-category**, **Attribute Name**, **Attribute Definition**, **Examples**, **Comments**, etc. To create a meaningful embedding, we concatenate key text fields (Attribute Name, Definition, Examples, Comments) into a single string. This combined text represents the semantic content of each attribute:

* **Attribute Name** – The name/title of the attribute.
* **Attribute Definition** – A description or definition of the attribute.
* **Examples** – Example values or usage of the attribute.
* **Comments** – Any additional remarks or context.

By joining these fields (with separators), we get a descriptive text for each row, which will be fed into the embedding model. We also handle missing values by replacing `NaN` with empty strings to avoid issues in concatenation.

## 2. Generating Embeddings with Azure OpenAI

For each prepared text string, we generate a vector embedding using Azure OpenAI’s `text-embedding-ada-002` model. This model produces a 1536-dimensional embedding vector for any given text. We use OpenAI’s Python SDK (or LangChain’s embedding utility) to call the model. Ensure that you have your Azure OpenAI endpoint and API key configured (for example, via environment variables or prior setup in the code).

**Note:** The code assumes that the OpenAI API is properly configured. For Azure OpenAI, you might need to set `openai.api_type = "azure"`, `openai.api_base`, `openai.api_version`, and `openai.api_key` according to your Azure OpenAI deployment. If using LangChain, you would initialize an `OpenAIEmbeddings` with the appropriate `deployment` name and `model` parameters.

We generate embeddings in batches to improve efficiency (the OpenAI API allows sending a list of texts in one request). Each embedding is a list of 1536 floating-point numbers.

## 3. Setting up PostgreSQL with pgvector

We use **pgvector** to store and query the embeddings. In the database, we create an extension and a table for our data:

* **Extension:** Enable the pgvector extension with `CREATE EXTENSION IF NOT EXISTS vector;`. This provides the new `VECTOR` data type and similarity operators.
* **Table Schema:** We create a table (e.g., `attributes`) with columns for all the Excel fields and an `embedding` column of type `VECTOR(1536)`. We specify 1536 dimensions to match the embedding size of the model. All other fields are stored as text (you can adjust types for numeric or date fields if needed).
* **Indexing:** To speed up similarity searches, we add an approximate index on the embedding column using the **IVFFlat** algorithm with cosine distance (`vector_cosine_ops`). OpenAI recommends using cosine similarity for embeddings, and pgvector provides a `vector_cosine_ops` index type for this purpose. We use `USING ivfflat (embedding vector_cosine_ops)` to create the index. (Note: pgvector also supports L2 `<->` and inner product distance with different operators; here we opt for cosine distance, which treats vectors as normalized direction vectors.)

## 4. Storing Embeddings in the Database

With the table ready, we insert all rows from the Excel along with their embeddings. Using **psycopg2** (and optionally SQLAlchemy), we connect to the database. We register pgvector’s vector type with psycopg2 so that Python lists/NumPy arrays can be directly written to the `VECTOR` column. We then batch-insert the data:

* We prepare a list of tuples, each tuple containing all field values and the embedding vector (as a NumPy array or list of floats).
* We use `psycopg2.extras.execute_values` for efficient batch insertion. This sends all rows in one call rather than executing one INSERT per row.
* Finally, we commit the transaction to save all data.

**Note:** The pgvector extension allows inserting a vector by simply passing a Python list (or NumPy array) for the vector parameter, as long as the dimension matches. In our code, we ensure the embedding vector has exactly 1536 floats to match the `VECTOR(1536)` column.

## 5. Similarity Search and Classification

To classify a new attribute, we perform the following steps:

1. **Embed the new attribute:** Given a new attribute's name, definition, examples, and comments, we construct the same kind of text string and obtain its embedding from the OpenAI model.
2. **Similarity search:** We query the database for the most similar existing entries. This is done by ordering the rows by vector distance to the new embedding. In SQL, `ORDER BY embedding <-> %s` uses the `<->` operator to compute Euclidean distance between vectors (with the cosine index, this effectively finds nearest neighbors by cosine similarity). We limit the results to the top **k** neighbors (e.g., k=5).
3. **Classification via neighbors:** Based on the nearest neighbors, we infer the likely classification for the new attribute. We retrieve each neighbor’s **CID status**, **CID category**, and **Attributes Sub-category**. Then, we can either:

   * Take the top-1 neighbor’s categories as the prediction (simple approach), or
   * Use a majority vote among the top-k neighbors. For majority voting, we consider each neighbor’s vote for each field. We can optionally weight votes by similarity (e.g., closer neighbors count more), but here we implement an unweighted majority for simplicity.

The output for a new attribute query will be the predicted **CID status**, **CID category**, and **Attributes Sub-category** that best match the new attribute, based on similarity to the stored data.

Below is the complete Python code implementing this pipeline. The code is modular: it defines functions for embedding & storing data and for classifying new entries. It also includes a simple CLI interface using `argparse` for running the ingestion or classification from the command line.

## Complete Code Implementation

```python
import os
import pandas as pd
import numpy as np
import openai
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import collections
import argparse

# **Database connection configuration** – Fill in your credentials or use environment variables.
DB_HOST = os.getenv("DB_HOST", "")      # e.g., "localhost"
DB_PORT = os.getenv("DB_PORT", "5432")  # default PostgreSQL port
DB_NAME = os.getenv("DB_NAME", "")      # e.g., "mydatabase"
DB_USER = os.getenv("DB_USER", "")      # e.g., "myuser"
DB_PASS = os.getenv("DB_PASS", "")      # e.g., "mypassword"

# **OpenAI API configuration** – Ensure your Azure OpenAI credentials are set.
# If using Azure OpenAI, set the necessary environment variables or directly assign here.
openai.api_key = os.getenv("OPENAI_API_KEY", "")  # Azure OpenAI Key or OpenAI API Key
# For Azure OpenAI, you may need to set:
# openai.api_type = "azure"
# openai.api_base = os.getenv("OPENAI_API_BASE", "")
# openai.api_version = os.getenv("OPENAI_API_VERSION", "2023-05-15")  # example version

# Connect to PostgreSQL database
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASS
)
# Register pgvector type for psycopg2 so we can use Python lists/arrays for the vector column
register_vector(conn, arrays=True)
cur = conn.cursor()

# Enable pgvector extension and create table if not exists
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute(sql.SQL("""
CREATE TABLE IF NOT EXISTS attributes (
    id SERIAL PRIMARY KEY,
    "CID status" TEXT,
    "CID category" TEXT,
    "Attributes Sub-category" TEXT,
    "Attribute Name" TEXT,
    "Attribute Definition" TEXT,
    "Usage (NSI - IPID/NSI-Non-IPID)" TEXT,
    "Exceptional usage" TEXT,
    "Comments" TEXT,
    "Jurisdiction Restricted" TEXT,
    "Examples" TEXT,
    "Structure" TEXT,
    "Format" TEXT,
    "Length" TEXT,
    "Unique" TEXT,
    "Attribute Bucket" TEXT,
    "Attribute Weight" TEXT,
    "CON ID" TEXT,
    "Version Added" TEXT,
    "Date Added" TEXT,
    "Mapping-ID-Old" TEXT,
    "Bucket Move" TEXT,
    "Comment" TEXT,
    embedding VECTOR(1536)
);
"""))
# Create an index on the embedding vector for efficient similarity search (cosine distance)
cur.execute(sql.SQL("""
CREATE INDEX IF NOT EXISTS idx_attributes_embedding
ON attributes
USING ivfflat (embedding vector_cosine_ops);
"""))
conn.commit()
cur.close()

def embed_and_store(xlsx_path: str):
    """
    Reads the Excel file at xlsx_path, generates embeddings for each row,
    and stores all data (fields + embedding) into the PostgreSQL database.
    """
    # Read Excel into DataFrame
    df = pd.read_excel(xlsx_path)
    # Replace NaN with empty string for text fields to avoid "nan" strings
    text_cols = ["Attribute Name", "Attribute Definition", "Examples", "Comments"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")
        else:
            df[col] = ""  # If missing column, create an empty one for consistency

    # Construct text for embedding by concatenating the relevant fields
    def make_embed_text(row):
        return f"{row['Attribute Name']} | Definition: {row['Attribute Definition']} | " \
               f"Examples: {row['Examples']} | Comments: {row['Comments']}"
    df["embedding_text"] = df.apply(make_embed_text, axis=1)

    # Generate embeddings for each row's text (batching for efficiency)
    texts = df["embedding_text"].tolist()
    embeddings = []  # will hold numpy arrays of shape (1536,)
    batch_size = 100  # you can adjust batch size based on API limits and performance
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        response = openai.Embedding.create(input=batch_texts, model="text-embedding-ada-002")
        # The response contains embeddings for each input text in order
        for data in response["data"]:
            vec = np.array(data["embedding"], dtype=float)  # convert to numpy array
            embeddings.append(vec)
    # Ensure we got an embedding for each text
    assert len(embeddings) == len(texts), "Embedding count does not match input texts count."

    # Prepare data for insertion
    cols = ["CID status", "CID category", "Attributes Sub-category", "Attribute Name",
            "Attribute Definition", "Usage (NSI - IPID/NSI-Non-IPID)", "Exceptional usage",
            "Comments", "Jurisdiction Restricted", "Examples", "Structure", "Format", "Length",
            "Unique", "Attribute Bucket", "Attribute Weight", "CON ID", "Version Added",
            "Date Added", "Mapping-ID-Old", "Bucket Move", "Comment"]
    data_to_insert = []
    for idx, row in df.iterrows():
        # tuple of all columns in the same order as `cols`, plus the embedding vector
        values = tuple(row[col] for col in cols)
        embed_vec = embeddings[idx]
        data_to_insert.append(values + (embed_vec,))

    # Insert data into the database
    insert_query = sql.SQL("""
        INSERT INTO attributes (
            {fields}, embedding
        ) VALUES %s;
    """).format(fields=sql.SQL(", ").join(sql.Identifier(c) for c in cols))
    # Using execute_values for batch insert
    cur = conn.cursor()
    execute_values(cur, insert_query.as_string(cur), data_to_insert)
    conn.commit()
    cur.close()
    print(f"Inserted {len(data_to_insert)} rows with embeddings into the database.")

def classify_attribute(name: str, definition: str, examples: str = "", comments: str = "", top_k: int = 5):
    """
    Given a new attribute's details, generate its embedding and find the top-k similar attributes.
    Returns a prediction for 'CID status', 'CID category', and 'Attributes Sub-category'.
    """
    # Combine the input fields into a single text in the same format as the stored data
    query_text = f"{name} | Definition: {definition} | Examples: {examples} | Comments: {comments}"
    # Get embedding for the query text
    response = openai.Embedding.create(input=[query_text], model="text-embedding-ada-002")
    query_embedding = np.array(response["data"][0]["embedding"], dtype=float)

    # Query the database for the nearest neighbors using the <-> (distance) operator
    cur = conn.cursor()
    cur.execute(
        """
        SELECT "CID status", "CID category", "Attributes Sub-category"
        FROM attributes
        ORDER BY embedding <-> %s
        LIMIT %s;
        """,
        (query_embedding, top_k)
    )
    neighbors = cur.fetchall()
    cur.close()
    if not neighbors:
        print("No similar records found for classification.")
        return None

    # Determine majority (or top-1) for each classification field from neighbors
    cid_status_votes = [n[0] for n in neighbors]
    cid_category_votes = [n[1] for n in neighbors]
    subcat_votes = [n[2] for n in neighbors]
    # Majority vote (most common) for each field
    pred_status = collections.Counter(cid_status_votes).most_common(1)[0][0]
    pred_category = collections.Counter(cid_category_votes).most_common(1)[0][0]
    pred_subcat = collections.Counter(subcat_votes).most_common(1)[0][0]
    # (If desired, one could implement similarity-weighted voting by weighting each neighbor's vote by its similarity score.)

    print("Nearest neighbors' classifications:", neighbors)
    print(f"Predicted CID status: {pred_status}")
    print(f"Predicted CID category: {pred_category}")
    print(f"Predicted Attributes Sub-category: {pred_subcat}")
    return {"CID status": pred_status, "CID category": pred_category, "Attributes Sub-category": pred_subcat}

# **Command-line interface**: allow running as a script for ingesting data or classifying a new entry.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding pipeline with pgvector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-command for embedding and storing data from an Excel file
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data from an Excel file and store embeddings in the database")
    ingest_parser.add_argument("file", help="Path to the Excel file containing attributes data")

    # Sub-command for classifying a new attribute
    classify_parser = subparsers.add_parser("classify", help="Classify a new attribute by finding similar entries")
    classify_parser.add_argument("--name", required=True, help="Attribute Name")
    classify_parser.add_argument("--definition", required=True, help="Attribute Definition")
    classify_parser.add_argument("--examples", default="", help="Attribute Examples")
    classify_parser.add_argument("--comments", default="", help="Attribute Comments")
    classify_parser.add_argument("--top_k", type=int, default=5, help="Number of neighbors to consider for classification")

    args = parser.parse_args()
    if args.command == "ingest":
        embed_and_store(args.file)
    elif args.command == "classify":
        classify_attribute(args.name, args.definition, args.examples, args.comments, args.top_k)
```

**Usage Examples:**

* To ingest data from an Excel file:

  ```bash
  python pipeline.py ingest path/to/attributes.xlsx
  ```

  This will read the Excel, generate embeddings for each row, and insert them into the database.

* To classify a new attribute via CLI:

  ```bash
  python pipeline.py classify --name "New Attribute" --definition "Description of the attribute" --examples "Example1, Example2" --comments "Additional info"
  ```

  This will output the nearest neighbors found and the predicted **CID status**, **CID category**, and **Attributes Sub-category** for the new attribute.

By following this pipeline, you can maintain a vector-searchable knowledge base of attributes. The pgvector-powered similarity search enables quick classification of new entries based on their semantic similarity to existing data. The combination of Azure OpenAI embeddings and PostgreSQL provides a self-contained solution for storing and querying high-dimensional vectors directly within a relational database, as demonstrated in the code above. The approach is scalable (with the help of the vector index) and can be integrated into larger systems for automated attribute categorization and retrieval.

**Sources:** The embedding dimensionality and usage of pgvector are based on OpenAI’s and pgvector’s documentation. OpenAI’s `text-embedding-ada-002` model outputs 1536-dimensional vectors, which we use as the vector size. The pgvector extension introduces the `VECTOR` type and similarity operators (`<->` for Euclidean, `<#>` for negative dot product, `<=>` for cosine). We use cosine similarity for nearest-neighbor search, with an IVFFlat index optimized for cosine distance. Insertion of embeddings is done by passing the vector as a list/array to psycopg2 (as shown in similar examples). This pipeline leverages these capabilities to enable efficient vector storage and search in PostgreSQL.
