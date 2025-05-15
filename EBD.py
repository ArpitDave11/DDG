import os
import sys
import json
import openai
import psycopg2
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Configure OpenAI API key (ensure the OPENAI_API_KEY environment variable is set)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")

# Retry decorator for OpenAI API calls to handle rate limits with exponential backoff
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """
    Generate an embedding for the given text using the specified OpenAI model.
    This function will retry on rate limit errors with exponential backoff.
    """
    response = openai.Embedding.create(input=[text], model=model)
    embedding_vector = response["data"][0]["embedding"]
    return embedding_vector

def main(json_path: str):
    # Load the JSON data from the provided file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract model and entity information
    model_name = data.get("Model", "")
    entity_info = data.get("Entity", {})
    entity_name = entity_info.get("ENTITY NAME", "")
    table_name = entity_info.get("TABLE_NAME", "")
    
    # Connect to PostgreSQL database
    # (Replace with your actual connection details)
    conn = psycopg2.connect(
        host     = os.getenv("PGHOST", "localhost"),
        port     = os.getenv("PGPORT", "5432"),
        dbname   = os.getenv("PGDATABASE", "your_database"),
        user     = os.getenv("PGUSER", "your_db_user"),
        password = os.getenv("PGPASSWORD", "your_db_password")
    )
    cur = conn.cursor()
    
    # Ensure the pgvector extension and target table exist
    # (Requires appropriate privileges to create extension/table)
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attribute_embeddings (
            model_name       TEXT,
            entity_name      TEXT,
            table_name       TEXT,
            attribute_name   TEXT,
            column_name      TEXT,
            column_data_type TEXT,
            pk               TEXT,
            embedding        VECTOR(1536)
        )
    """)
    conn.commit()  # commit DDL changes so table is ready
    
    # Iterate over each attribute in the JSON and process it
    attributes = data.get("Attributes", [])
    for attr in attributes:
        attr_name = attr.get("NAME", "")
        definition = attr.get("DEFINITION", "") or ""
        col_name = attr.get("Column Name", "")
        col_type = attr.get("Column Data Type", "")
        pk_flag = attr.get("PK?", "")
        
        # Prepare text for embedding: use definition if available, otherwise use the attribute name
        if definition.strip().lower() == "no definition available" or definition.strip() == "":
            text_to_embed = attr_name
        else:
            # Combine attribute name and definition for richer context
            text_to_embed = f"{attr_name}: {definition}"
        
        # Generate the embedding vector (with retry logic for robustness)
        try:
            embedding_vector = get_embedding(text_to_embed)
        except Exception as e:
            print(f"Error generating embedding for attribute '{attr_name}': {e}", file=sys.stderr)
            continue  # skip this attribute on failure (or handle as needed)
        
        # Insert the attribute metadata and embedding into the database
        cur.execute(
            """
            INSERT INTO attribute_embeddings 
                (model_name, entity_name, table_name, attribute_name, column_name, column_data_type, pk, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (model_name, entity_name, table_name, attr_name, col_name, col_type, pk_flag, embedding_vector)
        )
        # (Note: psycopg2 will handle Python list -> vector parameter adaptation)
    
    # Commit all inserts and close the connection
    conn.commit()
    cur.close()
    conn.close()
    print("Embedding generation and storage complete.")

if __name__ == "__main__":
    # Expect the JSON file path as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python embed_attributes.py <path_to_json_file>")
        sys.exit(1)
    json_file_path = sys.argv[1]
    main(json_file_path)
