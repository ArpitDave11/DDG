from fastapi import FastAPI
from pydantic import BaseModel
import logging
from metadata_loader import load_metadata_from_blob  # (if needed for verification)
from embedding_store import load_examples, build_vector_index, find_similar_examples
from description_generator import generate_description

# Initialize app
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Load examples and build index at startup
examples_cache = []
index = None
try:
    examples_cache = load_examples(config.get('fine_tune_data') or "fine_tune_data.jsonl")
    if examples_cache:
        index, examples_cache = build_vector_index(examples_cache)
        logging.info(f"Loaded {len(examples_cache)} examples into memory for API.")
    else:
        logging.warning("No historical examples loaded. API will generate descriptions without context.")
except Exception as e:
    logging.error(f"Failed to initialize example index for API: {e}")
    examples_cache = []
    index = None

class AttributeRequest(BaseModel):
    table: str
    column: str
    data_type: str

class AttributeResponse(BaseModel):
    table: str
    column: str
    data_type: str
    description: str

@app.post("/generate_description", response_model=AttributeResponse)
def generate_attribute_description(request: AttributeRequest):
    """
    Generate a description for the given attribute (table, column, data_type).
    """
    table = request.table
    column = request.column
    dtype = request.data_type
    query = f"Column: {table}.{column} (Type: {dtype})"
    similar = []
    if index:
        try:
            similar = find_similar_examples(query, index, examples_cache, top_k=config['settings']['top_k'])
        except Exception as e:
            logging.error(f"Example retrieval failed for API request {table}.{column}: {e}")
            similar = []
    try:
        description = generate_description(table, column, dtype, similar)
    except Exception as e:
        logging.error(f"Generation failed for API request {table}.{column}: {e}")
        # In case of failure, return an empty description with error message (or raise HTTPException)
        description = ""
    return {
        "table": table,
        "column": column,
        "data_type": dtype,
        "description": description
    }
