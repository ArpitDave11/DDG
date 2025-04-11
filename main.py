import logging
from metadata_loader import load_metadata_from_blob
from embedding_store import load_examples, build_vector_index, find_similar_examples
from description_generator import generate_description
from db_writer import ensure_table_exists, save_attribute_descriptions
from config import config

logging.basicConfig(level=logging.INFO)

def run_batch_generation():
    # 1. Load metadata from Azure Blob
    try:
        attributes = load_metadata_from_blob()
    except Exception as e:
        logging.error("Aborting: unable to load metadata.")
        return
    if not attributes:
        logging.error("No attributes found in metadata; nothing to process.")
        return

    # 2. Load fine-tune examples and build embeddings index
    examples = []
    index = None
    fine_tune_path = config.get('fine_tune_data') or "fine_tune_data.jsonl"
    try:
        examples = load_examples(fine_tune_path)
        if examples:
            index, examples = build_vector_index(examples)
        else:
            logging.warning("No examples loaded for RAG; generation will proceed without retrieved context.")
    except Exception as e:
        logging.error("Failed to prepare example embedding index, proceeding without retrieval.")
        # We continue even if RAG setup fails; model will generate without examples.

    # 3. Ensure database table exists
    try:
        ensure_table_exists()
    except Exception as e:
        logging.error("Aborting: could not ensure database table. Error: %s", e)
        return

    # 4. Generate descriptions for each attribute and collect results
    results = []
    for attr in attributes:
        table = attr['table']; column = attr['column']; dtype = attr['type']
        existing = attr.get('existing_desc', "")
        # Retrieve similar examples (if index is ready)
        query = f"Column: {table}.{column} (Type: {dtype})"
        try:
            similar = find_similar_examples(query, index, examples, top_k=config['settings']['top_k'])
        except Exception as e:
            logging.error(f"Example retrieval failed for {table}.{column}: {e}")
            similar = []
        # Generate description using GPT-4
        try:
            description = generate_description(table, column, dtype, similar)
        except Exception as e:
            logging.error(f"Generation failed for {table}.{column}: {e}")
            description = ""
        if description:
            results.append((table, column, dtype, description, existing))
            logging.info(f"Generated description for {table}.{column}")
        else:
            logging.warning(f"No description generated for {table}.{column} (left blank).")
    # 5. Save results to PostgreSQL
    try:
        rows_inserted = save_attribute_descriptions(results)
        logging.info(f"Batch generation complete. {rows_inserted} descriptions saved to database.")
    except Exception as e:
        logging.error(f"Failed to save results to database: {e}")
        # Not raising here; just log. In a real system, you might retry or handle accordingly.

if __name__ == "__main__":
    run_batch_generation()
