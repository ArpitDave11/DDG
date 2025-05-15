import os, json, csv
import openai
from glob import glob
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from openai.error import RateLimitError, APIError, Timeout, APIConnectionError, InvalidRequestError

# Configuration
folder_path = "/path/to/json/folder"          # TODO: set this to your directory path
output_csv = os.path.join(folder_path, "attributes_enriched.csv")
MAX_CHARS = 2000                              # max characters before chunking text
openai.api_key = os.getenv("OPENAI_API_KEY")  # or set your API key directly

# Prepare CSV writer
csvfile = open(output_csv, 'w', newline='', encoding='utf-8')
writer = csv.writer(csvfile)
writer.writerow(["model_name", "table_name", "entity_name", "attribute_name",
                 "column_name", "column_data_type", "pk", "embedding"])

# Tenacity retry decorator for OpenAI API calls (exponential backoff on certain errors)
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6),
       retry=retry_if_exception_type((RateLimitError, APIError, Timeout, APIConnectionError)))
def get_embedding_with_backoff(text, model="text-embedding-3-small"):
    """Call OpenAI API to get embedding, with retries on rate limit and transient errors."""
    response = openai.Embedding.create(input=text, model=model)
    return response['data'][0]['embedding']

# Process each JSON file in the folder
for file_path in glob(os.path.join(folder_path, "*.json")):  # find all JSON files:contentReference[oaicite:18]{index=18}
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Skipping {file_path}: JSON decode error: {e}")
            continue

    # Process each record in the JSON array
    for record in records:
        model_name = record.get("Model", "")
        table_name = record.get("TABLE NAME", "")
        entity_name = record.get("ENTITY NAME", "")
        # Some records might not have 'Attributes' or it might not be a list
        attributes = record.get("Attributes", [])
        if not isinstance(attributes, list):
            continue  # skip if attributes is not a list as expected

        for attr in attributes:
            # Compose text from attribute name and definition
            name = attr.get("NAME", "") 
            definition = attr.get("DEFINITION", "")
            text = f"{name}: {definition}" if name or definition else ""
            if not text:
                # If there's no text (empty name and definition), skip embedding
                attr["embedding"] = []
                continue

            # Chunk the text if it's too long for safety
            chunks = []
            if len(text) > MAX_CHARS:
                # Split by sentence (naive approach)
                sentences = text.split('. ')
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 <= MAX_CHARS:
                        current_chunk += sentence + '. '
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence + '. '
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks = [text]

            # Get embeddings for each chunk and average them if multiple
            try:
                if len(chunks) == 1:
                    # Single chunk - directly embed
                    embedding = get_embedding_with_backoff(chunks[0])
                else:
                    # Multiple chunks - embed each and then average
                    chunk_vectors = []
                    for chunk in chunks:
                        vec = get_embedding_with_backoff(chunk)
                        chunk_vectors.append(vec)
                    # Average the vectors element-wise
                    # (assuming all vectors are of equal length, as they should be)
                    avg_vector = [0] * len(chunk_vectors[0])
                    for vec in chunk_vectors:
                        for i, val in enumerate(vec):
                            avg_vector[i] += val
                    avg_vector = [val / len(chunk_vectors) for val in avg_vector]
                    embedding = avg_vector
                attr["embedding"] = embedding
            except InvalidRequestError as e:
                # This can happen if input is still too long or other issues; handle gracefully
                print(f"Embedding failed for attribute '{name}' in {file_path}: {e}")
                attr["embedding"] = []
            except Exception as e:
                # Catch-all for any other errors (should be rare with tenacity)
                print(f"Unexpected error embedding '{name}': {e}")
                attr["embedding"] = []

            # Write the attribute row to CSV
            embedding_json = json.dumps(attr.get("embedding", []))
            writer.writerow([
                model_name, 
                table_name, 
                entity_name, 
                name, 
                attr.get("Column Name", ""), 
                attr.get("Column Data Type", ""), 
                attr.get("PK?", ""), 
                embedding_json
            ])
    # Save enriched JSON for this file
    output_path = file_path.rsplit(".", 1)[0] + "_enriched.json"
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(records, out_f, ensure_ascii=False, indent=4)

# Close the CSV file
csvfile.close()
