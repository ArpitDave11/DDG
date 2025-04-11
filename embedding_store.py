import openai, faiss, numpy as np, logging, json
from config import config

# Initialize OpenAI API for Azure (for both embedding and GPT-4 usage)
openai.api_type = "azure"
openai.api_base = config['azure']['openai_endpoint']
openai.api_key = config['azure']['openai_api_key']
openai.api_version = config['azure']['openai_api_version']

def load_examples(finetune_file: str) -> list[dict]:
    """Load historical prompt-completion examples from a JSONL or JSON file."""
    examples = []
    if not finetune_file:
        return examples
    try:
        with open(finetune_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                data = json.loads(line)
                prompt = data.get('prompt') or data.get('instruction') or ""
                completion = data.get('completion') or data.get('output') or ""
                if prompt and completion:
                    # Optionally, parse table/column/type from prompt if needed (assuming prompt already in desired format)
                    examples.append({
                        "prompt": prompt,
                        "completion": completion.strip()
                    })
    except Exception as e:
        logging.error(f"Failed to load fine-tune examples: {e}")
        raise
    logging.info(f"Loaded {len(examples)} examples from fine-tune dataset.")
    return examples

def compute_embeddings(texts: list[str]) -> np.ndarray:
    """Compute embeddings for a list of texts using Azure OpenAI embedding deployment."""
    if not texts:
        return np.array([])
    deployment = config['azure']['openai_deployment_embed']  # embedding model deployment name
    # Call embedding API in batches to handle large lists
    batch_size = 16
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = openai.Embedding.create(input=batch, engine=deployment)
        except Exception as e:
            logging.error(f"Embedding API call failed: {e}")
            raise
        # Extract embeddings
        for item in response['data']:
            embeddings.append(item['embedding'])
    return np.array(embeddings, dtype='float32')

def build_vector_index(examples: list[dict]):
    """Build a FAISS vector index from the list of example prompts. Returns (index, examples_with_vectors)."""
    if not examples:
        return None, []
    # Prepare the text to embed (we use the prompt text for similarity search)
    texts = []
    for ex in examples:
        # Ensure prompt is formatted consistently (if not already). 
        # If the prompt isn't in "Column: X (Type: Y)" format, you may need to construct it from the data.
        texts.append(ex['prompt'])
    logging.info("Computing embeddings for historical examples...")
    embeddings = compute_embeddings(texts)  # shape (N, D)
    dim = embeddings.shape[1] if embeddings.size > 0 else 0
    index = faiss.IndexFlatL2(dim)  # exact L2 search index
    index.add(embeddings)  # add all vectors to the index
    logging.info(f"FAISS index built with {index.ntotal} vectors (dim={dim}).")
    # Optionally, store the embeddings or index to disk for reuse (faiss.write_index)
    return index, examples

def find_similar_examples(query_text: str, index, examples: list[dict], top_k: int = 3) -> list[dict]:
    """Find top-k similar examples to the query_text using the FAISS index."""
    if index is None or not query_text:
        return []
    try:
        query_embed = compute_embeddings([query_text])
    except Exception as e:
        logging.error(f"Failed to embed query text: {e}")
        return []
    if query_embed.size == 0:
        return []
    D, I = index.search(query_embed, top_k)  # I: indices of nearest examples
    similar_examples = []
    for idx in I[0]:
        if idx < len(examples):
            similar_examples.append(examples[idx])
    logging.info(f"Retrieved {len(similar_examples)} similar examples for query '{query_text}'.")
    return similar_examples
