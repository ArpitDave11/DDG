Great! I’ll now create a complete Databricks-compatible notebook script using LangChain 0.3.25 and FAISS 1.1.1. It will:

* Load your enriched JSON with attribute embeddings
* Build and persist a FAISS index to disk
* Load the index back in for retrieval
* Use LangChain’s RetrievalQA with OpenAI
* Output results to a pandas DataFrame with query, answer, top match attribute, entity name, table name, and score

I’ll let you know once the full notebook is ready.


# Databricks Notebook: LangChain RetrievalQA with FAISS

**This notebook demonstrates how to build a question-answering system using LangChain and FAISS**. We will use precomputed 1536-dimensional embeddings for entity attributes, store them in a FAISS vector index, and set up a LangChain Retrieval QA pipeline with OpenAI's GPT-3.5-Turbo as the LLM. The steps include:

* Installing the required libraries (LangChain 0.3.25 and FAISS 1.11.0)
* Loading a JSON file of enriched entity metadata (with attribute definitions and embeddings)
* Building a FAISS index from the attribute embeddings and saving it to DBFS
* Reloading the index and creating a retriever
* Setting up an OpenAI GPT-3.5-Turbo powered `RetrievalQA` chain
* Querying the system in natural language and returning answers with metadata (attribute name, entity name, table name, similarity score) of the top match

## Installation and Environment Setup

First, install specific versions of LangChain and FAISS that are compatible with Databricks Runtime 14.3 (Python 3.10). We also install `langchain-community` to ensure FAISS integration is available. Then import necessary modules and verify the versions:

```python
# Install specific versions of LangChain and FAISS
%pip install langchain==0.3.25 langchain-community==0.3.24 faiss-cpu==1.11.0

# After installation, import and check versions
import langchain
import faiss                   # FAISS library (via faiss-cpu)
import langchain_community     # LangChain community integrations (for vectorstores)
print(f"LangChain version: {langchain.__version__}")
print(f"FAISS version: {faiss.__version__}")
```

*Note:* The above `%pip install` command is intended for use in a Databricks notebook cell. It will install the required packages in the notebook environment. Ensure that the LangChain version is **0.3.25** and FAISS (faiss-cpu) is **1.11.0** for compatibility with Python 3.10.

## Loading the JSON Entity Metadata

We have a JSON file containing enriched entity metadata. Each entry in this JSON represents an entity (e.g., a database table or business entity) and includes:

* **Model**: The embedding model used (e.g., `"text-embedding-ada-002"`).
* **Entity**: An object with `TABLE_NAME`, `ENTITY NAME`, and a `DEFINITION` describing the entity.
* **Attributes**: A list of attribute objects, each with:

  * `NAME` – Attribute name
  * `DEFINITION` – Attribute definition (text description)
  * `Column Name` – Physical column name
  * `Column Data Type` – Data type of the column
  * `PK?` – Whether it’s a primary key
  * `embedding` – The precomputed 1536-d embedding vector (list of floats) for this attribute

We will load this JSON file from DBFS and inspect its structure:

```python
import json

# Path to the JSON file in DBFS (update this path as needed)
json_path = "/dbfs/FileStore/metadata/enriched_entity_metadata.json"

# Load the JSON data
with open(json_path, 'r') as f:
    entities_data = json.load(f)

print(f"Loaded {len(entities_data)} entities from JSON.")
# Print sample structure of the first entity for verification
print(json.dumps(entities_data[0], indent=2)[:500] + "...")
```

**Explanation:** We use the built-in `json` module to read the file. The path uses the `/dbfs/` prefix so that it can access the Databricks File System (DBFS). The snippet prints out the number of entities loaded and a truncated preview of the first entry to confirm the structure.

## Building the FAISS Vector Index from Attribute Embeddings

Next, we will construct a FAISS vector index using the precomputed attribute embeddings. We need to extract each attribute's embedding and some identifying metadata to store alongside it. We'll use LangChain's FAISS integration to build a vector store that holds:

* The text content for each attribute (we can use the attribute’s definition, possibly prefixed with its name)
* The attribute’s metadata (name, entity name, table name)
* The embedding vector for fast similarity search

**Steps to build the index:**

* Iterate through each entity entry and each attribute within it.
* For each attribute, prepare a text string (e.g., `"Attribute Name: Attribute Definition"`) that represents the attribute's content.
* Collect the attribute’s metadata (`attribute_name`, `entity_name`, `table_name`) in a dictionary.
* Collect the embedding vector (as a list of floats).
* Use `FAISS.from_embeddings` to create a vector store from the list of text–embedding pairs and metadata, using the same embedding model for queries.

We'll use the OpenAI Embeddings class from LangChain (with model `text-embedding-ada-002`) as the embedding function for the vector store, so that query embeddings are generated in the same vector space. We set `normalize_L2=True` to normalize embeddings, enabling cosine similarity-based search using the L2 distance metric.

```python
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Initialize the OpenAI embeddings function for queries (1536-dim Ada embeddings)
embedding_fn = OpenAIEmbeddings(model="text-embedding-ada-002")

# Prepare data for the vector store
text_embedding_pairs = []   # list of (text, embedding) tuples
metadatas = []              # list of metadata dicts

for entry in entities_data:
    entity_info = entry.get("entity", {})  # entity metadata
    table_name = entity_info.get("TABLE_NAME")
    entity_name = entity_info.get("ENTITY NAME")
    for attr in entry.get("attributes", []):
        attr_name = attr.get("NAME")
        attr_def  = attr.get("DEFINITION", "")
        # Combine attribute name and definition as the text content
        if attr_name and attr_def:
            text = f"{attr_name}: {attr_def}"
        elif attr_def:
            text = attr_def
        else:
            text = attr_name or ""
        # Get embedding vector (convert to list of floats if not already)
        embedding_vector = attr.get("embedding", [])
        # Only add if we have a valid embedding vector
        if embedding_vector:
            text_embedding_pairs.append((text, embedding_vector))
            metadatas.append({
                "attribute_name": attr_name,
                "entity_name": entity_name,
                "table_name": table_name
            })

# Build the FAISS vector store from the embeddings
faiss_vectorstore = FAISS.from_embeddings(
    text_embeddings=text_embedding_pairs,
    embedding=embedding_fn,
    metadatas=metadatas,
    normalize_L2=True   # normalize vectors for cosine similarity search
)

print(f"FAISS index built with {len(text_embedding_pairs)} attribute embeddings.")
```

In the code above, we use `FAISS.from_embeddings(...)` to create the vector index. This method takes an iterable of `(text, embedding)` pairs, the embedding function for queries, and the corresponding metadata. By setting `normalize_L2=True`, we ensure that both stored vectors and query vectors are normalized to unit length before comparison (making the L2 distance correspond to cosine similarity). After running this cell, `faiss_vectorstore` contains our vector index and an in-memory docstore of texts+metadata.

## Saving the FAISS Index and Metadata to DBFS

To use this index in future sessions or to share it between processes, we should save it to persistent storage. LangChain’s FAISS vectorstore provides a `save_local` method that saves:

* The FAISS index itself (to a `.faiss` file)
* The accompanying metadata (to a pickle `.pkl` file containing the docstore and index-to-doc ID mappings)

We'll choose a directory in DBFS to save these files. Make sure the path exists or use `save_local` to create it.

```python
# Define a directory path in DBFS to save the index and metadata
index_save_path = "/dbfs/FileStore/faiss_index_demo"

# Save the FAISS index and metadata to the specified path
faiss_vectorstore.save_local(folder_path=index_save_path)

# List saved files (should see 'index.faiss' and 'index.pkl')
import os
print("Saved files:", os.listdir(index_save_path.replace("dbfs:", "/dbfs")))
```

After this step, the FAISS index is saved to `index.faiss` and the metadata to `index.pkl` in the chosen DBFS directory. These can be used to reload the vector store without rebuilding it from scratch.

**Note:** In the above code, we replace the `dbfs:` scheme with the `/dbfs` mount point to use Python's `os.listdir` for verification (this is a common pattern in Databricks for file system operations).

## Reloading the FAISS Index and Creating a Retriever

Now we will demonstrate how to load the saved FAISS index and use it in a retrieval pipeline. We use the `FAISS.load_local` class method to load the index and docstore from disk. This requires providing the same embedding function used originally, so that query embeddings can be computed consistently.

```python
from langchain_community.vectorstores import FAISS

# Load the FAISS vector store from the saved files
# (allow_dangerous_deserialization=True is required to load the pickled metadata, assuming we trust this data source)
faiss_index_loaded = FAISS.load_local(
    folder_path=index_save_path,
    embeddings=embedding_fn,
    allow_dangerous_deserialization=True
)

# Create a retriever from the loaded vector store
retriever = faiss_index_loaded.as_retriever(search_kwargs={"k": 3})
```

We set `k=3` in the `search_kwargs` so that the retriever will fetch the top 3 most similar attribute definitions for each query. You can adjust this `k` value based on how many context pieces you want to provide to the LLM. The retriever uses the FAISS index to find relevant documents (attribute texts) given a query, using cosine similarity under the hood (since we normalized vectors earlier).

*Security note:* We passed `allow_dangerous_deserialization=True` to `load_local` because it reads a pickle file for the metadata. This is safe here since the data was generated by us and stored in a secure location. In general, be cautious and only load pickled data from trusted sources.

## Setting Up the LLM and Retrieval QA Chain

With the retriever ready, we can create a Retrieval QA chain. This chain will use OpenAI's GPT-3.5-Turbo model to generate answers to questions, given the context retrieved from our FAISS index.

**Components:**

* **LLM (ChatOpenAI):** We use the chat model `gpt-3.5-turbo` via LangChain’s `ChatOpenAI` wrapper. We set a temperature of 0 for deterministic output (you can adjust as needed). Make sure to configure your OpenAI API key, for example by setting the environment variable `OPENAI_API_KEY` or using Databricks secrets.
* **RetrievalQA Chain:** LangChain’s `RetrievalQA` will combine the retriever and LLM. It will automatically fetch relevant documents for a query and prompt the LLM to answer using those documents as context.

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Ensure your OpenAI API key is set (for example, via environment variable or Databricks secret)
# For example:
# import os
# os.environ["OPENAI_API_KEY"] = "YOUR-OPENAI-API-KEY"

# Initialize the OpenAI GPT-3.5 Turbo model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Set up the RetrievalQA chain with the LLM and retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",             # "stuff" method to directly stuff retrieved docs into the prompt
    retriever=retriever,
    return_source_documents=False   # we don't need to return source docs in the answer
)
```

Now, `qa_chain` is ready to answer questions. The chain will use our `retriever` to get the top matching attribute definitions for a query and then ask the LLM to formulate an answer based on that context.

*Note:* We used `chain_type="stuff"` which simply inserts the retrieved documents into the prompt. LangChain also supports other chain types like `"map_reduce"` or `"refine"` for more complex scenarios, but "stuff" is sufficient for many QA use-cases with smaller context sizes.

## Querying the QA System and Getting Results

Finally, we can query the system with natural language questions. The RetrievalQA chain will produce an answer string. We also want to capture the metadata of the top-matching attribute for each query (attribute name, entity name, table name, and similarity score), as requested.

We will run a few example queries and collect the results in a pandas DataFrame for clarity. Each row will contain:

* **query:** The input question asked.
* **answer:** The answer generated by GPT-3.5-Turbo.
* **top\_match\_attribute:** Name of the highest-scoring attribute that was retrieved.
* **entity\_name:** Name of the entity (or table) that this attribute belongs to.
* **table\_name:** The table name associated with the entity/attribute.
* **score:** The similarity score for the top attribute (note: this is the L2 distance if using normalized vectors, where a lower score means higher similarity).

```python
import pandas as pd

# Example queries to test the QA system
queries = [
    "What attribute contains the customer's email address?",
    "Give me the definition of the Order ID field."
]

results = []
for q in queries:
    # Get answer from the QA chain
    answer = qa_chain.run(q)
    # Retrieve the top matching attribute and its score
    top_docs_with_score = faiss_index_loaded.similarity_search_with_score(q, k=1)
    top_doc, top_score = top_docs_with_score[0]
    meta = top_doc.metadata
    results.append({
        "query": q,
        "answer": answer,
        "top_match_attribute": meta.get("attribute_name", ""),
        "entity_name": meta.get("entity_name", ""),
        "table_name": meta.get("table_name", ""),
        "score": top_score
    })

# Convert results to a pandas DataFrame for display
df_results = pd.DataFrame(results)
df_results
```

The resulting DataFrame `df_results` will show each query, the answer, and details of the top-matching attribute that was used to find the answer. For example, you might see something like:

| query                                     | answer                                    | top\_match\_attribute | entity\_name | table\_name   | score  |
| ----------------------------------------- | ----------------------------------------- | --------------------- | ------------ | ------------- | ------ |
| What attribute contains the customer's... | The attribute **Email Address** stores... | Email Address         | Customer     | CUSTOMER\_TBL | 0.0... |
| Give me the definition of the Order ID... | **Order ID** is the unique identifier...  | Order ID              | Orders       | ORDER\_TABLE  | 0.0... |

Each answer is generated by the LLM using the definition of the top attributes (and possibly a couple of others, since we set `k=3` for context). The **score** column shows the FAISS similarity score of the top attribute. If using normalized cosine similarity with L2 distance, a score closer to 0 indicates a very close match (0 would mean identical embedding, in theory).

You can now ask new questions by adding to the `queries` list or integrating this `qa_chain` into an application. The system will retrieve the most relevant attribute definitions and use them to answer your questions, providing traceability via the attribute metadata.
