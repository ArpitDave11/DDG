# Install necessary libraries with specific versions
%pip install langchain==0.3.25 langchain-openai==0.3.17 langchain-community==0.3.24 \
    faiss-gpu==1.7.2 SQLAlchemy==2.0.40 psycopg2-binary==2.9.10

# **Package Versions**:
# - LangChain == 0.3.25
# - LangChain-OpenAI == 0.3.17
# - LangChain-Community == 0.3.24
# - FAISS-GPU == 1.7.2
# - SQLAlchemy == 2.0.40
# - psycopg2-binary == 2.9.10
#
# These versions are chosen for compatibility with Databricks Runtime 14.3 (Python 3.10.12, NumPy 1.23.5, Pandas 1.5.3).
# They will not force an upgrade of NumPy/Pandas (LangChain's requirements allow NumPy 1.23.5:contentReference[oaicite:0]{index=0}, and Pandas 1.5.x is supported).
#
# **Issue Explanation & Fix**:
# The `qa_chain` was likely failing due to improper Azure OpenAI configuration or version mismatches. 
# Common issues include not using the Azure-specific LLM class or not providing the necessary Azure parameters (deployment name, API base, type, version).
# In earlier attempts, using the base `OpenAI` class or a mis-configured Azure client led to errors like **"Must provide an 'engine' or 'deployment_id' parameter"**:contentReference[oaicite:1]{index=1}, meaning the Azure deployment info wasn't being passed.
# We fix this by:
#   - Installing compatible versions of LangChain and its integration packages (to avoid mismatched interfaces).
#   - Using `AzureChatOpenAI` (from langchain_openai) for chat models with Azure, and specifying `deployment_name` (Azure deployment ID), as well as Azure-specific endpoint, API type, and version (either via environment variables or directly):contentReference[oaicite:2]{index=2}.
#   - Ensuring the OpenAI Python SDK is installed (automatically via langchain-openai) and that the correct Azure OpenAI credentials are provided.
#   - If using Azure OpenAI for embeddings, using `AzureOpenAIEmbeddings` with the corresponding Azure deployment (or setting `openai_api_type='azure'` in the embedding class).
#
# The code below demonstrates a full setup:
# 1. Connect to a data source or define documents (for example purposes, we create sample text documents).
# 2. Generate embeddings using Azure OpenAI (embedding model deployment).
# 3. Build a FAISS vector store for retrieval.
# 4. Initialize the Azure OpenAI chat model for Q&A (chat model deployment).
# 5. Create a RetrievalQA chain using the retriever and chat model.
# 6. Run a test query through the chain and print the answer and sources.
#
# **Note**: Replace the placeholder values for `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, and deployment names with your actual Azure OpenAI resource endpoint, API key, and the names of your deployed models.

import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Set Azure OpenAI credentials and endpoint (replace with your actual details)
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://<AZURE_OPENAI_ENDPOINT>.openai.azure.com/"  # e.g., "https://my-resource-name.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "<AZURE_OPENAI_KEY>"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"  # use the API version for your Azure OpenAI deployment

# (Alternatively, you can pass the above credentials as parameters directly to AzureOpenAI classes instead of environment variables.)

# 1. **Load or define documents** 
# (Here we define some sample text documents with metadata. In practice, you might load data from a database or files.)
texts = [
    "Databricks is a unified analytics platform, an innovation by the creators of Apache Spark.",
    "LangChain is a framework for developing applications powered by language models.",
    "Azure OpenAI Service allows access to OpenAI's powerful language models on Azure infrastructure."
]
docs = [Document(page_content=text, metadata={"source": f"Document {i+1}"}) for i, text in enumerate(texts)]

# 2. **Create embeddings for documents using Azure OpenAI** 
# Ensure you have an embedding model deployed on Azure (e.g., text-embedding-ada-002)
embedding_model_deployment = "<EMBEDDING_MODEL_DEPLOYMENT_NAME>"  # Azure deployment name for your embedding model
embeddings = AzureOpenAIEmbeddings(deployment_name=embedding_model_deployment)

# Embed and build the vector index (FAISS)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# 3. **Initialize Azure OpenAI chat model for Q&A** 
# Provide your Azure ChatGPT deployment name (e.g., for gpt-35-turbo or GPT-4)
chat_model_deployment = "<CHAT_MODEL_DEPLOYMENT_NAME>"  # Azure deployment name for your chat model (e.g., "gpt-35-turbo" deployment)
llm = AzureChatOpenAI(
    deployment_name=chat_model_deployment,
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type=os.environ["OPENAI_API_TYPE"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    # Note: model_name not needed for Azure; deployment_name is used
    streaming=False
)

# 4. **Create the RetrievalQA chain**
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True
)

# 5. **Test the chain with a sample query**
query = "What platform was created by the founders of Apache Spark?"
result = qa_chain({"query": query})

# 6. **Display the answer and its sources**
answer = result["result"]
source_docs = result["source_documents"]
print("Q:", query)
print("A:", answer)
print("Sources:")
for doc in source_docs:
    # Each document has 'source' metadata as set above
    print("-", doc.metadata.get("source", "Unknown"))
