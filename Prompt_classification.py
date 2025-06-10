import os
import psycopg2
import numpy as np
import collections
import pandas as pd

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage

# -----------------------------------------------------------------------------
# 0. Azure OpenAI setup (via LangChain’s AzureChatOpenAI)
# -----------------------------------------------------------------------------
azure_llm = AzureChatOpenAI(
    deployment_name    = os.getenv("AZURE_CHAT_DEPLOYMENT_NAME"),  # e.g. "gpt-35-turbo"
    openai_api_base    = os.getenv("AZURE_OPENAI_ENDPOINT"),       # e.g. "https://my-resource.openai.azure.com/"
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),    # e.g. "2023-05-15"
    openai_api_key     = os.getenv("AZURE_OPENAI_KEY"),
)

# -----------------------------------------------------------------------------
# 1. your embedding function (unchanged)
# -----------------------------------------------------------------------------
def get_embedding(text: str) -> np.ndarray:
    # … your Azure or OpenAI embedding call …
    raise NotImplementedError

# -----------------------------------------------------------------------------
# 2. vector‐lookup + voting classifier
# -----------------------------------------------------------------------------
def classify_cid(
    name: str,
    definition: str,
    examples: str,
    top_k: int = 5
) -> dict:
    query_text = f"Name: {name}\nDefinition: {definition}\nExamples: {examples}"
    query_emb = get_embedding(query_text)

    conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        database=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    cid_status,
                    cid_category,
                    attributes_sub_category,
                    embedding <-> %s::vector AS distance
                FROM public.knowledge_store_cid
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
                """,
                (query_emb, query_emb, top_k)
            )
            rows = cur.fetchall()

        if not rows:
            return {"CID status": None, "CID category": None, "Attributes sub-category": None}

        status_votes   = [r[0] for r in rows]
        category_votes = [r[1] for r in rows]
        subcat_votes   = [r[2] for r in rows]

        return {
            "CID status":             collections.Counter(status_votes).most_common(1)[0][0],
            "CID category":           collections.Counter(category_votes).most_common(1)[0][0],
            "Attributes sub-category": collections.Counter(subcat_votes).most_common(1)[0][0],
        }
    finally:
        conn.close()

# -----------------------------------------------------------------------------
# 3. map to RAG
# -----------------------------------------------------------------------------
def map_rag_category(cid_category: str) -> str:
    if cid_category in {"Category_A","Category_B","Category_C"}:
        return "Red"
    elif cid_category == "Category_D":
        return "Amber"
    elif cid_category is None or pd.isna(cid_category):
        return None
    else:
        return "Green"

# -----------------------------------------------------------------------------
# 4. justify via AzureChatOpenAI
# -----------------------------------------------------------------------------
def justify_classification(
    name: str,
    definition: str,
    examples: str,
    cid_status: str,
    cid_category: str,
    subcat: str
) -> str:
    prompt = f"""
An attribute has:
  • Name: "{name}"
  • Definition: "{definition}"
  • Examples: "{examples}"

It was classified as:
  • CID status: {cid_status}
  • CID category: {cid_category}
  • Sub-category: {subcat}

In 2–3 sentences of plain English, explain why this classification makes sense.
"""
    response = azure_llm([HumanMessage(content=prompt)])
    return response.content.strip()

# -----------------------------------------------------------------------------
# 5. apply to your DataFrame
# -----------------------------------------------------------------------------
# Assume df with ["COLUMN_NAME","COLUMN_DEFINITION","EXAMPLE"]
# Example:
# df = pd.DataFrame({
#     "COLUMN_NAME":       ["Military ID", "Other Attr"],
#     "COLUMN_DEFINITION": ["…",           "…"],
#     "EXAMPLE":           ["ID 12345",    "foo=bar"],
# })

# 5a) raw classification
cls = df.apply(
    lambda r: classify_cid(
        r["COLUMN_NAME"],
        r["COLUMN_DEFINITION"],
        r["EXAMPLE"]
    ),
    axis=1
)
cls_df = pd.DataFrame(cls.tolist(), index=df.index)

# 5b) merge & RAG
df = pd.concat([df, cls_df], axis=1)
df["RAG_Category"] = df["CID category"].apply(map_rag_category)

# 5c) generate justifications
df["Justification"] = df.apply(
    lambda r: justify_classification(
        r["COLUMN_NAME"],
        r["COLUMN_DEFINITION"],
        r["EXAMPLE"],
        r["CID status"],
        r["CID category"],
        r["Attributes sub-category"]
    ),
    axis=1
)

print(df.head())
