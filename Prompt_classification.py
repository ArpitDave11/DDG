import os
import psycopg2
import numpy as np
import collections
import pandas as pd

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# -----------------------------------------------------------------------------
# 0. Azure OpenAI via LangChain
# -----------------------------------------------------------------------------
azure_llm = AzureChatOpenAI(
    deployment_name    = os.getenv("AZURE_CHAT_DEPLOYMENT_NAME"),
    openai_api_base    = os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key     = os.getenv("AZURE_OPENAI_KEY"),
)

# -----------------------------------------------------------------------------
# 1. embedding function (user implements)
# -----------------------------------------------------------------------------
def get_embedding(text: str) -> np.ndarray:
    # … your Azure or OpenAI embedding call …
    raise NotImplementedError

# -----------------------------------------------------------------------------
# 2. improved classify_cid with distance threshold
# -----------------------------------------------------------------------------
def classify_cid(
    name: str,
    definition: str,
    examples: str,
    top_k: int = 5,
    threshold: float = 0.75  # tune this on a held-out set!
) -> dict:
    """
    Returns a dict with keys:
      • CID status         ("CID" / "Non-CID")
      • CID category       (e.g. "Category_A" or None)
      • Attributes sub-cat (e.g. "Natural Person" or None)
      • _min_distance      (float, for debugging)
    """
    # 1) build and embed the query
    query = f"Name: {name}\nDefinition: {definition}\nExamples: {examples}"
    q_emb  = get_embedding(query)

    # 2) fetch nearest neighbors
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
                (q_emb, q_emb, top_k)
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    # 3) if nothing or too far → rule‐based “No CID or NSI”
    if not rows:
        return {
            "CID status":             "Non-CID",
            "CID category":           None,
            "Attributes sub-category": None,
            "_min_distance":          None
        }
    distances = [r[3] for r in rows]
    min_dist  = min(distances)
    if min_dist > threshold:
        # fallback rule: attribute NOT used in UBS context
        return {
            "CID status":             "Non-CID",
            "CID category":           None,
            "Attributes sub-category": None,
            "_min_distance":          min_dist
        }

    # 4) otherwise, do your usual majority‐vote
    status_votes   = [r[0] for r in rows]
    category_votes = [r[1] for r in rows]
    subcat_votes   = [r[2] for r in rows]

    return {
        "CID status":             collections.Counter(status_votes).most_common(1)[0][0],
        "CID category":           collections.Counter(category_votes).most_common(1)[0][0],
        "Attributes sub-category": collections.Counter(subcat_votes).most_common(1)[0][0],
        "_min_distance":          min_dist
    }

# -----------------------------------------------------------------------------
# 3. RAG mapping (unchanged)
# -----------------------------------------------------------------------------
def map_rag_category(cid_cat: str) -> str:
    if cid_cat in {"Category_A","Category_B","Category_C"}:
        return "Red"
    elif cid_cat == "Category_D":
        return "Amber"
    elif cid_cat is None:
        return None
    else:
        return "Green"

# -----------------------------------------------------------------------------
# 4. LLM justification (unchanged)
# -----------------------------------------------------------------------------
def justify_classification(
    name, definition, examples,
    cid_status, cid_category, subcat
) -> str:
    sys   = SystemMessage(content="You are a helpful assistant that explains classification decisions clearly.")
    user  = HumanMessage(content=f"""
An attribute has:
  • Name: "{name}"
  • Definition: "{definition}"
  • Examples: "{examples}"

It was classified as:
  • CID status: {cid_status}
  • CID category: {cid_category}
  • Sub-category: {subcat}

In 2–3 sentences of plain English, explain why this classification makes sense.
""")
    resp = azure_llm([sys, user])
    return resp.content.strip()

# -----------------------------------------------------------------------------
# 5. apply to DataFrame
# -----------------------------------------------------------------------------
# assume df with ["COLUMN_NAME","COLUMN_DEFINITION","EXAMPLE"]
# e.g.:
# df = pd.DataFrame({
#   "COLUMN_NAME":       ["X","Y",…],
#   "COLUMN_DEFINITION": ["…","…",…],
#   "EXAMPLE":           ["…","…",…],
# })

# 5a) raw classify (with threshold fallback)
cls = df.apply(
    lambda r: classify_cid(
        r["COLUMN_NAME"],
        r["COLUMN_DEFINITION"],
        r["EXAMPLE"],
        top_k=5,
        threshold=0.75
    ),
    axis=1
)
cls_df = pd.DataFrame(cls.tolist(), index=df.index)

# 5b) merge, map RAG
df = pd.concat([df, cls_df], axis=1)
df["RAG_Category"] = df["CID category"].apply(map_rag_category)

# 5c) generate LLM justifications
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
