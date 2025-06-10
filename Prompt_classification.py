import os
import psycopg2
import numpy as np
import collections
import pandas as pd
import openai

# -----------------------------------------------------------------------------
# 0. OpenAI / Azure OpenAI setup
# -----------------------------------------------------------------------------
openai.api_key    = os.getenv("OPENAI_API_KEY")
# (or set api_type/base/version if you use Azure)

# -----------------------------------------------------------------------------
# 1. your embedding function
# -----------------------------------------------------------------------------
def get_embedding(text: str) -> np.ndarray:
    # … your implementation here …
    raise NotImplementedError

# -----------------------------------------------------------------------------
# 2. vector‐lookup + voting classifier, now using name/definition/examples
# -----------------------------------------------------------------------------
def classify_cid(
    name: str,
    definition: str,
    examples: str,
    top_k: int = 5
) -> dict:
    # build the prompt for embedding
    query_text = (
        f"Name: {name}\n"
        f"Definition: {definition}\n"
        f"Examples: {examples}"
    )
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
            return {
                "CID status":             None,
                "CID category":           None,
                "Attributes sub-category": None,
            }

        # majority vote each field
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
    if cid_category in {"Category_A", "Category_B", "Category_C"}:
        return "Red"
    elif cid_category == "Category_D":
        return "Amber"
    elif cid_category is None or pd.isna(cid_category):
        return None
    else:
        return "Green"

# -----------------------------------------------------------------------------
# 4. justification via LLM
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
An attribute with:
  • Name: "{name}"
  • Definition: "{definition}"
  • Examples: "{examples}"
was classified as:
  • CID status: {cid_status}
  • CID category: {cid_category}
  • Attributes sub-category: {subcat}

In 2–3 sentences, explain in plain English why this classification makes sense.
"""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content": prompt}],
        temperature=0.3,
        max_tokens=100
    )
    return resp.choices[0].message.content.strip()

# -----------------------------------------------------------------------------
# 5. apply to your DataFrame
# -----------------------------------------------------------------------------
# assume df has columns: "COLUMN_NAME", "COLUMN_DEFINITION", "EXAMPLE"
# e.g.
# df = pd.DataFrame({
#     "COLUMN_NAME":       ["Military ID",    "Other Attr", …],
#     "COLUMN_DEFINITION": ["…def of ID…",    "…def of foo…", …],
#     "EXAMPLE":           ["ID 12345",       "foo=bar",      …],
# })

# 5a) get raw classifications
cls = df.apply(
    lambda r: classify_cid(
        r["COLUMN_NAME"],
        r["COLUMN_DEFINITION"],
        r["EXAMPLE"]
    ),
    axis=1
)
cls_df = pd.DataFrame(cls.tolist(), index=df.index)

# 5b) merge back and map RAG
df = pd.concat([df, cls_df], axis=1)
df["RAG_Category"] = df["CID category"].apply(map_rag_category)

# 5c) generate human-readable justification
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

# final df columns:
# ["COLUMN_NAME","COLUMN_DEFINITION","EXAMPLE",
#  "CID status","CID category","Attributes sub-category",
#  "RAG_Category","Justification"]
print(df.head())
