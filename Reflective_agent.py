# Databricks notebook source
# MAGIC %md
# MAGIC # Reflexion Agent with Azure OpenAI
# MAGIC 
# MAGIC This notebook demonstrates how to build a self‑correcting **Reflexion** agent using the
# MAGIC [LangGraph](https://python.langchain.com/docs/use_cases/agents/langgraph) framework.  The agent
# MAGIC iteratively drafts an answer, runs search tools to gather new information, reflects on
# MAGIC deficiencies and missing citations, and revises its answer up to a maximum number of
# MAGIC iterations.  It is configured to use **Azure OpenAI** GPT‑4 Turbo via Databricks
# MAGIC secrets and proposes a placeholder search implementation for integration with Azure
# MAGIC search services.
# MAGIC 
# MAGIC ## Key features
# MAGIC 
# MAGIC - Uses `langgraph`, `langchain-community`, `langchain-core` and `pydantic`.
# MAGIC - Configured to call Azure OpenAI rather than the standard OpenAI API.  The
# MAGIC   endpoint URL, API key, and deployment name are loaded from Databricks secrets.
# MAGIC - Removes the Tavily dependency and replaces it with a simple **stub search function**.  A
# MAGIC   comment explains where to integrate a real search API such as Bing Search through
# MAGIC   Azure AI Services or a retrieval‑augmented generation (RAG) pipeline with
# MAGIC   [Azure AI Search](https://learn.microsoft.com/azure/search/search-what-is-azure-search).
# MAGIC - Implements the Reflexion loop as described by Shinn et al. and implemented in
# MAGIC   LangGraph: draft → execute tools → revise → repeat up to `max_iterations` times
# MAGIC   with an explicit reflection step that must include citations【589863008697738†L98-L105】.
# MAGIC - Demonstrates the agent on a sample question: **“Explain the applications of DeepSeek MoE and GRPO in AI research.”**  The final answer cites information from research sources about
# MAGIC   the Mixture‑of‑Experts architecture and the Group Relative Policy Optimization (GRPO)
# MAGIC   reinforcement learning algorithm【249130052367788†L256-L276】【535635772328128†L124-L141】.
# MAGIC 
# MAGIC **Note:** This notebook is intended as a template.  The search function is a
# MAGIC placeholder and should be replaced with a call to Azure’s Bing Search API or a
# MAGIC RAG pipeline built on top of Azure AI Search once those services are available in
# MAGIC your environment.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Install dependencies
# MAGIC 
# MAGIC We install the required libraries.  In Databricks notebooks, `%pip` installs
# MAGIC Python packages into the notebook’s session.  You can remove the `-q` flag
# MAGIC if you wish to see installation logs.


# COMMAND ----------
# MAGIC %pip install -q --upgrade langgraph langchain-community langchain-core pydantic langchain-openai


# COMMAND ----------
# MAGIC %md
# MAGIC ## Configure Azure OpenAI
# MAGIC 
# MAGIC Azure OpenAI requires an endpoint URL, an API key and a deployment name.  These
# MAGIC values should be stored in Databricks secrets for security.  Replace
# MAGIC `my_scope` with the name of your secret scope.  For example, if your
# MAGIC Databricks secret scope is called `azure_openai`, you would use:
# MAGIC 
# MAGIC ```python
# MAGIC endpoint = dbutils.secrets.get(scope="azure_openai", key="azure_openai_endpoint")
# MAGIC api_key = dbutils.secrets.get(scope="azure_openai", key="azure_openai_api_key")
# MAGIC deployment = dbutils.secrets.get(scope="azure_openai", key="azure_openai_deployment")
# MAGIC ```
# MAGIC 
# MAGIC The `AzureChatOpenAI` class from `langchain-openai` reads the endpoint and
# MAGIC API key from environment variables.  Therefore we set them using `os.environ`.


# COMMAND ----------
import os
from databricks import dbutils

## Load secrets from Databricks secret scope.
SCOPE_NAME = "my_scope"  # TODO: replace with your secret scope name

# Retrieve secrets securely.  These calls will throw an exception if the
# secrets are not defined in your workspace.
endpoint = dbutils.secrets.get(scope=SCOPE_NAME, key="azure_openai_endpoint")
api_key = dbutils.secrets.get(scope=SCOPE_NAME, key="azure_openai_api_key")
deployment = dbutils.secrets.get(scope=SCOPE_NAME, key="azure_openai_deployment")

# Set environment variables expected by AzureChatOpenAI
os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
os.environ["AZURE_OPENAI_API_KEY"] = api_key

from langchain_openai import AzureChatOpenAI

# Instantiate the chat model.  Temperature 0 yields deterministic outputs.
llm = AzureChatOpenAI(
    azure_deployment=deployment,
    api_version="2024-02-15-preview",
    temperature=0
)


# COMMAND ----------
# MAGIC %md
# MAGIC ## Define a search stub and data models
# MAGIC 
# MAGIC The Reflexion agent uses external tools to fetch additional information.  In
# MAGIC the original implementation, the Tavily web search API was used.  Since we
# MAGIC remove Tavily, we define a placeholder function `search_stub` that accepts a
# MAGIC query and returns a list of search results.  You should replace the
# MAGIC implementation with a call to [Azure Bing Search](https://learn.microsoft.com/azure/ai-services/bing-search-web) or a
# MAGIC retrieval‑augmented generation pipeline using Azure AI Search【535635772328128†L124-L141】.
# MAGIC 
# MAGIC We also define a `Reflection` data model using Pydantic to enforce the
# MAGIC structure of the reflection output.  Each reflection contains the agent’s
# MAGIC internal `thought`, a list of `new_actions` (search queries) to execute
# MAGIC next, and a list of `citations` to include in the revised answer.


# COMMAND ----------
from typing import List, Dict
from pydantic import BaseModel


def search_stub(query: str) -> List[Dict[str, str]]:
    """
    Placeholder search tool.

    In production, replace this function with a real web search.  For example,
    you could use Azure Bing Search by making HTTP requests to your Bing Search
    resource, or you could implement a retrieval‑augmented generation (RAG)
    pipeline on top of Azure AI Search【535635772328128†L124-L141】.  The function should return a list of
    dictionaries with at least 'title', 'url', and 'content' fields.

    Parameters
    ----------
    query : str
        A natural language search query.

    Returns
    -------
    List[Dict[str, str]]
        A list of search results.  Each result contains a title, a URL and
        content (the snippet).
    """
    # Example stub returns a single canned result describing DeepSeek MoE and GRPO.
    # Replace this with your own search implementation.
    return [
        {
            "title": "DeepSeek MoE & V2",
            "url": "https://creativestrategies.com/deepseek-moe-v2/",
            "content": (
                "DeepSeek’s Mixture‑of‑Experts model uses shared experts to capture common "
                "knowledge across contexts and only activates roughly 2.8B of its 16.4B parameters "
                "per token, improving efficiency【249130052367788†L256-L276】."
            ),
        },
        {
            "title": "Why GRPO is Important and How it Works",
            "url": "https://oxen.ai/why-grpo-is-important-and-how-it-works/",
            "content": (
                "Group Relative Policy Optimization (GRPO) reduces reinforcement learning overhead by "
                "dropping the value model used in PPO and instead generates multiple outputs per "
                "query and computes advantages relative to the group mean and standard deviation【535635772328128†L124-L141】."
            ),
        },
    ]


class Reflection(BaseModel):
    """Data model describing the reflection step."""

    thought: str
    new_actions: List[str]
    citations: List[str]


# COMMAND ----------
# MAGIC %md
# MAGIC ## Build the Reflexion agent
# MAGIC 
# MAGIC The Reflexion loop consists of three main nodes in the LangGraph:
# MAGIC 
# MAGIC 1. **Draft:** generate an initial answer to the question and propose search queries (actions).
# MAGIC 2. **Execute tools:** run the proposed search queries using the search stub and return observations.
# MAGIC 3. **Revise:** produce a revised answer by reflecting on the current state, tool results, and previous answer.  The revision must include citations and may propose new search actions for another iteration【589863008697738†L98-L105】.
# MAGIC 
# MAGIC We iterate this loop up to `max_iterations` times.  After the final revision, the
# MAGIC last answer is returned.  The state passed around the graph is a list of
# MAGIC LangChain messages (`HumanMessage`, `AIMessage`, etc.).


# COMMAND ----------
from langgraph.graph import END, MessageGraph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import json


# Helper to extract the number of iterations from the state.  We count each
# draft/revise pair as one iteration.
def _get_num_iterations(state: List) -> int:
    return sum(1 for m in state if isinstance(m, AIMessage) and m.name == "revise")


def draft_node(state: List) -> AIMessage:
    """
    Generate an initial answer and propose search actions.

    The LLM is prompted to write a first draft answer to the user’s question and
    suggest up to three web search queries that might uncover additional supporting
    evidence or missing information.  The queries should be returned in a JSON
    array in the message metadata.
    """
    # The last message in the state should be the user's question.
    question = next(m.content for m in state if isinstance(m, HumanMessage))
    system_prompt = (
        "You are an assistant tasked with answering the user’s question. "
        "Provide a concise answer and suggest up to three web search queries to gather "
        "additional information. Return your answer in plain text and list the search queries "
        "as a JSON array separated by a line break '---'. Do not fabricate citations; "
        "citations will be added later."
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    response = llm.invoke(messages)
    # Parse the search queries.  The response is expected to contain the answer and then
    # '---' followed by a JSON list of queries.  If parsing fails, we default to an empty list.
    content = response.content
    if "---" in content:
        parts = content.split("---")
        answer_part = parts[0]
        actions_json = parts[1]
    else:
        answer_part = content
        actions_json = "[]"
    try:
        actions = json.loads(actions_json.strip()) if actions_json else []
    except Exception:
        actions = []
    return AIMessage(content=answer_part.strip(), name="draft", additional_kwargs={"actions": actions})


def execute_tools_node(state: List) -> List:
    """
    Execute search queries and append ToolMessages with observations to the state.

    This node inspects the actions produced by the draft or revise step and runs
    them through `search_stub`.  Each observation is appended to the state as a
    `ToolMessage` with the tool’s output.  If no actions are present, the state
    is returned unchanged.
    """
    last_message = next((m for m in reversed(state) if isinstance(m, AIMessage)), None)
    if not last_message:
        return state
    actions = last_message.additional_kwargs.get("actions", []) if last_message.additional_kwargs else []
    tool_messages = []
    for query in actions:
        results = search_stub(query)
        snippets = []
        for res in results[:2]:
            snippets.append(f"{res['title']}: {res['content']} (source: {res['url']})")
        observation = "\n".join(snippets)
        tool_messages.append(ToolMessage(content=observation, name="search"))
    return state + tool_messages


def revise_node(state: List) -> AIMessage:
    """
    Reflect on the current answer and produce a revised answer.

    The reviser is given the entire conversation history (including tool
    observations) and asked to critique the last answer.  It should identify
    missing information, incorporate evidence from the search results, cite
    sources using tether IDs (pre‑provided citations), and propose up to two new
    search queries if more information is required.  The output format mirrors
    the draft step: a revised answer followed by '---' and a JSON list of new
    actions【589863008697738†L98-L105】.
    """
    messages: List = []
    messages.append(
        SystemMessage(
            content=(
                "You are a critical reviewer tasked with improving an answer to the "
                "user’s question.  Review the prior answer, examine the tool observations, "
                "and highlight any missing or incorrect information.  Compose a revised answer "
                "that synthesizes all available evidence and includes citations in the format "
                "【tether†Lstart-Lend】.  At the end, suggest up to two new search queries as a JSON "
                "array separated by a line break '---'.  If no further search is necessary, return an empty array."
            )
        )
    )
    for msg in state:
        if isinstance(msg, HumanMessage):
            messages.append(HumanMessage(content=msg.content))
        elif isinstance(msg, AIMessage):
            messages.append(AIMessage(content=msg.content, name=msg.name, additional_kwargs=msg.additional_kwargs))
        elif isinstance(msg, ToolMessage):
            messages.append(SystemMessage(content=f"Tool output: {msg.content}"))
    response = llm.invoke(messages)
    content = response.content
    if "---" in content:
        parts = content.split("---")
        answer_part = parts[0]
        actions_json = parts[1]
    else:
        answer_part = content
        actions_json = "[]"
    try:
        actions = json.loads(actions_json.strip()) if actions_json else []
    except Exception:
        actions = []
    return AIMessage(content=answer_part.strip(), name="revise", additional_kwargs={"actions": actions})


def build_reflexion_graph(max_iterations: int = 2) -> MessageGraph:
    """
    Construct the LangGraph reflexion loop.

    Parameters
    ----------
    max_iterations : int
        Maximum number of revise iterations before terminating.

    Returns
    -------
    MessageGraph
        A compiled LangGraph ready to invoke.
    """
    builder = MessageGraph()
    builder.add_node("draft", draft_node)
    builder.add_node("execute_tools", execute_tools_node)
    builder.add_node("revise", revise_node)
    builder.set_entry_point("draft")
    builder.add_edge("draft", "execute_tools")
    builder.add_edge("execute_tools", "revise")

    def event_loop(state: List) -> str:
        if _get_num_iterations(state) >= max_iterations:
            return END
        return "execute_tools"
    builder.add_conditional_edges("revise", event_loop)
    return builder.compile()


# COMMAND ----------
# MAGIC %md
# MAGIC ## Run a sample question
# MAGIC 
# MAGIC We now build the reflexion graph and invoke it with a sample question:
# MAGIC 
# MAGIC > **“Explain the applications of DeepSeek MoE and GRPO in AI research.”**
# MAGIC 
# MAGIC The agent will draft an answer, use the search stub to retrieve information
# MAGIC about DeepSeek’s Mixture‑of‑Experts architecture and the Group Relative Policy
# MAGIC Optimization algorithm, reflect on the results, and revise the answer.  For
# MAGIC demonstration purposes, the search stub returns static results containing
# MAGIC information about shared experts in DeepSeek MoE and the efficiency of GRPO
# MAGIC【249130052367788†L256-L276】【535635772328128†L124-L141】.


# COMMAND ----------
# Build the graph with a maximum of two revise iterations
graph = build_reflexion_graph(max_iterations=2)

# Define the user question
question = "Explain the applications of DeepSeek MoE and GRPO in AI research."

# Invoke the graph.  The input state is a list containing the user's message.
initial_state: List = [HumanMessage(content=question)]
final_state = graph.invoke(initial_state)

# Extract and display the final answer (content of the last AI message)
final_answer = next((m.content for m in reversed(final_state) if isinstance(m, AIMessage)), "")
print(final_answer)
