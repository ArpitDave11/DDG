# Databricks notebook source

"""
Databricks Notebook for Reflexion Agent
======================================

This notebook demonstrates how to build a self‑reflecting Q&A agent using
LangGraph, LangChain, and Azure OpenAI within a Databricks environment.
The workflow mirrors the architecture described in the provided reference code,
but restructures the components into separate modules and adds support for
parameterising the run via an input JSON. Each major part of the agent lives
in its own Python file, which are created on the fly the first time the
notebook runs. A DAG built with ``MessageGraph`` orchestrates the
interaction between modules.

**Overview of the workflow:**

1. **Dependency installation** – install required Python packages.
2. **Environment configuration** – fetch Azure OpenAI settings from
   Databricks secrets and write them into a ``.env`` file. This file is
   consumed by ``dotenv`` so that the agent can read the credentials.
3. **Module creation** – write ``schemas.py``, ``first_responder_agent.py``,
   ``revise_agent.py``, and ``tool_executor.py`` into the working directory.
4. **Graph construction** – define the DAG in ``main.py``. The DAG has three
   nodes: a draft node (first responder), a tool execution node, and a
   revise node. The graph loops through tool execution and revision up to
   ``max_iterations`` times.
5. **Parametrised execution** – accept a JSON string from a Databricks
   widget, parse it to extract the question and optional iteration limit,
   and run the graph accordingly.

Running this notebook in Databricks will install the necessary libraries,
generate the code modules, load environment variables from secrets, and
execute the reflexion agent end‑to‑end. To customise the question or
iterations, supply a JSON string via the ``params`` widget before executing
the final cell.

Note: The ``run_queries`` function in ``tool_executor.py`` is stubbed out to
return placeholder results. Replace this stub with calls to a search API of
your choice (e.g. Azure Bing Search or Azure AI Search) to enable real
web search functionality.
"""

# COMMAND ----------
"""Install required dependencies.

In Databricks notebooks, using ``%pip`` ensures that packages are scoped to
the session and properly installed. Adjust version constraints as needed.
"""

# MAGIC %pip install --quiet --disable-pip-version-check \
    langgraph==0.3.3 \
    langchain-community>=0.3.18,<0.4.0 \
    langchain-openai>=0.1.0,<0.2.0 \
    python-dotenv>=1.0.0,<2.0.0 \
    pydantic>=1.10.0,<2.0.0

# COMMAND ----------
"""Configure environment variables from Databricks secrets.

This cell retrieves the Azure OpenAI configuration from Databricks secret
scopes and writes them to a ``.env`` file. The agent will later read
from this file via ``load_dotenv``. You must ensure that the secrets
``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_DEPLOYMENT_NAME``,
and ``AZURE_OPENAI_API_VERSION`` exist in the specified secret scope.

If you prefer to set environment variables manually or use another secret
management mechanism, modify this cell accordingly.
"""

import os
from dotenv import load_dotenv

# Define the secret scope and keys. Update ``secret_scope`` to match your
# Databricks configuration.
secret_scope = "openai"

endpoint = dbutils.secrets.get(scope=secret_scope, key="AZURE_OPENAI_ENDPOINT")
api_key = dbutils.secrets.get(scope=secret_scope, key="AZURE_OPENAI_API_KEY")
deployment_name = dbutils.secrets.get(scope=secret_scope, key="AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = dbutils.secrets.get(scope=secret_scope, key="AZURE_OPENAI_API_VERSION")

# Write secrets into .env file in driver filesystem. The driver path persists
# during the notebook session.
env_path = "/databricks/driver/.env"
with open(env_path, "w") as env_file:
    env_file.write(f"AZURE_OPENAI_ENDPOINT={endpoint}\n")
    env_file.write(f"AZURE_OPENAI_API_KEY={api_key}\n")
    env_file.write(f"AZURE_OPENAI_DEPLOYMENT_NAME={deployment_name}\n")
    env_file.write(f"AZURE_OPENAI_API_VERSION={api_version}\n")

# Load the variables into the current process
load_dotenv(env_path)

print("Environment variables configured and loaded.")

# COMMAND ----------
"""Write the Pydantic schemas module.

Defines the core data structures used to communicate between nodes. The
``AnswerQuestion`` class includes an answer, a nested ``Reflection`` object,
and a list of search queries. ``ReviseAnswer`` extends ``AnswerQuestion`` by
adding a ``references`` field for citations.
"""

schemas_code = """from typing import List
from pydantic import BaseModel, Field


class Reflection(BaseModel):
    """Critiques on missing or superfluous information."""

    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")


class AnswerQuestion(BaseModel):
    """Response structure for the first answer along with reflection and search queries."""

    answer: str = Field(description="~250 words detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ReviseAnswer(AnswerQuestion):
    """Revision of the original answer with references for citations."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )
"""

schemas_path = "/databricks/driver/schemas.py"
with open(schemas_path, "w") as f:
    f.write(schemas_code)

print(f"Written schemas module to {schemas_path}")

# COMMAND ----------
"""Write the first responder agent module.

This module defines ``build_first_responder`` which returns an LLM chain
configured to produce the initial answer, reflection, and search queries.
The chain uses Azure OpenAI via LangChain’s ``AzureChatOpenAI`` class.
"""

first_responder_code = """import datetime
from typing import List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_openai import AzureChatOpenAI

from schemas import AnswerQuestion

def build_first_responder() -> AzureChatOpenAI:
    """Create the first responder chain.

    Returns a LangChain chain that, given a user question and message history,
    produces an AnswerQuestion Pydantic tool call. The model uses Azure
    OpenAI configured via environment variables ``AZURE_OPENAI_*``.
    """
    load_dotenv()
    # Build the prompt. The time is computed at execution to avoid stale timestamps.
    actor_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are expert researcher.\n\n"""
                "Current time: {time}\n\n"
                "1. {first_instruction}\n"
                "2. Reflect and critique your answer. Be severe to maximize improvement.\n"
                "3. Recommend search queries to research information and improve your answer."",
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Answer the user's question above using the required format."),
        ]
    ).partial(time=lambda: datetime.datetime.now().isoformat())
    first_responder_prompt_template = actor_prompt_template.partial(
        first_instruction="Provide a detailed ~250 words answer."
    )

    # Instantiate AzureChatOpenAI without explicitly passing credentials.  
    # When parameters are omitted, the library reads configuration from
    # environment variables (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    # AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME) loaded via
    # ``load_dotenv``.
    llm = AzureChatOpenAI()

    chain = first_responder_prompt_template | llm.bind_tools(
        tools=[AnswerQuestion], tool_choice="AnswerQuestion"
    )
    return chain
"""

first_responder_path = "/databricks/driver/first_responder_agent.py"
with open(first_responder_path, "w") as f:
    f.write(first_responder_code)

print(f"Written first responder agent module to {first_responder_path}")

# COMMAND ----------
"""Write the revise agent module.

This module defines ``build_revisor`` which returns an LLM chain
configured to revise the initial answer. The chain uses Azure OpenAI and
produces a ``ReviseAnswer`` Pydantic tool call with a constrained
word count and citation requirements.
"""

revise_agent_code = """import datetime
from typing import List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI

from schemas import ReviseAnswer

def build_revisor() -> AzureChatOpenAI:
    """Create the reviser chain.

    Returns a LangChain chain that revises an existing answer using
    critique and external information. The chain uses Azure OpenAI and
    produces a ``ReviseAnswer`` tool call.
    """
    load_dotenv()
    revise_instructions = """Revise your previous answer using the new information.\n    - You should use the previous critique to add important information to your answer.\n        - You MUST include numerical citations in your revised answer to ensure it can be verified.\n        - Add a \"References\" section to the bottom of your answer (which does not count towards the word limit). In form of:\n            - [1] https://example.com\n            - [2] https://example.com\n    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.\n"""

    actor_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are expert researcher.\n\n"""
                "Current time: {time}\n\n"
                "1. {first_instruction}\n"
                "2. Reflect and critique your answer. Be severe to maximize improvement.\n"
                "3. Recommend search queries to research information and improve your answer."",
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Answer the user's question above using the required format."),
        ]
    ).partial(time=lambda: datetime.datetime.now().isoformat())

    revisor_prompt_template = actor_prompt_template.partial(
        first_instruction=revise_instructions
    )

    # Instantiate AzureChatOpenAI without explicit credentials.  
    # The model pulls configuration from environment variables set via
    # ``load_dotenv``.
    llm = AzureChatOpenAI()

    chain = revisor_prompt_template | llm.bind_tools(
        tools=[ReviseAnswer], tool_choice="ReviseAnswer"
    )
    return chain
"""

revise_agent_path = "/databricks/driver/revise_agent.py"
with open(revise_agent_path, "w") as f:
    f.write(revise_agent_code)

print(f"Written revise agent module to {revise_agent_path}")

# COMMAND ----------
"""Write the tool executor module.

The ``run_queries`` function is intentionally a stub. It receives a list of
search queries and returns a dictionary mapping each query to a list of
dummy search results. Replace the body of this function with calls to a
search API (e.g. Azure Bing Search) to fetch real data. This module also
defines a ``ToolNode`` that registers two structured tools named after
``AnswerQuestion`` and ``ReviseAnswer`` to conform to the expected tool
names used by the agent.
"""

tool_executor_code = """from typing import List
from langchain_community.tools import TavilySearchResults  # noqa: F401 (unused import)
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool


def run_queries(search_queries: List[str], **kwargs) -> dict:
    """Stub function to run search queries.

    Parameters
    ----------
    search_queries : List[str]
        A list of search query strings.

    Returns
    -------
    dict
        A mapping from each query to a list of dummy results. Replace this
        implementation with calls to an actual search API to obtain
        meaningful results.
    """
    results = {}
    for query in search_queries:
        # Replace the dummy list below with actual search results. Each entry
        # could be a dict with keys like 'title', 'url', 'snippet', etc.
        results[query] = [
            {
                "title": "Dummy result for {}".format(query),
                "url": "https://example.com/{}".format(query.replace(" ", "_")),
                "snippet": "This is a placeholder result for '{}' generated by the stub.".format(query),
            }
        ]
    return results


tool_node = ToolNode(
    [
        StructuredTool.from_function(run_queries, name="AnswerQuestion"),
        StructuredTool.from_function(run_queries, name="ReviseAnswer"),
    ]
)
"""

tool_executor_path = "/databricks/driver/tool_executor.py"
with open(tool_executor_path, "w") as f:
    f.write(tool_executor_code)

print(f"Written tool executor module to {tool_executor_path}")

# COMMAND ----------
"""Write the main DAG script.

This script wires together the agents and tool executor into a LangGraph
``MessageGraph``. It supports parameterised execution based on a JSON
string provided via a Databricks widget named ``params``. The JSON may
contain ``question`` (string) and ``max_iterations`` (int) keys. If
``max_iterations`` is omitted, it defaults to 2.
"""

main_code = """import json
from typing import List

from langgraph.graph import MessageGraph, END
from langchain_core.messages import BaseMessage, ToolMessage

from schemas import AnswerQuestion, ReviseAnswer
from first_responder_agent import build_first_responder
from revise_agent import build_revisor
from tool_executor import tool_node


def build_graph(max_iterations: int = 2) -> MessageGraph:
    """Construct the reflexion graph.

    Parameters
    ----------
    max_iterations : int, default=2
        The maximum number of tool execution cycles. After this limit is
        reached, the graph terminates instead of looping back.

    Returns
    -------
    MessageGraph
        A compiled LangGraph message graph ready for invocation.
    """
    # Create the chains
    first_responder = build_first_responder()
    revisor = build_revisor()

    builder = MessageGraph()
    builder.add_node("draft", first_responder)
    builder.add_node("execute_tools", tool_node)
    builder.add_node("revise", revisor)
    builder.add_edge("draft", "execute_tools")
    builder.add_edge("execute_tools", "revise")

    def event_loop(state: List[BaseMessage]) -> str:
        # Count the number of ToolMessages to determine iteration count
        count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
        if count_tool_visits >= max_iterations:
            return END
        return "execute_tools"

    builder.add_conditional_edges("revise", event_loop)
    builder.set_entry_point("draft")
    return builder.compile()


def run_agent(input_json: str) -> str:
    """Run the reflexion agent given a JSON input.

    Parameters
    ----------
    input_json : str
        A JSON string with keys ``question`` (required) and
        ``max_iterations`` (optional).

    Returns
    -------
    str
        The final revised answer generated by the agent.
    """
    data = json.loads(input_json)
    question = data.get("question")
    if not question:
        raise ValueError("Input JSON must contain a 'question' field.")
    max_iters = int(data.get("max_iterations", 2))

    graph = build_graph(max_iterations=max_iters)
    result = graph.invoke(question)
    # The final message in the state is the revisor's AIMessage with a tool call
    last_message = result[-1]
    if not last_message.tool_calls:
        raise RuntimeError("Expected a tool call in the final message.")
    # Extract the answer from the tool call
    answer_payload = last_message.tool_calls[0]["args"]
    return answer_payload["answer"]


if __name__ == "__main__":
    # When running as a script (not in Databricks), parse parameters from
    # environment variable 'INPUT_JSON'. This allows testing locally.
    import os
    input_json = os.environ.get("INPUT_JSON")
    if not input_json:
        raise SystemExit("Please set the INPUT_JSON environment variable.")
    print(run_agent(input_json))
"""

main_path = "/databricks/driver/main.py"
with open(main_path, "w") as f:
    f.write(main_code)

print(f"Written main script to {main_path}")

# COMMAND ----------
"""Example: create a widget for input parameters and run the agent.

In Databricks, widgets allow you to parameterise notebook runs. Here we define
a text widget named ``params`` that should contain a JSON string. The
``run_agent`` function will parse this JSON and execute the reflexion
agent accordingly. If no widget is set, a default example question is used.
"""

import json

# Define a widget named 'params' if not already defined
try:
    dbutils.widgets.get("params")
except Exception:
    dbutils.widgets.text(
        "params",
        json.dumps(
            {
                "question": "Explain the applications of DeepSeek MoE and GRPO in AI research.",
                "max_iterations": 2,
            }
        ),
        "Input parameters (JSON)",
    )

params_json = dbutils.widgets.get("params")
print(f"Received params: {params_json}")

# Run the agent and display the final answer
from main import run_agent

final_answer = run_agent(params_json)
print("Final revised answer:\n")
print(final_answer)
