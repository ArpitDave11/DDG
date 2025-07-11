Multi-Agent Data Quality Rule Generation and Validation Pipeline
Implementing a multi-stage pipeline with multiple AI agents can greatly enhance and automate data quality checks. We will use Microsoft's AutoGen framework (v0.4) to create a series of agents, each performing a specific role in the pipeline. This approach leverages Azure OpenAI models of different sizes for efficiency (using powerful GPT-4 for complex tasks and lighter GPT-3.5 for simpler ones), as suggested by recent research on multi-model pipelines. Below are the components of our solution:
Part 1: Recommendation Agent (GPT-4) – Proposes data quality rules for each column based on schema and sample data.
Part 2: Review Agent (GPT-3.5) – Validates and refines the rules from the first agent to ensure correctness.
Part 3: Execution Agent (Rule Checker) – Executes each rule on the DataFrame to identify any violations.
Part 4: Analysis Agent (GPT-4) – Analyzes failed rules and explains why those data quality checks failed.
Each agent will be implemented as an AutoGen AssistantAgent with an appropriate Azure OpenAI model client. This allows us to easily interface with the model and even integrate tools if needed. We will keep each agent’s code in a separate file for clarity and modularity, and then combine them in a main orchestration script.
Part 1: Recommendation Agent (GPT-4)
Our Recommendation Agent uses Azure OpenAI’s GPT-4 model to suggest data quality rules for each column in the dataset. GPT-4’s advanced reasoning capabilities make it well-suited for generating insightful and context-aware rules. The agent will read the table schema and sample data (e.g., first 1000 rows of each column) to understand the data distribution and types. It will then prompt GPT-4 to output a set of recommended rules in JSON format for easy parsing. Key steps for the Recommendation Agent:
Data Profiling: For each column, gather metadata such as data type, number of rows, count of nulls, distinct count, min/max values, or example values.
Prompt Construction: Construct a prompt that provides the column name, type, and profile, and instructs the model to suggest appropriate data quality rules. We emphasize that the response must be a structured JSON (e.g., a list of rule objects with descriptions).
Azure GPT-4 Client: Initialize an Azure OpenAI chat client for GPT-4. We provide the Azure deployment name, endpoint, and API key or token. For example, using AzureOpenAIChatCompletionClient with your GPT-4 deployment.
Assistant Agent: Create an AssistantAgent with the GPT-4 client. We set a system message to guide the agent’s behavior (e.g., “You are an expert data quality analyst... output only JSON”) and possibly temperature=0 for deterministic output.
LLM Invocation: Run the agent with the prepared prompt to get the JSON rules. Parse the JSON string into Python objects (using json.loads) for further processing.
Below is recommend_agent.py, implementing this logic:
python
Copy
Edit
# recommend_agent.py
import os, json, asyncio
import pandas as pd
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

# Azure OpenAI configuration (ensure environment variables or values are set)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://<your-resource>.openai.azure.com/")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY", "<YOUR-API-KEY>")
DEPLOYMENT_NAME_GPT4 = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "<GPT4-DEPLOYMENT-NAME>")
API_VERSION = "2023-05-15"  # or appropriate API version for your Azure OpenAI

# Initialize Azure OpenAI client for GPT-4
client_gpt4 = AzureOpenAIChatCompletionClient(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=DEPLOYMENT_NAME_GPT4,
    model="gpt-4",  # GPT-4 model
    api_version=API_VERSION,
    azure_api_key=AZURE_OPENAI_KEY
)

# Create the AssistantAgent for recommending rules
recommendation_agent = AssistantAgent(
    name="recommend_agent",
    model_client=client_gpt4,
    system_message=(
        "You are a data quality expert AI. "
        "Given a column name, its data type, and profile statistics, "
        "suggest a list of data quality rules for that column. "
        "Provide the rules in a JSON format as an array of objects, "
        "where each object has a 'rule' and a 'description'. "
        "No explanation outside the JSON – just return the JSON."
    )
    # (We keep temperature low for consistent output; tools can be omitted here)
)

def suggest_rules_for_column(column_name: str, data_series: pd.Series):
    """Generate suggested data quality rules for a single column using GPT-4."""
    # Gather profile info for the column
    dtype = str(data_series.dtype)
    total_rows = len(data_series)
    null_count = int(data_series.isna().sum())
    distinct_count = int(data_series.nunique(dropna=True))
    sample_values = data_series.dropna().unique()[:5]  # first 5 unique values as sample
    sample_list = ', '.join(map(str, sample_values))
    profile = (
        f"Column: {column_name}\n"
        f"Type: {dtype}\n"
        f"Total Rows: {total_rows}, Nulls: {null_count}, Distinct Values: {distinct_count}\n"
        f"Sample Values: {sample_list}\n"
    )

    # Construct user prompt
    prompt = (
        f"{profile}\n"
        f"Based on the above information, list 3-5 plausible data quality rules for the '{column_name}' column. "
        "The response **must** be valid JSON. Each rule should be an object with 'rule' and 'description' fields. "
        "Do not include any explanation outside the JSON."
    )

    # Run the agent to get rules (ensure we run it within an async context)
    result = asyncio.run(recommendation_agent.run(task=prompt))
    # Extract the assistant's final message content (the JSON string)
    # The TaskResult messages include the conversation; the last message should be the assistant's answer.
    assistant_messages = [m for m in result.messages if m.source == "assistant"]
    if not assistant_messages:
        raise RuntimeError("No assistant response received for rules suggestion.")
    output_content = assistant_messages[-1].content  # this should be the JSON text

    # Parse JSON output into Python structure
    try:
        rules = json.loads(output_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from GPT-4 output: {e}\nOutput was: {output_content}")
    return rules
Explanation: In the code above, we instantiate a GPT-4 based AssistantAgent using the Azure OpenAI client. The system prompt is crafted to ensure the output is a JSON array of rule objects (with no extraneous text). We then define suggest_rules_for_column, which profiles the pandas Series (count of nulls, distinct, etc.) and builds a prompt including these details. The agent’s response is awaited (using asyncio.run to execute the coroutine synchronously in a script) and then parsed as JSON. We include error handling in case the model’s output isn’t valid JSON (which can be extended to re-prompt or fix the JSON if necessary). Note: Ensure that your Azure OpenAI deployment and model names match the ones provided to the AzureOpenAIChatCompletionClient. The client can authenticate via API key (as above) or Azure AD token (as shown in Microsoft’s example using DefaultAzureCredential). The AssistantAgent is a versatile agent class in AutoGen that uses the specified model to generate responses.





Part 2: Review Agent (GPT-3.5)
The Review Agent uses a secondary model (a cost-efficient GPT-3.5, referred to as "o3-mini") to validate and refine the rules from the Recommendation Agent. The goal is to double-check each proposed rule for relevance and correctness before execution, saving resources by catching obvious errors or irrelevant rules with a cheaper model. This agent will take the initially suggested rules and the column profile as input and output a final validated list of rules (again in JSON format). Key steps for the Review Agent:
Contextual Input: Provide the column’s profile (same as Part 1) and the initial rules (from GPT-4) to the GPT-3.5 model.
Validation Logic: Instruct the model to examine each rule: remove any rule that doesn’t make sense for the data, add any obvious missing rule, or adjust thresholds if needed based on the data profile.
Output Format: Require the output in the same JSON structure (so that downstream processing is consistent).
Azure GPT-3.5 Client: Initialize an Azure OpenAI client for the GPT-3.5 model (which is faster and cheaper). For example, Azure’s GPT-3.5 model is often deployed as gpt-35-turbo or similar.
Assistant Agent: Create an AssistantAgent for the review stage with a system prompt that emphasizes a critical reviewing role.
Below is review_agent.py:
python
Copy
Edit
# review_agent.py
import os, json, asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

# Azure OpenAI GPT-3.5 (o3-mini) setup
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://<your-resource>.openai.azure.com/")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY", "<YOUR-API-KEY>")
DEPLOYMENT_NAME_GPT35 = os.getenv("AZURE_OPENAI_GPT35_DEPLOYMENT", "<GPT35-DEPLOYMENT-NAME>")
API_VERSION = "2023-05-15"

client_gpt35 = AzureOpenAIChatCompletionClient(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=DEPLOYMENT_NAME_GPT35,
    model="gpt-35-turbo",  # Azure OpenAI GPT-3.5 Turbo
    api_version=API_VERSION,
    azure_api_key=AZURE_OPENAI_KEY
)

# AssistantAgent for reviewing/refining rules
review_agent = AssistantAgent(
    name="review_agent",
    model_client=client_gpt35,
    system_message=(
        "You are a data quality rule reviewer. You will be given a column's profile and a draft list of rules. "
        "Your job is to verify each suggested rule for correctness and relevance to the data. "
        "If a rule is inappropriate, remove it. If a rule needs refinement or correction, adjust it. "
        "Ensure the final rules are appropriate for the column data. Respond **only** with the final list of rules in JSON format."
    )
    # temperature can be low to avoid random changes
)

def review_rules_for_column(column_name: str, profile_info: str, initial_rules: list):
    """
    Validate/refine suggested rules for a column using GPT-3.5.
    - profile_info: the text describing column stats (same format used for recommendation prompt).
    - initial_rules: list of rule dicts from the recommendation agent.
    """
    # Convert initial rules list to JSON string for prompting
    initial_rules_json = json.dumps(initial_rules, indent=2)
    # Build prompt with profile and initial rules
    prompt = (
        f"{profile_info}\n"
        "The initial suggested rules for this column are as follows:\n"
        f"{initial_rules_json}\n\n"
        "Review these rules. Remove any irrelevant or incorrect rules. Suggest additions if something important is missing. "
        "Output the final set of data quality rules in JSON format (same structure)."
    )

    result = asyncio.run(review_agent.run(task=prompt))
    # Get assistant response (JSON text of refined rules)
    assistant_messages = [m for m in result.messages if m.source == "assistant"]
    if not assistant_messages:
        raise RuntimeError("No assistant response received for rule review.")
    output_content = assistant_messages[-1].content
    try:
        refined_rules = json.loads(output_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from GPT-3.5 output: {e}\nOutput was: {output_content}")
    return refined_rules
Explanation: The review agent is configured similarly to the recommend agent, but with a GPT-3.5 model client (here assumed to be gpt-35-turbo in Azure). The system prompt clearly defines the agent’s role as a reviewer or QA checker of the rules. The review_rules_for_column function takes the column profile (which we can reuse from Part 1) and the list of initial rules. We embed the initial rules JSON directly in the prompt for transparency. The agent’s output is expected to be a JSON of the final rules, which we parse and return. We again enforce that only JSON should be returned (no extra commentary), to make parsing reliable. Using a smaller model for this step is cost-effective while still ensuring quality control. As noted in research, combining a strong model with a secondary check by a cheaper model can maintain accuracy with lower costs. The refinement step helps catch any mistakes GPT-4 might have made or confirm that the rules align with the data characteristics.


PART 3
#CThe Execution Agent is responsible for applying each data quality rule to the actual DataFrame and determining whether the data passes or fails the rule. This component does not require a language model; it’s essentially a set of programmed checks. We implement it as a simple Python function (or class) that iterates through the rules for each column and tests them against the data.

# execution_agent.py
import os
import json
import asyncio
import pandas as pd

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from rule_executor import execute_rules_on_dataframe  # your prod-ready executor from Part 3

# ─── Azure OpenAI / Model Configuration ─────────────────────────────
AZURE_OPENAI_ENDPOINT      = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY       = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME_O4MINI     = os.getenv("AZURE_O4MINI_DEPLOYMENT")  # e.g. "o4-mini-deploy"
API_VERSION                = "2023-05-15"

# Initialize the Azure OpenAI client for o4-mini
client_o4mini = AzureOpenAIChatCompletionClient(
    azure_endpoint = AZURE_OPENAI_ENDPOINT,
    azure_deployment= DEPLOYMENT_NAME_O4MINI,
    model           = "o4-mini",             # or your o4-mini alias
    api_version     = API_VERSION,
    azure_api_key   = AZURE_OPENAI_API_KEY
)

# Create the ExecutionAgent
execution_agent = AssistantAgent(
    name         = "execute_agent",
    model_client = client_o4mini,
    system_message=(
        "You are the Data Quality Execution Agent. "
        "You will receive a raw JSON report of rule checks (pass/fail/details). "
        "Format and validate the JSON, ensuring it is syntactically correct and "
        "suitable for downstream processing. "
        "Return only the JSON—no commentary."
    )
)

async def run_execution_agent(df: pd.DataFrame, validated_rules: dict) -> dict:
    """
    1. Execute each validated rule on the DataFrame.
    2. Produce a raw report via `execute_rules_on_dataframe`.
    3. Ask the o4-mini agent to reformat/validate the JSON.
    4. Return the final JSON report as a Python dict.
    """
    # Step 1 & 2: run the Python rule executor
    raw_report = execute_rules_on_dataframe(df, validated_rules)
    
    # Step 3: serialize to JSON (compact)
    raw_json = json.dumps(raw_report)
    
    # Step 4: prompt the agent to validate/format
    result = await execution_agent.run(task=f"Raw execution report:\n{raw_json}")
    assistant_msgs = [m for m in result.messages if m.source=="assistant"]
    if not assistant_msgs:
        raise RuntimeError("No response from execution agent.")
    formatted = assistant_msgs[-1].content.strip()
    
    # Step 5: parse JSON back into Python
    try:
        return json.loads(formatted)
    except json.JSONDecodeError as e:
        # Fallback to raw report if parsing fails
        return raw_report


Part 4: Analysis Agent (GPT-4)
The Analysis Agent comes into play for any rules that failed in Part 3. Its role is to analyze the nature of each failure and provide an explanation or insight. This can help data engineers understand why a rule failed (for example, why there were null values or out-of-range values – perhaps due to data entry errors, missing data, outliers, etc.). We’ll use a model with strong reasoning ability (GPT-4, referred to as "o4-mini" in a cost-optimized context) to generate these explanations. Key steps for the Analysis Agent:
Input Context: Provide the agent with the column name, the rule description, and details about the failure (from the Execution Agent’s report).
Analytical Prompt: Ask the agent to explain why the rule might have failed, and possibly suggest what the pattern or cause of the bad data is. We can also ask for suggestions on how to address the issue.
Output: A textual explanation or hypothesis about the data quality issue.
We use another AssistantAgent with an Azure OpenAI GPT-4 client for this purpose. The model can be the same GPT-4 deployment as used in Part 1, or a separate one specialized for analysis if available. The system prompt should encourage the agent to be diagnostic and insightful. Below is analysis_agent.py:
python
Copy
Edit
# analysis_agent.py
import os, asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

# Azure OpenAI GPT-4 setup (reuse configuration from Part 1 or use a separate deployment)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://<your-resource>.openai.azure.com/")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY", "<YOUR-API-KEY>")
DEPLOYMENT_NAME_GPT4 = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "<GPT4-DEPLOYMENT-NAME>")
API_VERSION = "2023-05-15"

client_gpt4_analysis = AzureOpenAIChatCompletionClient(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=DEPLOYMENT_NAME_GPT4,
    model="gpt-4",  # using GPT-4 for in-depth analysis
    api_version=API_VERSION,
    azure_api_key=AZURE_OPENAI_KEY
)

analysis_agent = AssistantAgent(
    name="analysis_agent",
    model_client=client_gpt4_analysis,
    system_message=(
        "You are a data analyst AI tasked with diagnosing data quality issues. "
        "When a data quality rule fails, explain the possible reasons and context in a concise, insightful manner. "
        "Consider the nature of the rule and the provided failure details to hypothesize why the data might violate the rule. "
        "If relevant, suggest ways to address or prevent this issue."
    )
)

async def analyze_failed_rule(column_name: str, rule_text: str, failure_details: str) -> str:
    """
    Use GPT-4 to analyze a failed data quality rule and return an explanation.
    """
    prompt = (
        f"Column: {column_name}\n"
        f"Failed Rule: {rule_text}\n"
        f"Failure Details: {failure_details}\n\n"
        "Explain why this rule might have failed for this column. What could be the reasons behind the invalid data? "
        "Provide a brief analysis and any useful insights. If appropriate, suggest how to fix or prevent this issue."
    )
    result = await analysis_agent.run(task=prompt)
    # Extract assistant's response (text explanation)
    assistant_messages = [m for m in result.messages if m.source == "assistant"]
    explanation = assistant_messages[-1].content if assistant_messages else ""
    return explanation.strip()





# main.py
import pandas as pd
from recommend_agent import suggest_rules_for_column
from review_agent import review_rules_for_column
from execution_agent import execute_rules_on_dataframe
from analysis_agent import analyze_failed_rule

# Step 1: Load your data
df = pd.read_csv("your_dataset.csv")  # or any data source

# Step 2 & 3: Generate and review rules for each column
all_final_rules = {}  # {column: [rules]}
for col in df.columns:
    # Get initial rules from GPT-4 recommendation agent
    initial_rules = suggest_rules_for_column(col, df[col])
    # Prepare profile info string (to reuse in review; 
    # ideally, we'd modify suggest_rules_for_column to also return the profile string it constructed)
    dtype = str(df[col].dtype)
    total = len(df[col]); nulls = int(df[col].isna().sum()); distinct = int(df[col].nunique(dropna=True))
    sample_vals = df[col].dropna().unique()[:5]; sample_list = ', '.join(map(str, sample_vals))
    profile_text = (
        f"Column: {col}\nType: {dtype}\nTotal Rows: {total}, Nulls: {nulls}, Distinct: {distinct}\n"
        f"Sample Values: {sample_list}\n"
    )
    # Refine rules with GPT-3.5 review agent
    final_rules = review_rules_for_column(col, profile_text, initial_rules)
    all_final_rules[col] = final_rules

# Step 4: Execute all rules on the DataFrame
report = execute_rules_on_dataframe(df, all_final_rules)

# Step 5: Analyze failed rules using GPT-4 analysis agent
for col, results in report.items():
    for rule_result in results:
        if not rule_result["passed"]:
            rule_name = rule_result["rule"]
            details = rule_result.get("details", "")
            # Get analysis (note: analyze_failed_rule is async, so we use asyncio.run for simplicity in this script context)
            explanation = asyncio.run(analyze_failed_rule(col, rule_name, details))
            print(f"Analysis of failed rule '{rule_name}' for column '{col}':\n", explanation, "\n")

