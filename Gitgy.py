Architecture Diagram Generation with LangGraph and Azure OpenAI

System Architecture Overview

The solution is built as a multi-agent system orchestrated by LangGraph, designed to analyze a code repository and produce architecture diagrams in multiple formats. At a high level, it consists of a coordinator (graph) and several specialized agents (nodes) that carry out distinct tasks:
	•	Repository Parsing Agent – Ingests the GitHub repository (files and structure).
	•	Architecture Extraction Agent – Uses an LLM to infer the software architecture (components and their relationships) from the code.
	•	Summarization Agent – Refines or condenses the architectural information for clarity and brevity.
	•	Diagram Generation Agent – Converts the architecture description into diagram code (Mermaid, PlantUML, or Graphviz DOT).

All agents share a common state (defined via a TypedDict or dataclass) that carries data through the workflow (e.g. repository content, extracted components, diagram format, etc.). A LangGraph StateGraph orchestrates the flow: connecting agents, handling branching for diagram format, and managing state updates. This structured graph workflow allows conditional logic and even looping if needed (for example, to handle errors or iterative refinement). LangGraph’s design is well-suited here because it supports long-running, complex workflows with persistent state and even checkpointing for reliability ￼. The multi-agent approach ensures each task is handled in isolation but with a shared context, enabling complex reasoning and tool use in stages ￼.

Workflow: The process begins at the Repository Parser node and flows through extraction and summarization to diagram generation. A top-level Supervisor (the LangGraph controller) monitors the process. If a step fails or requires more information, the system can route to error-handling or retry nodes (for example, if the diagram code generation fails validation, an agent could adjust the prompt or clarify the architecture description and try again). This makes the solution robust and “production-ready.” Below is a summary of the workflow:
	1.	Parse Repo – Gather code and metadata from the repository into state.
	2.	Extract Architecture – Identify key modules/classes and their interactions via LLM.
	3.	Summarize – (Optional) Clean up and shorten the architecture description.
	4.	Generate Diagram Code – Produce diagram-as-code (Mermaid, PlantUML, or DOT) via LLM.
	5.	Validate & Output – Optionally validate the diagram syntax or render it to an image.

Each agent’s role is described in detail in the next section. The use of Azure OpenAI (GPT-4/GPT-3.5) provides the reasoning and code generation capabilities for the agents. LangGraph manages the sequence and can even visualize the agent flow for debugging (it can output a Mermaid flowchart of the agent graph for verification ￼). The overall architecture ensures modularity, so each agent can be developed and tested independently, and the system can handle polyglot codebases by leveraging the LLM’s understanding of multiple languages ￼.

Agent Roles and Workflow Details

1. Repository Parsing Agent (Code Ingestion)

Role: Ingest the GitHub repository content and prepare it for analysis.
Functionality: This agent could use a Git access token or public clone to fetch the repository. On Databricks, one can invoke a shell command or use the GitHub API to retrieve the repo. The agent then scans the file structure, reading in source code files (with a focus on relevant files: e.g. skipping binary or vendored files). It may create a structured representation of the codebase (e.g. a list of modules/packages, each with key classes/functions). To keep within token limits, the agent can select representative files or samples of code – as seen in the Swark tool, which automatically adjusts the number of files based on LLM context limits ￼. The parsed content (file names, code snippets, possibly a simplified AST or docstrings) is added to the shared state (e.g. state["repo_content"]).
Output: A structured summary of the code (or the raw text of important parts) stored in state for the next agent. It also records the repository language mix (e.g. via file extensions) to inform the LLM.

2. Architecture Extraction Agent

Role: Derive a high-level architecture model from the code.
Functionality: This agent prompts the LLM (Azure OpenAI GPT-4, for example) to analyze the repository content and identify the key architectural elements. It might ask the model to list the main components, modules, classes, and how they interact (e.g. which module calls which, layers like controllers/service/DB, etc.). The agent can include in the prompt the relevant code snippets or file summaries from the parser agent, along with an instruction like “Explain the software architecture represented by this code: identify major components and their relationships.” The logic of architecture identification is largely handled by the LLM’s reasoning on the code – an approach that allows natural support for any programming language present ￼ (the LLM’s knowledge covers multiple languages, frameworks, and common architectural patterns, so polyglot repositories are handled naturally). As the creator of Swark notes, encapsulating logic in the LLM means we avoid writing language-specific parsers and still get broad language support ￼.
Output: The agent produces an intermediate architecture description (stored in state, e.g. state["architecture_details"]). This could be a list of components and their connections or a narrative summary of the design. For example, it might output something like: “WebApp module calls into ServiceLayer (class OrderService), which uses Repository classes for data access. There is also a background Worker service pulling from Queue… etc.” This intermediate representation will be used to create diagrams.

3. Summarization Agent

Role: Refine and condense the architecture description.
Functionality: Depending on the size/complexity of the extracted architecture info, a summarization step may be applied. This agent uses the LLM to paraphrase or simplify the architecture details into a concise form suitable for diagramming. For instance, if the extraction agent returned a very detailed breakdown, the summarizer might produce a high-level summary focusing on major components only. It can also ensure consistent naming (e.g. making sure component names are succinct) and possibly format the info as needed (such as a bullet list of components and relationships). This makes the subsequent diagram generation more effective by reducing clutter. (This agent could be skipped if the extraction output is already succinct.)
Output: A polished architecture summary (e.g. state["architecture_summary"]). This might be text like: “Components: WebApp -> ServiceLayer -> Database. WebApp also calls external API. BatchProcessor -> Queue -> ServiceLayer.” – something that can be directly mapped to a diagram.

4. Diagram Generation Agent

Role: Generate diagram code (Mermaid, PlantUML, or Graphviz DOT) from the architecture summary.
Functionality: This agent formulates a prompt for the LLM to produce a diagram in the requested format. The desired format (e.g. "mermaid" or "plantuml" or "graphviz") would be indicated in the state (perhaps set initially or by user input). Conditional branching in the LangGraph can route to different sub-agents or prompt templates for each format, since the syntax differs – for example, a conditional edge in the graph can check state["diagram_format"] and go to a generate_mermaid node vs. generate_plantuml node, etc. In each case, the agent provides instructions to the LLM such as: “Convert the following architecture description into a <> diagram. Use proper syntax for <>.” The architecture summary from the previous step is included in the prompt.

- **Mermaid output:** The agent might request a Mermaid **flowchart** or **sequence diagram** or a **class diagram** (Mermaid supports flowcharts and has classDiagram syntax for class relationships). Mermaid is a good default since it’s widely supported (even GitHub can render it natively) [oai_citation:7‡medium.com](https://medium.com/@ozanani/introducing-swark-automatic-architecture-diagrams-from-code-cb5c8af7a7a5#:~:text=,refine%20the%20diagrams%20as%20needed). The LLM could produce a Mermaid code block showing nodes and links.  
- **PlantUML output:** The agent can ask for a UML diagram (e.g. a component or class diagram) in PlantUML syntax. PlantUML is more expressive for class relationships or deployment diagrams, if needed.  
- **Graphviz DOT output:** The agent can also request a Graphviz DOT graph description. This might be useful for generic network graphs or where precise control of layout is needed.  

Output: The result is a text block containing the diagram code in the chosen format (e.g. a Markdown snippet with a fenced code block for Mermaid/PlantUML/DOT). These three formats are popular “diagrams as code” standards (indeed, backend tools like Kroki support Mermaid, PlantUML, Graphviz and many more ￼). The system ensures we support all three to align with Nexus’s capabilities (Mermaid for quick Markdown rendering, PlantUML for rich UML, Graphviz for custom graphs).

After generation, a validation step can be included: for example, attempting to render the diagram using a library or checking for obvious syntax errors. If the diagram code is invalid or incomplete, the agent can loop back – e.g. an error state triggers a prompt to the LLM like “The diagram syntax had an error X, please fix it.” (In LangGraph, this can be modeled by a conditional edge that goes back to the diagram agent or to a corrective agent if validation fails, similar to how the Streamlit diagram generator app looped on errors ￼ ￼.) If all is well, the final diagram code is passed to output.

5. Orchestration & State Management

All the above agents are tied together by a LangGraph StateGraph. The StateGraph defines the nodes (agents) and edges (transitions). We set the entry point at the Repository Parser and end at the Diagram Generator (or its validation). Between agents, the state (a Python dict or dataclass) carries fields like repo_content, arch_details, arch_summary, diagram_format, etc. The orchestrator uses linear flow with a conditional branch for diagram format. Pseudocode for constructing the graph might look like:

graph = StateGraph(StateSchema)  
graph.add_node("parse_repo", parse_repo_agent)  
graph.add_node("extract_arch", extract_arch_agent)  
graph.add_node("summarize_arch", summarize_agent)  
graph.add_node("make_mermaid", gen_mermaid_agent)  
graph.add_node("make_plantuml", gen_plantuml_agent)  
graph.add_node("make_graphviz", gen_graphviz_agent)  
graph.set_entry_point("parse_repo")  
graph.add_edge("parse_repo", "extract_arch")  
graph.add_edge("extract_arch", "summarize_arch")  
# Conditional branch based on desired format:
graph.add_conditional_edges("summarize_arch", lambda st: st["diagram_format"], {  
    "mermaid": "make_mermaid",  
    "plantuml": "make_plantuml",  
    "graphviz": "make_graphviz"  
})  
graph.add_edge("make_mermaid", END); graph.add_edge("make_plantuml", END); graph.add_edge("make_graphviz", END)

The above is conceptual – in practice you could also use one generator agent that checks the format internally. The LangGraph approach makes the flow explicit and durable (the framework supports long workflows with checkpointing, so even if processing a large repo takes time, the state can persist and resume safely ￼). Each agent function is kept modular (single-responsibility), which improves maintainability. Logging can be added at each node (e.g. logging the state or decisions) for traceability in production.

Interactions: The agents interact only via the shared state and well-defined transitions. For example, the parser agent populates state["repo_content"], which the extraction agent reads. The extraction agent outputs state["architecture_details"], which the summarizer refines into state["architecture_summary"]. Finally the diagram generator reads the summary and state["diagram_format"] to produce state["diagram_code"]. The supervisor (LangGraph) ensures this happens in order. If an unexpected situation occurs (like repository content is empty or the LLM fails to parse something), the graph can route to an error handler agent (not detailed above, but you can include a node to capture exceptions and, say, return a graceful error message or attempt a fallback strategy).

Deployment Instructions (Azure OpenAI & Databricks)

Azure OpenAI Setup: First, provision an Azure OpenAI resource and deploy the desired model (e.g., GPT-4 or GPT-3.5 Turbo) with a custom deployment name. Note the endpoint URL and API key. In the code, you’ll use the Azure OpenAI API through an SDK like openai or langchain_openai. For example, using LangChain’s AzureChatOpenAI client you can initialize the LLM like this:

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint="https://<your-resource-name>.openai.azure.com/",
    api_key="<your-api-key>",
    api_version="2024-05-15",        # use the API version for your deployment
    deployment_name="gpt-4",        # the name you gave your model deployment
    temperature=0.0,
    max_tokens=1024
)

This configuration uses Azure OpenAI as the reasoning engine ￼. You can also set these via environment variables (OPENAI_API_TYPE=azure, OPENAI_API_BASE, OPENAI_API_KEY, etc.) and use the regular OpenAI classes. Ensure the Databricks environment is configured with these keys securely (e.g. using Databricks Secret Scope or environment variables on the cluster).

Dependencies: On Databricks, install the required libraries. This typically includes: langchain and langgraph (for orchestration), openai or langchain_openai (for Azure OpenAI integration), and possibly utility libs (gitpython or requests for fetching repos, etc.). For example, in a notebook cell:

%pip install langchain langgraph langchain-openai python-dotenv gitpython

(We include gitpython to clone repos via Python. Alternatively, one can use %sh git clone ... in a notebook to pull the repo.)

Code Organization: For maintainability, structure the solution as importable modules or notebooks corresponding to each agent, plus a main orchestrator. In Databricks, you could create a Repo with this project structure and use a main notebook to trigger it. For example:

repo-arch-diagram/
├── agents/
│   ├── parser.py             # RepoParsingAgent implementation
│   ├── extractor.py          # ArchitectureExtractionAgent implementation
│   ├── summarizer.py         # SummarizationAgent implementation
│   └── diagram_generator.py  # DiagramGenerationAgent implementation (with format handlers)
├── orchestrator.py           # constructs the LangGraph and provides a run() function
├── utils/
│   ├── git_loader.py         # utility to fetch/clone repo
│   └── prompt_templates.py   # (optional) predefined prompt strings for LLM
├── notebooks/
│   └── DemoNotebook          # (optional) a Databricks notebook showing usage
└── README.md

This modular layout is similar to other LangGraph projects (for example, a Streamlit diagram app organizes agents under an agent/ folder with helper utils) ￼. You can adapt it to either pure Python files (e.g. in Databricks Repo or %run magic to include them in notebooks) or keep each agent in a separate notebook if preferred. The key is that each agent module defines its function to process the state, and orchestrator.py assembles the graph.

Deployment Steps on Databricks:
	1.	Import Code: Add the project code to Databricks. This can be done by attaching a Repo (if stored in Git), or uploading the files. If using notebooks, copy each agent’s code into a notebook. If using files, ensure the cluster’s working directory has them (Databricks Repo feature allows a GitHub repo sync).
	2.	Cluster Setup: Attach the cluster and install dependencies (as mentioned above). Ensure internet access is available if you need to clone external repos at runtime, or pre-load the repository content by other means if internet is restricted (you could also attach the target repo as a Databricks Repo and read files directly from the workspace).
	3.	Configure Azure OpenAI: Set the Azure OpenAI credentials. In a notebook, one can do:

import os
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = "<YOUR_KEY>"
os.environ["OPENAI_API_BASE"] = "https://<YOUR_RESOURCE>.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2024-05-15"

Alternatively, store these in a Databricks Secret and fetch them in the code (to avoid hardcoding).

	4.	Run the Pipeline: Call the orchestrator with the repository reference and desired format. For example, in a notebook:

from orchestrator import run_pipeline
diagram_code = run_pipeline(repo_url="https://github.com/your-org/your-repo.git", format="mermaid")
print(diagram_code)

The run_pipeline function would internally clone/read the repo, initialize the LangGraph state (e.g. {"repo_url": ..., "diagram_format": "mermaid"}), and invoke the compiled graph. The output diagram_code could be a Markdown string containing the diagram. On Databricks, you might simply display it, or save it to a file in DBFS (e.g. diagram.md) for later use.

	5.	Rendering (Optional): If you want to produce an image of the diagram within Databricks, you can integrate a rendering step. For Mermaid, one could use the mermaid CLI or a library; for PlantUML, the plantuml package or an online server; for Graphviz, the graphviz python library. In production, a safer approach is to use a service like Kroki or Graphviz to render diagram code to SVG/PNG ￼. This could be an optional final agent: e.g. if running in an environment with access to a Kroki server or if Graphviz is installed on the cluster, have an agent take state["diagram_code"] and produce an image file. In a Databricks notebook, you can also use the displayHTML trick to render Mermaid diagrams by embedding HTML+JS, but an easier path is to save the diagram code (or image) and view it offline or in a docs site.

Testing & Verification: It’s advisable to test each agent individually (unit tests) by feeding it sample input states. For instance, test the DiagramGenerationAgent with a known small architecture summary to see if it produces correct Mermaid syntax. Also verify on a small repo (polyglot if possible, e.g. a repo with both Python and JS) that the pipeline completes and yields a sensible diagram. Because we are using LLMs, the output might vary – adding few-shot examples in prompts (in prompt_templates.py) can guide the style of diagrams (for example, provide an example of a Mermaid diagram for a simple 2-component system as part of the prompt).

Code Sample Highlights

To illustrate, below are brief examples of how parts of the implementation might look.
	•	State Definition: Using TypedDict for state schema (for clarity in code):

class ArchDiagramState(TypedDict):
    repo_url: str
    repo_content: dict    # e.g., {"path/to/file.py": "file content", ...}
    architecture_details: str
    architecture_summary: str
    diagram_format: str   # "mermaid" | "plantuml" | "graphviz"
    diagram_code: str

This state flows through the graph nodes.

	•	Repository Parsing (pseudo-code):

def parse_repo_agent(state: ArchDiagramState) -> ArchDiagramState:
    repo = clone_repository(state["repo_url"])        # utility to clone or fetch files
    files = select_relevant_files(repo, max_files=50) # limit files to avoid token overflow
    content_dict = {f.path: f.read() for f in files}
    state["repo_content"] = content_dict
    # Optionally, record primary languages:
    state["languages"] = detect_languages(files)
    return state

Here select_relevant_files might pick e.g. top-level modules and skip dependencies. This prevents overwhelming the LLM. (Swark’s approach of matching file count to token limit is one strategy ￼.)

	•	Architecture Extraction (pseudo-code):

def extract_arch_agent(state: ArchDiagramState) -> ArchDiagramState:
    code_summary = summarize_codebase(state["repo_content"])  # optional static summary
    prompt = (f"Analyze the following code and describe the high-level architecture:\n"
              f"{code_summary}\n"   # or embed select file contents
              "Provide the main components and how they interact.")
    response = llm.invoke(prompt)
    state["architecture_details"] = response.content
    return state

In practice, you might not summarize all code due to token limits, but rather include key parts (like content of app.py or major classes). The agent relies on GPT to interpret the code architecture. (In some cases, integrating retrieval of documentation (like the Qdrant vector DB in the earlier example ￼) can help if the code uses frameworks; however, for simplicity we assume the LLM itself can handle it.)

	•	Summarization:

def summarize_arch_agent(state: ArchDiagramState) -> ArchDiagramState:
    details = state.get("architecture_details", "")
    prompt = f"Summarize the following architecture description in a concise form:\n{details}"
    response = llm.invoke(prompt)
    state["architecture_summary"] = response.content
    return state

This yields a shorter description focusing on core components.

	•	Diagram Generation (Mermaid example):

def gen_mermaid_agent(state: ArchDiagramState) -> ArchDiagramState:
    summary = state.get("architecture_summary", "")
    prompt = ("Convert the following description into a Mermaid diagram showing the components "
              "and their relationships:\n"
              f"{summary}\n"
              "Use Mermaid flowchart syntax with nodes for each component and arrows for interactions.")
    response = llm.invoke(prompt)
    state["diagram_code"] = response.content  # e.g., a Mermaid code block
    return state

Similar functions (or a parameterized function) would handle PlantUML and Graphviz, with appropriate prompt tweaks (e.g. asking for UML class diagram syntax or DOT format). The LLM is instructed to output only code for the diagram in the correct syntax.
After this, a simple validation can be done. For instance, if using Mermaid, one could parse the output to ensure it starts with flowchart or valid mermaid syntax. If validation fails, you could modify the prompt or add an example diagram in the prompt for context and retry.

	•	Orchestrator Invocation: Finally, the main entry point could be:

def run_pipeline(repo_url: str, format: str = "mermaid") -> str:
    initial_state = {"repo_url": repo_url, "diagram_format": format.lower().strip()}
    final_state = compiled_graph.invoke(initial_state)
    return final_state.get("diagram_code", "")

This allows external callers (notebooks, APIs) to easily generate a diagram by calling run_pipeline.

Integration and Extensions

VS Code Extension Integration: Because the solution is packaged in modules, it’s feasible to integrate it with an editor or other interfaces. For example, one could create a VS Code command that sends the current workspace’s repository path and desired format to this pipeline (perhaps via an API endpoint or a CLI). In fact, Swark – a VS Code extension – demonstrates this concept by hooking into GitHub Copilot to generate Mermaid diagrams from the open folder ￼ ￼. Our system could replace Copilot with Azure OpenAI and use our LangGraph pipeline under the hood. The extension would just need to trigger the process and then display the resulting diagram (e.g., open the Markdown with the Mermaid/PlantUML code, which VS Code can render with appropriate plugins ￼). This means developers could get on-demand architecture diagrams in their IDE, which is great for onboarding or reviewing code ￼.

Web UI Integration: Another optional add-on is to provide a simple web interface. A lightweight approach is a Streamlit app or a minimal Flask API. For instance, a Streamlit app could allow the user to input a GitHub repo URL and select a diagram format from a dropdown, then call run_pipeline and display the output diagram. This is similar to the reference project that offered a chat-based interface for diagram generation ￼ ￼ – in our case the UI can be simpler (just one-shot generation). The diagram code could be rendered on the page using existing libraries (there are Streamlit components for Mermaid, or one could call out to an API like Kroki to get an image).

Production Considerations: In a production deployment (e.g. internal developer portal), you might wrap this solution in a service with caching. For example, if the same repository is requested multiple times, cache the architecture summary or diagram result to speed up responses. Also, consider setting limits on repository size or providing guidance: extremely large monorepos might need additional logic to chunk and iteratively process parts of the code (this could be an extension of the parsing agent to process directories one by one and perhaps use the LLM to focus on one subsystem at a time). The system’s modular nature means you can plug in such logic without disturbing other components.

Finally, ensure compliance and security: since code is being sent to an LLM, use Azure OpenAI’s data privacy features or filters as needed. In an enterprise setting, you might restrict the analysis to certain repositories or run the service offline. Azure OpenAI allows setting contexts and using on-premises deployment if required, which can be configured in this pipeline.

Conclusion

In summary, this solution leverages LangGraph’s multi-agent orchestration and Azure OpenAI’s LLM to automatically produce architecture diagrams from a given codebase. It supports Mermaid, PlantUML, and Graphviz DOT formats – three popular diagram-as-code notations ￼ – thereby integrating with various documentation and visualization workflows. The design is modular and extensible: new agent nodes or tools (e.g. a vector database for code docs, or a testing agent) can be added with minimal impact on the overall flow. By splitting the task into specialized agents (parsing, analyzing, summarizing, rendering), we achieve a robust pipeline that can handle polyglot projects and provide useful high-level documentation for software systems. This not only automates a labor-intensive process but also keeps architecture diagrams in sync with the code – a valuable capability for engineering teams.

References & Related Work:
	•	LangGraph official docs and examples of multi-agent workflows ￼ ￼.
	•	Swark – VS Code extension for LLM-based diagramming (inspiration for polyglot support and editor integration) ￼ ￼.
	•	Entelligence AI – a tool that auto-generates architecture diagrams and flowcharts from code (demonstrates the efficacy of using agents on codebases) ￼.
	•	Azure OpenAI integration with LangChain/LangGraph (example by V. Mishra) ￼.
	•	Diagram rendering tools like Kroki (supports Mermaid, PlantUML, Graphviz) for converting code to images ￼.

