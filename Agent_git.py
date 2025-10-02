Here’s a complete Python solution that uses LangGraph to orchestrate a multi-agent pipeline on Databricks for analysing public GitHub repositories and generating architecture diagrams.  The project is organized into clear modules with thorough documentation and can run as a CLI or be imported into notebooks.  A zip of the entire project is included at the end.

Project Overview

The pipeline uses four agents orchestrated by a StateGraph:
	1.	Repository Parsing Agent – clones the GitHub repo and summarises its contents (files, language distribution, LOC).
	2.	Architecture Extraction Agent – calls an Azure OpenAI chat model to infer components and relationships, returning a JSON spec.  When the API isn’t configured it falls back to a heuristic based on top‑level directories.
	3.	D2 Conversion Agent – converts the architecture spec into D2 diagram syntax.  Shapes are chosen based on component type (e.g. databases become cylinders).
	4.	Diagram Rendering Agent – uses the d2-python-wrapper to render D2 to PNG and SVG.  D2 supports exporting to PNG and SVG via a headless browser ￼, and the wrapper simplifies usage ￼.

The pipeline’s state is encapsulated in an ArchitectureState dataclass, and the LangGraph orchestrator wires the agents together sequentially ￼.  Azure OpenAI setup follows the guidelines from LangChain ￼, with environment variables controlling the deployment and API version.

Directory Structure

architecture_diagram/
├── agents/                  # Each agent lives in its own module
│   ├── repo_parser.py       # Clones and summarises the repository
│   ├── arch_extraction.py   # LLM-based architecture inference
│   ├── d2_conversion.py     # Converts architecture spec to D2
│   └── d2_render.py         # Renders D2 diagrams to PNG/SVG
├── utils/
│   ├── github_utils.py      # Repository cloning and summary helpers
│   └── d2_render_utils.py   # Wrapper around d2-python-wrapper
├── state/
│   └── state_schema.py      # Dataclass defining the shared state
├── orchestrator.py          # Builds and runs the LangGraph pipeline
├── main.py                  # CLI entrypoint for Databricks or local use
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

Key Modules

utils/github_utils.py

This module contains helper functions to clone a public GitHub repository and summarise its contents.  It uses GitPython when available and falls back to the git CLI.  A heuristic language detector counts file types, and a summary string is generated for the LLM.

import os, subprocess, tempfile
from git import Repo
from collections import Counter
from typing import Dict
...
def clone_repo(repo_url: str, dest_dir: str | None = None) -> str:
    # Clones the repository using GitPython or git CLI
    ...

def summarize_repository(repo_path: str) -> Dict[str, object]:
    # Walks the repository to collect file counts, language distribution,
    # total lines of code, and a list of files.
    ...

def generate_repo_summary_text(summary: Dict[str, object], max_files: int = 50) -> str:
    # Constructs a concise textual summary for the LLM.
    ...

agents/repo_parser.py

Clones the repository and populates the state with a summary and a textual description.

from ..utils import github_utils
from ..state.state_schema import ArchitectureState

def repository_parsing_agent(state: ArchitectureState) -> ArchitectureState:
    if state.repo_path is None:
        repo_path = github_utils.clone_repo(state.repo_url)
        state.repo_path = repo_path
    summary = github_utils.summarize_repository(repo_path)
    state.repo_summary = summary
    state.metadata['repo_summary_text'] = github_utils.generate_repo_summary_text(summary)
    return state

agents/arch_extraction.py

Uses AzureChatOpenAI to infer architecture components and relationships from the summary.  It falls back to a heuristic if the Azure configuration is missing or the API call fails.

from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
...
def architecture_extraction_agent(state: ArchitectureState) -> ArchitectureState:
    summary_text = state.metadata.get('repo_summary_text')
    # Read deployment info from env or metadata
    deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME') or state.metadata.get('azure_deployment')
    ...
    if not deployment_name:
        state.architecture_spec = _default_architecture_from_summary(state.repo_summary)
        return state
    llm = AzureChatOpenAI(deployment_name=deployment_name, api_version=api_version, temperature=temperature)
    system_prompt = (
        "You are a software architecture assistant. Given a summary of a repository, "
        "infer a high-level architecture consisting of components and relationships..."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Repository summary:\n{summary_text}\n\nReturn the architecture as JSON.")
    ]
    try:
        response = llm.invoke(messages)
        json_str = extract_json_from_response(response.content)
        spec = json.loads(json_str)
    except Exception:
        spec = _default_architecture_from_summary(state.repo_summary)
    state.architecture_spec = spec
    return state

agents/d2_conversion.py

Converts the architecture JSON into D2 language.  Shapes are chosen based on the component type.

SHAPE_MAPPING = {'module': 'rectangle', 'database': 'cylinder', ...}

def d2_conversion_agent(state: ArchitectureState) -> ArchitectureState:
    spec = state.architecture_spec
    lines = ["// Auto-generated D2 diagram"]
    for comp in spec.get('components', []):
        comp_id = comp['id']
        name = comp['name']
        shape = SHAPE_MAPPING.get(comp['type'].lower(), 'rectangle')
        lines.append(f"{comp_id}: \"{name}\" {{\n  shape: {shape}\n}}")
    lines.append("")
    for rel in spec.get('relationships', []):
        src, tgt = rel['source'], rel['target']
        label = rel.get('description') or rel.get('type') or ''
        label_part = f" : \"{label}\"" if label else ''
        lines.append(f"{src} -> {tgt}{label_part}")
    state.d2_code = "\n".join(lines)
    return state

agents/d2_render.py

Renders the D2 code to SVG and PNG using d2-python-wrapper, storing file paths in state.output_files.

from ..utils.d2_render_utils import render_d2_to_files
import tempfile

def diagram_rendering_agent(state: ArchitectureState) -> ArchitectureState:
    if not state.d2_code:
        raise ValueError("d2_code must be present.")
    output_dir = state.metadata.get('output_dir') or tempfile.mkdtemp(prefix="diagram_output_")
    state.metadata['output_dir'] = output_dir
    theme = state.metadata.get('d2_theme', 'neutral')
    files = render_d2_to_files(state.d2_code, output_dir, filename_prefix='architecture_diagram', theme=theme)
    state.output_files.update(files)
    return state

orchestrator.py

Builds the LangGraph, connects the agents, and provides a helper to run the pipeline.

from langgraph import StateGraph, END
from .state.state_schema import ArchitectureState
from .agents.repo_parser import repository_parsing_agent
from .agents.arch_extraction import architecture_extraction_agent
from .agents.d2_conversion import d2_conversion_agent
from .agents.d2_render import diagram_rendering_agent

def build_architecture_graph() -> StateGraph:
    graph = StateGraph(ArchitectureState)
    graph.add_node("repo_parser", repository_parsing_agent)
    graph.add_node("arch_extraction", architecture_extraction_agent)
    graph.add_node("d2_conversion", d2_conversion_agent)
    graph.add_node("diagram_rendering", diagram_rendering_agent)
    graph.set_entrypoint("repo_parser")
    graph.add_edge("repo_parser", "arch_extraction")
    graph.add_edge("arch_extraction", "d2_conversion")
    graph.add_edge("d2_conversion", "diagram_rendering")
    graph.add_edge("diagram_rendering", END)
    return graph.compile()

def run_pipeline(repo_url: str, **metadata) -> ArchitectureState:
    graph = build_architecture_graph()
    state = ArchitectureState(repo_url=repo_url)
    state.metadata.update(metadata)
    final_state = graph.invoke(state)
    return final_state

main.py

Provides a CLI wrapper suitable for Databricks or local execution.

import argparse, json, sys
from .orchestrator import run_pipeline

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--azure-deployment')
    parser.add_argument('--api-version', default='2023-07-01-preview')
    parser.add_argument('--d2-theme', default='neutral')
    parser.add_argument('--output-json')
    args = parser.parse_args(argv)

    metadata = {}
    if args.azure_deployment:
        metadata['azure_deployment'] = args.azure_deployment
    if args.api_version:
        metadata['azure_api_version'] = args.api_version
    if args.d2_theme:
        metadata['d2_theme'] = args.d2_theme

    state = run_pipeline(args.repo, **metadata)
    for fmt, path in state.output_files.items():
        print(f"Generated {fmt.upper()} diagram: {path}")
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump({...}, f, indent=2)

requirements.txt

langgraph>=0.0.4
langchain-openai>=0.0.5
gitpython>=3.1.31
d2-python-wrapper>=0.3.0
pyyaml>=6.0

Download the full project

You can download the complete project directory as a zip file:

￼

This solution is modular, extensible and suitable for running on Databricks or any Python environment.  It leverages LangGraph’s state graph to orchestrate multiple agents, uses Azure OpenAI to infer architecture, and produces diagrams in both SVG and PNG formats via D2 ￼ ￼.
