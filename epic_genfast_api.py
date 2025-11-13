1. Install dependencies
pip install fastapi uvicorn openai python-dotenv

2. .env (Azure OpenAI config)
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-azure-openai-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_GPT5=your-gpt5-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-15-preview

3. llm_client.py – Azure OpenAI client helper
# llm_client.py
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)

MODEL_GPT5 = os.environ["AZURE_OPENAI_DEPLOYMENT_GPT5"]


def chat_gpt5(messages, temperature: float = 0.4, max_tokens: int = 6000) -> str:
    """
    Call Azure OpenAI GPT-5 chat completion and return the assistant text.
    `messages` is a list of {role, content}.
    """
    resp = client.chat.completions.create(
        model=MODEL_GPT5,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""

4. main.py – FastAPI app with /api/generate-epic
# main.py
from typing import Optional, List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from llm_client import chat_gpt5

app = FastAPI(title="UBS GitLab Epic Generator API")


# ---------- Pydantic models ----------

class GenerateEpicRequest(BaseModel):
    projectName: str
    epicType: str
    customType: Optional[str] = None
    status: str
    groupProgram: Optional[str] = None
    targetDate: Optional[str] = None
    stakeholders: Optional[str] = None
    objective: str
    businessProblem: str
    successMetric: str
    inScope: str
    outOfScope: Optional[str] = None
    assumptions: Optional[str] = None
    architectureSecurity: Optional[str] = None
    backendAPI: Optional[str] = None
    uiUX: Optional[str] = None
    dataDBA: Optional[str] = None
    devOpsPlatform: Optional[str] = None
    productBA: Optional[str] = None
    keyFeatures: str
    nfrFocus: str
    dependencies: Optional[str] = None
    risks: Optional[str] = None


class Diagram(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class GenerateEpicResponse(BaseModel):
    markdown: str
    diagram: Diagram


# ---------- Utility functions ----------

def build_epic_prompt(body: GenerateEpicRequest) -> str:
    epic_type = body.epicType
    project_slug = body.projectName.lower().replace(" ", "-")

    return f"""You are a senior technical architect at a global bank (UBS) creating a comprehensive GitLab "Epic: Technical Design Blueprint" in Markdown.

You MUST:
- Use ONLY the information provided below.
- Where details are missing, write "TBD" rather than hallucinating specific tools, products, dates, URLs, or names.
- You may use generic roles (e.g., "Product Owner", "Tech Lead") if needed.
- Keep the tone professional and concise.

User Input:
- Project Name: {body.projectName}
- Type: {epic_type}
- Status: {body.status}
- Group/Program: {body.groupProgram or 'TBD'}
- Objective (free text): {body.objective}
- Business/Technical Problem: {body.businessProblem}
- Success Metric: {body.successMetric}
- In Scope: {body.inScope}
- Out of Scope: {body.outOfScope or 'TBD'}
- Assumptions: {body.assumptions or 'Standard development practices apply'}
- Stakeholders: {body.stakeholders or 'TBD'}
- Target Date: {body.targetDate or 'TBD'}
- Team Architecture & Security: {body.architectureSecurity or 'TBD'}
- Team Backend/API: {body.backendAPI or 'TBD'}
- Team UI/UX: {body.uiUX or 'TBD'}
- Team Data/DBA: {body.dataDBA or 'TBD'}
- Team DevOps/Platform: {body.devOpsPlatform or 'TBD'}
- Team Product/BA: {body.productBA or 'TBD'}
- Key Features (raw list): {body.keyFeatures}
- NFR Focus: {body.nfrFocus}
- Dependencies: {body.dependencies or 'None identified yet'}
- Risks: {body.risks or 'To be assessed during design phase'}

Using ONLY this information, generate a complete GitLab Epic in Markdown following this exact structure.
If the user did not provide something, explicitly write "TBD" instead of guessing.

# Epic: Technical Design Blueprint – {body.projectName}

Type: {epic_type}
Status: {body.status}
Group/Program: {body.groupProgram or 'TBD'}
Labels: Technical-Design, Blueprint, Architecture, {epic_type}
Target Date: {body.targetDate or 'TBD'}
Stakeholders: {body.stakeholders or 'TBD'}
Epic URL: <link after creation>

---

## Objective

Write the objective here, clearly summarizing the desired outcome and how it aligns to program/OKRs, using the provided input.

- Business/Technical Problem: {body.businessProblem}
- Success Metric: {body.successMetric}

---

## Background and Context

Generate a short paragraph about current state and context, based ONLY on the information above. Do not invent external systems or products.

- Links:
  - ADRs: <ADR index or specific ADR links / TBD>
  - Related epics / issues / MRs: <links / TBD>
  - Documents: <Confluence / Docs links / TBD>

---

## Scope

### In Scope
Convert these into bullet points (or "TBD" if empty):
{body.inScope}

### Out of Scope
{('Convert these into bullet points:\n' + body.outOfScope) if body.outOfScope else '- TBD'}

### Assumptions
{('Convert these into bullet points:\n' + body.assumptions) if body.assumptions else '- Standard development practices apply'}

---

## Architecture Overview

Write a high-level architecture narrative based on the key features, scope, and banking context. Use generic names where specifics are missing (e.g., "Core Banking Service", "Fraud Engine"), but do not reference tools or vendors that the user did not mention.

### Diagrams

- Option A (PlantUML URL preview)  
  - ![High-level architecture diagram](https://www.plantuml.com/plantuml/svg/REPLACE_WITH_ENCODED_PUML_URL)

- Option B (Local repo diagrams)  
  - Source PUML: diagrams/pumls/{project_slug}.puml
  - Exported SVG/PNG: diagrams/svg/{project_slug}.svg

---

## Team & Roles

- Architecture & Security: {body.architectureSecurity or 'TBD'}
- Backend / API: {body.backendAPI or 'TBD'}
- UI / UX: {body.uiUX or 'TBD'}
- Data / DBA: {body.dataDBA or 'TBD'}
- DevOps / Platform: {body.devOpsPlatform or 'TBD'}
- Product / BA: {body.productBA or 'TBD'}

---

## Key Milestones

- Design Complete (DoR for build): <YYYY-MM-DD / TBD>
- Dev / UAT / Prod rollouts: <dates or windows / TBD>
- External dependencies: <dates / TBD>

---

## Datastores, Services, and Interfaces

Based strictly on the features and description, suggest likely datastores, services, and interfaces using generic names, but do NOT invent vendor products or specific technologies.

- Databases: <names, purpose, RPO/RTO / TBD>
- External systems / integrations: <system, protocol, auth model / TBD>
- APIs / Events: <endpoints, topics, SLAs / TBD>

---

## Features & User Stories

From the key features list:
{body.keyFeatures}

Create user stories with acceptance criteria for each feature. Use generic user roles (e.g., "Operations User", "Customer", "Bank Staff") if not specified.

---

## Non-Functional Requirements (NFRs)

Base NFRs on the NFR focus: {body.nfrFocus}. Do NOT create unrealistic numbers; keep them reasonable but generic.

- Security: <authN/Z, data classification, encryption, key management, secrets>
- Privacy / Compliance: <PII/PCI/PHI handling, retention, audit>
- Reliability: <SLOs, availability, failover, DR, backups>
- Performance: <latency, throughput, load profile>
- Observability: <logging, metrics, tracing, alerting>
- Scalability / Capacity: <traffic growth, storage projections>
- Operability: <runbooks, dashboards, on-call>

---

## CI/CD and Environments

- Branching strategy: <GitFlow, trunk-based / TBD>
- GitLab CI/CD pipelines: <build, test, scans, deploy / TBD>
- Promotion flow: Dev → QA → UAT → Prod
- Feature flags: <strategy / TBD>

---

## Data Security and Access Controls

- Role model: <Green / Amber / Red / TBD>
- Column / row-level security: <approach / TBD>
- Group / role mapping: <AAD / ECMS / BBS / TBD>
- Secrets management: <vault approach / TBD>

---

## Dependencies & Risks

Dependencies:
{body.dependencies or '- None identified yet / TBD'}

Risks:
{body.risks or '- To be assessed during detailed design / TBD'}

---

## Deliverables

- Architecture diagrams (PUML + SVG/PNG)
- Design documentation and how-to guides
- API specifications / contracts
- Infrastructure as Code
- Runbooks and operational dashboards
- Test strategy and test results

---

## Next Steps

- [ ] Finalize architecture design and diagrams
- [ ] Align cross-team dependencies and timelines
- [ ] Prepare CI/CD pipeline templates and IaC skeletons
- [ ] Schedule design review sessions and obtain sign-offs
- [ ] Create child issues for implementation work

---

## Definition of Done (Epic)

- [ ] All features meet acceptance criteria and NFRs
- [ ] Security, compliance, and privacy reviews completed
- [ ] CI/CD pipelines validated across all environments
- [ ] Observability in place; runbooks and dashboards ready
- [ ] Documentation complete and published
- [ ] Stakeholder sign-offs captured

---

## Approvals

- Architecture: <name / date>
- Security: <name / date>
- Product / Business: <name / date>
- Platform / Operations: <name / date>

Return ONLY valid Markdown. Do not wrap the answer in ``` fences.
"""


def build_diagram_prompt(body: GenerateEpicRequest) -> str:
  return f"""You are designing a very simple high-level architecture diagram for a bank project.

Project name: {body.projectName}
Key features (raw list): {body.keyFeatures}
In scope: {body.inScope}

Create a JSON architecture diagram with nodes and connections.

Return ONLY valid JSON with this exact structure, no markdown fences, no explanation:

{{
  "nodes": [
    {{"id": "1", "label": "Component Name", "type": "ui"}},
    {{"id": "2", "label": "Another Component", "type": "service"}}
  ],
  "edges": [
    {{"from": "1", "to": "2", "label": "calls"}}
  ]
}}

Rules:
- Types can be: "ui", "service", "database", "api", "external"
- Include 4–8 nodes representing the main architectural components.
- Use only generic names (e.g., "Client App", "API Gateway", "Core Service", "Database").
- Do NOT mention specific products, vendors, or technologies.
"""


# ---------- Endpoint ----------

@app.post("/api/generate-epic", response_model=GenerateEpicResponse)
def generate_epic(req: GenerateEpicRequest):
    """
    Generate an Epic markdown document and a simple architecture diagram JSON
    using Azure OpenAI GPT-5.
    """
    # 1) EPIC MARKDOWN
    epic_prompt = build_epic_prompt(req)
    epic_messages = [
        {
            "role": "system",
            "content": (
                "You are Azure OpenAI GPT-5 acting as a senior UBS solution architect. "
                "You MUST not hallucinate unknown specifics. Use 'TBD' when unsure."
            ),
        },
        {"role": "user", "content": epic_prompt},
    ]
    markdown = chat_gpt5(epic_messages, temperature=0.35, max_tokens=5500).strip()

    # 2) DIAGRAM JSON
    diagram_prompt = build_diagram_prompt(req)
    diagram_messages = [
        {
            "role": "system",
            "content": (
                "You are Azure OpenAI GPT-5 returning ONLY strict JSON for diagrams. "
                "Do not include commentary or markdown."
            ),
        },
        {"role": "user", "content": diagram_prompt},
    ]
    diagram_raw = chat_gpt5(diagram_messages, temperature=0.2, max_tokens=800).strip()

    import json

    try:
        diagram_json = json.loads(diagram_raw)
    except Exception:
        # Fallback: empty diagram if parsing fails
        diagram_json = {"nodes": [], "edges": []}

    # Normalize structure
    nodes = diagram_json.get("nodes") or []
    edges = diagram_json.get("edges") or []

    return GenerateEpicResponse(
        markdown=markdown,
        diagram=Diagram(nodes=nodes, edges=edges),
    )

5. Run the backend
uvicorn main:app --reload --port 8000


And in your React code, make sure the fetch matches:

const response = await fetch('/api/generate-epic', { ... });


(or http://localhost:8000/api/generate-epic if you’re not using a proxy).
