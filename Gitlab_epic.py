Nice, let’s turn this into a ready-to-run project: one repo, backend + frontend, config via .env, very little you have to touch.

I’ll give you:
	•	📁 Project structure
	•	⚙️ Config (.env + config.py)
	•	🧠 Backend (FastAPI + Azure OpenAI GPT-5 + Whisper)
	•	🎨 Frontend (Vite + React + your UBS/GitLab Epic UI + mic → Whisper)
	•	🚀 How to run

You can literally copy this structure into a folder and start wiring your own keys.

⸻

1. Project Structure

ubs-epic-generator/
├── backend/
│   ├── main.py
│   ├── llm_client.py
│   ├── config.py
│   ├── requirements.txt
│   ├── .env.example
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── postcss.config.cjs
│   ├── tailwind.config.cjs
│   ├── .env.example
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── components/
│       │   └── EpicGenerator.tsx
│       └── styles.css
└── README.md

You’ll only need to tweak:
	•	backend/.env
	•	frontend/.env (for API base URL if needed)

Everything else is plug-and-play.

⸻

2. Backend

2.1 backend/.env.example

# Azure OpenAI general
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Chat (GPT-5 style) deployment name
AZURE_OPENAI_DEPLOYMENT_GPT5=your-gpt5-deployment-name

# Whisper deployment name
AZURE_OPENAI_DEPLOYMENT_WHISPER=your-whisper-deployment-name

# Server
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
BACKEND_CORS_ORIGINS=http://localhost:5173

Copy to backend/.env and fill in.

⸻

2.2 backend/requirements.txt

fastapi
uvicorn[standard]
python-dotenv
openai>=1.30.0


⸻

2.3 backend/config.py

import os
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

AZURE_OPENAI_DEPLOYMENT_GPT5 = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT5", "")
AZURE_OPENAI_DEPLOYMENT_WHISPER = os.getenv("AZURE_OPENAI_DEPLOYMENT_WHISPER", "")

BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
BACKEND_CORS_ORIGINS = os.getenv(
    "BACKEND_CORS_ORIGINS",
    "http://localhost:5173"
).split(",")


⸻

2.4 backend/llm_client.py

from openai import AzureOpenAI
from typing import List, Dict
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_GPT5,
    AZURE_OPENAI_DEPLOYMENT_WHISPER,
)

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

MODEL_GPT5 = AZURE_OPENAI_DEPLOYMENT_GPT5
MODEL_WHISPER = AZURE_OPENAI_DEPLOYMENT_WHISPER


def chat_gpt5(
    messages: List[Dict[str, str]],
    temperature: float = 0.35,
    max_tokens: int = 6000,
) -> str:
    resp = client.chat.completions.create(
        model=MODEL_GPT5,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


⸻

2.5 backend/main.py

from typing import Optional, List, Dict, Any
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import BACKEND_CORS_ORIGINS, BACKEND_HOST, BACKEND_PORT
from llm_client import chat_gpt5, client, MODEL_WHISPER


app = FastAPI(title="UBS GitLab Epic Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class TranscriptionResponse(BaseModel):
    text: str


# ---------- Prompt builders ----------

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
    return f"""You are designing a simple high-level architecture diagram for a bank project.

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


# ---------- Endpoints ----------

@app.post("/api/generate-epic", response_model=GenerateEpicResponse)
def generate_epic(req: GenerateEpicRequest):
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

    try:
        diagram_json = json.loads(diagram_raw)
    except Exception:
        diagram_json = {"nodes": [], "edges": []}

    nodes = diagram_json.get("nodes") or []
    edges = diagram_json.get("edges") or []

    return GenerateEpicResponse(
        markdown=markdown,
        diagram=Diagram(nodes=nodes, edges=edges),
    )


@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    if not MODEL_WHISPER:
        return TranscriptionResponse(text="")

    audio_bytes = await file.read()

    # If your SDK requires file-like, use io.BytesIO
    transcription = client.audio.transcriptions.create(
        model=MODEL_WHISPER,
        file=audio_bytes,
        response_format="text",
    )

    text = transcription if isinstance(transcription, str) else str(transcription)
    return TranscriptionResponse(text=text.strip())


⸻

3. Frontend (Vite + React + Tailwind)

3.1 frontend/.env.example

VITE_API_BASE_URL=http://localhost:8000

Copy to frontend/.env and adjust if backend URL changes.

⸻

3.2 frontend/package.json

{
  "name": "ubs-epic-generator-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "lucide-react": "^0.378.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.66",
    "@types/react-dom": "^18.2.22",
    "@vitejs/plugin-react-swc": "^3.6.0",
    "autoprefixer": "^10.4.19",
    "postcss": "^8.4.38",
    "tailwindcss": "^3.4.4",
    "typescript": "^5.4.0",
    "vite": "^5.2.0"
  }
}


⸻

3.3 frontend/vite.config.ts

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173
  }
});


⸻

3.4 frontend/tsconfig.json

{
  "compilerOptions": {
    "target": "ESNext",
    "useDefineForClassFields": true,
    "lib": ["DOM", "DOM.Iterable", "ESNext"],
    "allowJs": false,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src"]
}


⸻

3.5 Tailwind config

frontend/tailwind.config.cjs

module.exports = {
  content: ['./index.html', './src/**/*.{ts,tsx,js,jsx}'],
  theme: {
    extend: {}
  },
  plugins: []
};

frontend/postcss.config.cjs

module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {}
  }
};


⸻

3.6 frontend/index.html

<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>UBS · GitLab Epic Generator</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  </head>
  <body class="bg-gray-100">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>


⸻

3.7 frontend/src/styles.css

@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  @apply bg-gray-100;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}


⸻

3.8 frontend/src/main.tsx

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);


⸻

3.9 frontend/src/App.tsx

import React from 'react';
import EpicGenerator from './components/EpicGenerator';

const App: React.FC = () => {
  return <EpicGenerator />;
};

export default App;


⸻

3.10 frontend/src/components/EpicGenerator.tsx

This is your UBS-themed UI with mic and calls to backend.

import React, { useState, useRef } from 'react';
import {
  ChevronRight,
  ChevronLeft,
  Sparkles,
  FileText,
  Network,
  Download,
  CheckCircle2,
  Loader2,
  Edit3,
  Mic,
  MicOff
} from 'lucide-react';

const UBS_LOGO = '/assets/ubs-logo.svg';      // put logos in frontend/public/assets
const GITLAB_LOGO = '/assets/gitlab-logo.svg';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

type MicButtonProps = {
  onTranscription: (text: string) => void;
};

const MicButton: React.FC<MicButtonProps> = ({ onTranscription }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e: BlobEvent) => {
        if (e.data && e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        chunksRef.current = [];

        const formData = new FormData();
        formData.append('file', blob, 'recording.webm');

        try {
          const res = await fetch(`${API_BASE_URL}/api/transcribe`, {
            method: 'POST',
            body: formData
          });

          if (!res.ok) {
            throw new Error('Transcription failed');
          }

          const data = await res.json();
          if (data.text) {
            onTranscription(data.text);
          }
        } catch (err) {
          console.error('Transcription error:', err);
          alert('Could not transcribe audio. Please try again.');
        } finally {
          stream.getTracks().forEach(t => t.stop());
        }
      };

      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
    } catch (err) {
      console.error('Mic error:', err);
      alert('Microphone access denied or unavailable.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
    setIsRecording(false);
  };

  const handleClick = () => {
    if (!isRecording) startRecording();
    else stopRecording();
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      className={`flex items-center justify-center w-10 h-10 rounded-full border text-xs transition-colors ${
        isRecording
          ? 'bg-red-600 border-red-700 text-white animate-pulse'
          : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-100'
      }`}
      title={isRecording ? 'Stop recording' : 'Record with mic'}
    >
      {isRecording ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
    </button>
  );
};

const EpicGenerator: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [generatedEpic, setGeneratedEpic] = useState('');
  const [diagramData, setDiagramData] = useState<any | null>(null);
  const [isEditing, setIsEditing] = useState(false);

  const [formData, setFormData] = useState({
    projectName: '',
    epicType: 'initiative',
    customType: '',
    status: 'proposed',
    groupProgram: '',
    targetDate: '',
    stakeholders: '',
    objective: '',
    businessProblem: '',
    successMetric: '',
    inScope: '',
    outOfScope: '',
    assumptions: '',
    architectureSecurity: '',
    backendAPI: '',
    uiUX: '',
    dataDBA: '',
    devOpsPlatform: '',
    productBA: '',
    keyFeatures: '',
    nfrFocus: 'standard',
    dependencies: '',
    risks: ''
  });

  const steps = [
    {
      title: 'Epic Basics',
      description: 'Core information about your Epic',
      icon: '📋',
      fields: [
        {
          name: 'projectName',
          label: 'Project/Capability Name',
          type: 'text',
          placeholder: 'e.g., User Authentication System',
          required: true
        },
        {
          name: 'epicType',
          label: 'Epic Type',
          type: 'select',
          options: ['initiative', 'defect', 'custom']
        },
        {
          name: 'customType',
          label: 'Custom Type (if selected above)',
          type: 'text',
          placeholder: 'Enter custom type',
          conditional: 'epicType',
          conditionalValue: 'custom'
        },
        {
          name: 'status',
          label: 'Status',
          type: 'select',
          options: ['proposed', 'in-progress', 'approved', 'on-hold', 'completed']
        },
        {
          name: 'groupProgram',
          label: 'Group/Program ID',
          type: 'text',
          placeholder: 'e.g., Platform-Engineering-2025'
        }
      ]
    },
    {
      title: 'Objectives & Problem',
      description: 'What are you solving and why?',
      icon: '🎯',
      fields: [
        {
          name: 'objective',
          label: 'Main Objective',
          type: 'textarea',
          placeholder:
            'Describe the outcome this blueprint will achieve and how it aligns to program/OKRs...',
          required: true
        },
        {
          name: 'businessProblem',
          label: 'Business/Technical Problem',
          type: 'textarea',
          placeholder: 'What specific problem are you solving?',
          required: true
        },
        {
          name: 'successMetric',
          label: 'Success Metric/Measurable Outcome',
          type: 'text',
          placeholder:
            'e.g., 99.9% uptime, reduce auth time by 50%, support 10k concurrent users',
          required: true
        }
      ]
    },
    {
      title: 'Scope & Boundaries',
      description: 'Define what is in and out',
      icon: '📐',
      fields: [
        {
          name: 'inScope',
          label: 'In Scope Items',
          type: 'textarea',
          placeholder:
            'List key items in scope (one per line or comma-separated)...',
          required: true
        },
        {
          name: 'outOfScope',
          label: 'Out of Scope Items',
          type: 'textarea',
          placeholder: 'List what is explicitly out of scope...'
        },
        {
          name: 'assumptions',
          label: 'Key Assumptions',
          type: 'textarea',
          placeholder:
            'List your assumptions (e.g., existing infrastructure, team availability)...'
        }
      ]
    },
    {
      title: 'Team & Stakeholders',
      description: 'Who will work on this?',
      icon: '👥',
      fields: [
        {
          name: 'stakeholders',
          label: 'Key Stakeholders',
          type: 'text',
          placeholder: 'e.g., John Doe (VP Engineering), Jane Smith (Product Lead)',
          required: true
        },
        {
          name: 'architectureSecurity',
          label: 'Architecture & Security Lead',
          type: 'text',
          placeholder: 'Name(s)'
        },
        {
          name: 'backendAPI',
          label: 'Backend/API Team',
          type: 'text',
          placeholder: 'Name(s)'
        },
        {
          name: 'uiUX',
          label: 'UI/UX Team',
          type: 'text',
          placeholder: 'Name(s)'
        },
        {
          name: 'dataDBA',
          label: 'Data/DBA Team',
          type: 'text',
          placeholder: 'Name(s)'
        },
        {
          name: 'devOpsPlatform',
          label: 'DevOps/Platform Team',
          type: 'text',
          placeholder: 'Name(s)'
        },
        {
          name: 'productBA',
          label: 'Product/BA Team',
          type: 'text',
          placeholder: 'Name(s)'
        }
      ]
    },
    {
      title: 'Features & Requirements',
      description: 'What will be built?',
      icon: '⚙️',
      fields: [
        {
          name: 'keyFeatures',
          label: 'Key Features/Capabilities',
          type: 'textarea',
          placeholder:
            'List main features (one per line):\n- User login with MFA\n- OAuth integration\n- Session management\n- Audit logging',
          required: true
        },
        {
          name: 'nfrFocus',
          label: 'NFR Priority Focus',
          type: 'select',
          options: [
            'standard',
            'high-security',
            'high-performance',
            'high-availability',
            'compliance-heavy'
          ]
        },
        {
          name: 'targetDate',
          label: 'Target Completion Date',
          type: 'date'
        }
      ]
    },
    {
      title: 'Dependencies & Risks',
      description: 'What could block or impact this?',
      icon: '⚠️',
      fields: [
        {
          name: 'dependencies',
          label: 'Key Dependencies',
          type: 'textarea',
          placeholder:
            'List dependencies (e.g., SSO integration - owned by Platform Team - needed by Q2)...'
        },
        {
          name: 'risks',
          label: 'Known Risks',
          type: 'textarea',
          placeholder:
            'List risks and mitigation plans (e.g., Database migration complexity - High impact - Plan: phased rollout)...'
        }
      ]
    }
  ];

  const handleInputChange = (name: string, value: string) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(prev => prev + 1);
    } else {
      generateEpic();
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
    }
  };

  const generateEpic = async () => {
    setLoading(true);
    try {
      const epicType =
        formData.epicType === 'custom'
          ? formData.customType || 'custom'
          : formData.epicType;

      const res = await fetch(`${API_BASE_URL}/api/generate-epic`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...formData, epicType })
      });

      if (!res.ok) throw new Error('Backend error');

      const data = await res.json();
      setGeneratedEpic((data.markdown || '').trim());
      setDiagramData(data.diagram || { nodes: [], edges: [] });
      setCurrentStep(steps.length);
    } catch (err) {
      console.error('Error generating epic:', err);
      alert('Error generating Epic. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const downloadEpic = () => {
    const filename = `${formData.projectName
      .replace(/\s+/g, '-')
      .toLowerCase()}-epic.md`;
    const element = document.createElement('a');
    const file = new Blob([generatedEpic], { type: 'text/markdown' });
    element.href = URL.createObjectURL(file);
    element.download = filename;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const renderField = (field: any) => {
    if (field.conditional && (formData as any)[field.conditional] !== field.conditionalValue) {
      return null;
    }

    const value = (formData as any)[field.name];

    if (field.type === 'select') {
      return (
        <select
          value={value}
          onChange={e => handleInputChange(field.name, e.target.value)}
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:border-red-700 focus:outline-none transition-colors bg-white"
        >
          {field.options.map((opt: string) => (
            <option key={opt} value={opt}>
              {opt
                .split('-')
                .map(w => w.charAt(0).toUpperCase() + w.slice(1))
                .join(' ')}
            </option>
          ))}
        </select>
      );
    }

    if (field.type === 'textarea') {
      return (
        <div className="flex gap-2 items-start">
          <textarea
            value={value}
            onChange={e => handleInputChange(field.name, e.target.value)}
            placeholder={field.placeholder}
            rows={4}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:border-red-700 focus:outline-none transition-colors resize-none font-mono text-sm"
            required={field.required}
          />
          <MicButton
            onTranscription={text => {
              const prefix = value && value.length > 0 ? value + '\n' : '';
              handleInputChange(field.name, prefix + text);
            }}
          />
        </div>
      );
    }

    return (
      <input
        type={field.type}
        value={value}
        onChange={e => handleInputChange(field.name, e.target.value)}
        placeholder={field.placeholder}
        className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:border-red-700 focus:outline-none transition-colors"
        required={field.required}
      />
    );
  };

  const ArchitectureDiagram: React.FC<{ data: any }> = ({ data }) => {
    if (!data || !data.nodes || data.nodes.length === 0) {
      return (
        <div className="text-gray-500 text-center py-8">
          Diagram generation in progress...
        </div>
      );
    }

    const getNodeColor = (type?: string) => {
      const t = type ? type.toLowerCase() : 'default';
      switch (t) {
        case 'ui':
          return 'bg-red-600';
        case 'service':
          return 'bg-gray-800';
        case 'database':
          return 'bg-gray-600';
        case 'api':
          return 'bg-red-800';
        case 'external':
          return 'bg-gray-500';
        default:
          return 'bg-gray-700';
      }
    };

    return (
      <div className="relative bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg p-8 min-h-96 border border-gray-200">
        <div className="grid grid-cols-3 gap-6">
          {data.nodes.map((node: any) => (
            <div key={node.id} className="flex flex-col items-center">
              <div
                className={`${getNodeColor(
                  node.type
                )} text-white px-6 py-4 rounded-xl shadow-lg text-center min-w-40 transform hover:scale-105 transition-transform`}
              >
                <div className="font-bold text-sm">{node.label}</div>
                <div className="text-xs mt-1 opacity-80 uppercase">
                  {node.type || 'component'}
                </div>
              </div>
            </div>
          ))}
        </div>
        {data.edges && data.edges.length > 0 && (
          <div className="mt-8 flex flex-wrap gap-3 justify-center">
            {data.edges.map((edge: any, idx: number) => (
              <div
                key={idx}
                className="bg-white px-4 py-2 rounded-full text-xs shadow-md border border-gray-200"
              >
                {edge.label}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 via-gray-100 to-red-50 flex items-center justify-center">
        <div className="bg-white rounded-2xl shadow-2xl p-12 text-center max-w-md border border-gray-200">
          <Loader2 className="w-16 h-16 text-red-700 animate-spin mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            Generating Your Epic
          </h2>
          <p className="text-gray-600 mb-4">
            Azure OpenAI is crafting your technical design blueprint...
          </p>
          <div className="flex items-center justify-center gap-2 text-sm text-gray-500">
            <Sparkles className="w-4 h-4 text-red-600" />
            <span>Powered by Azure OpenAI GPT-5</span>
          </div>
        </div>
      </div>
    );
  }

  if (currentStep === steps.length) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 via-gray-100 to-red-50 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="bg-white rounded-2xl shadow-2xl overflow-hidden border border-gray-200">
            <div className="bg-gradient-to-r from-black via-gray-900 to-red-800 text-white p-6">
              <div className="flex items-center justify-between gap-6 flex-wrap">
                <div className="flex items-center gap-4">
                  <img src={UBS_LOGO} alt="UBS" className="h-10 w-auto object-contain" />
                  <span className="text-gray-400 text-sm">×</span>
                  <img src={GITLAB_LOGO} alt="GitLab" className="h-8 w-auto object-contain" />
                </div>
                <div className="flex items-center gap-3">
                  <Sparkles className="w-6 h-6 text-red-400" />
                  <span className="text-sm text-gray-200">
                    UBS Epic Technical Design Generator · Powered by Azure OpenAI
                  </span>
                </div>
              </div>
              <div className="mt-6 flex items-center justify-between flex-wrap gap-4">
                <div className="flex items-center gap-4">
                  <CheckCircle2 className="w-10 h-10 text-red-400" />
                  <div>
                    <h1 className="text-3xl font-bold">
                      {formData.projectName || 'Generated Epic'}
                    </h1>
                    <p className="text-gray-300 mt-1">
                      Technical Design Blueprint Generated
                    </p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={() => setIsEditing(prev => !prev)}
                    className="flex items-center gap-2 bg-white text-gray-900 px-6 py-3 rounded-lg hover:bg-gray-100 transition-colors font-semibold"
                  >
                    <Edit3 className="w-5 h-5" />
                    {isEditing ? 'Preview' : 'Edit'}
                  </button>
                  <button
                    onClick={downloadEpic}
                    className="flex items-center gap-2 bg-red-600 text-white px-6 py-3 rounded-lg hover:bg-red-700 transition-colors font-semibold"
                  >
                    <Download className="w-5 h-5" />
                    Download .md
                  </button>
                </div>
              </div>
            </div>

            <div className="p-8">
              <div className="mb-8">
                <div className="flex items-center gap-2 mb-4">
                  <Network className="w-6 h-6 text-red-700" />
                  <h2 className="text-2xl font-bold text-gray-900">
                    Architecture Overview
                  </h2>
                </div>
                <ArchitectureDiagram data={diagramData} />
              </div>

              <div>
                <div className="flex items-center gap-2 mb-4">
                  <FileText className="w-6 h-6 text-red-700" />
                  <h2 className="text-2xl font-bold text-gray-900">
                    Epic Document (.md)
                  </h2>
                </div>
                {isEditing ? (
                  <textarea
                    value={generatedEpic}
                    onChange={e => setGeneratedEpic(e.target.value)}
                    className="w-full h-96 p-6 bg-gray-50 rounded-lg border border-gray-300 focus:border-red-700 focus:outline-none font-mono text-sm"
                  />
                ) : (
                  <div className="bg-gray-50 rounded-lg p-6 max-h-96 overflow-y-auto border border-gray-300">
                    <pre className="whitespace-pre-wrap text-sm text-gray-800 font-mono">
                      {generatedEpic}
                    </pre>
                  </div>
                )}
              </div>

              <div className="mt-8 flex gap-4 flex-wrap">
                <button
                  onClick={() => {
                    setCurrentStep(0);
                    setGeneratedEpic('');
                    setDiagramData(null);
                    setIsEditing(false);
                  }}
                  className="px-6 py-3 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 transition-colors font-semibold"
                >
                  Create New Epic
                </button>
                <button
                  onClick={downloadEpic}
                  className="px-6 py-3 bg-black text-white rounded-lg hover:bg-gray-900 transition-colors font-semibold flex items-center gap-2"
                >
                  <Download className="w-5 h-5" />
                  Download for GitLab
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const currentStepData = steps[currentStep];
  const progress = ((currentStep + 1) / steps.length) * 100;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-gray-100 to-red-50 flex items-center justify-center p-4">
      <div className="w-full max-w-5xl">
        <div className="bg-white rounded-2xl shadow-2xl overflow-hidden border border-gray-200">
          <div className="bg-gradient-to-r from-black via-gray-900 to-red-800 text-white p-6">
            <div className="flex items-center justify-between gap-6 flex-wrap">
              <div className="flex items-center gap-4">
                <img src={UBS_LOGO} alt="UBS" className="h-10 w-auto object-contain" />
                <span className="text-gray-400 text-sm">×</span>
                <img src={GITLAB_LOGO} alt="GitLab" className="h-8 w-auto object-contain" />
              </div>
              <div className="flex flex-col items-end">
                <div className="flex items-center gap-2 mb-1">
                  <Sparkles className="w-6 h-6 text-red-400" />
                  <h1 className="text-2xl font-bold">
                    UBS · GitLab Epic Generator
                  </h1>
                </div>
                <p className="text-gray-300 text-sm">
                  AI-Powered Technical Design Blueprint · Azure OpenAI GPT-5
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gray-200 h-1.5">
            <div
              className="bg-gradient-to-r from-red-700 via-red-600 to-black h-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className="p-8">
            <div className="mb-8">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-4xl">{currentStepData.icon}</span>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="bg-black text-white w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm">
                      {currentStep + 1}
                    </span>
                    <h2 className="text-2xl font-bold text-gray-900">
                      {currentStepData.title}
                    </h2>
                  </div>
                  <p className="text-gray-600 mt-1 ml-10">
                    {currentStepData.description}
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-6">
              {currentStepData.fields.map(field => {
                const element = renderField(field);
                if (!element) return null;
                return (
                  <div key={field.name}>
                    <label className="block text-sm font-semibold text-gray-800 mb-2">
                      {field.label}
                      {field.required && (
                        <span className="text-red-600 ml-1">*</span>
                      )}
                    </label>
                    {element}
                  </div>
                );
              })}
            </div>

            <div className="flex justify-between mt-8 pt-6 border-t border-gray-200">
              <button
                onClick={handleBack}
                disabled={currentStep === 0}
                className="flex items-center gap-2 px-6 py-3 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 transition-colors font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-5 h-5" />
                Back
              </button>
              <button
                onClick={handleNext}
                className="flex items-center gap-2 px-8 py-3 bg-gradient-to-r from-red-700 via-red-600 to-black text-white rounded-lg hover:from-red-800 hover:via-red-700 hover:to-black transition-all font-semibold shadow-lg"
              >
                {currentStep === steps.length - 1 ? (
                  <>
                    <Sparkles className="w-5 h-5" />
                    Generate Epic
                  </>
                ) : (
                  <>
                    Next
                    <ChevronRight className="w-5 h-5" />
                  </>
                )}
              </button>
            </div>
          </div>

          <div className="bg-gray-50 px-8 py-4">
            <div className="flex justify-center gap-2">
              {steps.map((step, idx) => (
                <div
                  key={idx}
                  className={`h-1.5 rounded-full transition-all ${
                    idx <= currentStep
                      ? 'w-10 bg-gradient-to-r from-red-700 via-red-600 to-black'
                      : 'w-6 bg-gray-300'
                  }`}
                  title={step.title}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EpicGenerator;


⸻

4. How to Run

From project root:

Backend

cd backend
cp .env.example .env   # edit with your Azure keys & deployments
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

Frontend

cd ../frontend
cp .env.example .env   # set VITE_API_BASE_URL=http://localhost:8000
npm install
npm run dev

Open http://localhost:5173 in your browser.

You now have:
	•	UBS+GitLab-themed Epic Generator UI
	•	Mic icons on each textarea → Azure Whisper → auto-fill text
	•	“Generate Epic” → Azure GPT-5 → full markdown EPIC + diagram JSON
	•	Download .md ready for GitLab Epic.

If you want next, I can help you dockerize this whole thing, or add GitLab EPIC creation via API as a final button.
