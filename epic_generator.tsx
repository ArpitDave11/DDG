import React, { useState } from 'react';
import { ChevronRight, ChevronLeft, Sparkles, FileText, Network, Download, CheckCircle2, Loader2, Edit3 } from 'lucide-react';

const EpicGenerator = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [generatedEpic, setGeneratedEpic] = useState('');
  const [diagramData, setDiagramData] = useState(null);
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
        { name: 'projectName', label: 'Project/Capability Name', type: 'text', placeholder: 'e.g., User Authentication System', required: true },
        { name: 'epicType', label: 'Epic Type', type: 'select', options: ['initiative', 'defect', 'custom'] },
        { name: 'customType', label: 'Custom Type (if selected above)', type: 'text', placeholder: 'Enter custom type', conditional: 'epicType', conditionalValue: 'custom' },
        { name: 'status', label: 'Status', type: 'select', options: ['proposed', 'in-progress', 'approved', 'on-hold', 'completed'] },
        { name: 'groupProgram', label: 'Group/Program ID', type: 'text', placeholder: 'e.g., Platform-Engineering-2025' }
      ]
    },
    {
      title: 'Objectives & Problem',
      description: 'What are you solving and why?',
      icon: '🎯',
      fields: [
        { name: 'objective', label: 'Main Objective', type: 'textarea', placeholder: 'Describe the outcome this blueprint will achieve and how it aligns to program/OKRs...', required: true },
        { name: 'businessProblem', label: 'Business/Technical Problem', type: 'textarea', placeholder: 'What specific problem are you solving?', required: true },
        { name: 'successMetric', label: 'Success Metric/Measurable Outcome', type: 'text', placeholder: 'e.g., 99.9% uptime, reduce auth time by 50%, support 10k concurrent users', required: true }
      ]
    },
    {
      title: 'Scope & Boundaries',
      description: 'Define what is in and out',
      icon: '📐',
      fields: [
        { name: 'inScope', label: 'In Scope Items', type: 'textarea', placeholder: 'List key items in scope (one per line or comma-separated)...', required: true },
        { name: 'outOfScope', label: 'Out of Scope Items', type: 'textarea', placeholder: 'List what is explicitly out of scope...' },
        { name: 'assumptions', label: 'Key Assumptions', type: 'textarea', placeholder: 'List your assumptions (e.g., existing infrastructure, team availability)...' }
      ]
    },
    {
      title: 'Team & Stakeholders',
      description: 'Who will work on this?',
      icon: '👥',
      fields: [
        { name: 'stakeholders', label: 'Key Stakeholders', type: 'text', placeholder: 'e.g., John Doe (VP Engineering), Jane Smith (Product Lead)', required: true },
        { name: 'architectureSecurity', label: 'Architecture & Security Lead', type: 'text', placeholder: 'Name(s)' },
        { name: 'backendAPI', label: 'Backend/API Team', type: 'text', placeholder: 'Name(s)' },
        { name: 'uiUX', label: 'UI/UX Team', type: 'text', placeholder: 'Name(s)' },
        { name: 'dataDBA', label: 'Data/DBA Team', type: 'text', placeholder: 'Name(s)' },
        { name: 'devOpsPlatform', label: 'DevOps/Platform Team', type: 'text', placeholder: 'Name(s)' },
        { name: 'productBA', label: 'Product/BA Team', type: 'text', placeholder: 'Name(s)' }
      ]
    },
    {
      title: 'Features & Requirements',
      description: 'What will be built?',
      icon: '⚙️',
      fields: [
        { name: 'keyFeatures', label: 'Key Features/Capabilities', type: 'textarea', placeholder: 'List main features (one per line):\n- User login with MFA\n- OAuth integration\n- Session management\n- Audit logging', required: true },
        { name: 'nfrFocus', label: 'NFR Priority Focus', type: 'select', options: ['standard', 'high-security', 'high-performance', 'high-availability', 'compliance-heavy'] },
        { name: 'targetDate', label: 'Target Completion Date', type: 'date' }
      ]
    },
    {
      title: 'Dependencies & Risks',
      description: 'What could block or impact this?',
      icon: '⚠️',
      fields: [
        { name: 'dependencies', label: 'Key Dependencies', type: 'textarea', placeholder: 'List dependencies (e.g., SSO integration - owned by Platform Team - needed by Q2)...' },
        { name: 'risks', label: 'Known Risks', type: 'textarea', placeholder: 'List risks and mitigation plans (e.g., Database migration complexity - High impact - Plan: phased rollout)...' }
      ]
    }
  ];

  const handleInputChange = (name, value) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      generateEpic();
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const generateEpic = async () => {
    setLoading(true);
    
    try {
      const epicType = formData.epicType === 'custom' ? formData.customType : formData.epicType;
      const projectSlug = formData.projectName.toLowerCase().replace(/\s+/g, '-');
      
      const epicPrompt = `You are a technical architect creating a comprehensive GitLab Epic Technical Design Blueprint in Markdown format.

User Input:
- Project Name: ${formData.projectName}
- Type: ${epicType}
- Status: ${formData.status}
- Group/Program: ${formData.groupProgram || 'TBD'}
- Objective: ${formData.objective}
- Business/Technical Problem: ${formData.businessProblem}
- Success Metric: ${formData.successMetric}
- In Scope: ${formData.inScope}
- Out of Scope: ${formData.outOfScope || 'TBD'}
- Assumptions: ${formData.assumptions || 'Standard development practices apply'}
- Stakeholders: ${formData.stakeholders}
- Target Date: ${formData.targetDate || 'TBD'}
- Team Architecture & Security: ${formData.architectureSecurity || 'TBD'}
- Team Backend/API: ${formData.backendAPI || 'TBD'}
- Team UI/UX: ${formData.uiUX || 'TBD'}
- Team Data/DBA: ${formData.dataDBA || 'TBD'}
- Team DevOps/Platform: ${formData.devOpsPlatform || 'TBD'}
- Team Product/BA: ${formData.productBA || 'TBD'}
- Key Features: ${formData.keyFeatures}
- NFR Focus: ${formData.nfrFocus}
- Dependencies: ${formData.dependencies || 'None identified yet'}
- Risks: ${formData.risks || 'To be assessed during design phase'}

Generate a complete GitLab Epic in Markdown format following this structure:

# Epic: Technical Design Blueprint – ${formData.projectName}

Type: ${epicType}
Status: ${formData.status}
Group/Program: ${formData.groupProgram || 'TBD'}
Labels: Technical-Design, Blueprint, Architecture, ${epicType}
Target Date: ${formData.targetDate || 'TBD'}
Stakeholders: ${formData.stakeholders}
Epic URL: <link after creation>

---

## Objective

Write the objective here based on: ${formData.objective}

- Business/Technical Problem: ${formData.businessProblem}
- Success Metric: ${formData.successMetric}

---

## Background and Context

Generate a paragraph about current state and context.

- Links:
  - ADRs: <ADR index or specific ADR links / TBD>
  - Related epics / issues / MRs: <links / TBD>
  - Documents: <Confluence / Docs links / TBD>

---

## Scope

### In Scope
Convert these to bullet points: ${formData.inScope}

### Out of Scope
${formData.outOfScope ? 'Convert these to bullet points: ' + formData.outOfScope : '- TBD'}

### Assumptions
${formData.assumptions ? 'Convert these to bullet points: ' + formData.assumptions : '- Standard development practices apply'}

---

## Architecture Overview

Write a high-level architecture narrative based on features and scope.

### Diagrams

- Option A (PlantUML URL preview)  
  - ![High-level architecture diagram](https://www.plantuml.com/plantuml/svg/REPLACE_WITH_ENCODED_PUML_URL)

- Option B (Local repo diagrams)  
  - Source PUML: diagrams/pumls/${projectSlug}.puml
  - Exported SVG/PNG: diagrams/svg/${projectSlug}.svg

---

## Team & Roles

- Architecture & Security: ${formData.architectureSecurity || 'TBD'}
- Backend / API: ${formData.backendAPI || 'TBD'}
- UI / UX: ${formData.uiUX || 'TBD'}
- Data / DBA: ${formData.dataDBA || 'TBD'}
- DevOps / Platform: ${formData.devOpsPlatform || 'TBD'}
- Product / BA: ${formData.productBA || 'TBD'}

---

## Key Milestones

- Design Complete (DoR for build): <YYYY-MM-DD / TBD>
- Dev / UAT / Prod rollouts: <dates or windows / TBD>
- External dependencies: <dates / TBD>

---

## Datastores, Services, and Interfaces

Based on the features, suggest datastores, services, and interfaces.

- Databases: <names, purpose, RPO/RTO / TBD>
- External systems / integrations: <system, protocol, auth model / TBD>
- APIs / Events: <endpoints, topics, SLAs / TBD>

---

## Features & User Stories

For each feature from: ${formData.keyFeatures}

Create user stories with acceptance criteria.

1. Feature: [Name]

   User Story:  
   As a [role], I want [capability] so that [benefit].

   Acceptance Criteria:
   - [ ] [criterion 1]
   - [ ] [criterion 2]
   - [ ] [performance/SLO criterion]

---

## Non-Functional Requirements (NFRs)

Based on NFR focus: ${formData.nfrFocus}

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
${formData.dependencies || '- None identified yet / TBD'}

Risks:
${formData.risks || '- To be assessed during detailed design / TBD'}

---

## Deliverables

- Architecture diagrams (PUML + SVG/PNG)
- Design documentation and how-to guides
- API specifications / contracts
- Infrastructure as Code (Terraform/CloudFormation)
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

Generate complete, professional content for all sections. Use proper Markdown formatting.`;

      const epicResponse = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 8000,
          messages: [{ role: 'user', content: epicPrompt }]
        })
      });

      const epicData = await epicResponse.json();
      const epicContent = epicData.content.find(c => c.type === 'text')?.text || '';
      
      const cleanedEpic = epicContent.replace(/```markdown\n?/g, '').replace(/```\n?/g, '').trim();
      setGeneratedEpic(cleanedEpic);

      const diagramPrompt = `Based on this project: "${formData.projectName}"

Features: ${formData.keyFeatures}
Scope: ${formData.inScope}

Create a JSON architecture diagram with nodes and connections. Return ONLY valid JSON with this exact structure:
{
  "nodes": [
    {"id": "1", "label": "Component Name", "type": "ui"},
    {"id": "2", "label": "Another Component", "type": "service"}
  ],
  "edges": [
    {"from": "1", "to": "2", "label": "calls"}
  ]
}

Types can be: ui, service, database, api, external
Include 4-8 nodes representing the main architectural components.`;

      const diagramResponse = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 1500,
          messages: [{ role: 'user', content: diagramPrompt }]
        })
      });

      const diagramData = await diagramResponse.json();
      const diagramText = diagramData.content.find(c => c.type === 'text')?.text || '';
      
      try {
        const cleanedJson = diagramText.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
        const parsedDiagram = JSON.parse(cleanedJson);
        setDiagramData(parsedDiagram);
      } catch (e) {
        console.error('Diagram parse error:', e);
        setDiagramData({ nodes: [], edges: [] });
      }

      setCurrentStep(steps.length);
    } catch (error) {
      console.error('Error generating epic:', error);
      alert('Error generating Epic. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const downloadEpic = () => {
    const filename = `${formData.projectName.replace(/\s+/g, '-').toLowerCase()}-epic.md`;
    const element = document.createElement('a');
    const file = new Blob([generatedEpic], { type: 'text/markdown' });
    element.href = URL.createObjectURL(file);
    element.download = filename;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const renderField = (field) => {
    if (field.conditional && formData[field.conditional] !== field.conditionalValue) {
      return null;
    }

    const value = formData[field.name];

    if (field.type === 'select') {
      return (
        <select
          value={value}
          onChange={(e) => handleInputChange(field.name, e.target.value)}
          className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-blue-500 focus:outline-none transition-colors bg-white"
        >
          {field.options.map(opt => (
            <option key={opt} value={opt}>
              {opt.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
            </option>
          ))}
        </select>
      );
    }

    if (field.type === 'textarea') {
      return (
        <textarea
          value={value}
          onChange={(e) => handleInputChange(field.name, e.target.value)}
          placeholder={field.placeholder}
          rows={4}
          className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-blue-500 focus:outline-none transition-colors resize-none font-mono text-sm"
          required={field.required}
        />
      );
    }

    return (
      <input
        type={field.type}
        value={value}
        onChange={(e) => handleInputChange(field.name, e.target.value)}
        placeholder={field.placeholder}
        className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-blue-500 focus:outline-none transition-colors"
        required={field.required}
      />
    );
  };

  const ArchitectureDiagram = ({ data }) => {
    if (!data || !data.nodes || data.nodes.length === 0) {
      return <div className="text-gray-500 text-center py-8">Diagram generation in progress...</div>;
    }

    const getNodeColor = (type) => {
      const typeStr = type ? type.toLowerCase() : 'default';
      switch(typeStr) {
        case 'ui': return 'bg-orange-500';
        case 'service': return 'bg-blue-500';
        case 'database': return 'bg-green-500';
        case 'api': return 'bg-purple-500';
        case 'external': return 'bg-red-500';
        default: return 'bg-gray-500';
      }
    };

    return (
      <div className="relative bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg p-8 min-h-96 border-2 border-gray-200">
        <div className="grid grid-cols-3 gap-6">
          {data.nodes.map((node, idx) => (
            <div key={node.id} className="flex flex-col items-center">
              <div className={`${getNodeColor(node.type)} text-white px-6 py-4 rounded-xl shadow-lg text-center min-w-40 transform hover:scale-105 transition-transform`}>
                <div className="font-bold text-sm">{node.label}</div>
                <div className="text-xs mt-1 opacity-80 uppercase">{node.type || 'component'}</div>
              </div>
            </div>
          ))}
        </div>
        {data.edges && data.edges.length > 0 && (
          <div className="mt-8 flex flex-wrap gap-3 justify-center">
            {data.edges.map((edge, idx) => (
              <div key={idx} className="bg-white px-4 py-2 rounded-full text-xs shadow-md border border-gray-200">
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
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
        <div className="bg-white rounded-2xl shadow-2xl p-12 text-center max-w-md">
          <Loader2 className="w-16 h-16 text-blue-600 animate-spin mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Generating Your Epic</h2>
          <p className="text-gray-600 mb-4">AI is crafting your comprehensive technical blueprint...</p>
          <div className="flex items-center justify-center gap-2 text-sm text-gray-500">
            <Sparkles className="w-4 h-4" />
            <span>Powered by Claude AI</span>
          </div>
        </div>
      </div>
    );
  }

  if (currentStep === steps.length) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="bg-white rounded-2xl shadow-2xl overflow-hidden">
            <div className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white p-8">
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div className="flex items-center gap-4">
                  <CheckCircle2 className="w-12 h-12" />
                  <div>
                    <h1 className="text-3xl font-bold">{formData.projectName}</h1>
                    <p className="text-blue-100 mt-1">Technical Design Blueprint Generated</p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={() => setIsEditing(!isEditing)}
                    className="flex items-center gap-2 bg-white text-blue-600 px-6 py-3 rounded-lg hover:bg-blue-50 transition-colors font-semibold"
                  >
                    <Edit3 className="w-5 h-5" />
                    {isEditing ? 'Preview' : 'Edit'}
                  </button>
                  <button
                    onClick={downloadEpic}
                    className="flex items-center gap-2 bg-green-500 text-white px-6 py-3 rounded-lg hover:bg-green-600 transition-colors font-semibold"
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
                  <Network className="w-6 h-6 text-indigo-600" />
                  <h2 className="text-2xl font-bold text-gray-800">Architecture Overview</h2>
                </div>
                <ArchitectureDiagram data={diagramData} />
              </div>

              <div>
                <div className="flex items-center gap-2 mb-4">
                  <FileText className="w-6 h-6 text-indigo-600" />
                  <h2 className="text-2xl font-bold text-gray-800">Epic Document (.md)</h2>
                </div>
                
                {isEditing ? (
                  <textarea
                    value={generatedEpic}
                    onChange={(e) => setGeneratedEpic(e.target.value)}
                    className="w-full h-96 p-6 bg-gray-50 rounded-lg border-2 border-gray-200 focus:border-blue-500 focus:outline-none font-mono text-sm"
                  />
                ) : (
                  <div className="bg-gray-50 rounded-lg p-6 max-h-96 overflow-y-auto border-2 border-gray-200">
                    <pre className="whitespace-pre-wrap text-sm text-gray-700 font-mono">
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
                  className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors font-semibold"
                >
                  Create New Epic
                </button>
                <button
                  onClick={downloadEpic}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-semibold flex items-center gap-2"
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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center p-4">
      <div className="w-full max-w-4xl">
        <div className="bg-white rounded-2xl shadow-2xl overflow-hidden">
          <div className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white p-8">
            <div className="flex items-center gap-3 mb-4">
              <Sparkles className="w-8 h-8" />
              <h1 className="text-3xl font-bold">GitLab Epic Generator</h1>
            </div>
            <p className="text-blue-100">AI-Powered Technical Design Blueprint Creator</p>
          </div>

          <div className="bg-gray-200 h-2">
            <div
              className="bg-gradient-to-r from-blue-600 to-purple-600 h-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className="p-8">
            <div className="mb-8">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-4xl">{currentStepData.icon}</span>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm">
                      {currentStep + 1}
                    </span>
                    <h2 className="text-2xl font-bold text-gray-800">{currentStepData.title}</h2>
                  </div>
                  <p className="text-gray-600 mt-1 ml-10">{currentStepData.description}</p>
                </div>
              </div>
            </div>

            <div className="space-y-6">
              {currentStepData.fields.map(field => {
                const fieldElement = renderField(field);
                if (!fieldElement) return null;
                
                return (
                  <div key={field.name}>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      {field.label}
                      {field.required && <span className="text-red-500 ml-1">*</span>}
                    </label>
                    {fieldElement}
                  </div>
                );
              })}
            </div>

            <div className="flex justify-between mt-8 pt-6 border-t-2 border-gray-100">
              <button
                onClick={handleBack}
                disabled={currentStep === 0}
                className="flex items-center gap-2 px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-5 h-5" />
                Back
              </button>

              <button
                onClick={handleNext}
                className="flex items-center gap-2 px-8 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all font-semibold shadow-lg"
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
                  className={`h-2 rounded-full transition-all ${
                    idx <= currentStep ? 'w-12 bg-gradient-to-r from-blue-600 to-indigo-600' : 'w-8 bg-gray-300'
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
