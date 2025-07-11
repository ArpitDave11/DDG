# enhanced_config.py
import os
import hashlib
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI services."""
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    gpt4_deployment: str = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4")
    gpt35_deployment: str = os.getenv("AZURE_OPENAI_GPT35_DEPLOYMENT", "gpt-35-turbo")
    o4_mini_deployment: str = os.getenv("AZURE_OPENAI_O4_MINI_DEPLOYMENT", "o1-mini")
    o3_mini_deployment: str = os.getenv("AZURE_OPENAI_O3_MINI_DEPLOYMENT", "o1-mini")
    
    def __post_init__(self):
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure OpenAI endpoint and API key must be provided")

@dataclass
class DatabaseConfig:
    """Database configuration for SQL execution."""
    connection_string: str = os.getenv("DATABASE_CONNECTION_STRING", "")
    db_type: str = os.getenv("DATABASE_TYPE", "postgresql")

@dataclass
class VectorDBConfig:
    """Configuration for vector database (Chroma/Azure Cognitive Search)."""
    provider: str = os.getenv("VECTOR_DB_PROVIDER", "chroma")  # chroma, azure_search
    connection_string: str = os.getenv("VECTOR_DB_CONNECTION", "")
    collection_name: str = "dq_conversation_memory"

@dataclass
class NotificationConfig:
    """Configuration for notifications."""
    slack_webhook: str = os.getenv("SLACK_WEBHOOK_URL", "")
    email_smtp_server: str = os.getenv("EMAIL_SMTP_SERVER", "")
    email_username: str = os.getenv("EMAIL_USERNAME", "")
    email_password: str = os.getenv("EMAIL_PASSWORD", "")

def generate_schema_hash(schema: Dict[str, Any]) -> str:
    """Generate SHA-256 hash of schema for versioning."""
    schema_str = str(sorted(schema.items()))
    return hashlib.sha256(schema_str.encode()).hexdigest()

# enhanced_models.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json
import yaml
from datetime import datetime
from semantic_version import Version

class RuleType(Enum):
    """Types of data quality rules."""
    NULL_CHECK = "null_check"
    NOT_NULL = "not_null"
    RANGE_CHECK = "range"
    FORMAT_CHECK = "format"
    PATTERN_CHECK = "pattern"
    UNIQUENESS_CHECK = "uniqueness"
    COMPLETENESS_CHECK = "completeness"
    CONSISTENCY_CHECK = "consistency"
    CUSTOM = "custom"

@dataclass
class DQRuleSpec:
    """Lightweight DSL for DQ rules."""
    col: str
    rule: str
    description: str = ""
    severity: str = "medium"
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_yaml(self) -> str:
        """Convert to YAML format."""
        rule_dict = {
            'col': self.col,
            'rule': self.rule,
            'description': self.description,
            'severity': self.severity
        }
        if self.parameters:
            rule_dict.update(self.parameters)
        return yaml.dump(rule_dict, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'DQRuleSpec':
        """Create from YAML string."""
        data = yaml.safe_load(yaml_str)
        parameters = {k: v for k, v in data.items() 
                     if k not in ['col', 'rule', 'description', 'severity']}
        return cls(
            col=data['col'],
            rule=data['rule'],
            description=data.get('description', ''),
            severity=data.get('severity', 'medium'),
            parameters=parameters
        )

@dataclass
class VersionedArtifact:
    """Base class for versioned artifacts."""
    version: str
    schema_hash: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_metadata_header(self) -> Dict[str, Any]:
        """Get metadata header for file storage."""
        return {
            "dq_version": self.version,
            "schema_hash": self.schema_hash,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class EnhancedDataQualityRule(VersionedArtifact):
    """Enhanced DQ rule with versioning and DSL support."""
    rule_spec: DQRuleSpec
    sql_statement: str = ""
    rule_id: str = ""
    
    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = f"{self.rule_spec.col}_{self.rule_spec.rule}_{hash(self.rule_spec.col + self.rule_spec.rule) % 10000}"

@dataclass
class FailureTrend:
    """Represents failure trend analysis."""
    rule_id: str
    sparkline_counts: List[int]  # Last 10 runs
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_percentage: float
    last_runs_summary: str  # Human-readable summary

@dataclass
class EnhancedDQFailureRecord:
    """Enhanced failure record with trend analysis."""
    rule_id: str
    column_name: str
    table_name: str
    failure_data: Dict[str, Any]
    timestamp: datetime
    record_count: int
    trend: Optional[FailureTrend] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "rule_id": self.rule_id,
            "column_name": self.column_name,
            "table_name": self.table_name,
            "failure_data": self.failure_data,
            "timestamp": self.timestamp.isoformat(),
            "record_count": self.record_count
        }
        if self.trend:
            result["trend"] = {
                "sparkline_counts": self.trend.sparkline_counts,
                "trend_direction": self.trend.trend_direction,
                "trend_percentage": self.trend.trend_percentage,
                "summary": self.trend.last_runs_summary
            }
        return result

@dataclass
class AutoFixSuggestion:
    """Represents an auto-fix suggestion."""
    issue_type: str
    description: str
    proposed_sql: str
    confidence_score: float
    requires_approval: bool = True
    approved_by: Optional[str] = None
    applied_at: Optional[datetime] = None

# rule_planner_agent.py
import json
import yaml
from typing import Dict, List, Optional
from base_agent import BaseAgent
from enhanced_models import DQRuleSpec, RuleType
from enhanced_config import AzureOpenAIConfig

class RulePlannerAgent(BaseAgent):
    """Agent responsible for planning what DQ checks are needed."""
    
    def __init__(self, config: AzureOpenAIConfig):
        system_message = """
        You are a data quality expert responsible for planning comprehensive data quality checks.
        Given a data schema and column profiles, determine what quality rules should be applied.
        
        Your output should be a YAML list of rules using this DSL format:
        - col: column_name
          rule: rule_type
          description: human-readable description
          severity: critical|high|medium|low
          param1: value1  # additional parameters as needed
        
        Available rule types:
        - not_null: Column should not contain null values
        - range: Numeric values should be within min/max bounds
        - pattern: Text should match a regex pattern
        - uniqueness: Values should be unique
        - format: Values should follow a specific format (email, phone, etc.)
        - completeness: Column should have sufficient non-empty values
        - consistency: Values should be consistent with business rules
        
        Consider:
        1. Data type and domain
        2. Business context
        3. Statistical properties
        4. Regulatory requirements
        5. Data lineage and source systems
        
        Plan comprehensive but practical rules - avoid over-engineering.
        """
        super().__init__(
            name="rule_planner_agent",
            config=config,
            deployment_name=config.o4_mini_deployment,  # Use o4-mini for planning
            model_name="o1-mini",
            system_message=system_message
        )
    
    async def plan_rules(self, table_name: str, schema: Dict[str, Any], 
                        profiles: Dict[str, Any], business_context: str = "") -> List[DQRuleSpec]:
        """Plan DQ rules for a table."""
        
        prompt = f"""
        Plan data quality rules for table: {table_name}
        
        Schema information:
        {json.dumps(schema, indent=2)}
        
        Column profiles:
        {json.dumps(profiles, indent=2)}
        
        Business context:
        {business_context}
        
        Generate a comprehensive set of data quality rules in YAML format.
        Focus on practical, actionable rules that align with the data characteristics.
        
        Consider edge cases and business requirements.
        """
        
        response = await self.run_with_retry(prompt)
        
        try:
            # Parse YAML response
            rules_data = yaml.safe_load(response)
            rule_specs = []
            
            for rule_data in rules_data:
                # Extract parameters
                parameters = {k: v for k, v in rule_data.items() 
                            if k not in ['col', 'rule', 'description', 'severity']}
                
                rule_spec = DQRuleSpec(
                    col=rule_data['col'],
                    rule=rule_data['rule'],
                    description=rule_data.get('description', ''),
                    severity=rule_data.get('severity', 'medium'),
                    parameters=parameters
                )
                rule_specs.append(rule_spec)
            
            return rule_specs
            
        except (yaml.YAMLError, KeyError) as e:
            logger.error(f"Failed to parse rule plan: {e}")
            return []

# sql_writer_agent.py
import json
from typing import Dict, List
from jinja2 import Template
from base_agent import BaseAgent
from enhanced_models import DQRuleSpec, EnhancedDataQualityRule
from enhanced_config import AzureOpenAIConfig

class SQLWriterAgent(BaseAgent):
    """Agent responsible for converting rules to SQL using templates."""
    
    def __init__(self, config: AzureOpenAIConfig):
        system_message = """
        You are a SQL code generation expert. Convert data quality rule specifications 
        into executable SQL queries using provided templates.
        
        Your job is to:
        1. Take a rule specification in DSL format
        2. Apply the appropriate SQL template
        3. Fill in table names, column names, and parameters
        4. Generate optimized, executable SQL
        
        Focus on:
        - Correct SQL syntax for the target database
        - Optimal performance (use indexes, limit result sets)
        - Clear, readable queries
        - Proper parameterization
        
        Return only valid SQL statements that identify violations.
        """
        super().__init__(
            name="sql_writer_agent",
            config=config,
            deployment_name=config.o3_mini_deployment,  # Use o3-mini for code generation
            model_name="o1-mini",
            system_message=system_message
        )
        
        # SQL templates using Jinja2
        self.templates = {
            'not_null': Template("""
                SELECT *
                FROM {{ table_name }}
                WHERE {{ column_name }} IS NULL
            """),
            'range': Template("""
                SELECT *
                FROM {{ table_name }}
                WHERE {{ column_name }} IS NOT NULL
                AND ({{ column_name }} < {{ min_value }} OR {{ column_name }} > {{ max_value }})
            """),
            'pattern': Template("""
                SELECT *
                FROM {{ table_name }}
                WHERE {{ column_name }} IS NOT NULL
                AND {{ column_name }} !~ '{{ pattern }}'
            """),
            'uniqueness': Template("""
                SELECT {{ column_name }}, COUNT(*) as duplicate_count
                FROM {{ table_name }}
                WHERE {{ column_name }} IS NOT NULL
                GROUP BY {{ column_name }}
                HAVING COUNT(*) > 1
            """),
            'completeness': Template("""
                SELECT *
                FROM {{ table_name }}
                WHERE {{ column_name }} IS NULL 
                OR TRIM({{ column_name }}) = ''
            """),
            'format': Template("""
                SELECT *
                FROM {{ table_name }}
                WHERE {{ column_name }} IS NOT NULL
                AND {{ column_name }} !~ '{{ format_pattern }}'
            """)
        }
    
    async def generate_sql(self, table_name: str, rule_specs: List[DQRuleSpec]) -> Dict[str, str]:
        """Generate SQL for all rule specifications."""
        sql_statements = {}
        
        for rule_spec in rule_specs:
            try:
                sql = self._generate_sql_for_rule(table_name, rule_spec)
                rule_id = f"{table_name}_{rule_spec.col}_{rule_spec.rule}"
                sql_statements[rule_id] = sql
            except Exception as e:
                logger.error(f"Failed to generate SQL for rule {rule_spec}: {e}")
                # Fallback to AI generation
                ai_sql = await self._generate_ai_sql(table_name, rule_spec)
                if ai_sql:
                    rule_id = f"{table_name}_{rule_spec.col}_{rule_spec.rule}"
                    sql_statements[rule_id] = ai_sql
        
        return sql_statements
    
    def _generate_sql_for_rule(self, table_name: str, rule_spec: DQRuleSpec) -> str:
        """Generate SQL using templates."""
        template = self.templates.get(rule_spec.rule)
        if not template:
            raise ValueError(f"No template for rule type: {rule_spec.rule}")
        
        # Prepare template variables
        template_vars = {
            'table_name': table_name,
            'column_name': rule_spec.col,
            **rule_spec.parameters
        }
        
        sql = template.render(**template_vars)
        return sql.strip()
    
    async def _generate_ai_sql(self, table_name: str, rule_spec: DQRuleSpec) -> str:
        """Fallback AI generation for complex rules."""
        prompt = f"""
        Generate SQL to find violations of this data quality rule:
        
        Table: {table_name}
        Column: {rule_spec.col}
        Rule: {rule_spec.rule}
        Description: {rule_spec.description}
        Parameters: {json.dumps(rule_spec.parameters)}
        
        Return only the SQL SELECT statement that identifies violating records.
        """
        
        try:
            response = await self.run_with_retry(prompt)
            # Extract SQL from response (remove markdown formatting if present)
            sql = response.strip()
            if sql.startswith('```sql'):
                sql = sql[6:-3].strip()
            elif sql.startswith('```'):
                sql = sql[3:-3].strip()
            return sql
        except Exception as e:
            logger.error(f"AI SQL generation failed: {e}")
            return ""

# vector_memory_manager.py
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import chromadb
from chromadb.config import Settings
from enhanced_models import EnhancedDQFailureRecord

logger = logging.getLogger(__name__)

class VectorMemoryManager:
    """Manages conversation memory using vector database."""
    
    def __init__(self, config):
        self.config = config
        if config.provider == "chroma":
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
        else:
            # Azure Cognitive Search implementation would go here
            raise NotImplementedError("Azure Cognitive Search not implemented yet")
        
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get or create the conversation memory collection."""
        try:
            return self.client.get_collection(name=self.config.collection_name)
        except Exception:
            return self.client.create_collection(
                name=self.config.collection_name,
                metadata={"description": "DQ conversation memory"}
            )
    
    def store_failure_summary(self, table_name: str, column_name: str, 
                             rule_id: str, failure_summary: str, 
                             failure_record: EnhancedDQFailureRecord):
        """Store a failure summary in vector memory."""
        document_id = f"{table_name}_{column_name}_{rule_id}_{datetime.now().isoformat()}"
        
        metadata = {
            "table_name": table_name,
            "column_name": column_name,
            "rule_id": rule_id,
            "timestamp": failure_record.timestamp.isoformat(),
            "record_count": failure_record.record_count,
            "trend_direction": failure_record.trend.trend_direction if failure_record.trend else "unknown"
        }
        
        self.collection.add(
            documents=[failure_summary],
            metadatas=[metadata],
            ids=[document_id]
        )
    
    def query_similar_failures(self, table_name: str, column_name: str, 
                             query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query for similar failures using semantic search."""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where={
                    "$and": [
                        {"table_name": {"$eq": table_name}},
                        {"column_name": {"$eq": column_name}}
                    ]
                }
            )
            
            similar_failures = []
            for i, doc in enumerate(results['documents'][0]):
                similar_failures.append({
                    "summary": doc,
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
            
            return similar_failures
        except Exception as e:
            logger.error(f"Failed to query similar failures: {e}")
            return []
    
    def get_conversation_context(self, table_name: str, column_name: str, 
                               rule_id: str) -> str:
        """Get conversation context for a specific rule."""
        try:
            results = self.collection.query(
                query_texts=[f"What happened with {rule_id} in {table_name}.{column_name}?"],
                n_results=3,
                where={
                    "$and": [
                        {"table_name": {"$eq": table_name}},
                        {"column_name": {"$eq": column_name}},
                        {"rule_id": {"$eq": rule_id}}
                    ]
                }
            )
            
            if results['documents'] and results['documents'][0]:
                context_parts = []
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    context_parts.append(f"Previous analysis: {doc} (from {metadata['timestamp']})")
                return "\n".join(context_parts)
            
            return "No previous conversation context found."
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return "Unable to retrieve conversation context."

# trend_aware_failure_manager.py
import json
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
from enhanced_models import EnhancedDQFailureRecord, FailureTrend, VersionedArtifact

logger = logging.getLogger(__name__)

class TrendAwareFailureManager:
    """Enhanced failure manager with trend analysis."""
    
    def __init__(self, history_dir: str = "dq_history"):
        self.history_dir = history_dir
        os.makedirs(history_dir, exist_ok=True)
    
    def load_failure_history(self, table_name: str, 
                           max_runs: int = 10) -> List[EnhancedDQFailureRecord]:
        """Load failure history with trend analysis."""
        history_files = self._get_history_files(table_name)
        all_failures = []
        
        # Load recent history files
        for file_path in sorted(history_files, reverse=True)[:max_runs]:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                for failure_data in data.get('failures', []):
                    failure_record = EnhancedDQFailureRecord(
                        rule_id=failure_data['rule_id'],
                        column_name=failure_data['column_name'],
                        table_name=failure_data['table_name'],
                        failure_data=failure_data['failure_data'],
                        timestamp=datetime.fromisoformat(failure_data['timestamp']),
                        record_count=failure_data['record_count']
                    )
                    all_failures.append(failure_record)
                    
            except Exception as e:
                logger.warning(f"Failed to load history file {file_path}: {e}")
        
        # Add trend analysis
        self._add_trend_analysis(all_failures)
        return all_failures
    
    def _get_history_files(self, table_name: str) -> List[str]:
        """Get all history files for a table."""
        pattern = f"DQ_failureHistory_{table_name}_"
        files = []
        
        for filename in os.listdir(self.history_dir):
            if filename.startswith(pattern) and filename.endswith('.json'):
                files.append(os.path.join(self.history_dir, filename))
        
        return files
    
    def _add_trend_analysis(self, failures: List[EnhancedDQFailureRecord]):
        """Add trend analysis to failure records."""
        # Group failures by rule_id
        failures_by_rule = defaultdict(list)
        for failure in failures:
            failures_by_rule[failure.rule_id].append(failure)
        
        # Calculate trends for each rule
        for rule_id, rule_failures in failures_by_rule.items():
            # Sort by timestamp
            rule_failures.sort(key=lambda x: x.timestamp)
            
            # Get sparkline data (last 10 runs)
            sparkline_counts = [f.record_count for f in rule_failures[-10:]]
            
            # Calculate trend
            trend = self._calculate_trend(sparkline_counts)
            
            # Apply trend to the most recent failure
            if rule_failures:
                latest_failure = rule_failures[-1]
                latest_failure.trend = trend
    
    def _calculate_trend(self, counts: List[int]) -> FailureTrend:
        """Calculate trend from count history."""
        if len(counts) < 2:
            return FailureTrend(
                rule_id="",
                sparkline_counts=counts,
                trend_direction="insufficient_data",
                trend_percentage=0.0,
                last_runs_summary=f"Only {len(counts)} run(s) available"
            )
        
        # Simple trend calculation
        recent_avg = sum(counts[-3:]) / min(3, len(counts))
        older_avg = sum(counts[:-3]) / max(1, len(counts) - 3) if len(counts) > 3 else counts[0]
        
        if recent_avg > older_avg * 1.1:
            trend_direction = "increasing"
        elif recent_avg < older_avg * 0.9:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        trend_percentage = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        
        # Generate summary
        summary = f"Violations trend: {trend_direction} ({trend_percentage:+.1f}%) over last {len(counts)} runs"
        
        return FailureTrend(
            rule_id="",
            sparkline_counts=counts,
            trend_direction=trend_direction,
            trend_percentage=trend_percentage,
            last_runs_summary=summary
        )
    
    def save_failure_history(self, table_name: str, failures: List[EnhancedDQFailureRecord], 
                           version: str, schema_hash: str):
        """Save failure history with versioning."""
        timestamp = datetime.now()
        filename = f"DQ_failureHistory_{table_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.history_dir, filename)
        
        # Create versioned artifact
        artifact = VersionedArtifact(
            version=version,
            schema_hash=schema_hash,
            created_at=timestamp,
            metadata={"table_name": table_name, "failure_count": len(failures)}
        )
        
        data = {
            **artifact.get_metadata_header(),
            "failures": [f.to_dict() for f in failures]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved failure history to {filepath}")

# auto_fix_agent.py
import json
import logging
from typing import Dict, List, Optional, Any
from base_agent import BaseAgent
from enhanced_models import AutoFixSuggestion, EnhancedDQFailureRecord
from enhanced_config import AzureOpenAIConfig

logger = logging.getLogger(__name__)

class AutoFixAgent(BaseAgent):
    """Agent that suggests automatic fixes for data quality issues."""
    
    def __init__(self, config: AzureOpenAIConfig):
        system_message = """
        You are a data quality remediation expert. Analyze data quality failures 
        and suggest automated fixes when appropriate.
        
        For each failure, consider:
        1. Root cause analysis
        2. Systematic vs. isolated issues
        3. Safety of automated fixes
        4. Business impact of changes
        
        Suggest fixes for common issues like:
        - Trailing/leading spaces → TRIM operations
        - Consistent formatting issues → REPLACE/REGEXP_REPLACE
        - Null values that should have defaults → UPDATE with defaults
        - Duplicates with clear merge strategy → DELETE/UPDATE
        
        Always include:
        - Confidence score (0-1)
        - Whether human approval is required
        - Potential risks
        - Rollback strategy
        
        Be conservative - only suggest fixes you're confident about.
        """
        super().__init__(
            name="auto_fix_agent",
            config=config,
            deployment_name=config.gpt4_deployment,
            model_name="gpt-4",
            system_message=system_message
        )
    
    async def analyze_and_suggest_fixes(self, failures: List[EnhancedDQFailureRecord], 
                                      table_schema: Dict[str, Any]) -> List[AutoFixSuggestion]:
        """Analyze failures and suggest automated fixes."""
        suggestions = []
        
        # Group failures by type for pattern analysis
        failure_groups = self._group_failures_by_pattern(failures)
        
        for pattern, pattern_failures in failure_groups.items():
            try:
                suggestion = await self._analyze_failure_pattern(pattern, pattern_failures, table_schema)
                if suggestion:
                    suggestions.append(suggestion)
            except Exception as e:
                logger.error(f"Failed to analyze pattern {pattern}: {e}")
        
        return suggestions
    
    def _group_failures_by_pattern(self, failures: List[EnhancedDQFailureRecord]) -> Dict[str, List[EnhancedDQFailureRecord]]:
        """Group failures by common patterns."""
        groups = {
            "trailing_spaces": [],
            "null_values": [],
            "format_violations": [],
            "duplicates": [],
            "range_violations": [],
            "other": []
        }
        
        for failure in failures:
            # Simple pattern detection based on failure data
            if self._is_trailing_space_issue(failure):
                groups["trailing_spaces"].append(failure)
            elif self._is_null_issue(failure):
                groups["null_values"].append(failure)
            elif self._is_format_issue(failure):
                groups["format_violations"].append(failure)
            elif self._is_duplicate_issue(failure):
                groups["duplicates"].append(failure)
            elif self._is_range_issue(failure):
                groups["range_violations"].append(failure)
            else:
                groups["other"].append(failure)
        
        return {k: v for k, v in groups.items() if v}  # Remove empty groups
    
    def _is_trailing_space_issue(self, failure: EnhancedDQFailureRecord) -> bool:
        """Check if failure is related to trailing spaces."""
        # Look for patterns in failure data that suggest trailing spaces
        if isinstance(failure.failure_data, dict):
            for value in failure.failure_data.values():
                if isinstance(value, str) and value != value.strip():
                    return True
        return False
    
    def _is_null_issue(self, failure: EnhancedDQFailureRecord) -> bool:
        """Check if failure is related to null values."""
        return "null" in failure.rule_id.lower()
    
    def _is_format_issue(self, failure: EnhancedDQFailureRecord) -> bool:
        """Check if failure is related to format violations."""
        return "format" in failure.rule_id.lower() or "pattern" in failure.rule_id.lower()
    
    def _is_duplicate_issue(self, failure: EnhancedDQFailureRecord) -> bool:
        """Check if failure is related to duplicates."""
        return "unique" in failure.rule_id.lower() or "duplicate" in failure.rule_id.lower()
    
    def _is_range_issue(self, failure: EnhancedDQFailureRecord) -> bool:
        """Check if failure is related to range violations."""
        return "range" in failure.rule_id.lower()
    
    async def _analyze_failure_pattern(self, pattern: str, failures: List[EnhancedDQFailureRecord], 
                                     table_schema: Dict[str, Any]) -> Optional[AutoFixSuggestion]:
        """

