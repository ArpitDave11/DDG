
Proposed enhancement	Why it matters & how to implement agentically
1	Split the “SQL‑writer” responsibility into two agents
▪ RulePlanner Agent (GPT‑4/o4‑mini) – decides what checks are needed, given schema/data profile.
▪ SQLWriter Agent (o3‑mini) – converts each rule into SQL, templated with table & column names.	Separation of planning vs. 
  code‑generation aligns with multi‑agent best practice: a stronger model does the reasoning; a cheaper model turns that plan into concrete code. It also lets you unit‑test the SQL writer in isolation (feed it a synthetic rule → expect SQL).

Version every artefact (rules JSON, SQL, failure JSON) with semantic versioning & a SHA‑256 hash of the schema.

Store rules in a lightweight DSL, but also write in plain English for human to read only.
Example YAML:
yaml\n- col: age\n rule: range\n min: 0\n max: 120\n- col: email\n rule: not_null\n

Add an “Auto‑Fix Agent” (optional human‑in‑the‑loop) that, if the Analysis Agent finds a systematic issue (e.g., trailing spaces), 
can propose a data‐fix SQL or Spark job. Escalate to a human approver.
	Version every artefact (rules JSON, SQL, failure JSON) with semantic versioning & a SHA‑256 hash of the schema.	
Ensures you can trace which rule set was used for a past run and re‑run the exact SQL if needed. Persist metadata section at top of each file: { "dq_version": "1.3.0", "schema_hash": "<…>" }.

Store rules in a lightweight DSL, not plain English.
Example YAML:
yaml\n- col: age\n rule: range\n min: 0\n max: 120\n- col: email\n rule: not_null\n	A structured DSL makes it trivial for agents or non‑AI code to transform into SQL, Spark, Great Expectations, etc. 
The Planner agent emits YAML; the SQLWriter agent renders SQL via Jinja templates.

Add an “Auto‑Fix Agent” (optional human‑in‑the‑loop) that, if the Analysis Agent finds a systematic issue (e.g., trailing spaces), can propose a data‐fix SQL or Spark job. Escalate to a human approver via slack/email.	Turns the pipeline from detect‑only to detect‑and‑suggest‑remediate. 
Keeps remediation suggestions auditable because each fix is an agent message requiring approval.

Maintain conversation memory in a small Vector DB (Chroma / Azure Cognitive Search) keyed by {table, column, rule_signature}.	Lets the Analysis Agent retrieve “what happened last week” in natural language rather than reading JSON blobs. 
AutoGen supports custom “memory” classes—feed the failure summaries as embeddings.

Trend‑aware FailureHistoryManager: compute a sparkline of failure counts (e.g., last 10 runs) and pass it to the Analysis Agent.	Prompts like “Violations of rule X have decreased 
from 15→3 rows over the last 5 runs” allow the LLM to reason about trends (improving vs. regressing).




















##################







# config.py
import os
from dataclasses import dataclass

@dataclass
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI services."""
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    gpt4_deployment: str = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4")
    gpt35_deployment: str = os.getenv("AZURE_OPENAI_GPT35_DEPLOYMENT", "gpt-35-turbo")
    
    def __post_init__(self):
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure OpenAI endpoint and API key must be provided")

@dataclass
class DatabaseConfig:
    """Database configuration for SQL execution."""
    connection_string: str = os.getenv("DATABASE_CONNECTION_STRING", "")
    db_type: str = os.getenv("DATABASE_TYPE", "postgresql")  # postgresql, mysql, sqlite, sqlserver

# models.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json
from datetime import datetime

class RuleType(Enum):
    """Types of data quality rules."""
    NULL_CHECK = "null_check"
    RANGE_CHECK = "range_check"
    FORMAT_CHECK = "format_check"
    UNIQUENESS_CHECK = "uniqueness_check"
    COMPLETENESS_CHECK = "completeness_check"
    CONSISTENCY_CHECK = "consistency_check"

@dataclass
class DataQualityRule:
    """Represents a single data quality rule."""
    rule_id: str
    rule_type: RuleType
    description: str
    column_name: str
    table_name: str
    sql_statement: str
    severity: str = "medium"
    threshold: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type.value,
            "description": self.description,
            "column_name": self.column_name,
            "table_name": self.table_name,
            "sql_statement": self.sql_statement,
            "severity": self.severity,
            "threshold": self.threshold,
            "parameters": self.parameters
        }

@dataclass
class DQFailureRecord:
    """Represents a single DQ failure record."""
    rule_id: str
    column_name: str
    table_name: str
    failure_data: Dict[str, Any]
    timestamp: datetime
    record_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "column_name": self.column_name,
            "table_name": self.table_name,
            "failure_data": self.failure_data,
            "timestamp": self.timestamp.isoformat(),
            "record_count": self.record_count
        }

@dataclass
class DQFailureHistory:
    """Holds the conversation context and failure history."""
    table_name: str
    failures: List[DQFailureRecord] = field(default_factory=list)
    last_check_timestamp: Optional[datetime] = None
    conversation_context: Dict[str, Any] = field(default_factory=dict)
    
    def add_failure(self, failure_record: DQFailureRecord):
        """Add a new failure record to history."""
        self.failures.append(failure_record)
        self.last_check_timestamp = datetime.now()
    
    def get_previous_failures(self, rule_id: str) -> List[DQFailureRecord]:
        """Get previous failures for a specific rule."""
        return [f for f in self.failures if f.rule_id == rule_id]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "failures": [f.to_dict() for f in self.failures],
            "last_check_timestamp": self.last_check_timestamp.isoformat() if self.last_check_timestamp else None,
            "conversation_context": self.conversation_context
        }

# sql_writer_agent.py
import json
from typing import Dict, List
from base_agent import BaseAgent
from models import ColumnProfile, DataQualityRule, RuleType
from config import AzureOpenAIConfig

class SQLWriterAgent(BaseAgent):
    """Agent responsible for writing SQL statements for data quality checks."""
    
    def __init__(self, config: AzureOpenAIConfig):
        system_message = """
        You are a SQL expert specialized in writing data quality check queries.
        Generate SQL statements that identify records violating data quality rules.
        
        For each rule type, write queries that:
        - null_check: Find records where column IS NULL
        - range_check: Find records outside specified ranges
        - format_check: Find records not matching patterns
        - uniqueness_check: Find duplicate records
        - completeness_check: Find empty or null records
        - consistency_check: Find records violating business rules
        
        Return only valid JSON with SQL statements.
        """
        super().__init__(
            name="sql_writer_agent",
            config=config,
            deployment_name=config.gpt4_deployment,
            model_name="gpt-4",
            system_message=system_message
        )
    
    async def process(self, table_name: str, rules: List[DataQualityRule]) -> Dict[str, str]:
        """Generate SQL statements for all rules of a table."""
        rules_data = [rule.to_dict() for rule in rules]
        
        prompt = f"""
        Generate SQL statements for data quality checks on table: {table_name}
        
        Rules to implement:
        {json.dumps(rules_data, indent=2)}
        
        For each rule, write a SQL query that returns records violating the rule.
        Return JSON format:
        {{
            "rule_id_1": "SELECT * FROM {table_name} WHERE condition_for_violation",
            "rule_id_2": "SELECT * FROM {table_name} WHERE condition_for_violation",
            ...
        }}
        
        Ensure queries are optimized and return only violating records.
        """
        
        response = await self.run_with_retry(prompt)
        
        try:
            sql_statements = json.loads(response)
            return sql_statements
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse SQL statements: {e}")
            return {}

# dq_rule_generator.py
import json
import os
from typing import Dict, List
from sql_writer_agent import SQLWriterAgent
from models import DataQualityRule, RuleType
from config import AzureOpenAIConfig

class DQRuleGenerator:
    """Generates and saves DQ rules with SQL statements."""
    
    def __init__(self, config: AzureOpenAIConfig, output_dir: str = "dq_rules"):
        self.sql_writer = SQLWriterAgent(config)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    async def generate_table_rules(self, table_name: str, profiles: Dict[str, ColumnProfile]) -> Dict[str, List[DataQualityRule]]:
        """Generate rules for a table and save as JSON and SQL files."""
        all_rules = {}
        
        for column_name, profile in profiles.items():
            # Generate basic rules based on column profile
            column_rules = self._generate_column_rules(table_name, column_name, profile)
            all_rules[column_name] = column_rules
        
        # Flatten rules for SQL generation
        flat_rules = []
        for column_rules in all_rules.values():
            flat_rules.extend(column_rules)
        
        # Generate SQL statements
        sql_statements = await self.sql_writer.process(table_name, flat_rules)
        
        # Update rules with SQL statements
        for column_rules in all_rules.values():
            for rule in column_rules:
                if rule.rule_id in sql_statements:
                    rule.sql_statement = sql_statements[rule.rule_id]
        
        # Save rules and SQL
        await self._save_rules(table_name, all_rules, sql_statements)
        
        return all_rules
    
    def _generate_column_rules(self, table_name: str, column_name: str, profile: ColumnProfile) -> List[DataQualityRule]:
        """Generate basic rules for a column based on its profile."""
        rules = []
        
        # Null check rule
        if profile.null_count > 0:
            rules.append(DataQualityRule(
                rule_id=f"{table_name}_{column_name}_null_check",
                rule_type=RuleType.NULL_CHECK,
                description=f"Check for null values in {column_name}",
                column_name=column_name,
                table_name=table_name,
                sql_statement="",  # Will be populated by SQL writer
                severity="high" if profile.null_count / profile.total_rows > 0.1 else "medium"
            ))
        
        # Range check for numeric columns
        if profile.min_value is not None and profile.max_value is not None:
            rules.append(DataQualityRule(
                rule_id=f"{table_name}_{column_name}_range_check",
                rule_type=RuleType.RANGE_CHECK,
                description=f"Check value range for {column_name}",
                column_name=column_name,
                table_name=table_name,
                sql_statement="",
                parameters={
                    "min_value": profile.min_value,
                    "max_value": profile.max_value
                }
            ))
        
        # Uniqueness check if high distinct count
        if profile.distinct_count == profile.total_rows:
            rules.append(DataQualityRule(
                rule_id=f"{table_name}_{column_name}_uniqueness_check",
                rule_type=RuleType.UNIQUENESS_CHECK,
                description=f"Check for duplicate values in {column_name}",
                column_name=column_name,
                table_name=table_name,
                sql_statement="",
                severity="critical"
            ))
        
        return rules
    
    async def _save_rules(self, table_name: str, rules: Dict[str, List[DataQualityRule]], sql_statements: Dict[str, str]):
        """Save rules as JSON and SQL files."""
        # Save rules JSON
        rules_dict = {
            table_name: {
                column: [rule.to_dict() for rule in column_rules]
                for column, column_rules in rules.items()
            }
        }
        
        json_file = os.path.join(self.output_dir, f"{table_name}_dq_rules.json")
        with open(json_file, 'w') as f:
            json.dump(rules_dict, f, indent=2)
        
        # Save SQL statements
        sql_file = os.path.join(self.output_dir, f"{table_name}_DQ_check.sql")
        with open(sql_file, 'w') as f:
            f.write(f"-- Data Quality Check SQL for {table_name}\n")
            f.write(f"-- Generated on: {datetime.now().isoformat()}\n\n")
            
            for rule_id, sql in sql_statements.items():
                f.write(f"-- Rule: {rule_id}\n")
                f.write(f"{sql};\n\n")

# failure_history_manager.py
import json
import os
from typing import Dict, Optional
from datetime import datetime
from models import DQFailureHistory, DQFailureRecord

class FailureHistoryManager:
    """Manages DQ failure history persistence and retrieval."""
    
    def __init__(self, history_dir: str = "dq_history"):
        self.history_dir = history_dir
        os.makedirs(history_dir, exist_ok=True)
    
    def load_failure_history(self, table_name: str) -> DQFailureHistory:
        """Load failure history for a table."""
        history_file = os.path.join(self.history_dir, f"DQ_failureHistory_{table_name}.json")
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                
                history = DQFailureHistory(table_name=table_name)
                history.last_check_timestamp = datetime.fromisoformat(data['last_check_timestamp']) if data.get('last_check_timestamp') else None
                history.conversation_context = data.get('conversation_context', {})
                
                for failure_data in data.get('failures', []):
                    failure_record = DQFailureRecord(
                        rule_id=failure_data['rule_id'],
                        column_name=failure_data['column_name'],
                        table_name=failure_data['table_name'],
                        failure_data=failure_data['failure_data'],
                        timestamp=datetime.fromisoformat(failure_data['timestamp']),
                        record_count=failure_data['record_count']
                    )
                    history.failures.append(failure_record)
                
                return history
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to load history for {table_name}: {e}")
        
        return DQFailureHistory(table_name=table_name)
    
    def save_failure_history(self, history: DQFailureHistory):
        """Save failure history for a table."""
        history_file = os.path.join(self.history_dir, f"DQ_failureHistory_{history.table_name}.json")
        
        with open(history_file, 'w') as f:
            json.dump(history.to_dict(), f, indent=2)
    
    def update_conversation_context(self, table_name: str, context: Dict[str, Any]):
        """Update conversation context for a table."""
        history = self.load_failure_history(table_name)
        history.conversation_context.update(context)
        self.save_failure_history(history)

# sql_executor.py
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
from models import DataQualityRule, DQFailureRecord, DQFailureHistory
from failure_history_manager import FailureHistoryManager

logger = logging.getLogger(__name__)

class SQLExecutor:
    """Executes SQL-based data quality checks."""
    
    def __init__(self, connection_string: str, history_manager: FailureHistoryManager):
        self.connection_string = connection_string
        self.history_manager = history_manager
    
    def execute_dq_rules(self, table_name: str, rules: List[DataQualityRule]) -> DQFailureHistory:
        """Execute DQ rules and update failure history."""
        # Load existing failure history
        history = self.history_manager.load_failure_history(table_name)
        
        logger.info(f"Executing {len(rules)} DQ rules for table {table_name}")
        logger.info(f"Previous failures: {len(history.failures)}")
        
        new_failures = []
        
        for rule in rules:
            try:
                # Execute SQL to find violations
                violation_df = self._execute_sql(rule.sql_statement)
                
                if not violation_df.empty:
                    # Create failure record
                    failure_record = DQFailureRecord(
                        rule_id=rule.rule_id,
                        column_name=rule.column_name,
                        table_name=rule.table_name,
                        failure_data=violation_df.to_dict('records'),
                        timestamp=datetime.now(),
                        record_count=len(violation_df)
                    )
                    
                    new_failures.append(failure_record)
                    history.add_failure(failure_record)
                    
                    logger.warning(f"Rule {rule.rule_id} failed: {len(violation_df)} violations")
                else:
                    logger.info(f"Rule {rule.rule_id} passed: No violations")
                    
            except Exception as e:
                logger.error(f"Failed to execute rule {rule.rule_id}: {e}")
                # Create error record
                error_record = DQFailureRecord(
                    rule_id=rule.rule_id,
                    column_name=rule.column_name,
                    table_name=rule.table_name,
                    failure_data={"error": str(e)},
                    timestamp=datetime.now(),
                    record_count=0
                )
                new_failures.append(error_record)
                history.add_failure(error_record)
        
        # Save updated history
        self.history_manager.save_failure_history(history)
        
        logger.info(f"DQ check completed. New failures: {len(new_failures)}")
        return history
    
    def _execute_sql(self, sql_statement: str) -> pd.DataFrame:
        """Execute SQL statement and return results as DataFrame."""
        # This is a simplified implementation
        # In production, you'd use appropriate database connectors
        try:
            import sqlite3
            conn = sqlite3.connect(self.connection_string)
            df = pd.read_sql_query(sql_statement, conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return pd.DataFrame()
    
    def get_previous_failures_context(self, table_name: str, rule_id: str) -> Dict[str, Any]:
        """Get context about previous failures for a specific rule."""
        history = self.history_manager.load_failure_history(table_name)
        previous_failures = history.get_previous_failures(rule_id)
        
        if not previous_failures:
            return {"message": "No previous failures found for this rule"}
        
        return {
            "rule_id": rule_id,
            "total_previous_failures": len(previous_failures),
            "latest_failure": previous_failures[-1].to_dict() if previous_failures else None,
            "failure_trend": self._analyze_failure_trend(previous_failures)
        }
    
    def _analyze_failure_trend(self, failures: List[DQFailureRecord]) -> Dict[str, Any]:
        """Analyze trend in failures over time."""
        if len(failures) < 2:
            return {"trend": "insufficient_data"}
        
        # Simple trend analysis
        recent_count = failures[-1].record_count
        previous_count = failures[-2].record_count
        
        if recent_count > previous_count:
            trend = "increasing"
        elif recent_count < previous_count:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_count": recent_count,
            "previous_count": previous_count,
            "change_percentage": ((recent_count - previous_count) / previous_count * 100) if previous_count > 0 else 0
        }

# enhanced_pipeline.py
import asyncio
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from config import AzureOpenAIConfig, DatabaseConfig
from profiler import DataProfiler
from dq_rule_generator import DQRuleGenerator
from sql_executor import SQLExecutor
from failure_history_manager import FailureHistoryManager
from analysis_agent import AnalysisAgent

logger = logging.getLogger(__name__)

class EnhancedDataQualityPipeline:
    """Enhanced DQ pipeline with failure history and SQL-based checks."""
    
    def __init__(self, azure_config: AzureOpenAIConfig, db_config: DatabaseConfig):
        self.azure_config = azure_config
        self.db_config = db_config
        self.profiler = DataProfiler()
        self.rule_generator = DQRuleGenerator(azure_config)
        self.history_manager = FailureHistoryManager()
        self.sql_executor = SQLExecutor(db_config.connection_string, self.history_manager)
        self.analysis_agent = AnalysisAgent(azure_config)
    
    async def run_dq_pipeline(self, table_name: str, df: pd.DataFrame, 
                             columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the enhanced DQ pipeline with failure history."""
        if columns is None:
            columns = df.columns.tolist()
        
        logger.info(f"Starting enhanced DQ pipeline for table: {table_name}")
        
        # Step 1: Profile data
        logger.info("Step 1: Profiling data")
        profiles = {col: self.profiler.profile_column(df[col], col) for col in columns}
        
        # Step 2: Generate rules and SQL
        logger.info("Step 2: Generating DQ rules and SQL statements")
        table_rules = await self.rule_generator.generate_table_rules(table_name, profiles)
        
        # Flatten rules for execution
        all_rules = []
        for column_rules in table_rules.values():
            all_rules.extend(column_rules)
        
        # Step 3: Check previous failures
        logger.info("Step 3: Checking previous failure history")
        previous_context = {}
        for rule in all_rules:
            context = self.sql_executor.get_previous_failures_context(table_name, rule.rule_id)
            previous_context[rule.rule_id] = context
        
        # Step 4: Execute DQ rules
        logger.info("Step 4: Executing DQ rules")
        failure_history = self.sql_executor.execute_dq_rules(table_name, all_rules)
        
        # Step 5: Analyze failures with context
        logger.info("Step 5: Analyzing failures with historical context")
        analysis_results = {}
        
        if failure_history.failures:
            # Group failures by rule
            failure_groups = {}
            for failure in failure_history.failures:
                if failure.rule_id not in failure_groups:
                    failure_groups[failure.rule_id] = []
                failure_groups[failure.rule_id].append(failure)
            
            # Analyze each group
            for rule_id, failures in failure_groups.items():
                context = {
                    "rule_id": rule_id,
                    "current_failures": [f.to_dict() for f in failures],
                    "previous_context": previous_context.get(rule_id, {}),
                    "conversation_context": failure_history.conversation_context
                }
                
                # Get analysis from AI agent
                analysis = await self._get_failure_analysis(context)
                analysis_results[rule_id] = analysis
        
        # Step 6: Update conversation context
        self.history_manager.update_conversation_context(table_name, {
            "last_pipeline_run": datetime.now().isoformat(),
            "rules_executed": len(all_rules),
            "failures_found": len(failure_history.failures),
            "analysis_results": analysis_results
        })
        
        # Compile final report
        report = self._compile_enhanced_report(
            table_name, profiles, table_rules, failure_history, 
            previous_context, analysis_results
        )
        
        logger.info("Enhanced DQ pipeline completed successfully")
        return report
    
    async def _get_failure_analysis(self, context: Dict[str, Any]) -> str:
        """Get AI analysis for failures with historical context."""
        prompt = f"""
        Analyze these data quality failures with historical context:
        
        {json.dumps(context, indent=2)}
        
        Provide insights considering:
        1. Current failure patterns
        2. Historical trends
        3. Previous conversation context
        4. Root cause analysis
        5. Remediation recommendations
        
        Focus on actionable insights and trend analysis.
        """
        
        try:
            analysis = await self.analysis_agent.run_with_retry(prompt)
            return analysis
        except Exception as e:
            logger.error(f"Failed to get analysis: {e}")
            return f"Analysis failed: {str(e)}"
    
    def _compile_enhanced_report(self, table_name: str, profiles: Dict[str, ColumnProfile],
                               rules: Dict[str, List[DataQualityRule]], 
                               failure_history: DQFailureHistory,
                               previous_context: Dict[str, Any],
                               analysis_results: Dict[str, str]) -> Dict[str, Any]:
        """Compile the enhanced pipeline report."""
        return {
            "table_name": table_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_columns": len(profiles),
                "total_rules": sum(len(rule_list) for rule_list in rules.values()),
                "total_failures": len(failure_history.failures),
                "new_failures": len([f for f in failure_history.failures 
                                   if f.timestamp.date() == datetime.now().date()]),
                "success_rate": self._calculate_success_rate(rules, failure_history)
            },
            "column_profiles": {col: profile.to_dict() for col, profile in profiles.items()},
            "generated_rules": {
                col: [rule.to_dict() for rule in rule_list] 
                for col, rule_list in rules.items()
            },
            "failure_history": failure_history.to_dict(),
            "previous_context": previous_context,
            "failure_analysis": analysis_results,
            "recommendations": self._generate_recommendations(failure_history, analysis_results)
        }
    
    def _calculate_success_rate(self, rules: Dict[str, List[DataQualityRule]], 
                              failure_history: DQFailureHistory) -> float:
        """Calculate overall success rate."""
        total_rules = sum(len(rule_list) for rule_list in rules.values())
        failed_rules = len(set(f.rule_id for f in failure_history.failures))
        return ((total_rules - failed_rules) / total_rules * 100) if total_rules > 0 else 0
    
    def _generate_recommendations(self, failure_history: DQFailureHistory, 
                                analysis_results: Dict[str, str]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if failure_history.failures:
            # High-level recommendations based on failure patterns
            critical_failures = [f for f in failure_history.failures 
                               if any(rule.severity == "critical" for rule in self._get_rules_by_id(f.rule_id))]
            
            if critical_failures:
                recommendations.append("Address critical data quality issues immediately")
            
            if len(failure_history.failures) > 10:
                recommendations.append("Consider implementing data validation at source")
            
            # Add AI-generated recommendations
            for rule_id, analysis in analysis_results.items():
                if "recommend" in analysis.lower():
                    recommendations.append(f"Rule {rule_id}: {analysis}")
        
        return recommendations
    
    def _get_rules_by_id(self, rule_id: str) -> List[DataQualityRule]:
        """Helper to get rules by ID - simplified for this example."""
        return []

# main.py
import asyncio
import logging
from config import AzureOpenAIConfig, DatabaseConfig
from enhanced_pipeline import EnhancedDataQualityPipeline
import pandas as pd

logging.basicConfig(level=logging.INFO)

async def main():
    """Main function to run the enhanced DQ pipeline."""
    # Load configurations
    azure_config = AzureOpenAIConfig()
    db_config = DatabaseConfig()
    
    # Load your data
    df = pd.read_csv("your_dataset.csv")
    table_name = "your_table_name"
    
    # Initialize and run pipeline
    pipeline = EnhancedDataQualityPipeline(azure_config, db_config)
    
    try:
        report = await pipeline.run_dq_pipeline(table_name, df)
        
        print("\n" + "="*60)
        print("ENHANCED DATA QUALITY PIPELINE REPORT")
        print("="*60)
        print(f"Table: {report['table_name']}")
        print(f"Timestamp: {report['timestamp']}")
        print(f"Total Rules: {report['summary']['total_rules']}")
        print(f"Total Failures: {report['summary']['total_failures']}")
        print(f"New Failures: {report['summary']['new_failures']}")
        print(f"Success Rate: {report['summary']['success_rate']:.2f}%")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        return report
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
