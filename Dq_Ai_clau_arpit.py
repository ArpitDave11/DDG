
# config.py
import os
from dataclasses import dataclass
from typing import Optional

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

# models.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json

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
    expression: str
    severity: str = "medium"  # low, medium, high, critical
    threshold: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type.value,
            "description": self.description,
            "expression": self.expression,
            "severity": self.severity,
            "threshold": self.threshold,
            "parameters": self.parameters
        }

@dataclass
class ColumnProfile:
    """Profile information for a column."""
    name: str
    dtype: str
    total_rows: int
    null_count: int
    distinct_count: int
    sample_values: List[str]
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean_value: Optional[float] = None
    std_dev: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "total_rows": self.total_rows,
            "null_count": self.null_count,
            "distinct_count": self.distinct_count,
            "sample_values": self.sample_values,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": self.mean_value,
            "std_dev": self.std_dev
        }

@dataclass
class RuleExecutionResult:
    """Result of executing a data quality rule."""
    rule_id: str
    column_name: str
    passed: bool
    violation_count: int
    total_records: int
    violation_percentage: float
    details: str
    sample_violations: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "column_name": self.column_name,
            "passed": self.passed,
            "violation_count": self.violation_count,
            "total_records": self.total_records,
            "violation_percentage": self.violation_percentage,
            "details": self.details,
            "sample_violations": self.sample_violations
        }

# profiler.py
import pandas as pd
import numpy as np
from typing import Dict, List
from models import ColumnProfile

class DataProfiler:
    """Handles data profiling for columns."""
    
    def __init__(self, sample_size: int = 1000):
        self.sample_size = sample_size
    
    def profile_column(self, data: pd.Series, column_name: str) -> ColumnProfile:
        """Create a comprehensive profile for a column."""
        # Basic statistics
        total_rows = len(data)
        null_count = int(data.isna().sum())
        distinct_count = int(data.nunique(dropna=True))
        
        # Sample values
        sample_values = data.dropna().unique()[:5].tolist()
        sample_values = [str(val) for val in sample_values]
        
        # Numeric-specific statistics
        min_value = None
        max_value = None
        mean_value = None
        std_dev = None
        
        if pd.api.types.is_numeric_dtype(data):
            try:
                min_value = float(data.min()) if not data.empty else None
                max_value = float(data.max()) if not data.empty else None
                mean_value = float(data.mean()) if not data.empty else None
                std_dev = float(data.std()) if not data.empty else None
            except (ValueError, TypeError):
                pass
        
        return ColumnProfile(
            name=column_name,
            dtype=str(data.dtype),
            total_rows=total_rows,
            null_count=null_count,
            distinct_count=distinct_count,
            sample_values=sample_values,
            min_value=min_value,
            max_value=max_value,
            mean_value=mean_value,
            std_dev=std_dev
        )
    
    def profile_dataframe(self, df: pd.DataFrame) -> Dict[str, ColumnProfile]:
        """Profile all columns in a DataFrame."""
        profiles = {}
        for column in df.columns:
            profiles[column] = self.profile_column(df[column], column)
        return profiles

# base_agent.py
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from config import AzureOpenAIConfig

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all agents in the pipeline."""
    
    def __init__(self, name: str, config: AzureOpenAIConfig, deployment_name: str, 
                 model_name: str, system_message: str, temperature: float = 0.1):
        self.name = name
        self.config = config
        self.deployment_name = deployment_name
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAIChatCompletionClient(
            azure_endpoint=config.endpoint,
            azure_deployment=deployment_name,
            model=model_name,
            api_version=config.api_version,
            azure_api_key=config.api_key
        )
        
        # Create AssistantAgent
        self.agent = AssistantAgent(
            name=name,
            model_client=self.client,
            system_message=system_message
        )
    
    async def run_with_retry(self, task: str, max_retries: int = 3) -> str:
        """Run agent with retry logic."""
        for attempt in range(max_retries):
            try:
                result = await self.agent.run(task=task)
                assistant_messages = [m for m in result.messages if m.source == "assistant"]
                if not assistant_messages:
                    raise RuntimeError(f"No response from {self.name}")
                return assistant_messages[-1].content.strip()
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {self.name}: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    @abstractmethod
    async def process(self, *args, **kwargs) -> Any:
        """Process the agent's specific task."""
        pass

# recommendation_agent.py
import json
from typing import List, Dict, Any
from base_agent import BaseAgent
from models import ColumnProfile, DataQualityRule, RuleType
from config import AzureOpenAIConfig

class RecommendationAgent(BaseAgent):
    """Agent responsible for recommending data quality rules."""
    
    def __init__(self, config: AzureOpenAIConfig):
        system_message = """
        You are an expert data quality analyst. Generate comprehensive data quality rules 
        for columns based on their profiles. Return only valid JSON with the following structure:
        {
            "rules": [
                {
                    "rule_id": "unique_identifier",
                    "rule_type": "null_check|range_check|format_check|uniqueness_check|completeness_check|consistency_check",
                    "description": "Human-readable description",
                    "expression": "Executable rule expression",
                    "severity": "low|medium|high|critical",
                    "threshold": null_or_numeric_value,
                    "parameters": {}
                }
            ]
        }
        Focus on practical, actionable rules that can be executed programmatically.
        """
        super().__init__(
            name="recommendation_agent",
            config=config,
            deployment_name=config.gpt4_deployment,
            model_name="gpt-4",
            system_message=system_message
        )
    
    async def process(self, profile: ColumnProfile) -> List[DataQualityRule]:
        """Generate rules for a column profile."""
        prompt = f"""
        Analyze this column profile and suggest 3-5 data quality rules:
        
        Column Profile:
        {json.dumps(profile.to_dict(), indent=2)}
        
        Consider the data type, null patterns, value ranges, and distributions.
        Generate rules that are:
        1. Specific to this column's characteristics
        2. Executable programmatically
        3. Appropriate for the data type and distribution
        4. Ranked by severity (critical issues first)
        
        Return only the JSON response.
        """
        
        response = await self.run_with_retry(prompt)
        
        try:
            parsed = json.loads(response)
            rules = []
            
            for rule_data in parsed.get("rules", []):
                rule = DataQualityRule(
                    rule_id=rule_data["rule_id"],
                    rule_type=RuleType(rule_data["rule_type"]),
                    description=rule_data["description"],
                    expression=rule_data["expression"],
                    severity=rule_data.get("severity", "medium"),
                    threshold=rule_data.get("threshold"),
                    parameters=rule_data.get("parameters", {})
                )
                rules.append(rule)
            
            return rules
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse recommendation response: {e}")
            return []

# review_agent.py
import json
from typing import List
from base_agent import BaseAgent
from models import ColumnProfile, DataQualityRule, RuleType
from config import AzureOpenAIConfig

class ReviewAgent(BaseAgent):
    """Agent responsible for reviewing and refining data quality rules."""
    
    def __init__(self, config: AzureOpenAIConfig):
        system_message = """
        You are a data quality rule reviewer. Review proposed rules for correctness,
        relevance, and completeness. Return only valid JSON with the same structure
        as the input, containing the refined rules. Remove inappropriate rules,
        adjust thresholds, and add missing critical rules.
        """
        super().__init__(
            name="review_agent",
            config=config,
            deployment_name=config.gpt35_deployment,
            model_name="gpt-35-turbo",
            system_message=system_message
        )
    
    async def process(self, profile: ColumnProfile, initial_rules: List[DataQualityRule]) -> List[DataQualityRule]:
        """Review and refine rules for a column."""
        rules_dict = {"rules": [rule.to_dict() for rule in initial_rules]}
        
        prompt = f"""
        Review these data quality rules for the column:
        
        Column Profile:
        {json.dumps(profile.to_dict(), indent=2)}
        
        Proposed Rules:
        {json.dumps(rules_dict, indent=2)}
        
        Review each rule for:
        1. Appropriateness for this column type and data distribution
        2. Correct thresholds and parameters
        3. Missing critical rules
        4. Redundant or conflicting rules
        
        Return the refined rules in the same JSON format.
        """
        
        response = await self.run_with_retry(prompt)
        
        try:
            parsed = json.loads(response)
            refined_rules = []
            
            for rule_data in parsed.get("rules", []):
                rule = DataQualityRule(
                    rule_id=rule_data["rule_id"],
                    rule_type=RuleType(rule_data["rule_type"]),
                    description=rule_data["description"],
                    expression=rule_data["expression"],
                    severity=rule_data.get("severity", "medium"),
                    threshold=rule_data.get("threshold"),
                    parameters=rule_data.get("parameters", {})
                )
                refined_rules.append(rule)
            
            return refined_rules
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse review response: {e}")
            return initial_rules  # Return original rules if parsing fails

# rule_executor.py
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any
from models import DataQualityRule, RuleExecutionResult, RuleType

class RuleExecutor:
    """Executes data quality rules on DataFrames."""
    
    def __init__(self, max_sample_violations: int = 10):
        self.max_sample_violations = max_sample_violations
    
    def execute_rule(self, df: pd.DataFrame, column_name: str, rule: DataQualityRule) -> RuleExecutionResult:
        """Execute a single rule on a column."""
        column_data = df[column_name]
        total_records = len(column_data)
        
        try:
            violations = self._evaluate_rule(column_data, rule)
            violation_count = len(violations) if isinstance(violations, list) else int(violations.sum())
            violation_percentage = (violation_count / total_records) * 100 if total_records > 0 else 0
            
            # Check if rule passed based on threshold
            passed = self._determine_pass_status(violation_percentage, rule)
            
            # Sample violations for analysis
            sample_violations = []
            if isinstance(violations, list):
                sample_violations = violations[:self.max_sample_violations]
            elif hasattr(violations, 'index'):
                violation_indices = violations[violations].index[:self.max_sample_violations]
                sample_violations = column_data.iloc[violation_indices].tolist()
            
            details = self._generate_details(rule, violation_count, total_records, violation_percentage)
            
            return RuleExecutionResult(
                rule_id=rule.rule_id,
                column_name=column_name,
                passed=passed,
                violation_count=violation_count,
                total_records=total_records,
                violation_percentage=violation_percentage,
                details=details,
                sample_violations=sample_violations
            )
            
        except Exception as e:
            return RuleExecutionResult(
                rule_id=rule.rule_id,
                column_name=column_name,
                passed=False,
                violation_count=0,
                total_records=total_records,
                violation_percentage=0,
                details=f"Rule execution failed: {str(e)}",
                sample_violations=[]
            )
    
    def _evaluate_rule(self, data: pd.Series, rule: DataQualityRule) -> Any:
        """Evaluate a rule against column data."""
        if rule.rule_type == RuleType.NULL_CHECK:
            return data.isna()
        
        elif rule.rule_type == RuleType.RANGE_CHECK:
            min_val = rule.parameters.get('min_value')
            max_val = rule.parameters.get('max_value')
            violations = pd.Series([False] * len(data))
            
            if min_val is not None:
                violations |= (data < min_val)
            if max_val is not None:
                violations |= (data > max_val)
            
            return violations
        
        elif rule.rule_type == RuleType.FORMAT_CHECK:
            pattern = rule.parameters.get('pattern')
            if pattern:
                return ~data.astype(str).str.match(pattern, na=False)
            return pd.Series([False] * len(data))
        
        elif rule.rule_type == RuleType.UNIQUENESS_CHECK:
            return data.duplicated(keep=False)
        
        elif rule.rule_type == RuleType.COMPLETENESS_CHECK:
            return data.isna() | (data.astype(str).str.strip() == '')
        
        elif rule.rule_type == RuleType.CONSISTENCY_CHECK:
            # Custom consistency checks would be implemented here
            return pd.Series([False] * len(data))
        
        else:
            raise ValueError(f"Unknown rule type: {rule.rule_type}")
    
    def _determine_pass_status(self, violation_percentage: float, rule: DataQualityRule) -> bool:
        """Determine if a rule passed based on violation percentage and threshold."""
        if rule.threshold is None:
            return violation_percentage == 0  # No violations allowed
        return violation_percentage <= rule.threshold
    
    def _generate_details(self, rule: DataQualityRule, violation_count: int, 
                         total_records: int, violation_percentage: float) -> str:
        """Generate details string for rule execution result."""
        return (f"Rule '{rule.description}' found {violation_count} violations "
                f"out of {total_records} records ({violation_percentage:.2f}%)")
    
    def execute_all_rules(self, df: pd.DataFrame, 
                         rules_by_column: Dict[str, List[DataQualityRule]]) -> Dict[str, List[RuleExecutionResult]]:
        """Execute all rules for all columns."""
        results = {}
        
        for column_name, rules in rules_by_column.items():
            if column_name not in df.columns:
                continue
                
            column_results = []
            for rule in rules:
                result = self.execute_rule(df, column_name, rule)
                column_results.append(result)
            
            results[column_name] = column_results
        
        return results

# analysis_agent.py
import json
from typing import List
from base_agent import BaseAgent
from models import RuleExecutionResult, ColumnProfile
from config import AzureOpenAIConfig

class AnalysisAgent(BaseAgent):
    """Agent responsible for analyzing failed data quality rules."""
    
    def __init__(self, config: AzureOpenAIConfig):
        system_message = """
        You are a data quality analyst. Analyze failed data quality rules and provide
        insights about root causes, patterns, and remediation suggestions.
        Provide concise, actionable analysis in a structured format.
        """
        super().__init__(
            name="analysis_agent",
            config=config,
            deployment_name=config.gpt4_deployment,
            model_name="gpt-4",
            system_message=system_message
        )
    
    async def process(self, profile: ColumnProfile, failed_results: List[RuleExecutionResult]) -> Dict[str, str]:
        """Analyze failed rules and provide insights."""
        if not failed_results:
            return {}
        
        analysis_data = {
            "column_profile": profile.to_dict(),
            "failed_rules": [result.to_dict() for result in failed_results]
        }
        
        prompt = f"""
        Analyze these failed data quality rules:
        
        {json.dumps(analysis_data, indent=2)}
        
        For each failed rule, provide:
        1. Root cause analysis
        2. Patterns in the violations
        3. Remediation suggestions
        4. Impact assessment
        
        Focus on actionable insights that can help improve data quality.
        """
        
        response = await self.run_with_retry(prompt)
        
        # Create analysis dictionary keyed by rule_id
        analyses = {}
        for result in failed_results:
            analyses[result.rule_id] = response
        
        return analyses

# pipeline.py
import asyncio
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from config import AzureOpenAIConfig
from profiler import DataProfiler
from recommendation_agent import RecommendationAgent
from review_agent import ReviewAgent
from rule_executor import RuleExecutor
from analysis_agent import AnalysisAgent
from models import DataQualityRule, RuleExecutionResult, ColumnProfile

logger = logging.getLogger(__name__)

class DataQualityPipeline:
    """Main orchestrator for the multi-agent data quality pipeline."""
    
    def __init__(self, config: AzureOpenAIConfig):
        self.config = config
        self.profiler = DataProfiler()
        self.recommendation_agent = RecommendationAgent(config)
        self.review_agent = ReviewAgent(config)
        self.rule_executor = RuleExecutor()
        self.analysis_agent = AnalysisAgent(config)
    
    async def run(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the complete data quality pipeline."""
        if columns is None:
            columns = df.columns.tolist()
        
        logger.info(f"Starting data quality pipeline for {len(columns)} columns")
        
        # Step 1: Profile data
        logger.info("Step 1: Profiling data")
        profiles = {col: self.profiler.profile_column(df[col], col) for col in columns}
        
        # Step 2: Generate and review rules
        logger.info("Step 2: Generating and reviewing rules")
        all_rules = {}
        
        # Process columns concurrently
        tasks = []
        for column in columns:
            task = self._process_column_rules(profiles[column])
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            column = columns[i]
            if isinstance(result, Exception):
                logger.error(f"Error processing column {column}: {result}")
                all_rules[column] = []
            else:
                all_rules[column] = result
        
        # Step 3: Execute rules
        logger.info("Step 3: Executing rules")
        execution_results = self.rule_executor.execute_all_rules(df, all_rules)
        
        # Step 4: Analyze failures
        logger.info("Step 4: Analyzing failures")
        analysis_results = {}
        
        analysis_tasks = []
        for column in columns:
            failed_results = [r for r in execution_results.get(column, []) if not r.passed]
            if failed_results:
                task = self._analyze_failures(profiles[column], failed_results)
                analysis_tasks.append((column, task))
        
        if analysis_tasks:
            analysis_responses = await asyncio.gather(
                *[task for _, task in analysis_tasks], 
                return_exceptions=True
            )
            
            for i, (column, _) in enumerate(analysis_tasks):
                response = analysis_responses[i]
                if isinstance(response, Exception):
                    logger.error(f"Error analyzing failures for column {column}: {response}")
                    analysis_results[column] = {}
                else:
                    analysis_results[column] = response
        
        # Compile final report
        report = self._compile_report(profiles, all_rules, execution_results, analysis_results)
        
        logger.info("Pipeline completed successfully")
        return report
    
    async def _process_column_rules(self, profile: ColumnProfile) -> List[DataQualityRule]:
        """Process rules for a single column."""
        try:
            # Generate initial rules
            initial_rules = await self.recommendation_agent.process(profile)
            
            if not initial_rules:
                logger.warning(f"No rules generated for column {profile.name}")
                return []
            
            # Review and refine rules
            final_rules = await self.review_agent.process(profile, initial_rules)
            
            logger.info(f"Generated {len(final_rules)} rules for column {profile.name}")
            return final_rules
            
        except Exception as e:
            logger.error(f"Error processing rules for column {profile.name}: {e}")
            return []
    
    async def _analyze_failures(self, profile: ColumnProfile, 
                              failed_results: List[RuleExecutionResult]) -> Dict[str, str]:
        """Analyze failures for a column."""
        try:
            return await self.analysis_agent.process(profile, failed_results)
        except Exception as e:
            logger.error(f"Error analyzing failures for column {profile.name}: {e}")
            return {}
    
    def _compile_report(self, profiles: Dict[str, ColumnProfile],
                       rules: Dict[str, List[DataQualityRule]],
                       execution_results: Dict[str, List[RuleExecutionResult]],
                       analysis_results: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """Compile the final pipeline report."""
        total_rules = sum(len(rule_list) for rule_list in rules.values())
        total_failed = sum(
            len([r for r in results if not r.passed])
            for results in execution_results.values()
        )
        
        return {
            "summary": {
                "total_columns": len(profiles),
                "total_rules": total_rules,
                "total_failed_rules": total_failed,
                "success_rate": ((total_rules - total_failed) / total_rules * 100) if total_rules > 0 else 0
            },
            "column_profiles": {col: profile.to_dict() for col, profile in profiles.items()},
            "generated_rules": {
                col: [rule.to_dict() for rule in rule_list] 
                for col, rule_list in rules.items()
            },
            "execution_results": {
                col: [result.to_dict() for result in results]
                for col, results in execution_results.items()
            },
            "failure_analysis": analysis_results
        }

# main.py
import asyncio
import logging
import pandas as pd
from config import AzureOpenAIConfig
from pipeline import DataQualityPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Main function to run the data quality pipeline."""
    try:
        # Load configuration
        config = AzureOpenAIConfig()
        
        # Load your data
        df = pd.read_csv("your_dataset.csv")  # Replace with your data source
        
        # Initialize and run pipeline
        pipeline = DataQualityPipeline(config)
        report = await pipeline.run(df)
        
        # Print summary
        print("\n" + "="*50)
        print("DATA QUALITY PIPELINE REPORT")
        print("="*50)
        print(f"Columns processed: {report['summary']['total_columns']}")
        print(f"Total rules generated: {report['summary']['total_rules']}")
        print(f"Failed rules: {report['summary']['total_failed_rules']}")
        print(f"Success rate: {report['summary']['success_rate']:.2f}%")
        
        # Print detailed results for columns with failures
        for column, results in report['execution_results'].items():
            failed_rules = [r for r in results if not r['passed']]
            if failed_rules:
                print(f"\n{column} - {len(failed_rules)} failed rules:")
                for rule in failed_rules:
                    print(f"  - {rule['details']}")
                    
                # Print analysis if available
                if column in report['failure_analysis']:
                    print(f"  Analysis: {report['failure_analysis'][column]}")
        
        return report
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
