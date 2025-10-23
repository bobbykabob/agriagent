from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from langchain_anthropic import ChatAnthropic
from src.config.settings import config
from src.utils.logger import logger
import pandas as pd

class BaseAgent(ABC):
    """Base class for all AgriAgent agents"""

    def __init__(self, name: str, model_name: str = None):
        self.name = name
        self.model_name = model_name or config.GENOTYPE_MODEL
        self.llm = ChatAnthropic(
            model=self.model_name,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=0.1
        )
        self.data = None
        self.analysis_results = {}
        self.thinking_process = []

    def set_data(self, data: Dict[str, pd.DataFrame]):
        """Set the processed data for analysis"""
        self.data = data

    @abstractmethod
    def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze data based on query and context"""
        pass

    def get_relevant_data(self, data_type: str) -> pd.DataFrame:
        """Get relevant data for this agent"""
        if self.data and data_type in self.data:
            return self.data[data_type]
        return pd.DataFrame()

    def format_data_summary(self, data: pd.DataFrame) -> str:
        """Format data summary for LLM consumption"""
        if data.empty:
            return "No data available for analysis."

        summary = []
        summary.append(f"Dataset contains {len(data)} entries and {len(data.columns)} columns.")
        summary.append(f"Columns: {', '.join(data.columns.tolist())}")

        # Add some basic statistics for numerical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary.append("Basic statistics:")
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                if col in data.columns:
                    stats = data[col].describe()
                    summary.append(f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")

        return "\n".join(summary)

    def log_analysis(self, analysis_type: str, results: Dict[str, Any]):
        """Log analysis results"""
        logger.info(f"{self.name} - {analysis_type} analysis completed")
        self.analysis_results[analysis_type] = results

    def add_thinking_step(self, thought: str):
        """Add a step to the thinking process"""
        self.thinking_process.append(thought)

    def get_thinking_process(self) -> List[str]:
        """Get the current thinking process"""
        return self.thinking_process.copy()

    def clear_thinking_process(self):
        """Clear the thinking process"""
        self.thinking_process = []

    def generate_chain_of_thought_prompt(self, query: str, context: Dict[str, Any] = None) -> str:
        """Generate a chain of thought prompt"""
        context = context or {}

        # Base chain of thought prompt
        cot_prompt = f"""
        You are an expert agricultural breeding analyst. Please analyze the following query step by step, showing your reasoning process.

        Query: {query}

        Data Context:
        - Available data types: {list(self.data.keys()) if self.data else 'None'}
        - Analysis context: {context}

        Please think step by step:

        1. First, understand what the query is asking for
        2. Identify what data is relevant to answer this query
        3. Plan your analysis approach
        4. Execute the analysis
        5. Draw conclusions based on the results

        Show your detailed reasoning process for each step.
        """

        return cot_prompt

    def analyze_with_thinking(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze with chain of thought reasoning"""
        self.clear_thinking_process()

        # Generate chain of thought prompt
        cot_prompt = self.generate_chain_of_thought_prompt(query, context)

        # Add initial thinking step
        self.add_thinking_step(f"Starting analysis for query: {query}")

        try:
            # Use the LLM with chain of thought
            response = self.llm.invoke([{"role": "user", "content": cot_prompt}])

            # Extract thinking from response (this is a simplified approach)
            thinking_content = response.content if hasattr(response, 'content') else str(response)

            # Add the thinking to our process (full response, no truncation)
            self.add_thinking_step(f"LLM Response: {thinking_content}")

            # Call the main analysis method
            result = self.analyze(query, context)

            # Add final thinking step
            self.add_thinking_step(f"Analysis completed with status: {result.get('status', 'unknown')}")

            # Include thinking process in results
            result['thinking_process'] = self.get_thinking_process()

            return result

        except Exception as e:
            self.add_thinking_step(f"Error during analysis: {str(e)}")
            result = {
                "status": "error",
                "message": str(e),
                "thinking_process": self.get_thinking_process()
            }
            return result
