from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from langchain_anthropic import ChatAnthropic
from src.config.settings import config
from src.utils.logger import logger
import pandas as pd
import numpy as np

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

        # Add initial thinking step with data inspection
        self.add_thinking_step(f"=== STARTING ANALYSIS ===")
        self.add_thinking_step(f"Query: {query}")
        self.add_thinking_step(f"Context: {context}")
        
        # Inspect available data
        if self.data:
            self.add_thinking_step(f"\n=== DATA INSPECTION ===")
            for data_type, df in self.data.items():
                if df is not None and not df.empty:
                    self.add_thinking_step(f"\n{data_type.upper()} DATA:")
                    self.add_thinking_step(f"  - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    self.add_thinking_step(f"  - Columns: {list(df.columns)}")
                    
                    # Show sample data
                    if len(df) > 0:
                        self.add_thinking_step(f"  - First 3 entries: {df['entry'].head(3).tolist() if 'entry' in df.columns else 'N/A'}")
                        
                        # Show numeric column ranges
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            self.add_thinking_step(f"  - Numeric columns: {list(numeric_cols)}")
                            for col in numeric_cols[:3]:  # First 3 numeric columns
                                if col != 'entry':
                                    self.add_thinking_step(f"    * {col}: range [{df[col].min():.2f}, {df[col].max():.2f}], mean {df[col].mean():.2f}")
        else:
            self.add_thinking_step("WARNING: No data available for analysis")

        # Generate chain of thought prompt
        cot_prompt = self.generate_chain_of_thought_prompt(query, context)

        try:
            # Use the LLM with chain of thought
            self.add_thinking_step(f"\n=== REASONING PROCESS ===")
            response = self.llm.invoke([{"role": "user", "content": cot_prompt}])

            # Extract thinking from response
            thinking_content = response.content if hasattr(response, 'content') else str(response)
            self.add_thinking_step(f"LLM Reasoning: {thinking_content}")

            # Call the main analysis method with explicit thinking
            self.add_thinking_step(f"\n=== PERFORMING ANALYSIS ===")
            result = self.analyze(query, context)

            # Add final thinking step with results summary
            self.add_thinking_step(f"\n=== ANALYSIS COMPLETE ===")
            self.add_thinking_step(f"Status: {result.get('status', 'unknown')}")
            self.add_thinking_step(f"Summary: {result.get('summary', 'No summary available')}")

            # Include thinking process in results
            result['thinking_process'] = self.get_thinking_process()

            return result

        except Exception as e:
            self.add_thinking_step(f"ERROR during analysis: {str(e)}")
            result = {
                "status": "error",
                "message": str(e),
                "thinking_process": self.get_thinking_process()
            }
            return result

    def chat(self, user_message: str, chat_history: List[Dict[str, str]] = None, analysis_context: Dict[str, Any] = None) -> str:
        """
        Interactive chat method for conversing with the agent about its analysis.
        
        Args:
            user_message: The user's question or message
            chat_history: Previous chat messages in format [{"role": "user"/"assistant", "content": "..."}]
            analysis_context: Previous analysis results to provide context
        
        Returns:
            Agent's response as a string
        """
        chat_history = chat_history or []
        analysis_context = analysis_context or self.analysis_results
        
        # Build context-aware system prompt
        system_prompt = self._build_chat_system_prompt(analysis_context)
        
        # Build messages for LLM
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        try:
            # Get response from LLM
            response = self.llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"{self.name} - Chat response generated for: {user_message[:50]}...")
            return response_text
            
        except Exception as e:
            logger.error(f"{self.name} - Chat error: {e}")
            return f"I apologize, but I encountered an error processing your question: {str(e)}"
    
    def _build_chat_system_prompt(self, analysis_context: Dict[str, Any]) -> str:
        """Build a context-aware system prompt for chat"""
        
        # Get detailed data summary
        data_summary = ""
        if self.data:
            data_types = list(self.data.keys())
            data_summary = f"Data Sources Available:\n"
            
            for data_type in data_types:
                df = self.data.get(data_type)
                if df is not None and not df.empty:
                    data_summary += f"- {data_type.title()} data: {len(df)} entries"
                    if 'entry' in df.columns:
                        data_summary += f", Columns: {', '.join(df.columns.tolist())}"
                    data_summary += "\n"
            
            data_summary += f"\nIMPORTANT: You HAVE analyzed this data from the Excel file (data/IYT_DATA_UCLA.xls). "
            data_summary += f"All your previous analyses were performed on this dataset. "
            data_summary += f"When discussing results, reference this data as the source."
        
        # Get detailed analysis summary with actual data
        analysis_summary = ""
        if analysis_context:
            analysis_summary = f"\nYour Previous Analysis Results (with detailed findings):\n"
            for analysis_type, results in analysis_context.items():
                if isinstance(results, dict):
                    status = results.get('status', 'unknown')
                    summary = results.get('summary', '')
                    analysis_summary += f"\n**{analysis_type.title().replace('_', ' ')}:**\n"
                    analysis_summary += f"  Status: {status}\n"
                    if summary:
                        analysis_summary += f"  Summary: {summary}\n"
                    
                    # Include specific detailed findings based on analysis type
                    if status == "success":
                        # For correlations
                        if 'significant_correlations' in results:
                            correlations = results.get('significant_correlations', [])
                            if correlations:
                                analysis_summary += f"  Significant correlations found:\n"
                                for corr in correlations[:5]:  # Top 5
                                    analysis_summary += f"    - {corr.get('trait1', 'T1')} ↔ {corr.get('trait2', 'T2')}: r={corr.get('correlation', 0):.3f}\n"
                            else:
                                analysis_summary += f"  No significant correlations found (threshold |r| > 0.3)\n"
                        
                        # For performance rankings
                        if 'top_performers' in results:
                            top_lines = results.get('top_performers', [])
                            if top_lines:
                                analysis_summary += f"  Top {len(top_lines)} performing lines: {', '.join(map(str, top_lines[:10]))}\n"
                        
                        # For rankings with scores
                        if 'rankings' in results:
                            rankings = results.get('rankings', {})
                            if rankings:
                                analysis_summary += f"  Total lines ranked: {len(rankings)}\n"
                                # Show top 5 with scores
                                sorted_rankings = sorted(rankings.items(), key=lambda x: x[1].get('rank', 999))[:5]
                                if sorted_rankings:
                                    analysis_summary += f"  Top 5 lines with composite scores:\n"
                                    for line_id, rank_info in sorted_rankings:
                                        score = rank_info.get('composite_score', 0)
                                        category = rank_info.get('category', 'N/A')
                                        analysis_summary += f"    - Line {line_id}: score={score:.3f}, category={category}\n"
                        
                        # For diversity scores
                        if 'top_diverse_lines' in results:
                            diverse_lines = results.get('top_diverse_lines', [])
                            if diverse_lines:
                                analysis_summary += f"  Most diverse lines: {', '.join(map(str, diverse_lines[:10]))}\n"
                        
                        # For breeding values
                        if 'high_value_lines' in results:
                            high_value = results.get('high_value_lines', [])
                            if high_value:
                                analysis_summary += f"  High breeding value lines: {', '.join(map(str, high_value[:10]))}\n"
                        
                        # For trait statistics
                        if 'trait_statistics' in results:
                            trait_stats = results.get('trait_statistics', {})
                            if trait_stats:
                                analysis_summary += f"  Trait statistics:\n"
                                for trait, stats in list(trait_stats.items())[:3]:  # Top 3 traits
                                    mean = stats.get('mean', 0)
                                    std = stats.get('std', 0)
                                    analysis_summary += f"    - {trait}: mean={mean:.2f}, std={std:.2f}\n"
        
        # Determine agent specialization
        if 'Genotype' in self.name:
            specialization = 'genetic marker analysis, haplotype diversity, and kinship matrices'
            expertise_focus = 'genetic diversity, relatedness, and marker-based selection'
        elif 'Phenotype' in self.name:
            specialization = 'phenotypic trait analysis, performance ranking, and breeding value estimation'
            expertise_focus = 'trait performance, correlations, and phenotypic selection'
        elif 'Environment' in self.name:
            specialization = 'environmental analysis, genotype-by-environment interactions, and location effects'
            expertise_focus = 'environmental adaptation, stability, and GxE patterns'
        elif 'Controller' in self.name:
            specialization = 'integrated multi-agent decision making, combining insights from genetic, phenotypic, and environmental analyses to make holistic breeding decisions'
            expertise_focus = 'integrated decision making, risk assessment, and strategic breeding recommendations'
        else:
            specialization = 'agricultural breeding analysis and decision support'
            expertise_focus = 'general breeding program support'
        
        # Add controller-specific guidelines
        controller_guidelines = ""
        if 'Controller' in self.name:
            controller_guidelines = "\n- As the Controller Agent, you synthesize insights from all three domain agents (Genotype, Phenotype, Environment) to make integrated decisions\n- Explain how different agents' perspectives complement each other in decision making"
        
        system_prompt = f"""You are {self.name}, an expert agricultural breeding analyst AI assistant.

You specialize in {specialization}.

{data_summary}

{analysis_summary}

Your role in this conversation:
1. Answer questions about the analysis you performed
2. Explain your reasoning and methodology in detail
3. Provide insights and recommendations based on the data
4. Help users understand complex breeding concepts
5. Suggest follow-up analyses or considerations

Guidelines:
- You HAVE access to the Excel data analysis results shown above - use them!
- Reference specific lines, traits, scores, and findings from your analysis
- Be specific and cite actual numbers from the analysis results
- Use technical terminology but explain concepts clearly
- Never say you don't have access to data - you DO have the analysis results above
- If asked about the data, explain what you analyzed and show specific findings
- Stay focused on your domain expertise ({expertise_focus})
- Be critical and rigorous in your assessments
- When discussing breeding decisions, consider both benefits and risks{controller_guidelines}

IMPORTANT: The detailed analysis results above are YOUR results from analyzing the Excel file. 
Reference specific lines, scores, correlations, and rankings from these results when answering questions.
Do not ask users to provide data - you already have the analysis results.

Always provide scientifically sound, actionable insights based on the available data."""

        return system_prompt
