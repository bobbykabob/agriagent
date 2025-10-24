from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Dict, List, Any, TypedDict, Annotated

def merge_agent_analyses(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer function to merge agent analyses results"""
    if existing is None:
        return new or {}
    if new is None:
        return existing
    return {**existing, **new}

def merge_current_step(existing: str, new: str) -> str:
    """Reducer function to handle current_step updates"""
    # When multiple values are passed, take the "newest" one based on workflow progression
    if not new:
        return existing

    # Define workflow step progression order
    step_order = [
        "initializing",
        "data_loaded",
        "genotype_analyzed",
        "phenotype_analyzed",
        "environment_analyzed",
        "analyses_integrated",
        "decisions_made",
        "report_generated"
    ]

    # If existing is None or empty, return new
    if not existing:
        return new

    # If new step comes after existing step in workflow, use new
    try:
        existing_idx = step_order.index(existing)
        new_idx = step_order.index(new)
        return new if new_idx > existing_idx else existing
    except ValueError:
        # If step not in order, prefer non-empty values
        return new if new else existing
import asyncio
from src.data_processing.data_loader import DataLoader
from src.agents.genotype_agent import GenotypeAgent
from src.agents.phenotype_agent import PhenotypeAgent
from src.agents.environment_agent import EnvironmentAgent
from src.agents.controller_agent import ControllerAgent
from src.config.settings import config
from src.utils.logger import logger

class WorkflowState(TypedDict):
    """State maintained throughout the workflow"""
    messages: Annotated[List[Any], add_messages]
    data: Dict[str, Any]
    agent_analyses: Annotated[Dict[str, Any], merge_agent_analyses]
    current_step: Annotated[str, merge_current_step]
    query: str
    final_decision: Dict[str, Any]

class AgriAgentWorkflow:
    """Main workflow orchestrating the multi-agent breeding decision system"""

    def __init__(self):
        self.data_loader = DataLoader()
        self.genotype_agent = GenotypeAgent()
        self.phenotype_agent = PhenotypeAgent()
        self.environment_agent = EnvironmentAgent()
        self.controller_agent = ControllerAgent()

        # Build the workflow graph
        self.graph = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""

        # Create the graph
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("load_data", self._load_data)
        workflow.add_node("analyze_genotype", self._analyze_genotype)
        workflow.add_node("analyze_phenotype", self._analyze_phenotype)
        workflow.add_node("analyze_environment", self._analyze_environment)
        workflow.add_node("integrate_analyses", self._integrate_analyses)
        workflow.add_node("make_decisions", self._make_decisions)
        workflow.add_node("generate_report", self._generate_report)

        # Define the flow
        workflow.set_entry_point("load_data")

        workflow.add_edge("load_data", "analyze_genotype")
        workflow.add_edge("load_data", "analyze_phenotype")
        workflow.add_edge("load_data", "analyze_environment")

        # All analysis nodes lead to integration
        workflow.add_edge("analyze_genotype", "integrate_analyses")
        workflow.add_edge("analyze_phenotype", "integrate_analyses")
        workflow.add_edge("analyze_environment", "integrate_analyses")

        # Integration leads to decision making
        workflow.add_edge("integrate_analyses", "make_decisions")
        workflow.add_edge("make_decisions", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    async def run_workflow(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the complete workflow"""
        logger.info(f"Starting AgriAgent workflow with query: {query}")

        # Initial state
        initial_state = WorkflowState(
            messages=[SystemMessage(content="You are part of an agricultural breeding decision support system.")],
            data={},
            agent_analyses={},
            current_step="initializing",
            query=query,
            final_decision={}
        )

        # Run the workflow
        try:
            result_state = await self.graph.ainvoke(initial_state)
            logger.info("Workflow completed successfully")
            return result_state
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise

    def _load_data(self, state: WorkflowState) -> Dict[str, Any]:
        """Load and preprocess data"""
        logger.info("Loading and preprocessing data...")

        try:
            # Load raw data
            raw_data = self.data_loader.load_data()

            # Preprocess data
            processed_data = self.data_loader.preprocess_data()

            # Set data for all agents
            self.genotype_agent.set_data(processed_data)
            self.phenotype_agent.set_data(processed_data)
            self.environment_agent.set_data(processed_data)

            # Update state
            new_state = state.copy()
            new_state["data"] = processed_data
            new_state["current_step"] = "data_loaded"

            logger.info("Data loading completed")

            # Get data summary for the message
            data_summary = self.data_loader.get_summary_statistics()

            # Return new state with message
            return {
                "data": processed_data,
                "current_step": "data_loaded",
                "messages": [AIMessage(content=f"Data loaded successfully. Summary: {data_summary}")]
            }

        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            error_msg = f"Failed to load data: {str(e)}"
            return {
                "messages": [AIMessage(content=error_msg)]
            }

    def _analyze_genotype(self, state: WorkflowState) -> Dict[str, Any]:
        """Analyze genotype data"""
        logger.info("Analyzing genotype data...")

        try:
            # Perform multiple genotype analyses
            genotype_analyses = {}

            # Diversity analysis with chain of thought
            diversity_analysis = self.genotype_agent.analyze_with_thinking(
                "Analyze genetic diversity and haplotype patterns",
                {"analysis_type": "diversity"}
            )
            genotype_analyses["diversity"] = diversity_analysis

            # Kinship analysis with chain of thought
            kinship_analysis = self.genotype_agent.analyze_with_thinking(
                "Analyze kinship relationships between breeding lines",
                {"analysis_type": "kinship"}
            )
            genotype_analyses["kinship"] = kinship_analysis

            # Selection analysis with chain of thought
            selection_analysis = self.genotype_agent.analyze_with_thinking(
                "Analyze lines for genetic advancement selection",
                {"analysis_type": "selection", "target_traits": config.TARGET_TRAITS}
            )
            genotype_analyses["selection"] = selection_analysis

            logger.info("Genotype analysis completed")

            # Return new state with message
            status_msg = f"Genotype analysis completed. Found {len(genotype_analyses.get('selection', {}).get('recommended_lines', []))} recommended lines."
            return {
                "agent_analyses": {"genotype": genotype_analyses},
                "current_step": "genotype_analyzed",
                "messages": [AIMessage(content=status_msg)]
            }

        except Exception as e:
            logger.error(f"Genotype analysis failed: {e}")
            error_msg = f"Genotype analysis failed: {str(e)}"
            return {
                "messages": [AIMessage(content=error_msg)]
            }

    def _analyze_phenotype(self, state: WorkflowState) -> Dict[str, Any]:
        """Analyze phenotype data"""
        logger.info("Analyzing phenotype data...")

        try:
            # Perform multiple phenotype analyses
            phenotype_analyses = {}

            # Trait correlation analysis with chain of thought
            correlation_analysis = self.phenotype_agent.analyze_with_thinking(
                "Analyze correlations between different traits",
                {"analysis_type": "trait_correlation"}
            )
            phenotype_analyses["correlations"] = correlation_analysis

            # Performance ranking with chain of thought
            ranking_analysis = self.phenotype_agent.analyze_with_thinking(
                "Rank lines based on phenotype performance",
                {"analysis_type": "performance_ranking", "target_traits": config.TARGET_TRAITS}
            )
            phenotype_analyses["performance"] = ranking_analysis

            # Stability analysis with chain of thought
            stability_analysis = self.phenotype_agent.analyze_with_thinking(
                "Analyze phenotypic stability across locations",
                {"analysis_type": "stability_analysis"}
            )
            phenotype_analyses["stability"] = stability_analysis

            # Breeding value estimation with chain of thought
            breeding_value_analysis = self.phenotype_agent.analyze_with_thinking(
                "Estimate breeding values for selection",
                {"analysis_type": "breeding_value"}
            )
            phenotype_analyses["breeding_values"] = breeding_value_analysis

            logger.info("Phenotype analysis completed")

            # Return new state with message
            rankings = phenotype_analyses.get("performance", {}).get("rankings", {})
            status_msg = f"Phenotype analysis completed. Ranked {len(rankings)} lines with {len(phenotype_analyses.get('performance', {}).get('top_performers', []))} top performers."
            return {
                "agent_analyses": {"phenotype": phenotype_analyses},
                "current_step": "phenotype_analyzed",
                "messages": [AIMessage(content=status_msg)]
            }

        except Exception as e:
            logger.error(f"Phenotype analysis failed: {e}")
            error_msg = f"Phenotype analysis failed: {str(e)}"
            return {
                "messages": [AIMessage(content=error_msg)]
            }

    def _analyze_environment(self, state: WorkflowState) -> Dict[str, Any]:
        """Analyze environmental data"""
        logger.info("Analyzing environmental data...")

        try:
            # Perform environmental analyses
            environment_analyses = {}

            # Location effects analysis with chain of thought
            location_analysis = self.environment_agent.analyze_with_thinking(
                "Analyze location effects on phenotype performance",
                {"analysis_type": "location_effects"}
            )
            environment_analyses["location_effects"] = location_analysis

            logger.info("Environmental analysis completed")

            # Return new state with message
            location_summary = environment_analyses.get("location_effects", {}).get("location_summary", {})
            status_msg = f"Environmental analysis completed. Analyzed {len(location_summary)} locations."
            return {
                "agent_analyses": {"environment": environment_analyses},
                "current_step": "environment_analyzed",
                "messages": [AIMessage(content=status_msg)]
            }

        except Exception as e:
            logger.error(f"Environmental analysis failed: {e}")
            error_msg = f"Environmental analysis failed: {str(e)}"
            return {
                "messages": [AIMessage(content=error_msg)]
            }

    def _integrate_analyses(self, state: WorkflowState) -> Dict[str, Any]:
        """Integrate analyses from all agents"""
        logger.info("Integrating analyses from all agents...")

        try:
            # Set agent analyses for controller
            self.controller_agent.set_agent_analyses(state["agent_analyses"])

            logger.info("Analysis integration completed")

            # Return new state with message
            agent_count = len([k for k in state["agent_analyses"].keys() if state["agent_analyses"][k]])
            status_msg = f"Integrated analyses from {agent_count} agents successfully."
            return {
                "current_step": "analyses_integrated",
                "messages": [AIMessage(content=status_msg)]
            }

        except Exception as e:
            logger.error(f"Analysis integration failed: {e}")
            error_msg = f"Analysis integration failed: {str(e)}"
            return {
                "messages": [AIMessage(content=error_msg)]
            }

    def _make_decisions(self, state: WorkflowState) -> Dict[str, Any]:
        """Make integrated advancement decisions"""
        logger.info("Making advancement decisions...")

        try:
            # Ensure controller has access to agent analyses
            self.controller_agent.set_agent_analyses(state["agent_analyses"])
            
            logger.info(f"Agent analyses available: {list(state['agent_analyses'].keys())}")
            
            # Make advancement decisions
            advancement_decisions = self.controller_agent.analyze(
                "Make line advancement decisions based on integrated analysis",
                {"decision_type": "advancement"}
            )
            
            logger.info(f"Advancement decisions status: {advancement_decisions.get('status')}")
            logger.info(f"Lines advanced: {len(advancement_decisions.get('advanced_lines', []))}")

            # Make prioritization decisions
            prioritization_decisions = self.controller_agent.analyze(
                "Prioritize lines for different breeding activities",
                {"decision_type": "prioritization"}
            )

            # Assess risks
            risk_assessment = self.controller_agent.analyze(
                "Assess risks associated with line advancement",
                {"decision_type": "risk_assessment"}
            )

            logger.info("Decision making completed")

            # Return new state with message
            advanced_count = len(advancement_decisions.get("advanced_lines", []))
            status_msg = f"Decision making completed. {advanced_count} lines recommended for advancement."
            return {
                "final_decision": {
                    "advancement": advancement_decisions,
                    "prioritization": prioritization_decisions,
                    "risk_assessment": risk_assessment
                },
                "current_step": "decisions_made",
                "messages": [AIMessage(content=status_msg)]
            }

        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            error_msg = f"Decision making failed: {str(e)}"
            return {
                "final_decision": {
                    "advancement": {
                        "status": "error",
                        "message": f"Error: {str(e)}",
                        "advanced_lines": [],
                        "not_advanced_lines": []
                    }
                },
                "messages": [AIMessage(content=error_msg)]
            }

    def _generate_report(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate final report"""
        logger.info("Generating final report...")

        try:
            # Generate comprehensive report
            report = {
                "workflow_summary": self._generate_workflow_summary(state),
                "agent_analyses_summary": self._generate_agent_analyses_summary(state),
                "decision_summary": self._generate_decision_summary(state),
                "recommendations": self._generate_recommendations(state),
                "next_steps": self._generate_next_steps(state)
            }

            logger.info("Report generation completed")

            # Return new state with message
            status_msg = "Final report generated successfully. Analysis complete."
            final_decision = state.get("final_decision", {})
            final_decision["report"] = report

            return {
                "final_decision": final_decision,
                "current_step": "report_generated",
                "messages": [AIMessage(content=status_msg)]
            }

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            error_msg = f"Report generation failed: {str(e)}"
            return {
                "messages": [AIMessage(content=error_msg)]
            }

    def _generate_workflow_summary(self, state: WorkflowState) -> str:
        """Generate workflow execution summary"""
        data_summary = self.data_loader.get_summary_statistics()

        return f"""
        AgriAgent Breeding Decision Support System - Workflow Summary

        Data Overview:
        - Total entries processed: {data_summary.get('phenotype', {}).get('num_lines', 0)}
        - Analysis completed for {len(state['agent_analyses'])} agent domains
        - Workflow steps executed: {len([m for m in state['messages'] if isinstance(m, AIMessage)])}

        Query: {state['query']}

        Status: {'Completed successfully' if state['current_step'] == 'report_generated' else 'In progress'}
        """

    def _generate_agent_analyses_summary(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate summary of all agent analyses"""
        summaries = {}

        for agent_type, analyses in state["agent_analyses"].items():
            if analyses:
                agent_name = agent_type.title()
                summaries[agent_type] = f"{agent_name} analysis completed with {len(analyses)} analysis types"

        return summaries

    def _generate_decision_summary(self, state: WorkflowState) -> str:
        """Generate decision summary"""
        decisions = state["final_decision"]

        advancement = decisions.get("advancement", {})
        advanced_lines = advancement.get("advanced_lines", [])

        return f"""
        Decision Summary:
        - Lines recommended for advancement: {len(advanced_lines)}
        - Total candidates evaluated: {len(advancement.get('advancement_decisions', {}))}
        - Decision confidence: High (integrated multi-agent analysis)
        """

    def _generate_recommendations(self, state: WorkflowState) -> List[str]:
        """Generate key recommendations"""
        recommendations = []

        decisions = state["final_decision"]
        advancement = decisions.get("advancement", {})

        if advancement.get("advanced_lines"):
            recommendations.append(f"Advance {len(advancement['advanced_lines'])} high-priority breeding lines to next generation")
            recommendations.append("Monitor performance of advanced lines in expanded testing environments")
            recommendations.append("Consider the identified prioritization strategies for resource allocation")

        # Add agent-specific recommendations
        for agent_type, analyses in state["agent_analyses"].items():
            if analyses:
                if agent_type == "genotype":
                    recommendations.append("Maintain genetic diversity in breeding population through strategic crossing")
                elif agent_type == "phenotype":
                    recommendations.append("Focus selection pressure on high-performing trait combinations")
                elif agent_type == "environment":
                    recommendations.append("Expand testing to additional locations for better environmental adaptation data")

        return recommendations

    def _generate_next_steps(self, state: WorkflowState) -> List[str]:
        """Generate next steps for breeding program"""
        next_steps = [
            "Validate agent recommendations with field observations",
            "Plan crosses between advanced lines to combine desirable traits",
            "Expand environmental testing for stability assessment",
            "Update breeding database with new performance data",
            "Schedule follow-up analysis after next growing season"
        ]

        return next_steps

    def run_sync_workflow(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Synchronous version of workflow execution"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.run_workflow(query, context))
