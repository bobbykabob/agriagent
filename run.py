#!/usr/bin/env python3
"""
AgriAgent - AI-Powered Agricultural Breeding Decision Support System

This script provides multiple ways to run the AgriAgent system:
1. Run complete workflow analysis
2. Start the web interface
3. Test individual agents
4. Process data only

Usage:
    python run.py [mode] [options]

Modes:
    workflow    - Run complete breeding analysis workflow
    web         - Start the Streamlit web interface
    test        - Test individual agents
    data        - Process data only
"""

import argparse
import sys
import os
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.workflows.agriagent_workflow import AgriAgentWorkflow
from src.data_processing.data_loader import DataLoader
from src.agents.genotype_agent import GenotypeAgent
from src.agents.phenotype_agent import PhenotypeAgent
from src.agents.environment_agent import EnvironmentAgent
from src.agents.controller_agent import ControllerAgent
from src.config.settings import config
from src.utils.logger import logger

def run_workflow_analysis(query: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run the complete AgriAgent workflow"""
    print("🌱 Starting AgriAgent Breeding Analysis Workflow...")

    try:
        # Initialize workflow
        workflow = AgriAgentWorkflow()

        # Set default query if not provided
        if query is None:
            query = "Perform comprehensive breeding line analysis for advancement decisions"

        # Run workflow
        result = workflow.run_sync_workflow(query, context)

        print("✅ Workflow completed successfully!")
        return result

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        logger.error(f"Workflow execution failed: {e}")
        return {}

def start_web_interface(host: str = None, port: int = None):
    """Start the Streamlit web interface"""
    print("🌐 Starting AgriAgent Web Interface...")

    try:
        import streamlit as st
        from web_interface.app import main

        # Use config values if not provided
        host = host or config.STREAMLIT_HOST
        port = port or config.STREAMLIT_PORT

        print(f"📡 Web interface starting at http://{host}:{port}")
        print("🔗 Open your browser and navigate to the URL above")

        # Set environment variables for Streamlit
        os.environ['STREAMLIT_SERVER_HOST'] = host
        os.environ['STREAMLIT_SERVER_PORT'] = str(port)

        # Run the web interface
        main()

    except ImportError:
        print("❌ Streamlit not installed. Install with: pip install streamlit")
    except Exception as e:
        print(f"❌ Failed to start web interface: {e}")

def test_agents():
    """Test individual agents"""
    print("🧪 Testing Individual Agents...")

    try:
        # Load data
        data_loader = DataLoader()
        data = data_loader.load_data()
        processed_data = data_loader.preprocess_data()

        print(f"📊 Loaded {len(data)} breeding lines")

        # Test Genotype Agent
        print("\n🧬 Testing Genotype Agent...")
        genotype_agent = GenotypeAgent()
        genotype_agent.set_data(processed_data)

        genotype_result = genotype_agent.analyze(
            "Analyze genetic diversity",
            {"analysis_type": "diversity"}
        )
        print(f"✅ Genotype analysis: {genotype_result.get('status', 'unknown')}")

        # Test Phenotype Agent
        print("\n🌿 Testing Phenotype Agent...")
        phenotype_agent = PhenotypeAgent()
        phenotype_agent.set_data(processed_data)

        phenotype_result = phenotype_agent.analyze(
            "Analyze trait performance",
            {"analysis_type": "performance_ranking"}
        )
        print(f"✅ Phenotype analysis: {phenotype_result.get('status', 'unknown')}")

        # Test Environment Agent
        print("\n🌍 Testing Environment Agent...")
        environment_agent = EnvironmentAgent()
        environment_agent.set_data(processed_data)

        environment_result = environment_agent.analyze(
            "Analyze location effects",
            {"analysis_type": "location_effects"}
        )
        print(f"✅ Environment analysis: {environment_result.get('status', 'unknown')}")

        # Test Controller Agent
        print("\n🎛️ Testing Controller Agent...")
        controller_agent = ControllerAgent()
        controller_agent.set_agent_analyses({
            "genotype": genotype_result,
            "phenotype": phenotype_result,
            "environment": environment_result
        })

        controller_result = controller_agent.analyze(
            "Make advancement decisions",
            {"decision_type": "advancement"}
        )
        print(f"✅ Controller analysis: {controller_result.get('status', 'unknown')}")

        print("\n✅ All agents tested successfully!")

    except Exception as e:
        print(f"❌ Agent testing failed: {e}")
        logger.error(f"Agent testing failed: {e}")

def process_data_only():
    """Process data only (no analysis)"""
    print("📋 Processing Data Only...")

    try:
        # Load and process data
        data_loader = DataLoader()
        raw_data = data_loader.load_data()
        processed_data = data_loader.preprocess_data()

        # Display data summary
        summary = data_loader.get_summary_statistics()

        print("✅ Data processed successfully!")
        print(f"📊 Raw data: {len(raw_data)} rows, {len(raw_data.columns)} columns")
        print(f"📋 Processed data: {len(processed_data)} data types")

        for data_type, stats in summary.items():
            if isinstance(stats, dict) and 'num_lines' in stats:
                print(f"  • {data_type}: {stats['num_lines']} lines, {stats['num_features']} features")

        return processed_data

    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        logger.error(f"Data processing failed: {e}")
        return {}

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AgriAgent - AI-Powered Agricultural Breeding Decision Support System")

    parser.add_argument(
        "mode",
        choices=["workflow", "web", "test", "data"],
        help="Mode to run the system in"
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Query for workflow analysis"
    )

    parser.add_argument(
        "--host",
        type=str,
        default=config.STREAMLIT_HOST,
        help="Host for web interface"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=config.STREAMLIT_PORT,
        help="Port for web interface"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=config.ADVANCEMENT_THRESHOLD,
        help="Advancement threshold for decisions"
    )

    args = parser.parse_args()

    # Set up context for workflow
    context = {
        "advancement_threshold": args.threshold,
        "top_percentage": config.TOP_LINES_PERCENTAGE
    }

    try:
        if args.mode == "workflow":
            result = run_workflow_analysis(args.query, context)

            if result:
                # Display key results
                final_decision = result.get("final_decision", {})
                advancement = final_decision.get("advancement", {})

                if advancement.get("status") == "success":
                    advanced_lines = advancement.get("advanced_lines", [])
                    print("\n🎯 Key Results:")
                    print(f"   • Lines recommended for advancement: {len(advanced_lines)}")
                    print(f"   • Total candidates evaluated: {len(advancement.get('advancement_decisions', {}))}")

                    if advanced_lines:
                        print(f"   • Top recommended lines: {', '.join(advanced_lines[:5])}")

        elif args.mode == "web":
            start_web_interface(args.host, args.port)

        elif args.mode == "test":
            test_agents()

        elif args.mode == "data":
            processed_data = process_data_only()

    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
    except Exception as e:
        print(f"❌ Operation failed: {e}")
        logger.error(f"Main execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()