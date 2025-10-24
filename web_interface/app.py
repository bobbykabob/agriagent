import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import os
import sys
import time
import hashlib

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.workflows.agriagent_workflow import AgriAgentWorkflow
from src.data_processing.data_loader import DataLoader
from src.config.settings import config
from src.utils.logger import logger

# Page configuration
st.set_page_config(
    page_title="AgriAgent - AI-Powered Breeding Decisions",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Password protection
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == hashlib.sha256("gobruins".encode()).hexdigest():
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # First run, show password input
    if "password_correct" not in st.session_state:
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <img src="https://brand.ucla.edu/images/logos-and-marks/campus-logo.jpg" alt="UCLA Logo" style="height: 100px; margin-bottom: 20px;">
            <h1 style="color: #2774AE;">🌱 AgriAgent</h1>
            <h3 style="color: #666;">AI-Powered Agricultural Breeding Decision Support</h3>
            <p style="color: #888; margin-top: 30px;">Please enter the password to access the system</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.text_input(
                "Password", 
                type="password", 
                on_change=password_entered, 
                key="password",
                label_visibility="collapsed"
            )
        
        st.markdown("""
        <div style="text-align: center; margin-top: 50px; color: #888; font-size: 0.85rem;">
            <p>🏫 UCLA • 🌾 North Dakota State University • 🤝 @structures.computer Lab</p>
        </div>
        """, unsafe_allow_html=True)
        return False
    
    # Password not correct, show input + error
    elif not st.session_state["password_correct"]:
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <img src="https://brand.ucla.edu/images/logos-and-marks/campus-logo.jpg" alt="UCLA Logo" style="height: 100px; margin-bottom: 20px;">
            <h1 style="color: #2774AE;">🌱 AgriAgent</h1>
            <h3 style="color: #666;">AI-Powered Agricultural Breeding Decision Support</h3>
            <p style="color: #888; margin-top: 30px;">Please enter the password to access the system</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.text_input(
                "Password", 
                type="password", 
                on_change=password_entered, 
                key="password",
                label_visibility="collapsed"
            )
            st.error("❌ Incorrect password. Please try again.")
        
        st.markdown("""
        <div style="text-align: center; margin-top: 50px; color: #888; font-size: 0.85rem;">
            <p>🏫 UCLA • 🌾 North Dakota State University • 🤝 @structures.computer Lab</p>
        </div>
        """, unsafe_allow_html=True)
        return False
    
    # Password correct
    else:
        return True

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .decision-card {
        background: linear-gradient(135deg, #fff8dc 0%, #f5deb3 100%);
        border: 2px solid #daa520;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: black;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2E8B57;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Check password first
    if not check_password():
        return

    # Header
    st.markdown('<h1 class="main-header">🌱 AgriAgent: AI-Powered Breeding Decisions</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Multi-Agent AI Framework for Agricultural Breeding Line Advancement</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🎛️ Controls</div>', unsafe_allow_html=True)

        # Analysis type selection
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Complete Breeding Analysis", "Genotype Analysis", "Phenotype Analysis", "Environmental Analysis", "Decision Support"],
            help="Select the type of analysis to perform"
        )

        # Configuration options
        st.markdown("**Configuration**")
        advancement_threshold = st.slider(
            "Advancement Threshold",
            min_value=0.0,
            max_value=1.0,
            value=config.ADVANCEMENT_THRESHOLD,
            step=0.05,
            help="Minimum score required for line advancement"
        )

        top_percentage = st.slider(
            "Top Lines Percentage",
            min_value=0.05,
            max_value=0.3,
            value=config.TOP_LINES_PERCENTAGE,
            step=0.05,
            help="Percentage of top lines to consider for advancement"
        )

        # Run analysis button
        run_analysis = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True
        )

    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔬 Agent Analysis", "📈 Data Visualization", "📋 Reports", "💬 Chat with Agents"])

    # Initialize workflow
    @st.cache_resource
    def get_workflow():
        return AgriAgentWorkflow()

    @st.cache_resource
    def get_data_loader():
        return DataLoader()

    workflow = get_workflow()
    data_loader = get_data_loader()

    # Load and display data summary
    with tab1:
        st.header("📊 System Dashboard")

        # Data overview
        col1, col2, col3 = st.columns(3)

        try:
            raw_data = data_loader.load_data()
            processed_data = data_loader.preprocess_data()

            with col1:
                st.metric(
                    label="📁 Total Entries",
                    value=f"{len(raw_data):,}",
                    help="Total number of breeding lines in dataset"
                )

            with col2:
                phenotype_data = processed_data.get('phenotype', pd.DataFrame())
                st.metric(
                    label="🌿 Phenotype Traits",
                    value=len(phenotype_data.columns) if not phenotype_data.empty else 0,
                    help="Number of measured phenotypic traits"
                )

            with col3:
                st.metric(
                    label="📊 Data Completeness",
                    value="High" if len(raw_data) > 500 else "Medium",
                    help="Overall data quality assessment"
                )

        except Exception as e:
            st.error(f"Error loading data: {e}")

        # System status
        st.subheader("🔧 System Status")
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)

        with status_col1:
            st.markdown('<div class="metric-card">✅ Data Loading</div>', unsafe_allow_html=True)
        with status_col2:
            st.markdown('<div class="metric-card">🔬 Agent System Ready</div>', unsafe_allow_html=True)
        with status_col3:
            st.markdown('<div class="metric-card">📊 Analysis Engine</div>', unsafe_allow_html=True)
        with status_col4:
            st.markdown('<div class="metric-card">🌐 Web Interface</div>', unsafe_allow_html=True)

    # Agent Analysis Tab
    with tab2:
        st.header("🔬 Multi-Agent Analysis")

        if run_analysis:
            # Create placeholders for real-time updates
            status_placeholder = st.empty()
            progress_placeholder = st.empty()

            with st.spinner("🤖 Running AgriAgent Analysis..."):
                try:
                    # Run the workflow
                    query = f"Perform {analysis_type.lower()} for breeding line advancement"
                    context = {
                        "advancement_threshold": advancement_threshold,
                        "top_percentage": top_percentage
                    }

                    # Show initial status
                    status_placeholder.info("🔄 Initializing multi-agent analysis...")
                    progress_placeholder.progress(0)

                    result_state = workflow.run_sync_workflow(query, context)

                    # Update status as analysis progresses
                    progress_placeholder.progress(25)
                    status_placeholder.info("🧬 Genotype Agent analyzing genetic data...")
                    time.sleep(0.5)

                    progress_placeholder.progress(50)
                    status_placeholder.info("🌿 Phenotype Agent analyzing trait data...")
                    time.sleep(0.5)

                    progress_placeholder.progress(75)
                    status_placeholder.info("🌍 Environment Agent analyzing environmental factors...")
                    time.sleep(0.5)

                    progress_placeholder.progress(100)
                    status_placeholder.success("✅ Analysis completed successfully!")

                    # Agent status
                    agent_analyses = result_state.get("agent_analyses", {})

                    st.subheader("🤖 Agent Execution Status")
                    agent_status = {
                        "Genotype Agent": "✅ Completed" if agent_analyses.get("genotype") else "⏸️ Not Run",
                        "Phenotype Agent": "✅ Completed" if agent_analyses.get("phenotype") else "⏸️ Not Run",
                        "Environment Agent": "✅ Completed" if agent_analyses.get("environment") else "⏸️ Not Run"
                    }

                    for agent, status in agent_status.items():
                        st.write(f"**{agent}:** {status}")

                    # Store results in session state for other tabs
                    st.session_state.analysis_results = result_state

                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    logger.error(f"Workflow execution failed: {e}")

        # Display stored results if available
        if "analysis_results" in st.session_state:
            results = st.session_state.analysis_results

            # Multi-Agent Workflow Visualization
            st.subheader("🔄 Multi-Agent Workflow")

            # Interactive workflow visualization using Streamlit components
            st.markdown("### 🤖 Agentic AI Workflow")

            # Create columns for the workflow visualization
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                st.markdown("""
                **📊 Input Data**
                - Excel breeding data
                - Trait measurements
                - Environmental records
                """)

            with col2:
                st.markdown("""
                **🔄 Parallel Processing**
                """)

                # Create a custom graph visualization using Streamlit components
                st.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; gap: 20px;">
                    <div style="display: flex; gap: 20px; align-items: center;">
                        <div style="background: #b3d9ff; border: 2px solid #004d7a; border-radius: 10px; padding: 15px; text-align: center; min-width: 120px; color: black;">
                            📊 Raw Data<br>Excel File
                        </div>
                        <div style="font-size: 24px;">➜</div>
                        <div style="font-size: 24px;">➜</div>
                        <div style="font-size: 24px;">➜</div>
                    </div>
                    <div style="display: flex; gap: 20px; justify-content: center;">
                        <div style="background: #e1bee7; border: 2px solid #3a1e5a; border-radius: 10px; padding: 15px; text-align: center; min-width: 150px; color: black;">
                            🧬 Genotype Agent<br>Genetic Analysis
                        </div>
                        <div style="background: #c8e6c9; border: 2px solid #1e5a1e; border-radius: 10px; padding: 15px; text-align: center; min-width: 150px; color: black;">
                            🌿 Phenotype Agent<br>Trait Analysis
                        </div>
                        <div style="background: #ffe0b2; border: 2px solid #bf360c; border-radius: 10px; padding: 15px; text-align: center; min-width: 150px; color: black;">
                            🌍 Environment Agent<br>Environmental Analysis
                        </div>
                    </div>
                    <div style="display: flex; justify-content: center;">
                        <div style="font-size: 24px;">⬇</div>
                    </div>
                    <div style="display: flex; justify-content: center;">
                        <div style="background: #f8bbd9; border: 2px solid #8e0038; border-radius: 10px; padding: 15px; text-align: center; min-width: 200px; color: black;">
                            🎛️ Controller Agent<br>Decision Integration
                        </div>
                    </div>
                    <div style="display: flex; justify-content: center;">
                        <div style="font-size: 24px;">⬇</div>
                    </div>
                    <div style="display: flex; justify-content: center;">
                        <div style="background: #dcedc8; border: 2px solid #1e5a1e; border-radius: 10px; padding: 15px; text-align: center; min-width: 180px; color: black;">
                            📋 Final Report<br>Advancement Recommendations
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                **Key Features:**
                - ⚡ Parallel agent execution
                - 🔄 Real-time progress updates
                - 💭 Chain-of-thought reasoning
                - 📊 Integrated decision making
                """)

            with col3:
                st.markdown("""
                **📋 Output Results**
                - Advancement recommendations
                - Trait correlations
                - Risk assessments
                - Breeding strategies
                """)

            # Add some spacing
            st.markdown("<br>", unsafe_allow_html=True)

            # Agent insights
            st.subheader("💡 Agent Insights")

            # Add tabs for agent insights and thinking process
            insight_tabs = st.tabs(["💡 Agent Insights", "🧠 Train of Thought"])

            with insight_tabs[0]:
                agent_tabs = st.tabs(["🧬 Genotype", "🌿 Phenotype", "🌍 Environment", "🎛️ Controller"])

                with agent_tabs[0]:
                    genotype_data = results.get("agent_analyses", {}).get("genotype", {})
                    if genotype_data:
                        st.markdown("**🧬 Genotype Agent Results**")

                        # Show diversity analysis
                        diversity_analysis = genotype_data.get("diversity", {})
                        if diversity_analysis.get("status") == "success":
                            st.markdown("**Genetic Diversity Analysis:**")
                            top_lines = diversity_analysis.get("top_diverse_lines", [])[:5]
                            if top_lines:
                                st.write(f"• Top 5 genetically diverse lines: {', '.join(top_lines)}")
                            else:
                                st.write("• Diversity analysis completed - no specific recommendations")
                        else:
                            st.write(f"• Status: {diversity_analysis.get('message', 'No data available')}")

                        # Show kinship analysis
                        kinship_analysis = genotype_data.get("kinship", {})
                        if kinship_analysis.get("status") == "success":
                            st.markdown("**Kinship Analysis:**")
                            related_pairs = kinship_analysis.get("related_pairs", [])
                            if related_pairs:
                                st.write(f"• Found {len(related_pairs)} highly related pairs")
                            else:
                                st.write("• Kinship analysis completed")
                        else:
                            st.write(f"• Kinship: {kinship_analysis.get('message', 'No data available')}")

            with agent_tabs[1]:
                phenotype_data = results.get("agent_analyses", {}).get("phenotype", {})
                if phenotype_data:
                    st.markdown("**🌿 Phenotype Agent Results**")

                    # Show trait correlations
                    correlation_analysis = phenotype_data.get("correlations", {})
                    if correlation_analysis.get("status") == "success":
                        st.markdown("**Trait Correlation Analysis:**")
                        significant_correlations = correlation_analysis.get("significant_correlations", [])
                        if significant_correlations:
                            st.write(f"• Found {len(significant_correlations)} significant trait correlations")
                            # Show top correlations
                            for corr in significant_correlations[:3]:
                                st.write(f"  • {corr['trait1']} ↔ {corr['trait2']}: {corr['correlation']:.3f} ({corr['interpretation']})")
                        else:
                            st.write("• No significant correlations found")
                    else:
                        st.write(f"• Correlations: {correlation_analysis.get('message', 'No data available')}")

                    # Show performance ranking
                    performance_analysis = phenotype_data.get("performance", {})
                    if performance_analysis.get("status") == "success":
                        st.markdown("**Performance Ranking:**")
                        rankings = performance_analysis.get("rankings", {})
                        if rankings:
                            st.write(f"• Ranked {len(rankings)} lines")
                            top_performers = performance_analysis.get("top_performers", [])[:5]
                            if top_performers:
                                st.write(f"• Top 5 performers: {', '.join(top_performers)}")
                        else:
                            st.write("• Performance ranking completed")
                    else:
                        st.write(f"• Performance: {performance_analysis.get('message', 'No data available')}")

                    # Show breeding values
                    breeding_values = phenotype_data.get("breeding_values", {})
                    if breeding_values.get("status") == "success":
                        st.markdown("**Breeding Value Estimation:**")
                        high_value_lines = breeding_values.get("high_value_lines", [])
                        if high_value_lines:
                            st.write(f"• {len(high_value_lines)} high-value lines identified")
                        else:
                            st.write("• Breeding values calculated")
                    else:
                        st.write(f"• Breeding values: {breeding_values.get('message', 'No data available')}")

            with agent_tabs[2]:
                environment_data = results.get("agent_analyses", {}).get("environment", {})
                if environment_data:
                    st.markdown("**🌍 Environment Agent Results**")

                    # Show location effects
                    location_analysis = environment_data.get("location_effects", {})
                    if location_analysis.get("status") == "success":
                        st.markdown("**Location Effects Analysis:**")
                        location_summary = location_analysis.get("location_summary", {})
                        if location_summary:
                            st.write(f"• Analyzed {len(location_summary)} locations")
                            # Show top locations
                            sorted_locations = location_analysis.get("sorted_locations", [])[:3]
                            if sorted_locations:
                                st.write(f"• Top locations: {', '.join(sorted_locations)}")
                        else:
                            st.write("• Location analysis completed")
                    else:
                        st.write(f"• Location effects: {location_analysis.get('message', 'No data available')}")

            with agent_tabs[3]:
                final_decision = results.get("final_decision", {})
                
                # DEBUG: Show what data is available
                with st.expander("🔍 Debug Info - Click to expand"):
                    st.write("**final_decision keys:**", list(final_decision.keys()) if final_decision else "Empty!")
                    if final_decision:
                        advancement = final_decision.get("advancement", {})
                        st.write("**advancement keys:**", list(advancement.keys()) if advancement else "Empty!")
                        st.write("**advancement status:**", advancement.get("status", "No status"))
                        st.write("**advanced_lines count:**", len(advancement.get("advanced_lines", [])))
                        st.write("**not_advanced_lines count:**", len(advancement.get("not_advanced_lines", [])))
                
                if final_decision:
                    st.markdown("**🎛️ Controller Agent Results**")

                    # Show advancement decisions
                    advancement = final_decision.get("advancement", {})
                    if advancement.get("status") == "success":
                        st.markdown("**Advancement Decisions:**")
                        
                        advanced_lines = advancement.get("advanced_lines", [])
                        not_advanced_lines = advancement.get("not_advanced_lines", [])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("✅ Lines Advanced", len(advanced_lines))
                        with col2:
                            st.metric("❌ Lines Not Advanced", len(not_advanced_lines))
                        
                        if advanced_lines:
                            with st.expander(f"View {len(advanced_lines)} Advanced Lines"):
                                for i, line in enumerate(advanced_lines[:20], 1):  # Show first 20
                                    st.write(f"{i}. {line}")
                                if len(advanced_lines) > 20:
                                    st.write(f"... and {len(advanced_lines) - 20} more")
                    else:
                        st.write(f"• Status: {advancement.get('message', 'No advancement decisions available')}")
                    
                    # Show recommendations
                    if "report" in final_decision:
                        report = final_decision["report"]
                        recommendations = report.get("recommendations", [])
                        
                        if recommendations:
                            st.markdown("**💡 Key Recommendations:**")
                            for i, rec in enumerate(recommendations, 1):
                                st.write(f"{i}. {rec}")
                        
                        next_steps = report.get("next_steps", [])
                        if next_steps:
                            st.markdown("**🚀 Next Steps:**")
                            for i, step in enumerate(next_steps, 1):
                                st.write(f"{i}. {step}")
                        
                        risk_assessment = report.get("risk_assessment", "")
                        if risk_assessment:
                            st.markdown("**⚠️ Risk Assessment:**")
                            st.write(risk_assessment)
                else:
                    st.info("Controller Agent has not generated any decisions yet. Run a complete analysis first.")

            # Train of Thought Tab
            with insight_tabs[1]:
                st.subheader("🧠 Agent Thinking Process")

                # Create tabs for each agent's thinking process
                thought_tabs = st.tabs(["🧬 Genotype Thinking", "🌿 Phenotype Thinking", "🌍 Environment Thinking", "🎛️ Controller Thinking"])

                with thought_tabs[0]:
                    genotype_data = results.get("agent_analyses", {}).get("genotype", {})
                    if genotype_data:
                        # Show thinking process for each analysis type
                        for analysis_type in ["diversity", "kinship", "selection"]:
                            if analysis_type in genotype_data:
                                analysis_data = genotype_data[analysis_type]
                                thinking_process = analysis_data.get("thinking_process", [])

                                if thinking_process:
                                    st.markdown(f"**{analysis_type.title()} Analysis Thinking:**")
                                    for i, thought in enumerate(thinking_process, 1):
                                        # Show full thinking process, not truncated
                                        st.markdown(f"**Step {i}:**")
                                        st.write(thought)
                                        st.markdown("---")

                with thought_tabs[1]:
                    phenotype_data = results.get("agent_analyses", {}).get("phenotype", {})
                    if phenotype_data:
                        # Show thinking process for each analysis type
                        for analysis_type in ["correlations", "performance", "stability", "breeding_values"]:
                            if analysis_type in phenotype_data:
                                analysis_data = phenotype_data[analysis_type]
                                thinking_process = analysis_data.get("thinking_process", [])

                                if thinking_process:
                                    st.markdown(f"**{analysis_type.replace('_', ' ').title()} Analysis Thinking:**")
                                    for i, thought in enumerate(thinking_process, 1):
                                        # Show full thinking process, not truncated
                                        st.markdown(f"**Step {i}:**")
                                        st.write(thought)
                                        st.markdown("---")

                with thought_tabs[2]:
                    environment_data = results.get("agent_analyses", {}).get("environment", {})
                    if environment_data:
                        # Show thinking process for location effects
                        if "location_effects" in environment_data:
                            analysis_data = environment_data["location_effects"]
                            thinking_process = analysis_data.get("thinking_process", [])

                            if thinking_process:
                                st.markdown("**Location Effects Analysis Thinking:**")
                                for i, thought in enumerate(thinking_process, 1):
                                    # Show full thinking process, not truncated
                                    st.markdown(f"**Step {i}:**")
                                    st.write(thought)
                                    st.markdown("---")

                with thought_tabs[3]:
                    final_decision = results.get("final_decision", {})
                    if final_decision:
                        # Show thinking process for advancement decisions
                        if "advancement" in final_decision:
                            advancement_data = final_decision["advancement"]
                            thinking_process = advancement_data.get("thinking_process", [])

                            if thinking_process:
                                st.markdown("**Advancement Decision Thinking:**")
                                for i, thought in enumerate(thinking_process, 1):
                                    # Show full thinking process, not truncated
                                    st.markdown(f"**Step {i}:**")
                                    st.write(thought)
                                    st.markdown("---")
                        else:
                            st.info("No thinking process available for Controller Agent decisions.")
                    else:
                        st.info("Run a complete analysis to see Controller Agent thinking process.")

    # Data Visualization Tab
    with tab3:
        st.header("📈 Data Visualization")

        try:
            # Load data for visualization
            raw_data = data_loader.load_data()
            processed_data = data_loader.preprocess_data()

            # Trait distribution plots
            phenotype_data = processed_data.get('phenotype', pd.DataFrame())
            if not phenotype_data.empty:
                st.subheader("🌿 Phenotype Trait Distributions")

                numeric_traits = phenotype_data.select_dtypes(include=[float, int]).columns

                if len(numeric_traits) > 0:
                    # Create subplot for multiple traits
                    cols = st.columns(min(len(numeric_traits), 3))

                    for i, trait in enumerate(numeric_traits[:6]):  # Limit to 6 traits
                        with cols[i % 3]:
                            fig = px.histogram(
                                phenotype_data,
                                x=trait,
                                title=f"{trait.replace('_', ' ').title()}",
                                marginal="box",
                                opacity=0.7
                            )
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)

                # Correlation heatmap
                if len(numeric_traits) > 1:
                    st.subheader("🔗 Trait Correlations")
                    correlation_matrix = phenotype_data[numeric_traits].corr()

                    fig = go.Figure(data=go.Heatmap(
                        z=correlation_matrix.values,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.columns,
                        colorscale='RdBu',
                        zmin=-1, zmax=1
                    ))
                    fig.update_layout(title="Trait Correlation Matrix", height=500)
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating visualizations: {e}")

    # Reports Tab
    with tab4:
        st.header("📋 Analysis Reports")

        if "analysis_results" in st.session_state:
            results = st.session_state.analysis_results
            final_decision = results.get("final_decision", {})

            # Advancement decisions
            advancement = final_decision.get("advancement", {})
            if advancement.get("status") == "success":
                st.subheader("🎯 Advancement Recommendations")

                advanced_lines = advancement.get("advanced_lines", [])
                not_advanced_lines = advancement.get("not_advanced_lines", [])

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown('<div class="decision-card">', unsafe_allow_html=True)
                    st.markdown("**✅ Lines to Advance**")
                    st.write(f"**Count:** {len(advanced_lines)}")
                    if advanced_lines:
                        st.write("**Top Lines:**")
                        for i, line in enumerate(advanced_lines[:10], 1):
                            st.write(f"{i}. {line}")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="decision-card">', unsafe_allow_html=True)
                    st.markdown("**❌ Lines Not Advanced**")
                    st.write(f"**Count:** {len(not_advanced_lines)}")
                    st.markdown('</div>', unsafe_allow_html=True)

            # Recommendations
            if "report" in final_decision:
                report = final_decision["report"]
                recommendations = report.get("recommendations", [])

                if recommendations:
                    st.subheader("💡 Key Recommendations")
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")

                next_steps = report.get("next_steps", [])
                if next_steps:
                    st.subheader("🚀 Next Steps")
                    for i, step in enumerate(next_steps, 1):
                        st.write(f"{i}. {step}")

        else:
            st.info("👆 Run an analysis in the Agent Analysis tab to generate reports")

    # Chat with Agents Tab
    with tab5:
        st.header("💬 Chat with Agents")
        
        if "analysis_results" not in st.session_state:
            st.info("⚠️ Please run an analysis first in the 'Agent Analysis' tab before chatting with agents.")
            st.markdown("""
            ### How to Use This Feature:
            1. Go to the **Agent Analysis** tab
            2. Run a complete breeding analysis
            3. Return here to chat with specific agents about their analysis
            
            ### What You Can Learn:
            - 🧬 **Genotype Agent**: Ask about genetic diversity, kinship relationships, and marker-based selection
            - 🌿 **Phenotype Agent**: Inquire about trait performance, correlations, and breeding values
            - 🌍 **Environment Agent**: Learn about location effects and environmental adaptations
            - 🎛️ **Controller Agent**: Get integrated recommendations combining all three agents' perspectives
            """)
        else:
            st.markdown("""
            Select an agent below to learn more about their analysis. You can ask questions about:
            - Specific lines or traits
            - Methodology and reasoning
            - Alternative interpretations
            - Recommendations for breeding decisions
            """)
            
            # Agent selection
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.subheader("Select Agent")
                agent_choice = st.radio(
                    "Choose an agent to chat with:",
                    ["🧬 Genotype Agent", "🌿 Phenotype Agent", "🌍 Environment Agent", "🎛️ Controller Agent"],
                    key="agent_selector"
                )
                
                # Show agent info
                st.markdown("---")
                if "Genotype" in agent_choice:
                    st.markdown("""
                    **Genotype Agent Expertise:**
                    - Genetic diversity analysis
                    - Kinship relationships
                    - Marker-based selection
                    - Population genetics
                    """)
                elif "Phenotype" in agent_choice:
                    st.markdown("""
                    **Phenotype Agent Expertise:**
                    - Trait correlations
                    - Performance ranking
                    - Breeding value estimation
                    - Stability analysis
                    """)
                elif "Environment" in agent_choice:
                    st.markdown("""
                    **Environment Agent Expertise:**
                    - Location effects
                    - GxE interactions
                    - Environmental adaptation
                    - Climate impact
                    """)
                else:
                    st.markdown("""
                    **Controller Agent Expertise:**
                    - Integrated decision making
                    - Multi-criteria optimization
                    - Risk assessment
                    - Strategic recommendations
                    - Synthesizing all agents' insights
                    """)
                
                # Clear chat button
                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    agent_key = agent_choice.split()[1].lower()
                    if f"chat_history_{agent_key}" in st.session_state:
                        del st.session_state[f"chat_history_{agent_key}"]
                    st.rerun()
            
            with col2:
                st.subheader(f"Chat with {agent_choice}")
                
                # Get the appropriate agent
                agent_key = agent_choice.split()[1].lower()  # "genotype", "phenotype", "environment", or "controller"
                
                # Initialize chat history for this agent if not exists
                if f"chat_history_{agent_key}" not in st.session_state:
                    st.session_state[f"chat_history_{agent_key}"] = []
                
                # Get agent analysis context
                results = st.session_state.analysis_results
                agent_analyses = results.get("agent_analyses", {})
                
                # For controller agent, use all analyses and decisions
                if agent_key == "controller":
                    agent_context = {
                        "agent_analyses": agent_analyses,
                        "final_decision": results.get("final_decision", {})
                    }
                else:
                    agent_context = agent_analyses.get(agent_key, {})
                
                # Display chat history
                chat_container = st.container()
                with chat_container:
                    if len(st.session_state[f"chat_history_{agent_key}"]) == 0:
                        st.info(f"👋 Hello! I'm the {agent_choice}. Ask me anything about my analysis!")
                    
                    for msg in st.session_state[f"chat_history_{agent_key}"]:
                        if msg["role"] == "user":
                            st.markdown(f"**🧑 You:** {msg['content']}")
                        else:
                            st.markdown(f"**🤖 {agent_choice}:** {msg['content']}")
                        st.markdown("---")
                
                # Chat input
                with st.form(key=f"chat_form_{agent_key}", clear_on_submit=True):
                    user_input = st.text_area(
                        "Your question:",
                        placeholder="E.g., 'Which lines showed the highest genetic diversity?' or 'Why did you recommend line 123 for advancement?'",
                        height=100,
                        key=f"chat_input_{agent_key}"
                    )
                    
                    col_a, col_b, col_c = st.columns([1, 1, 2])
                    with col_a:
                        submit_button = st.form_submit_button("💬 Send", use_container_width=True)
                    with col_b:
                        if st.form_submit_button("💡 Example Questions", use_container_width=True):
                            st.session_state[f"show_examples_{agent_key}"] = True
                
                # Show example questions if requested
                if st.session_state.get(f"show_examples_{agent_key}", False):
                    st.markdown("### 💡 Example Questions:")
                    if "genotype" in agent_key:
                        st.markdown("""
                        - What are the top 5 most genetically diverse lines?
                        - Are there any highly related pairs in the population?
                        - How did you calculate the genetic diversity scores?
                        - Which lines would you recommend for maintaining diversity?
                        - What's the average kinship in this population?
                        """)
                    elif "phenotype" in agent_key:
                        st.markdown("""
                        - Which traits are most strongly correlated?
                        - What are the top performing lines for yield?
                        - How stable are the elite lines across environments?
                        - Which lines have the highest breeding values?
                        - Are there any trade-offs between traits I should know about?
                        """)
                    elif "environment" in agent_key:
                        st.markdown("""
                        - Which locations were most favorable for testing?
                        - Are there strong genotype-by-environment interactions?
                        - How do environmental conditions affect trait expression?
                        - Which lines show the best environmental adaptation?
                        - Should we expand testing to additional locations?
                        """)
                    else:  # controller agent
                        st.markdown("""
                        - Which lines should I advance to the next generation and why?
                        - How do the three agents' recommendations compare?
                        - What are the biggest risks with my top candidates?
                        - Give me an integrated assessment of line X.
                        - What would you prioritize: yield, stability, or diversity?
                        - How confident are you in these advancement decisions?
                        """)
                    st.session_state[f"show_examples_{agent_key}"] = False
                
                # Process user input
                if submit_button and user_input.strip():
                    # Add user message to history
                    st.session_state[f"chat_history_{agent_key}"].append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    # Get the appropriate agent from workflow
                    with st.spinner(f"🤔 {agent_choice} is thinking..."):
                        try:
                            # Get agent instance and ensure it has data
                            if agent_key == "genotype":
                                agent = workflow.genotype_agent
                            elif agent_key == "phenotype":
                                agent = workflow.phenotype_agent
                            elif agent_key == "environment":
                                agent = workflow.environment_agent
                            else:  # controller agent
                                agent = workflow.controller_agent
                            
                            # Ensure agent has data set (in case workflow was cached without data)
                            if agent.data is None or not agent.data:
                                processed_data = data_loader.preprocess_data()
                                agent.set_data(processed_data)
                            
                            # Debug: Show what context we have
                            if not agent_context:
                                st.warning(f"⚠️ No analysis context found for {agent_key} agent. Running analysis first...")
                                # Run a quick analysis if context is missing
                                if agent_key == "phenotype":
                                    agent_context = agent.analyze(
                                        "Analyze trait correlations",
                                        {"analysis_type": "trait_correlation"}
                                    )
                                elif agent_key == "genotype":
                                    agent_context = agent.analyze(
                                        "Analyze genetic diversity",
                                        {"analysis_type": "diversity"}
                                    )
                                elif agent_key == "environment":
                                    agent_context = agent.analyze(
                                        "Analyze location effects",
                                        {"analysis_type": "location_effects"}
                                    )
                                else:  # controller
                                    # For controller, we need all agent analyses
                                    st.info("Running integrated analysis...")
                                    agent_context = {
                                        "agent_analyses": agent_analyses,
                                        "final_decision": results.get("final_decision", {})
                                    }
                            
                            # Get response from agent
                            response = agent.chat(
                                user_message=user_input,
                                chat_history=st.session_state[f"chat_history_{agent_key}"][:-1],  # Exclude the current message
                                analysis_context=agent_context
                            )
                            
                            # Add agent response to history
                            st.session_state[f"chat_history_{agent_key}"].append({
                                "role": "assistant",
                                "content": response
                            })
                            
                        except Exception as e:
                            st.error(f"❌ Error getting response: {str(e)}")
                            logger.error(f"Chat error with {agent_key}: {e}")
                            import traceback
                            st.error(f"Traceback: {traceback.format_exc()}")
                    
                    # Rerun to show new messages
                    st.rerun()

    # Footer
    st.markdown("---")

    # Create footer with UCLA logo and credits
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.markdown("""
        <div style="text-align: center;">
            <img src="https://brand.ucla.edu/images/logos-and-marks/campus-logo.jpg" alt="UCLA Logo" style="height: 80px; width: auto; margin-bottom: 5px;">
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align: center; color: black; font-size: 0.9rem;">
            <strong>AgriAgent v1.0</strong><br>
            AI-Powered Agricultural Breeding Decision Support System<br>
            Built with LangGraph, Streamlit, and Claude AI<br>
            For Research Use Only
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="text-align: center;">
            <p style="color: #2E8B57; font-weight: bold; margin-bottom: 8px; font-size: 0.9rem;">🌱 From</p>
            <p style="color: black; font-size: 0.8rem; margin: 0; line-height: 1.6;">
                <strong>UCLA</strong><br>
                <strong>North Dakota State University</strong><br>
                <a href="https://structures.computer/" target="_blank" style="color: #2E8B57; text-decoration: none;">
                    <strong>@structures.computer</strong>
                </a> Lab
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
