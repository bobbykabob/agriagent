import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import os
import sys
import time
import hashlib
import base64

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.workflows.agriagent_workflow import AgriAgentWorkflow
from src.data_processing.data_loader import DataLoader
from src.config.settings import config
from src.utils.logger import logger

# Helper functions for plant management
def _get_data_path(filename):
    """Get absolute path to data file, works in both local and deployed environments"""
    # Try multiple possible paths
    possible_paths = [
        # Path relative to app.py (local development)
        os.path.join(os.path.dirname(__file__), '..', 'data', 'correlated', filename),
        # Path relative to project root
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'correlated', filename),
        # Absolute path from current working directory
        os.path.join(os.getcwd(), 'data', 'correlated', filename),
        # Direct path (for deployed environments)
        os.path.join('data', 'correlated', filename),
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    
    # Return the first path as default (will show error if file doesn't exist)
    return os.path.abspath(possible_paths[0])

@st.cache_data
def load_genotype_data():
    """Load genotype data from processed_genotype.csv"""
    try:
        genotype_path = _get_data_path('processed_genotype.csv')
        if not os.path.exists(genotype_path):
            logger.error(f"Genotype file not found at: {genotype_path}")
            return pd.DataFrame()
        df = pd.read_csv(genotype_path, nrows=100)
        logger.info(f"Successfully loaded genotype data from: {genotype_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading genotype data: {e}")
        logger.error(f"Attempted path: {_get_data_path('processed_genotype.csv')}")
        return pd.DataFrame()

@st.cache_data
def load_phenotype_data():
    """Load phenotype data from Excel file"""
    try:
        phenotype_path = _get_data_path('Pros_96 plot_Seq_Yeld_25.xlsx')
        if not os.path.exists(phenotype_path):
            logger.warning(f"Phenotype file not found at: {phenotype_path}")
            return pd.DataFrame()
        df = pd.read_excel(phenotype_path)
        logger.info(f"Successfully loaded phenotype data from: {phenotype_path}")
        return df
    except Exception as e:
        logger.warning(f"Error loading phenotype data: {e}")
        logger.warning(f"Attempted path: {_get_data_path('Pros_96 plot_Seq_Yeld_25.xlsx')}")
        return pd.DataFrame()

def get_plant_ids_from_genotype(genotype_df):
    """Extract plant IDs from genotype dataframe columns"""
    if genotype_df.empty:
        return []
    # Plant IDs are columns that start with 'C0' followed by digits
    plant_cols = [col for col in genotype_df.columns if col.startswith('C0') and col[1:].isdigit()]
    return sorted(plant_cols)

def get_plant_genotype(genotype_df, plant_id):
    """Get genotype data for a specific plant"""
    if genotype_df.empty or plant_id not in genotype_df.columns:
        return pd.DataFrame()
    
    # Get marker information and plant's genotype
    marker_cols = ['rs#', 'alleles', 'chrom', 'pos']
    available_marker_cols = [col for col in marker_cols if col in genotype_df.columns]
    
    result = genotype_df[available_marker_cols + [plant_id]].copy()
    result = result[result[plant_id].notna()]  # Remove rows with missing data
    return result

def get_plant_phenotype(phenotype_df, plant_id):
    """Get phenotype data for a specific plant"""
    if phenotype_df.empty:
        return pd.DataFrame()
    
    # Try to find the plant ID in various possible column formats
    # Check if plant_id exists as a column or in entry/plot/name columns
    if plant_id in phenotype_df.columns:
        return phenotype_df[[plant_id]]
    
    # Check common identifier columns
    id_cols = ['entry', 'plot', 'name', 'Entry', 'Plot', 'Name']
    available_id_cols = [col for col in id_cols if col in phenotype_df.columns]
    
    if available_id_cols:
        # Filter rows where any identifier column matches the plant_id
        mask = pd.Series([False] * len(phenotype_df))
        for col in available_id_cols:
            mask |= phenotype_df[col].astype(str).str.contains(plant_id, na=False, case=False)
        
        if mask.any():
            return phenotype_df[mask]
    
    return pd.DataFrame()

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
        if "password" in st.session_state:
            if hashlib.sha256(st.session_state["password"].strip().encode()).hexdigest() == hashlib.sha256("gobruins".encode()).hexdigest():
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
    /* Reduce default spacing */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 0.5rem;
        margin-top: 0;
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
    
    /* Floating Chat Button */
    .chat-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        cursor: pointer;
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        transition: all 0.3s ease;
    }
    
    .chat-button:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 16px rgba(0,0,0,0.4);
    }
    
    .chat-button:active {
        transform: scale(0.95);
    }
    
    /* Chat notification badge */
    .chat-badge {
        position: absolute;
        top: -5px;
        right: -5px;
        background: #ff4444;
        color: white;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def _render_user_message(content) -> None:
    """Render a user message that may be plain text or text + images."""
    if isinstance(content, dict):
        text = content.get("text", "")
        images = content.get("images", [])
        if text:
            st.markdown(f"**🧑 You:** {text}")
        for img in images:
            b64 = img.get("base64", "")
            if b64:
                try:
                    st.image(base64.b64decode(b64), caption="Uploaded image", use_container_width=True)
                except Exception:
                    st.caption("(Image)")
    else:
        st.markdown(f"**🧑 You:** {content}")


def _user_content_to_history_content(text: str, image_dicts: List[Dict[str, str]]) -> Any:
    """Build the content to store in chat history (and to send as history to the API)."""
    if not image_dicts:
        return text
    return {"text": text or "(image)", "images": image_dicts}


def _history_content_to_api_content(content: Any) -> Any:
    """Convert stored history content to the format expected by the LLM (list of blocks or string)."""
    if isinstance(content, dict):
        blocks = [{"type": "text", "text": content.get("text", "")}]
        for img in content.get("images", []):
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": img.get("mime_type", "image/jpeg"), "data": img.get("base64", "")},
            })
        return blocks
    return content


def render_chat_popup(workflow, data_loader):
    """Render the chat interface as a popup panel"""
    # Don't add header here since it's added in the modal wrapper
    
    if "analysis_results" not in st.session_state:
        st.info("⚠️ Please run an analysis first in the 'Agent Analysis' tab before chatting with agents.")
        st.markdown("""
        **How to Use:**
        1. Go to **Agent Analysis** tab
        2. Run a complete breeding analysis
        3. Return here to chat with agents
        """)
        if st.button("❌ Close", key="close_chat_no_analysis"):
            st.session_state.chat_popup_open = False
            st.rerun()
        return
    
    # Close button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("❌ Close", key="close_chat_popup"):
            st.session_state.chat_popup_open = False
            st.rerun()
    
    # Agent selection
    agent_choice = st.radio(
        "Choose an agent:",
        ["🧬 Genotype Agent", "🌿 Phenotype Agent", "🌍 Environment Agent", "🎛️ Controller Agent"],
        key="popup_agent_selector"
    )
    
    # Get the appropriate agent
    agent_key = agent_choice.split()[1].lower()
    
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
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", key=f"clear_popup_{agent_key}"):
        if f"chat_history_{agent_key}" in st.session_state:
            del st.session_state[f"chat_history_{agent_key}"]
        st.rerun()
    
    st.markdown("---")
    
    # Chat history display
    chat_container = st.container(height=300)
    with chat_container:
        if len(st.session_state[f"chat_history_{agent_key}"]) == 0:
            st.info(f"👋 Hello! I'm the {agent_choice}. Ask me anything about my analysis!")
        
        # Display messages
        for msg in st.session_state[f"chat_history_{agent_key}"][-10:]:  # Show last 10 messages
            if msg["role"] == "user":
                _render_user_message(msg["content"])
            else:
                st.markdown(f"**🤖 {agent_choice}:** {msg['content']}")
            st.markdown("---")
    
    # Chat input (text + optional image upload)
    with st.form(key=f"popup_chat_form_{agent_key}", clear_on_submit=True):
        user_input = st.text_area(
            "Your question:",
            placeholder="E.g., 'Which lines showed the highest genetic diversity?' or describe an uploaded image.",
            height=80,
            key=f"popup_chat_input_{agent_key}"
        )
        uploaded_files = st.file_uploader(
            "📷 Add images (optional)",
            type=["png", "jpg", "jpeg", "gif", "webp"],
            accept_multiple_files=True,
            key=f"popup_image_upload_{agent_key}"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            submit_button = st.form_submit_button("💬 Send", use_container_width=True)
        with col_b:
            if st.form_submit_button("💡 Examples", use_container_width=True):
                st.session_state[f"show_examples_popup_{agent_key}"] = True
    
    # Show example questions if requested
    if st.session_state.get(f"show_examples_popup_{agent_key}", False):
        st.markdown("### 💡 Example Questions:")
        if "genotype" in agent_key:
            st.markdown("""
            - What are the top 5 most genetically diverse lines?
            - Are there any highly related pairs?
            - How did you calculate genetic diversity?
            """)
        elif "phenotype" in agent_key:
            st.markdown("""
            - Which traits are most strongly correlated?
            - What are the top performing lines for yield?
            - Which lines have the highest breeding values?
            """)
        elif "environment" in agent_key:
            st.markdown("""
            - Which locations were most favorable?
            - Are there strong GxE interactions?
            - Which lines show best adaptation?
            """)
        else:  # controller agent
            st.markdown("""
            - Which lines should I advance and why?
            - What are the biggest risks?
            - Give me an integrated assessment.
            """)
        st.session_state[f"show_examples_popup_{agent_key}"] = False
    
    # Process user input (allow send if there is text and/or images)
    has_text = bool(user_input and user_input.strip())
    has_images = bool(uploaded_files)
    if submit_button and (has_text or has_images):
        # Build image list for this turn (base64 + mime_type)
        user_images_this_turn = []
        if uploaded_files:
            for uf in uploaded_files:
                raw = uf.read()
                b64 = base64.b64encode(raw).decode("utf-8")
                mime = uf.type or "image/jpeg"
                user_images_this_turn.append({"base64": b64, "mime_type": mime})
        # Store in history (content can be string or dict with text + images)
        user_content = _user_content_to_history_content((user_input or "").strip(), user_images_this_turn)
        st.session_state[f"chat_history_{agent_key}"].append({
            "role": "user",
            "content": user_content
        })
        # Build chat_history for API: same list but content converted to API format
        api_history = [
            {"role": m["role"], "content": _history_content_to_api_content(m["content"])}
            for m in st.session_state[f"chat_history_{agent_key}"][:-1]
        ]
        # Get the appropriate agent from workflow
        with st.spinner(f"🤔 {agent_choice} is thinking..."):
            try:
                if agent_key == "genotype":
                    agent = workflow.genotype_agent
                elif agent_key == "phenotype":
                    agent = workflow.phenotype_agent
                elif agent_key == "environment":
                    agent = workflow.environment_agent
                else:
                    agent = workflow.controller_agent
                if agent.data is None or not agent.data:
                    processed_data = data_loader.preprocess_data()
                    agent.set_data(processed_data)
                if not agent_context:
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
                    else:
                        agent_context = {
                            "agent_analyses": agent_analyses,
                            "final_decision": results.get("final_decision", {})
                        }
                response = agent.chat(
                    user_message=(user_input or "").strip() or "",
                    chat_history=api_history,
                    analysis_context=agent_context,
                    user_images=user_images_this_turn if user_images_this_turn else None,
                )
                st.session_state[f"chat_history_{agent_key}"].append({
                    "role": "assistant",
                    "content": response
                })
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Chat error with {agent_key}: {e}")
        st.rerun()

def render_chat_interface(workflow, data_loader):
    """Render the chat interface in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💬 Chat with Agents")
    
    if "analysis_results" not in st.session_state:
        st.sidebar.info("⚠️ Please run an analysis first in the 'Agent Analysis' tab before chatting with agents.")
        st.sidebar.markdown("""
        **How to Use:**
        1. Go to **Agent Analysis** tab
        2. Run a complete breeding analysis
        3. Return here to chat with agents
        
        **Available Agents:**
        - 🧬 **Genotype Agent**: Genetic diversity, kinship, markers
        - 🌿 **Phenotype Agent**: Trait performance, correlations
        - 🌍 **Environment Agent**: Location effects, adaptations
        - 🎛️ **Controller Agent**: Integrated recommendations
        """)
        return
    
    # Agent selection
    agent_choice = st.sidebar.radio(
        "Choose an agent:",
        ["🧬 Genotype Agent", "🌿 Phenotype Agent", "🌍 Environment Agent", "🎛️ Controller Agent"],
        key="agent_selector"
    )
    
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
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
        if f"chat_history_{agent_key}" in st.session_state:
            del st.session_state[f"chat_history_{agent_key}"]
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Display chat history in expandable area
    with st.sidebar.expander(f"💬 Chat with {agent_choice}", expanded=True):
        chat_container = st.container()
        with chat_container:
            if len(st.session_state[f"chat_history_{agent_key}"]) == 0:
                st.info(f"👋 Hello! I'm the {agent_choice}. Ask me anything about my analysis!")
            
            # Display messages in reverse order for better UX in sidebar
            for msg in reversed(st.session_state[f"chat_history_{agent_key}"][-10:]):  # Show last 10 messages
                if msg["role"] == "user":
                    _render_user_message(msg["content"])
                else:
                    st.markdown(f"**🤖 {agent_choice}:** {msg['content']}")
                st.markdown("---")
        
        # Chat input (text + optional image upload)
        with st.form(key=f"chat_form_{agent_key}", clear_on_submit=True):
            user_input = st.text_area(
                "Your question:",
                placeholder="E.g., 'Which lines showed the highest genetic diversity?' or describe an uploaded image.",
                height=80,
                key=f"chat_input_{agent_key}"
            )
            uploaded_files = st.file_uploader(
                "📷 Add images (optional)",
                type=["png", "jpg", "jpeg", "gif", "webp"],
                accept_multiple_files=True,
                key=f"sidebar_image_upload_{agent_key}"
            )
            col_a, col_b = st.columns(2)
            with col_a:
                submit_button = st.form_submit_button("💬 Send", use_container_width=True)
            with col_b:
                if st.form_submit_button("💡 Examples", use_container_width=True):
                    st.session_state[f"show_examples_{agent_key}"] = True
        
        # Show example questions if requested
        if st.session_state.get(f"show_examples_{agent_key}", False):
            st.markdown("### 💡 Example Questions:")
            if "genotype" in agent_key:
                st.markdown("""
                - What are the top 5 most genetically diverse lines?
                - Are there any highly related pairs?
                - How did you calculate genetic diversity?
                """)
            elif "phenotype" in agent_key:
                st.markdown("""
                - Which traits are most strongly correlated?
                - What are the top performing lines for yield?
                - Which lines have the highest breeding values?
                """)
            elif "environment" in agent_key:
                st.markdown("""
                - Which locations were most favorable?
                - Are there strong GxE interactions?
                - Which lines show best adaptation?
                """)
            else:  # controller agent
                st.markdown("""
                - Which lines should I advance and why?
                - What are the biggest risks?
                - Give me an integrated assessment.
                """)
            st.session_state[f"show_examples_{agent_key}"] = False
        
        # Process user input (allow send if there is text and/or images)
        has_text = bool(user_input and user_input.strip())
        has_images = bool(uploaded_files)
        if submit_button and (has_text or has_images):
            user_images_this_turn = []
            if uploaded_files:
                for uf in uploaded_files:
                    raw = uf.read()
                    b64 = base64.b64encode(raw).decode("utf-8")
                    mime = uf.type or "image/jpeg"
                    user_images_this_turn.append({"base64": b64, "mime_type": mime})
            user_content = _user_content_to_history_content((user_input or "").strip(), user_images_this_turn)
            st.session_state[f"chat_history_{agent_key}"].append({
                "role": "user",
                "content": user_content
            })
            api_history = [
                {"role": m["role"], "content": _history_content_to_api_content(m["content"])}
                for m in st.session_state[f"chat_history_{agent_key}"][:-1]
            ]
            with st.spinner(f"🤔 {agent_choice} is thinking..."):
                try:
                    if agent_key == "genotype":
                        agent = workflow.genotype_agent
                    elif agent_key == "phenotype":
                        agent = workflow.phenotype_agent
                    elif agent_key == "environment":
                        agent = workflow.environment_agent
                    else:
                        agent = workflow.controller_agent
                    if agent.data is None or not agent.data:
                        processed_data = data_loader.preprocess_data()
                        agent.set_data(processed_data)
                    if not agent_context:
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
                        else:
                            agent_context = {
                                "agent_analyses": agent_analyses,
                                "final_decision": results.get("final_decision", {})
                            }
                    response = agent.chat(
                        user_message=(user_input or "").strip() or "",
                        chat_history=api_history,
                        analysis_context=agent_context,
                        user_images=user_images_this_turn if user_images_this_turn else None,
                    )
                    st.session_state[f"chat_history_{agent_key}"].append({
                        "role": "assistant",
                        "content": response
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Chat error with {agent_key}: {e}")
            st.rerun()

def main():
    """Main application function"""
    
    # Check password first
    if not check_password():
        return

    # Initialize chat popup state - auto-enabled by default
    if "chat_popup_open" not in st.session_state:
        st.session_state.chat_popup_open = True

    # Header
    st.markdown('<h1 class="main-header">🌱 AgriAgent: AI-Powered Breeding Decisions</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 1rem;">Multi-Agent AI Framework for Agricultural Breeding Line Advancement</p>', unsafe_allow_html=True)
    
    # Initialize workflow and data loader early (needed for sidebar chat)
    @st.cache_resource
    def get_workflow():
        return AgriAgentWorkflow()

    @st.cache_resource
    def get_data_loader():
        return DataLoader()

    workflow = get_workflow()
    data_loader = get_data_loader()
    
    # Add CSS for right sidebar styling (always open)
    if st.session_state.chat_popup_open:
        st.markdown("""
        <style>
            /* Style the right sidebar column */
            div[data-testid="column"]:last-of-type {
                background: white;
                border-left: 3px solid #2E8B57;
                padding: 10px !important;
            }
        </style>
        """, unsafe_allow_html=True)

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

        num_runs = st.number_input(
            "Number of runs",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            help="Run the analysis this many times; use the selector below to view each run."
        )

        # Run analysis button
        run_analysis = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True
        )

        # Initialize multi-run state
        if "analysis_runs" not in st.session_state:
            st.session_state.analysis_runs = []
        if "current_run_index" not in st.session_state:
            st.session_state.current_run_index = 0

        # Run selector: which run we're currently viewing
        if st.session_state.analysis_runs:
            runs = st.session_state.analysis_runs
            current = min(st.session_state.current_run_index, len(runs) - 1)
            options = [f"Run {i+1}" for i in range(len(runs))]
            selected = st.selectbox(
                "View run",
                options,
                index=current,
                key="sidebar_run_selector",
                help="Switch which run's results are shown in the app."
            )
            idx = options.index(selected)
            st.session_state.current_run_index = idx
            st.session_state.analysis_results = runs[idx]
        elif "analysis_results" in st.session_state:
            # Legacy: we had a single run stored as analysis_results; keep it
            pass

    # Main content area with optional right sidebar
    if st.session_state.chat_popup_open:
        # Create two-column layout: main content on left, chat sidebar on right
        main_col, chat_sidebar_col = st.columns([2.5, 1])
        
        # Main content in left column
        with main_col:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔬 Agent Analysis", "📈 Data Visualization", "📋 Reports", "🌱 Plant Management"])
        
        # Right sidebar chat panel (always visible)
        with chat_sidebar_col:
            # Sidebar header
            st.markdown("""
            <div style="background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%); 
                       color: white; padding: 12px 15px; border-radius: 8px; margin-bottom: 15px;">
                <h4 style="margin: 0; color: white; font-size: 1.1rem;">💬 Chat with Agents</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Render chat content
            render_chat_popup(workflow, data_loader)

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
            status_placeholder = st.empty()
            progress_placeholder = st.empty()

            with st.spinner("🤖 Running AgriAgent Analysis..."):
                try:
                    query = f"Perform {analysis_type.lower()} for breeding line advancement"
                    context = {
                        "advancement_threshold": advancement_threshold,
                        "top_percentage": top_percentage
                    }

                    all_runs = []
                    for run_i in range(num_runs):
                        pct_base = (run_i / num_runs) * 100
                        pct_next = ((run_i + 1) / num_runs) * 100
                        status_placeholder.info(f"🔄 Run {run_i + 1} of {num_runs} — initializing...")
                        progress_placeholder.progress(pct_base / 100)

                        result_state = workflow.run_sync_workflow(query, context)
                        all_runs.append(result_state)

                        progress_placeholder.progress(pct_next / 100)
                        status_placeholder.info(f"✅ Run {run_i + 1} of {num_runs} completed.")

                    status_placeholder.success(f"✅ All {num_runs} run(s) completed successfully!")
                    progress_placeholder.progress(1.0)

                    # Store all runs and set current view to first run
                    st.session_state.analysis_runs = all_runs
                    st.session_state.current_run_index = 0
                    st.session_state.analysis_results = all_runs[0]

                    # Show agent status for the last run
                    agent_analyses = all_runs[-1].get("agent_analyses", {})
                    st.subheader("🤖 Agent Execution Status (last run)")
                    agent_status = {
                        "Genotype Agent": "✅ Completed" if agent_analyses.get("genotype") else "⏸️ Not Run",
                        "Phenotype Agent": "✅ Completed" if agent_analyses.get("phenotype") else "⏸️ Not Run",
                        "Environment Agent": "✅ Completed" if agent_analyses.get("environment") else "⏸️ Not Run"
                    }
                    for agent, status in agent_status.items():
                        st.write(f"**{agent}:** {status}")

                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    logger.error(f"Workflow execution failed: {e}")

        # Display stored results if available
        if "analysis_results" in st.session_state:
            results = st.session_state.analysis_results
            runs = st.session_state.get("analysis_runs", [])
            current_idx = st.session_state.get("current_run_index", 0)
            if len(runs) > 1:
                run_options = [f"Run {i+1}" for i in range(len(runs))]
                sel = st.selectbox(
                    "**View run** (which run’s results are shown)",
                    run_options,
                    index=current_idx,
                    key="tab2_run_selector"
                )
                new_idx = run_options.index(sel)
                if new_idx != current_idx:
                    st.session_state.current_run_index = new_idx
                    st.session_state.analysis_results = runs[new_idx]
                    results = st.session_state.analysis_results
                st.caption(f"Showing run **{st.session_state.current_run_index + 1}** of **{len(runs)}**.")

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

    # Plant Management Tab
    with tab5:
        st.header("🌱 Plant Management")
        st.markdown("Browse and explore individual plants with their genotype and phenotype information")
        
        # Load plant data
        genotype_df = load_genotype_data()
        phenotype_df = load_phenotype_data()
        
        if genotype_df.empty:
            st.error("❌ Could not load genotype data. Please check if processed_genotype.csv exists in data/correlated/")
            st.info(f"**Debug Info:** Attempted to load from: `{_get_data_path('processed_genotype.csv')}`")
            st.markdown("""
            **Troubleshooting:**
            - Ensure `data/correlated/processed_genotype.csv` exists in your project
            - Check file permissions
            - Verify the file path is correct in your deployment environment
            """)
        else:
            # Get all plant IDs
            plant_ids = get_plant_ids_from_genotype(genotype_df)
            
            if not plant_ids:
                st.warning("⚠️ No plant IDs found in genotype data")
            else:
                # Search and filter section
                col1, col2 = st.columns([2, 1])
                with col1:
                    search_query = st.text_input(
                        "🔍 Search plants",
                        placeholder="Enter plant ID (e.g., C002, C003)...",
                        help="Type a plant ID to filter the list"
                    )
                with col2:
                    sort_option = st.selectbox(
                        "Sort by",
                        ["ID (Ascending)", "ID (Descending)"],
                        help="Sort plant list"
                    )
                
                # Filter plants based on search
                filtered_plants = plant_ids
                if search_query:
                    filtered_plants = [p for p in plant_ids if search_query.upper() in p.upper()]
                
                # Sort plants
                if sort_option == "ID (Descending)":
                    filtered_plants = sorted(filtered_plants, reverse=True)
                else:
                    filtered_plants = sorted(filtered_plants)
                
                st.markdown(f"**Total Plants:** {len(plant_ids)} | **Filtered:** {len(filtered_plants)}")
                st.markdown("---")
                
                # Plant selection
                if filtered_plants:
                    # Create a grid layout for plant cards
                    cols_per_row = 4
                    num_rows = (len(filtered_plants) + cols_per_row - 1) // cols_per_row
                    
                    # Initialize selected plant if not in session state or if current selection is not in filtered list
                    if 'selected_plant' not in st.session_state or st.session_state.get('selected_plant') not in filtered_plants:
                        st.session_state.selected_plant = filtered_plants[0]
                    
                    selected_plant = st.session_state.get('selected_plant')
                    
                    # Display plant cards in a grid
                    for row in range(num_rows):
                        cols = st.columns(cols_per_row)
                        for col_idx, col in enumerate(cols):
                            plant_idx = row * cols_per_row + col_idx
                            if plant_idx < len(filtered_plants):
                                plant_id = filtered_plants[plant_idx]
                                with col:
                                    # Create a card-like button for each plant
                                    is_selected = (selected_plant == plant_id)
                                    
                                    if st.button(
                                        f"🌱 {plant_id}",
                                        key=f"plant_btn_{plant_id}",
                                        use_container_width=True,
                                        type="primary" if is_selected else "secondary"
                                    ):
                                        st.session_state.selected_plant = plant_id
                                        st.rerun()
                                    
                                    # Show selection indicator
                                    if is_selected:
                                        st.markdown(f"<div style='text-align: center; color: #2E8B57; font-weight: bold;'>✓ Selected</div>", 
                                                   unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Display detailed information for selected plant
                    if selected_plant and selected_plant in plant_ids:
                        st.subheader(f"📋 Plant Details: {selected_plant}")
                        
                        # Create tabs for different information views
                        detail_tabs = st.tabs(["🧬 Genotype", "🌿 Phenotype", "📊 Summary"])
                        
                        with detail_tabs[0]:
                            st.markdown(f"### 🧬 Genotype Information for {selected_plant}")
                            
                            plant_genotype = get_plant_genotype(genotype_df, selected_plant)
                            
                            if not plant_genotype.empty:
                                # Summary statistics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                total_markers = len(plant_genotype)
                                with col1:
                                    st.metric("Total Markers", f"{total_markers:,}")
                                
                                # Count different allele types
                                allele_counts = plant_genotype[selected_plant].value_counts()
                                with col2:
                                    st.metric("Unique Alleles", len(allele_counts))
                                
                                # Count chromosomes
                                if 'chrom' in plant_genotype.columns:
                                    unique_chroms = plant_genotype['chrom'].nunique()
                                    with col3:
                                        st.metric("Chromosomes", unique_chroms)
                                
                                # Missing data
                                missing_count = plant_genotype[selected_plant].isna().sum()
                                missing_pct = (missing_count / total_markers * 100) if total_markers > 0 else 0
                                with col4:
                                    st.metric("Missing Data", f"{missing_pct:.1f}%")
                                
                                st.markdown("---")
                                
                                # Allele distribution
                                if len(allele_counts) > 0:
                                    st.markdown("**Allele Distribution:**")
                                    fig = px.bar(
                                        x=allele_counts.index,
                                        y=allele_counts.values,
                                        labels={'x': 'Allele', 'y': 'Count'},
                                        title=f"Allele Distribution for {selected_plant}"
                                    )
                                    fig.update_layout(height=300)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Data table with search and pagination
                                st.markdown("**Marker Details:**")
                                
                                # Search within markers
                                marker_search = st.text_input(
                                    "🔍 Search markers",
                                    placeholder="Search by marker ID, chromosome, or position...",
                                    key=f"marker_search_{selected_plant}"
                                )
                                
                                display_df = plant_genotype.copy()
                                if marker_search:
                                    mask = pd.Series([False] * len(display_df))
                                    for col in display_df.columns:
                                        if col != selected_plant:
                                            mask |= display_df[col].astype(str).str.contains(
                                                marker_search, na=False, case=False
                                            )
                                    display_df = display_df[mask]
                                
                                # Show data table
                                st.dataframe(
                                    display_df,
                                    use_container_width=True,
                                    height=400,
                                    hide_index=True
                                )
                                
                                # Download button
                                csv = display_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Genotype Data (CSV)",
                                    data=csv,
                                    file_name=f"{selected_plant}_genotype.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning(f"No genotype data available for {selected_plant}")
                        
                        with detail_tabs[1]:
                            st.markdown(f"### 🌿 Phenotype Information for {selected_plant}")
                            
                            plant_phenotype = get_plant_phenotype(phenotype_df, selected_plant)
                            
                            if not plant_phenotype.empty:
                                # Summary statistics
                                numeric_cols = plant_phenotype.select_dtypes(include=[float, int]).columns
                                
                                if len(numeric_cols) > 0:
                                    st.markdown("**Trait Summary:**")
                                    
                                    # Create metrics for key traits
                                    num_traits = len(numeric_cols)
                                    cols_per_row = min(4, num_traits)
                                    if num_traits > 0:
                                        cols = st.columns(cols_per_row)
                                        for idx, trait in enumerate(numeric_cols[:cols_per_row]):
                                            with cols[idx % cols_per_row]:
                                                mean_val = plant_phenotype[trait].mean()
                                                st.metric(
                                                    trait.replace('_', ' ').title(),
                                                    f"{mean_val:.2f}" if pd.notna(mean_val) else "N/A"
                                                )
                                    
                                    # Trait distributions
                                    if len(numeric_cols) > 0:
                                        st.markdown("**Trait Distributions:**")
                                        selected_trait = st.selectbox(
                                            "Select trait to visualize",
                                            numeric_cols.tolist(),
                                            key=f"trait_select_{selected_plant}"
                                        )
                                        
                                        if selected_trait:
                                            fig = px.histogram(
                                                plant_phenotype,
                                                x=selected_trait,
                                                title=f"{selected_trait.replace('_', ' ').title()} Distribution for {selected_plant}",
                                                marginal="box"
                                            )
                                            fig.update_layout(height=400)
                                            st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("---")
                                
                                # Data table
                                st.markdown("**Phenotype Data:**")
                                st.dataframe(
                                    plant_phenotype,
                                    use_container_width=True,
                                    height=400,
                                    hide_index=True
                                )
                                
                                # Download button
                                csv = plant_phenotype.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Phenotype Data (CSV)",
                                    data=csv,
                                    file_name=f"{selected_plant}_phenotype.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.info(f"🌿 No phenotype data found for {selected_plant} in the dataset")
                                st.markdown("""
                                **Note:** Phenotype data may be stored under different identifiers.
                                Try searching for the plant ID in the original data files.
                                """)
                        
                        with detail_tabs[2]:
                            st.markdown(f"### 📊 Summary for {selected_plant}")
                            
                            # Combined summary
                            summary_col1, summary_col2 = st.columns(2)
                            
                            with summary_col1:
                                st.markdown("**🧬 Genotype Summary**")
                                plant_genotype = get_plant_genotype(genotype_df, selected_plant)
                                if not plant_genotype.empty:
                                    st.write(f"• **Total Markers:** {len(plant_genotype):,}")
                                    if 'chrom' in plant_genotype.columns:
                                        st.write(f"• **Chromosomes:** {plant_genotype['chrom'].nunique()}")
                                    allele_counts = plant_genotype[selected_plant].value_counts()
                                    st.write(f"• **Unique Alleles:** {len(allele_counts)}")
                                    if len(allele_counts) > 0:
                                        st.write(f"• **Most Common Allele:** {allele_counts.index[0]} ({allele_counts.iloc[0]} occurrences)")
                                else:
                                    st.write("• No genotype data available")
                            
                            with summary_col2:
                                st.markdown("**🌿 Phenotype Summary**")
                                plant_phenotype = get_plant_phenotype(phenotype_df, selected_plant)
                                if not plant_phenotype.empty:
                                    numeric_cols = plant_phenotype.select_dtypes(include=[float, int]).columns
                                    st.write(f"• **Records:** {len(plant_phenotype)}")
                                    st.write(f"• **Traits:** {len(numeric_cols)}")
                                    if len(numeric_cols) > 0:
                                        # Show a few key traits
                                        for trait in numeric_cols[:3]:
                                            mean_val = plant_phenotype[trait].mean()
                                            if pd.notna(mean_val):
                                                st.write(f"• **{trait.replace('_', ' ').title()}:** {mean_val:.2f}")
                                else:
                                    st.write("• No phenotype data available")
                            
                            # Quick actions
                            st.markdown("---")
                            st.markdown("**⚡ Quick Actions**")
                            action_col1, action_col2, action_col3 = st.columns(3)
                            
                            with action_col1:
                                if st.button("📊 View Full Genotype", key=f"view_geno_{selected_plant}"):
                                    st.session_state[f"show_full_geno_{selected_plant}"] = True
                            
                            with action_col2:
                                if st.button("🌿 View Full Phenotype", key=f"view_pheno_{selected_plant}"):
                                    st.session_state[f"show_full_pheno_{selected_plant}"] = True
                            
                            with action_col3:
                                if st.button("📥 Export All Data", key=f"export_{selected_plant}"):
                                    # Combine genotype and phenotype
                                    combined_data = {}
                                    if not plant_genotype.empty:
                                        combined_data['genotype'] = plant_genotype
                                    if not plant_phenotype.empty:
                                        combined_data['phenotype'] = plant_phenotype
                                    
                                    if combined_data:
                                        st.success(f"✅ Data for {selected_plant} is ready for export")
                                        # In a real implementation, you could create a zip file or combined CSV
                    else:
                        st.info("👆 Select a plant from the grid above to view details")
                else:
                    st.warning(f"⚠️ No plants found matching '{search_query}'")

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
