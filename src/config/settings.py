import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_streamlit_secret(section, key, default=None):
    """Safely get Streamlit secrets for cloud deployment"""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and section in st.secrets:
            return st.secrets[section].get(key, default)
    except:
        pass
    return default

class Config:
    # Claude API Configuration (Anthropic)
    # Try Streamlit secrets first (for cloud deployment), then env vars (for local)
    ANTHROPIC_API_KEY = (
        get_streamlit_secret('anthropic', 'ANTHROPIC_API_KEY') or
        os.getenv('ANTHROPIC_API_KEY', 'your_claude_api_key_here')
    )

    # Data Configuration
    DATA_FILE_PATH = (
        get_streamlit_secret('data', 'DATA_FILE_PATH') or
        os.getenv('DATA_FILE_PATH', 'data/IYT_DATA_UCLA.xls')
    )
    PROCESSED_DATA_DIR = (
        get_streamlit_secret('data', 'PROCESSED_DATA_DIR') or
        os.getenv('PROCESSED_DATA_DIR', 'outputs/processed/')
    )

    # Logging Configuration
    LOG_LEVEL = (
        get_streamlit_secret('logging', 'LOG_LEVEL') or
        os.getenv('LOG_LEVEL', 'INFO')
    )
    LOG_FILE = (
        get_streamlit_secret('logging', 'LOG_FILE') or
        os.getenv('LOG_FILE', 'logs/agriagent.log')
    )

    # Web Interface Configuration
    STREAMLIT_PORT = int(os.getenv('STREAMLIT_PORT', 8501))
    STREAMLIT_HOST = os.getenv('STREAMLIT_HOST', 'localhost')

    # Model Configuration (Claude models)
    GENOTYPE_MODEL = (
        get_streamlit_secret('models', 'GENOTYPE_MODEL') or
        os.getenv('GENOTYPE_MODEL', 'claude-3-5-sonnet-20241022')
    )
    PHENOTYPE_MODEL = (
        get_streamlit_secret('models', 'PHENOTYPE_MODEL') or
        os.getenv('PHENOTYPE_MODEL', 'claude-3-5-sonnet-20241022')
    )
    ENVIRONMENT_MODEL = (
        get_streamlit_secret('models', 'ENVIRONMENT_MODEL') or
        os.getenv('ENVIRONMENT_MODEL', 'claude-3-5-sonnet-20241022')
    )
    CONTROLLER_MODEL = (
        get_streamlit_secret('models', 'CONTROLLER_MODEL') or
        os.getenv('CONTROLLER_MODEL', 'claude-3-5-sonnet-20241022')
    )

    # Data Processing Configuration
    CHUNK_SIZE = 1000
    MAX_TOKENS_PER_CHUNK = 4000

    # Breeding Program Configuration
    TARGET_TRAITS = [
        'yield', 'pod_density', 'plant_height', 'days_to_maturity',
        'oil_content', 'protein_content', 'lodging_resistance'
    ]

    # Decision Thresholds
    ADVANCEMENT_THRESHOLD = float(
        get_streamlit_secret('thresholds', 'ADVANCEMENT_THRESHOLD') or
        os.getenv('ADVANCEMENT_THRESHOLD', 0.7)
    )
    TOP_LINES_PERCENTAGE = float(
        get_streamlit_secret('thresholds', 'TOP_LINES_PERCENTAGE') or
        os.getenv('TOP_LINES_PERCENTAGE', 0.1)
    )

# Global configuration instance
config = Config()
