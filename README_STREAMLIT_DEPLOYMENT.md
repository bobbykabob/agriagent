# 🚀 Deploying AgriAgent to Streamlit Cloud

## Overview
This guide walks you through deploying AgriAgent to Streamlit Cloud, ensuring your API keys remain secure.

## 📋 Prerequisites

1. **GitHub Account** - To host your code repository
2. **Streamlit Cloud Account** - Free at [share.streamlit.io](https://share.streamlit.io)
3. **Claude API Key** - From [Anthropic Console](https://console.anthropic.com/)

## 🔐 Step 1: Prepare Your Repository

### 1.1 Verify .gitignore

Make sure `.gitignore` includes:
```
.env
.env.local
.streamlit/secrets.toml
```

⚠️ **Critical:** Never commit API keys!

### 1.2 Check Your Files

Your repository should have:
- ✅ `requirements.txt` - All dependencies listed
- ✅ `web_interface/app.py` - Main application
- ✅ `data/IYT_DATA_UCLA.xls` - Your data file
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `.streamlit/secrets.toml.example` - Example secrets file
- ✅ `.gitignore` - Excluding sensitive files

### 1.3 Update src/config/settings.py

Ensure it reads from both environment variables AND Streamlit secrets:

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Try Streamlit secrets first, then environment variables
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            ANTHROPIC_API_KEY = st.secrets.get('anthropic', {}).get('ANTHROPIC_API_KEY', 
                                              os.getenv('ANTHROPIC_API_KEY', 'your_claude_api_key_here'))
        else:
            ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', 'your_claude_api_key_here')
    except:
        ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', 'your_claude_api_key_here')
    
    DATA_FILE_PATH = os.getenv('DATA_FILE_PATH', 'data/IYT_DATA_UCLA.xls')
    # ... rest of config
```

## 🌐 Step 2: Create GitHub Repository

```bash
# Initialize git (if not already done)
cd /path/to/flax-and-canola
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: AgriAgent with UCLA and structures.computer credits"

# Create a new repository on GitHub
# Then push to it
git remote add origin https://github.com/yourusername/agriagent.git
git branch -M main
git push -u origin main
```

⚠️ **Before pushing:** Verify `.env` is NOT being tracked:
```bash
git status
# Should NOT see .env or secrets.toml listed
```

## ☁️ Step 3: Deploy to Streamlit Cloud

### 3.1 Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `yourusername/agriagent`
5. Set main file path: `web_interface/app.py`
6. Click "Advanced settings"

### 3.2 Configure Secrets

In the "Secrets" section, paste your configuration:

```toml
[anthropic]
ANTHROPIC_API_KEY = "sk-ant-api03-YOUR-ACTUAL-KEY-HERE"

[data]
DATA_FILE_PATH = "data/IYT_DATA_UCLA.xls"
PROCESSED_DATA_DIR = "outputs/processed/"

[logging]
LOG_LEVEL = "INFO"
LOG_FILE = "logs/agriagent.log"

[models]
GENOTYPE_MODEL = "claude-3-5-sonnet-20241022"
PHENOTYPE_MODEL = "claude-3-5-sonnet-20241022"
ENVIRONMENT_MODEL = "claude-3-5-sonnet-20241022"
CONTROLLER_MODEL = "claude-3-5-sonnet-20241022"

[thresholds]
ADVANCEMENT_THRESHOLD = 0.7
TOP_LINES_PERCENTAGE = 0.1
```

### 3.3 Deploy

Click "Deploy!" and wait for the app to build.

Your app will be available at: `https://yourusername-agriagent-xxxxx.streamlit.app`

## 🔧 Step 4: Update Config for Streamlit Cloud

Update `src/config/settings.py` to read from Streamlit secrets:

```python
import os
from dotenv import load_dotenv

load_dotenv()

def get_streamlit_secret(section, key, default=None):
    """Safely get Streamlit secrets"""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and section in st.secrets:
            return st.secrets[section].get(key, default)
    except:
        pass
    return default

class Config:
    # API Key - Try Streamlit secrets first, then env vars
    ANTHROPIC_API_KEY = (
        get_streamlit_secret('anthropic', 'ANTHROPIC_API_KEY') or
        os.getenv('ANTHROPIC_API_KEY', 'your_claude_api_key_here')
    )
    
    # Data paths
    DATA_FILE_PATH = (
        get_streamlit_secret('data', 'DATA_FILE_PATH') or
        os.getenv('DATA_FILE_PATH', 'data/IYT_DATA_UCLA.xls')
    )
    
    PROCESSED_DATA_DIR = (
        get_streamlit_secret('data', 'PROCESSED_DATA_DIR') or
        os.getenv('PROCESSED_DATA_DIR', 'outputs/processed/')
    )
    
    # Logging
    LOG_LEVEL = (
        get_streamlit_secret('logging', 'LOG_LEVEL') or
        os.getenv('LOG_LEVEL', 'INFO')
    )
    
    LOG_FILE = (
        get_streamlit_secret('logging', 'LOG_FILE') or
        os.getenv('LOG_FILE', 'logs/agriagent.log')
    )
    
    # Model names
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
    
    # Thresholds
    ADVANCEMENT_THRESHOLD = float(
        get_streamlit_secret('thresholds', 'ADVANCEMENT_THRESHOLD') or
        os.getenv('ADVANCEMENT_THRESHOLD', 0.7)
    )
    
    TOP_LINES_PERCENTAGE = float(
        get_streamlit_secret('thresholds', 'TOP_LINES_PERCENTAGE') or
        os.getenv('TOP_LINES_PERCENTAGE', 0.1)
    )
    
    # Static configs
    STREAMLIT_PORT = int(os.getenv('STREAMLIT_PORT', 8501))
    STREAMLIT_HOST = os.getenv('STREAMLIT_HOST', 'localhost')
    CHUNK_SIZE = 1000
    MAX_TOKENS_PER_CHUNK = 4000
    TARGET_TRAITS = [
        'yield', 'pod_density', 'plant_height', 'days_to_maturity',
        'oil_content', 'protein_content', 'lodging_resistance'
    ]

config = Config()
```

## ✅ Step 5: Verify Deployment

1. **Check Logs** - Look for any errors in Streamlit Cloud dashboard
2. **Test Analysis** - Run a multi-agent analysis
3. **Verify Data** - Ensure Excel file is being loaded correctly
4. **Check Credits** - Confirm UCLA logo and @structures.computer appear in footer

## 🔍 Troubleshooting

### Issue: "API Key Not Found"
- **Solution:** Check Streamlit Cloud secrets are configured correctly
- Verify the secret section names match: `[anthropic]`, not `[ANTHROPIC]`

### Issue: "Data File Not Found"
- **Solution:** Ensure `data/IYT_DATA_UCLA.xls` is committed to your repo
- Check the path in secrets: `DATA_FILE_PATH = "data/IYT_DATA_UCLA.xls"`

### Issue: "Module Not Found"
- **Solution:** Verify all dependencies are in `requirements.txt`
- Check Streamlit Cloud build logs

### Issue: "Out of Memory"
- **Solution:** Streamlit Cloud free tier has memory limits
- Consider reducing data processing or upgrading to paid tier

## 📊 Post-Deployment Checklist

- [ ] Application loads without errors
- [ ] Excel data is being processed correctly
- [ ] Multi-agent analysis runs successfully
- [ ] UCLA logo appears in footer
- [ ] @structures.computer credit appears in footer
- [ ] Train of thought displays properly
- [ ] Workflow diagram renders with correct colors
- [ ] System status shows black text
- [ ] No API keys are visible in the UI or source code

## 🎉 Success!

Your AgriAgent is now deployed and accessible to collaborators!

**Share your app URL:**
`https://yourusername-agriagent-xxxxx.streamlit.app`

## 🔐 Security Best Practices

1. ✅ Never commit `.env` or `secrets.toml`
2. ✅ Use Streamlit Cloud secrets for sensitive data
3. ✅ Rotate API keys regularly
4. ✅ Monitor API usage in Anthropic Console
5. ✅ Set up billing alerts
6. ✅ Keep dependencies updated

## 📞 Support Resources

- **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)
- **Anthropic Console:** [console.anthropic.com](https://console.anthropic.com/)
- **GitHub Issues:** Create issues in your repository

---

**🌱 Happy Breeding Analysis with AgriAgent!**

*Built with ❤️ at UCLA in collaboration with @structures.computer Lab*

