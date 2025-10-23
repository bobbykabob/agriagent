# 🚀 Quick Deployment Guide

## Local Testing (Right Now)

```bash
# 1. Make sure your .env file has your Claude API key
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# 2. Run the application
streamlit run web_interface/app.py

# 3. Access at http://localhost:8501
```

## Deploy to Streamlit Cloud (5 Minutes)

### Step 1: Create GitHub Repository

```bash
cd /Users/zhenyusong/Desktop/mae199/flax-and-canola

# Initialize git if needed
git init

# Add all files (API keys are excluded by .gitignore)
git add .

# Commit
git commit -m "AgriAgent: Multi-agent breeding analysis system"

# Create new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/agriagent.git
git branch -M main
git push -u origin main
```

⚠️ **VERIFY:** Run `git status` - you should NOT see `.env` listed!

### Step 2: Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Select:
   - Repository: `YOUR_USERNAME/agriagent`
   - Branch: `main`
   - Main file: `web_interface/app.py`

5. Click **"Advanced settings"**

6. In **Secrets**, paste this (with YOUR actual key):

```toml
[anthropic]
ANTHROPIC_API_KEY = "sk-ant-api03-YOUR-ACTUAL-CLAUDE-KEY-HERE"

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

7. Click **"Deploy"**

### Step 3: Share Your App

Your app will be live at:
```
https://YOUR_USERNAME-agriagent-xxxx.streamlit.app
```

## ✅ Features You'll See

- 🔄 **Visual Workflow Diagram** - Shows agent collaboration
- 🧬 **Genotype Agent** - Genetic diversity analysis
- 🌿 **Phenotype Agent** - Trait correlation analysis
- 🌍 **Environment Agent** - Location effects
- 🎛️ **Controller Agent** - Integrated decisions
- 🧠 **Train of Thought** - Complete Claude reasoning
- 🏫 **UCLA Logo** in footer
- 🤝 **@structures.computer Lab** credits

## 🔐 Security Checklist

- ✅ `.env` is in `.gitignore`
- ✅ `.streamlit/secrets.toml` is in `.gitignore`
- ✅ API key is only in Streamlit Cloud secrets
- ✅ No keys visible in code or UI

## 📊 Excel Data Integration

Your `data/IYT_DATA_UCLA.xls` is:
- ✅ Automatically loaded on startup
- ✅ Preprocessed into genotype/phenotype/environment categories
- ✅ Fed directly into the multi-agent system
- ✅ Used for all breeding line recommendations

**Data Flow:**
```
Excel File → DataLoader → Agents → Analysis → Recommendations
```

## 🎯 What's Working

1. **System Status Cards** - Now with black text ✅
2. **Excel Integration** - 900 breeding lines loaded ✅
3. **UCLA Logo** - Displayed in footer ✅
4. **@structures.computer** - Credited in footer ✅
5. **Claude AI** - Powering all agents ✅
6. **Train of Thought** - Full reasoning visible ✅
7. **Workflow Diagram** - Darker colors, black text ✅

## 🐛 Troubleshooting

**App won't start?**
- Check Streamlit Cloud logs
- Verify secrets are pasted correctly
- Ensure `requirements.txt` has all dependencies

**Data not loading?**
- Verify `data/IYT_DATA_UCLA.xls` is in your repo
- Check file path in secrets

**API errors?**
- Verify Claude API key is valid
- Check you have credits in Anthropic Console

## 📞 Need Help?

Check the full guide: `README_STREAMLIT_DEPLOYMENT.md`

---

**🌱 You're ready to deploy AgriAgent!**

*Built at UCLA 🏫 | Collaboration with @structures.computer 🤝 | Powered by Claude AI 🤖*

