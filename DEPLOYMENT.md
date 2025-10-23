# AgriAgent Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git
- Your Claude API key

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd flax-and-canola

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env file with your Claude API key
```

### 2. Configure API Key

**Edit `.env` file:**
```bash
# Claude API Configuration (Anthropic)
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here

# Data Configuration
DATA_FILE_PATH=data/IYT_DATA_UCLA.xls
PROCESSED_DATA_DIR=outputs/processed/

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/agriagent.log

# Web Interface Configuration
STREAMLIT_PORT=8501
STREAMLIT_HOST=localhost

# Model Configuration (Claude models)
GENOTYPE_MODEL=claude-3-5-sonnet-20241022
PHENOTYPE_MODEL=claude-3-5-sonnet-20241022
ENVIRONMENT_MODEL=claude-3-5-sonnet-20241022
CONTROLLER_MODEL=claude-3-5-sonnet-20241022

# Decision Thresholds
ADVANCEMENT_THRESHOLD=0.7
TOP_LINES_PERCENTAGE=0.1
```

### 3. Run the Application

```bash
# Start the web interface
python run.py web

# Or run directly with Streamlit
streamlit run web_interface/app.py
```

### 4. Access the Application

Open your browser and navigate to: **http://localhost:8501**

## 📊 Features Available

### 🔬 Multi-Agent Analysis
- **🧬 Genotype Agent** - Genetic diversity and kinship analysis
- **🌿 Phenotype Agent** - Trait correlations and performance ranking
- **🌍 Environment Agent** - Location effects and stability analysis
- **🎛️ Controller Agent** - Integrated decision making

### 🎯 Enhanced Visualizations
- **🔄 Interactive Workflow Diagrams** - Visual representation of agent collaboration
- **💡 Agent Insights Tabs** - Detailed results from each agent
- **🧠 Train of Thought Display** - Complete Claude reasoning process
- **📈 Data Visualizations** - Interactive plots and charts

### 🤖 AI Integration
- **Claude 3.5 Sonnet** - Advanced reasoning for breeding analysis
- **Chain of Thought** - Transparent decision-making process
- **Parallel Processing** - Multiple agents working simultaneously

## 🔐 Security Notes

⚠️ **Important:** Never commit API keys to version control!

**Safe Deployment:**
1. Use environment variables (`.env` files)
2. Add `.env` to `.gitignore`
3. Use deployment platforms with secret management
4. Rotate API keys regularly

## 🚀 Deployment Options

### Option 1: Local Development
```bash
python run.py web
# Access at http://localhost:8501
```

### Option 2: Production Deployment
```bash
# Using Streamlit Cloud, Heroku, or similar platforms
streamlit run web_interface/app.py --server.port $PORT --server.address 0.0.0.0
```

### Option 3: Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "web_interface/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

## 📋 Data Requirements

- **Excel file** (`.xls` format) with breeding data
- **Required columns**: entry/plot identifiers, trait measurements
- **Optional**: location/environmental data for enhanced analysis

## 🔧 Configuration

### Environment Variables
- `ANTHROPIC_API_KEY` - Your Claude API key
- `DATA_FILE_PATH` - Path to your Excel file
- `STREAMLIT_PORT` - Web interface port (default: 8501)
- `LOG_LEVEL` - Logging level (INFO, DEBUG, ERROR)

### Model Selection
- All agents use `claude-3-5-sonnet-20241022` by default
- Can be changed in `.env` file

## 📈 Usage

1. **Upload your Excel file** to the `data/` directory
2. **Update `DATA_FILE_PATH`** in `.env` if needed
3. **Add your Claude API key** to `.env`
4. **Start the application** and access the web interface
5. **Run multi-agent analysis** to get breeding recommendations

## 🤝 Credits & Collaborations

- **UCLA** - University of California, Los Angeles
- **@structures.computer Lab** - Collaborative research partner
- **Claude AI** - Anthropic's advanced reasoning model

## 📞 Support

For questions or issues:
- Check the logs in `logs/agriagent.log`
- Verify your `.env` configuration
- Ensure API key is valid and has sufficient credits

---

**🎉 Happy Breeding Analysis!** 🌱
