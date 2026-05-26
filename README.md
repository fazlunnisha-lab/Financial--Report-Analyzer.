# Financial--Report-Analyzer.
AI-powered financial document intelligence system. Ingests SEC 10-K filings, builds vector search with ChromaDB, enables natural language Q&amp;A via Groq LLM. Features automated ratio calculation, peer comparison, trend visualization, and red flag detection for management discussion patterns. Built for equity research and fundamental analysis.
Project Overview
FinSight is a Retrieval-Augmented Generation (RAG) based system designed for equity research analysts, portfolio managers, retail investors, and business journalists. It ingests company annual reports (SEC 10-K filings), processes Management Discussion & Analysis (MD&A) sections, and enables intelligent natural language querying with source citations.




Key Capabilities:
🔍 Natural language Q&A over financial documents
📊 Automated financial ratio calculation & trend visualization
⚖️ Peer comparison across companies
🚩 Red flag detection for concerning management language patterns
📈 Interactive Plotly charts for margin trends
🏗️ System Architecture
plain
Copy
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   SEC EDGAR     │────▶│  Text Extraction │────▶│  Chunking &     │
│   10-K Filings  │     │  (PyMuPDF/HTML)  │     │  Vectorization  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit UI  │◀────│   Groq LLM       │◀────│  ChromaDB       │
│   (4 Tabs)      │     │  (Llama 3.3 70B) │     │  Vector Store   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Analysis Layer: Ratios | Peer Compare | Red Flags | Trends   │
└─────────────────────────────────────────────────────────────────┘


#Tech Stack
Table
Component	Technology	Why
LLM	Groq (Llama 3.3 70B)	Free tier, fast inference
Embeddings	ChromaDB Default (all-MiniLM-L6-v2)	Local, no API cost
Vector DB	ChromaDB	Lightweight, persistent
UI	Streamlit	Rapid prototyping, interactive widgets
Charts	Plotly	Interactive, professional visuals
Data Source	SEC EDGAR	Public, legally compliant


📁 Project Structure
plain
Copy
finsight/
├── app.py                      # Main Streamlit application
├── data/
│   └── text/
│       ├── AAPL_2024_item7.txt   # Apple MD&A text
│       ├── AMZN_2024_item7.txt   # Amazon MD&A text
│       ├── GOOGL_2024_item7.txt  # Google MD&A text
│       ├── META_2024_item7.txt   # Meta MD&A text
│       └── NVDA_2024_item7.txt   # NVIDIA MD&A text (optional)
├── .chroma_db/                 # Persistent vector database
├── requirements.txt            # Python dependencies
└── README.md                   # This file
🚀 Installation & Setup
Prerequisites
Python 3.9+
Groq API key (free tier available at console.groq.com)
Step 1: Clone & Install
bash
Copy
git clone <your-repo-url>
cd finsight
pip install -r requirements.txt
Step 2: Configure API Key
Set your Groq API key as an environment variable:
bash
Copy

# Linux/Mac
export GROQ_API_KEY="gsk-your-key-here"

# Windows
set GROQ_API_KEY=gsk-your-key-here
Or add it to Replit Secrets / Streamlit Cloud secrets.
Step 3: Prepare Data
Extract Item 7 (MD&A) sections from 10-K filings:
Python
Copy
# In Google Colab or local environment
from sec_edgar_downloader import Downloader
import PyPDF2, re

# Download filings
dl = Downloader("YourName", "your@email.com")
for ticker in ["AAPL", "AMZN", "GOOGL", "META", "NVDA"]:
    dl.get("10-K", ticker, limit=2)

# Extract Item 7 text (see ingestion scripts)
# Save to data/text/{TICKER}_2024_item7.txt
Step 4: Run Application
bash
Copy
streamlit run app.py --server.port 8501
Open http://localhost:8501 in your browser.

💡 Usage Guide
Tab 1: 📄 Overview
View document statistics (character count, word count)
Compare document sizes across companies
Preview raw MD&A text excerpts
Tab 2: 💬 Q&A
Ask natural language questions about filings
Examples:
"What were the main revenue drivers?"
"What risks did management highlight?"
"How did operating income change year over year?"
Filter by specific companies
View source passages with relevance scores
Tab 3: ⚖️ Compare
Side-by-side comparison of two companies
AI-generated summaries on any topic
Head-to-head analysis with winner assessment
Tab 4: 📈 Analysis (Major Project Feature)
Ratio Calculation: Gross margin, operating margin, net margin, ROE
Trend Visualization: 3-year interactive line charts
Peer Comparison: Bar charts across all companies
Red Flag Detection: Pattern-based analysis of MD&A language
Vague attributions
Temporal vagueness
Passive deflection
Accounting euphemisms
Guidance softening

📊 Financial Data
Hardcoded financials (2022-2024) for ratio calculations:
Table
Company	Revenue (2024)	Operating Margin	Net Margin
AAPL	$391B	31.5%	24.0%
AMZN	$638B	9.2%	9.3%
GOOGL	$350B	28.3%	23.6%
META	$165B	35.9%	37.9%
NVDA	$131B	57.1%	55.8%
Source: Yahoo Finance / SEC 10-K filings

🚩 Red Flag Detection
The system scans MD&A text for concerning linguistic patterns:
Table
Pattern Type	Example	Severity
Vague Attribution	"due to certain market factors"	2/5
Temporal Vagueness	"in the foreseeable future"	2/5
Passive Deflection	"was impacted by external conditions"	2/5
Accounting Euphemism	"non-recurring restructuring charge"	3/5
Guidance Softening	"withdrawing forward guidance"	3/5

🎓 Academic Context
Project Type: Internship Major Project
Learning Outcomes:
RAG pipeline construction with vector databases
LLM integration for domain-specific Q&A
Financial statement analysis and ratio computation
Pattern recognition in unstructured text
Interactive data visualization

🔒 Privacy & Compliance
All data used is publicly available from the U.S. Securities and Exchange Commission (SEC). No private company data is accessed. This project complies with:
SEC EDGAR fair access policies
Groq API terms of service
Open-source license requirements (MIT/Apache 2.0 dependencies)

⚠️ Known Limitations
Storage Constraints: NVIDIA (NVDA) excluded from some deployments due to Replit 1GB storage limit
Single-Year Text: Current implementation uses 2024 MD&A only; multi-year RAG requires additional ingestion
Hardcoded Financials: Ratio data manually extracted; production would use automated XBRL parsing
English Only: Pattern detection optimized for English SEC filings
Mitigation: Architecture supports N companies and multi-year ingestion via configuration changes.

🔮 Future Enhancements
[ ] Real-time 10-Q quarterly ingestion via SEC RSS feeds
[ ] Automated XBRL financial data extraction
[ ] Fine-tuned FinBERT sentiment analysis
[ ] Multi-year temporal RAG (compare 2022 vs 2024 language)
[ ] PDF table extraction with Camelot/Tabula
[ ] User authentication and portfolio tracking
[ ] Alert system for new red flags in filings

🙏 Acknowledgments
SEC EDGAR for public access to corporate filings
Groq for free-tier LLM inference
ChromaDB for open-source vector storage
Streamlit for rapid UI prototyping
Sentence Transformers for local embeddings

📜 License
This project is for educational purposes only. Not intended for production investment decisions. Financial data approximations used for demonstration.
Built with ❤️ for equity research and fundamental analysis.
