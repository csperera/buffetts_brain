# Buffett's Brain 🧠

**An Educational RAG-Enabled AI Agent for Investment Wisdom**

Buffett's Brain is an intelligent chatbot powered by Retrieval-Augmented Generation (RAG) that allows you to interactively explore the investment philosophy of Warren Buffett and Charlie Munger. Ask questions about value investing, business analysis, mental models, and more—all grounded in decades of shareholder letters, speeches, and writings.

---

## 🎯 What is RAG and Why Does It Matter?

### Understanding Agentic AI Chatbots

Traditional AI chatbots rely solely on their training data, which can lead to several problems:
- **Hallucinations**: The model may generate plausible-sounding but factually incorrect information
- **Knowledge Cutoffs**: They lack access to information beyond their training date
- **Domain Limitations**: They may not have deep knowledge in specialized areas like investment philosophy

### How RAG Enhancement Works

**Retrieval-Augmented Generation (RAG)** transforms a standard AI chatbot into an **Agentic AI system** by combining two powerful capabilities:

1. **Knowledge Retrieval**: Before generating a response, the system searches through a curated knowledge base (vector database) to find relevant information
2. **Contextual Generation**: The AI then uses these retrieved documents as context to generate accurate, grounded responses

**Key Benefits of RAG:**
- ✅ **Reduces Hallucinations**: Responses are anchored to real documents, not just the model's training data
- ✅ **Domain Expertise**: The system becomes an expert in your specific knowledge domain (in this case, Buffett & Munger's investment wisdom)
- ✅ **Up-to-date Information**: Combined with web search capabilities (via Tavily), the agent can access current market data and news
- ✅ **Transparent Sources**: Users can trace answers back to specific shareholder letters or speeches
- ✅ **Customizable Knowledge**: You control what information the agent has access to

In Buffett's Brain, RAG enables the agent to:
- Quote directly from Berkshire Hathaway annual letters spanning 1977-2024
- Reference specific mental models from Charlie Munger's speeches
- Combine historical investment wisdom with current market information via web search
- Provide nuanced answers that reflect the actual philosophy documented in primary sources

---

## 🚀 Features

- **RAG-Powered Responses**: Answers grounded in authentic Buffett and Munger documents
- **Web Search Integration**: Real-time market data and news via Tavily search
- **Interactive Web UI**: Clean Streamlit interface with scrolling chat history
- **CLI Alternative**: Terminal-based chat for lightweight interaction
- **Comprehensive Knowledge Base**: 
  - Berkshire Hathaway Shareholder Letters (1977-2024)
  - Poor Charlie's Almanack
  - Charlie Munger Daily Journal Meeting transcripts
  - Selected speeches and writings

---

## 📋 Prerequisites

- **Python**: Latest stable version (3.11+ recommended)
- **API Keys**: 
  - `GEMINI_API_KEY` from [Google AI Studio](https://ai.google.dev/)
  - `TAVILY_API_KEY` from [Tavily](https://tavily.com/)

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd buffetts-brain
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate the environment:
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Where to get API keys:**
- **Gemini API Key**: Sign up at [Google AI Studio](https://ai.google.dev/)
- **Tavily API Key**: Register at [Tavily](https://tavily.com/)

---

## 📚 Setup Knowledge Base

### Step 1: Download Source Documents

Run the automated download script to fetch Berkshire Hathaway letters, Poor Charlie's Almanack, and other materials:

```bash
python download_data.py
```

This will populate the `knowledge_base/docs/` directory with PDFs and create placeholder `.txt` files with URLs for documents that require manual collection in the future.

**What gets downloaded:**
- ✅ Berkshire Hathaway Annual Letters (1999-2024, plus combined 1977-2002 archive)
- ✅ Poor Charlie's Almanack (PDF)
- ✅ Charlie Munger speech: "The Psychology of Human Misjudgment"
- 📝 URL bookmarks for Daily Journal transcripts (for future manual scraping)

### Step 2: Process Documents & Build Vector Database

```bash
python process_documents.py
```

This script will:
1. Load all PDFs from `knowledge_base/docs/`
2. Split documents into chunks (1000 characters, 200 overlap)
3. Generate embeddings using Google's `text-embedding-004` model
4. Store embeddings in ChromaDB vector store at `knowledge_base/vector_db/`

---

## 💬 Usage

### Primary Interface: Streamlit Web App (Recommended)

Launch the interactive web interface:

```bash
streamlit run app2.py
```

The app will open in your browser at `http://localhost:8501`

**Features:**
- Clean, modern chat interface
- Scrolling message history (fixed 500px height)
- Real-time responses with "Thinking..." indicator
- Powered by Gemini 2.5 Flash for fast performance

### Alternative: Command-Line Interface

For a terminal-based chat experience:

```bash
python chat_with_brain.py
```

Type your questions and press Enter. Type `exit` or `quit` to end the session.

---

## 🧪 Example Questions

Try asking Buffett's Brain:

- "What is Buffett's circle of competence concept?"
- "How does Charlie Munger define a 'lollapalooza effect'?"
- "What are the key principles of value investing according to Buffett?"
- "Explain the moat concept in business analysis"
- "What is the current price of Berkshire Hathaway stock?" *(uses web search)*
- "What did Buffett say about cryptocurrency in his 2023 letter?"

---

## 📂 Project Structure

```
buffetts-brain/
├── knowledge_base/
│   ├── docs/                      # Source PDFs (populated by download_data.py)
│   │   ├── Berkshire_Letters/     # Annual shareholder letters
│   │   ├── Munger_Transcripts/    # Munger speeches and transcripts
│   │   └── Poor_Charlies_Almanack.pdf
│   └── vector_db/                 # ChromaDB vector store (generated)
├── src/
│   ├── download_data.py           # Automated document downloader
│   ├── process_documents.py       # Document processor & vector DB builder
│   ├── chat_with_brain.py         # CLI chat interface
│   └── app2.py                    # Streamlit web interface
├── .env                           # API keys (create this)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## ⚙️ Technical Details

### RAG Pipeline Architecture

1. **Document Loading**: `PyPDFDirectoryLoader` recursively loads PDFs
2. **Text Chunking**: `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)
3. **Embedding Model**: Google `text-embedding-004`
4. **Vector Store**: ChromaDB with persistent storage
5. **Retriever**: Top-k=4 similarity search
6. **LLM**: Gemini 2.5 Flash (temperature=0.0 for factual responses)
7. **Web Search**: Tavily integration for real-time information

### Key Technologies

- **LangChain**: Orchestration framework for RAG pipeline
- **Google Generative AI**: Embeddings and language model
- **ChromaDB**: Vector database for semantic search
- **Streamlit**: Web UI framework
- **Tavily**: Web search API

---

## ⚠️ Known Limitations

- **Incomplete Document Coverage**: Not all Berkshire Hathaway shareholder letters are currently included in the knowledge base
- **Manual Collection Required**: Some Munger transcripts are saved as URL bookmarks (`.txt` files) and require manual web scraping
- **Knowledge Scope**: The agent is limited to information in the loaded documents plus web search results

---

## 🔮 Future Enhancements

Potential improvements for future versions:

- 📄 **Expand Knowledge Base**: Add remaining Berkshire letters and historical documents
- 👥 **Additional Investors**: Include writings from other legendary investors (Peter Lynch, Benjamin Graham, etc.)
- 🔍 **Enhanced Search**: Implement hybrid search (semantic + keyword)
- 💾 **Conversation Memory**: Persist chat history across sessions
- 📊 **Citation System**: Display source documents for each response
- 🌐 **Web Scraping**: Automate collection of Munger transcripts

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## ⚖️ Disclaimers

### Educational Purpose
This project is designed for **educational purposes only**. It is intended to demonstrate the capabilities of RAG-enabled AI systems and provide a learning tool for understanding investment philosophy.

### Not Financial Advice
**This tool does NOT provide financial advice.** Any information, analysis, or insights generated by Buffett's Brain should not be considered as recommendations to buy, sell, or hold any securities. Always consult with a qualified financial advisor before making investment decisions.

### Data Attribution
This project uses publicly available materials including:
- Berkshire Hathaway Inc. shareholder letters (© Berkshire Hathaway Inc.)
- "Poor Charlie's Almanack" by Charles T. Munger
- Public speeches and transcripts by Warren Buffett and Charlie Munger

All materials are used for educational purposes under fair use principles. The authors of this project claim no ownership of the original source materials. Users should respect the intellectual property rights of the original authors.

If you are a copyright holder and believe any content should be removed, please contact the project maintainer.

---

## 🤝 Contributing

Contributions are welcome! Potential areas for contribution:
- Adding more source documents to the knowledge base
- Improving the RAG pipeline performance
- Enhancing the UI/UX
- Bug fixes and documentation improvements

---

## 📧 Contact

For questions, feedback, or issues, please open an issue on the project repository.

---

**Happy Learning! 📈**

*"Risk comes from not knowing what you're doing." - Warren Buffett*