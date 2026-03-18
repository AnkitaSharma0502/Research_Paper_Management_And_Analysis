# 🔬 Research Paper Management & Analysis System

An AI-powered research assistant that helps you **discover, organize, analyze, and interact** with academic research papers using LLMs, vector search, and intelligent tool integration.

Built with LangChain, Groq, FAISS, and Streamlit.

Link:https://researchpapermanagementandanalysis-ke6tnztrz7kma55zwlazse.streamlit.app/
---

## Screenshots




| Library Dashboard | Chat Assistant | Trend Analytics |
|:-:|:-:|:-:|
| <img width="1915" height="848" alt="image" src="https://github.com/user-attachments/assets/d955b07a-c694-4067-b380-2f6913c88ead" />|<img width="1890" height="871" alt="image" src="https://github.com/user-attachments/assets/0e187856-78b8-4dd1-9857-d06d6faeb66b" />|<img width="1863" height="861" alt="image" src="https://github.com/user-attachments/assets/c013ac4d-9dae-4d60-a88d-2b555a20be12" />

---

##  Features

-  **Smart PDF Parsing** — Handles single-column, two-column, and mixed-format academic papers. Detects sections using font-size analysis, not just keyword matching.
-  **Semantic Search** — FAISS-powered vector search across your entire paper library using natural language queries.
-  **Research Chat Assistant** — Ask questions about a single paper, your entire library, or compare multiple papers side by side.
-  **Web Search Fallback** — When your library doesn't have the answer, the assistant automatically searches the web via Tavily.
-  **Summarization** — Generates structured summaries (problem, approach, contributions, results, limitations) for any paper.
-  **AI Topic Categorization** — On-demand LLM categorization of papers into research topics with trend charts.
-  **Trend Analytics** — Publication frequency charts, venue breakdowns, category evolution over time, and citation network visualization.
-  **Citation Network** — Extracts and visualizes citation relationships between papers.
-  **Metadata Editing** — Manually edit Year, Venue, and Reading Status for any paper directly in the dashboard.
-  **Deployment Ready** — Switchable between local HuggingFace embeddings (development) and HuggingFace Inference API (cloud deployment).

---

##  Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq (llama-3.3-70b-versatile) via LangChain |
| **Embeddings** | HuggingFace sentence-transformers/all-MiniLM-L6-v2 |
| **Vector Store** | FAISS |
| **PDF Parsing** | PyMuPDF (fitz) |
| **Orchestration** | LangChain |
| **External Tools** | Tavily Search |
| **Frontend** | Streamlit |
| **Data Models** | Pydantic  |

---

##  Project Structure

```
research_intelligence_system/
│
├── analytics/
│   └── trends.py              # Trend analysis, citation table, topic distribution
│
├── core/
│   ├── parser.py              # PDF parsing, section extraction, metadata extraction
│   ├── indexer.py             # FAISS vector store, content-aware chunking
│   └── rag_pipeline.py        # RAG Q&A, summarization, cross-paper comparison
│
├── data/                      # FAISS index storage (auto-created)
│
├── models/
│   └── schemas.py             # Pydantic data models (ResearchPaper, PaperSection, Citation)
│
├── tools/
│   ├── mcp_tools.py           # LangChain tools: metadata lookup, related work, trends
│   └── search_tool.py         # Tavily web search wrapper
│
├── ui/
│   ├── __init__.py
│   ├── library_view.py        # Paper upload, inventory, paper viewer
│   ├── chat_view.py           # Chat assistant with 3 modes + web fallback
│   └── analytics_view.py      # Trend charts, topic categories, citation network
│
├── .env                       
├── app.py                     # Streamlit entry point
└── requirements.txt
```

---
## High level Architecture

 <img width="1174" height="704" alt="image" src="https://github.com/user-attachments/assets/b98e141a-c33e-4dc3-9b05-6cc03e2291d6" />

---
## Low Level Architecture

<img width="1265" height="540" alt="image" src="https://github.com/user-attachments/assets/ee284885-777b-406c-9c70-63230bc00b7d" />

---
##  How to Use

### Uploading Papers
1. Go to **📚 Library Dashboard**
2. Expand **Upload New Research Papers**
3. Select one or more PDF files
4. Click **Process & Index Papers**

The system will:
- Detect the PDF format (single-column, two-column, or scanned)
- Extract sections, metadata, and references
- Generate embeddings and index into FAISS

### Asking Questions
1. Go to **🤖 Chat Assistant**
2. Choose a mode:
   - **🌐 Entire Library** — searches all indexed papers
   - **📄 Single Paper** — focuses on one paper
   - **⚖️ Compare Papers** — compares two or more papers
3. Enable **Web Search Fallback** toggle to automatically search the web when your library doesn't have the answer

### Viewing Trends
1. Go to **📈 Trend Insights**
2. **Library Overview** tab — see publication frequency and venue breakdown
3. **Research Topics** tab — click **Analyze Topics** to auto-categorize papers using AI
4. **Emerging Trends** tab — filter by year to see newest papers
5. **Citation Network** tab — explore citation relationships



##  Limitations

- **Scanned PDFs** are not supported (image-only PDFs require OCR). The system will display a clear warning message.
- **Metadata extraction accuracy** depends on the PDF format. Metadata can be manually corrected in the Library Dashboard.
- **Citation extraction** works best for numbered reference formats (`[1]`, `1.`). Complex formats fall back to raw text display.
- The **HuggingFace embedding model** requires ~500MB RAM. Ensure sufficient memory when deploying.

---

##  Future Improvements

- [ ] OCR support for scanned PDFs using Tesseract
- [ ] arXiv / Semantic Scholar API integration for direct paper import
- [ ] Export library as BibTeX or CSV
- [ ] Multi-user support with persistent storage
- [ ] Fine-tuned summarization prompts per research domain
- [ ] Graph-based citation visualization (NetworkX + PyVis)

---

##  Architecture

```
PDF Upload
    │
    ▼
PDFParser
  ├── Detect format (single / two-column / scanned)
  ├── Extract clean text (PyMuPDF)
  ├── Detect sections via font-size analysis
  └── Extract metadata (PDF built-in → fonts → regex → LLM fallback)
    │
    ▼
ResearchIndexer
  ├── Content-aware chunking
  │     ├── Short sections → keep whole
  │     ├── Long sections → paragraph-aware splitting
  │     └── References → one chunk per entry
  └── FAISS vector store (HuggingFace embeddings)
    │
    ▼
ResearchRAG
  ├── ask_question()     → single paper or full library Q&A
  ├── compare_papers()   → cross-paper comparison
  └── generate_summary() → structured paper summary
    │
    ▼
Streamlit UI
  ├── Library Dashboard  → upload, view, edit
  ├── Chat Assistant     → Q&A + web fallback
  └── Trend Insights     → analytics + citation network
```



##  Acknowledgements

- [LangChain](https://langchain.com) — LLM orchestration framework
- [Groq](https://groq.com) — Ultra-fast LLM inference
- [Semantic Scholar](https://www.semanticscholar.org) — Free academic metadata API
- [Tavily](https://tavily.com) — Web search API for AI agents
- [HuggingFace](https://huggingface.co) — Open source embedding models
- [Streamlit](https://streamlit.io) — Rapid UI development for Python

---
