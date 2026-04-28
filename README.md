<<<<<<< HEAD
# 🔬 Research Paper Management & Analysis System

An AI-powered research assistant that helps you **discover, organize, analyze, and interact** with academic research papers using LLMs, vector search, and intelligent tool integration.

Built with LangChain, Groq, FAISS, and Streamlit.

Link:https://researchpapermanagementandanalysis-ke6tnztrz7kma55zwlazse.streamlit.app/
---

=======
# Research Intelligence System

An AI-powered research assistant for managing, analyzing, and querying academic papers. Upload PDFs and the system extracts structured metadata, indexes content into a vector store, and lets you ask questions across single papers or your whole library — with an optional tool-calling agent that reaches Semantic Scholar and Tavily when local context isn't enough.


Link:https://researchpapermanagementandanalysis-ke6tnztrz7kma55zwlazse.streamlit.app/

---

## Features


- **Three Q&A modes** — single-paper, entire-library, and cross-paper comparison (with reference filtering to prevent citation confusion)
- **Tool-calling agent** — LangChain v1 `create_agent` bound to MCP-style tools (Semantic Scholar metadata, related-work, trend analytics) with Tavily fallback
- **Auto-generated summaries** — on-demand Short + Structured summaries grounded in the paper's own text
- **Trend analytics** — papers/year, venue breakdown, LLM-assigned categories, emerging-trends filter, citation network with internal/external citation tracking

## Architecture

```
PDF upload
    - PDFParser (PyMuPDF + heuristics + LLM refine)
    - ResearchIndexer (HuggingFace embeddings + FAISS)
    - ResearchPaper objects in session state

Chat query
    - ResearchRAG (plain LangChain) ─OR─ ResearchAgent (tool-calling)
    - Groq Llama 3.3-70B
    - answer + sources + tool-call trace
```



## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Orchestration | LangChain v1 (`create_agent`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (CPU) |
| Vector store | FAISS (in-memory) |
| PDF | PyMuPDF (`fitz`) |
| External research | Semantic Scholar API |
| Web search | Tavily |
| Schema | Pydantic 2 |

>>>>>>> c329d43 (changes in parser and mcp tools)
## Screenshots




| Library Dashboard | Chat Assistant | Trend Analytics |
|:-:|:-:|:-:|
| <img width="1915" height="848" alt="image" src="https://github.com/user-attachments/assets/d955b07a-c694-4067-b380-2f6913c88ead" />|<img width="1890" height="871" alt="image" src="https://github.com/user-attachments/assets/0e187856-78b8-4dd1-9857-d06d6faeb66b" />|<img width="1863" height="861" alt="image" src="https://github.com/user-attachments/assets/c013ac4d-9dae-4d60-a88d-2b555a20be12" />

---
<<<<<<< HEAD

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
=======
## Overview

1. **📚 Library Dashboard** — upload PDFs, edit metadata inline
2. **🤖 Chat Assistant** — ask questions in three modes:
   - 🌐 Entire Library — RAG across all papers
   - 📄 Single Paper — RAG filtered to one paper, with optional research agent
   - ⚖️ Compare Papers — cross-paper synthesis (per-paper retrieval merged into one prompt)
3. **📈 Trend Insights** — library overview, LLM-tagged categories, emerging trends, citation network

## Project Layout

```
.
├── app.py                       # Streamlit entry point + session state
├── core/
│   ├── parser.py                # PDF → ResearchPaper (layout-aware)
│   ├── indexer.py               # Chunking + FAISS
│   ├── rag_pipeline.py          # Plain RAG (summary, ask, compare)
│   └── agent.py                 # LangChain tool-calling agent
├── tools/
│   ├── mcp_tools.py             # Semantic Scholar + Tavily tools (agent-bound)
│   └── search_tool.py           # Tavily fallback wrapper (non-agent)
├── analytics/
│   └── trends.py                # Library/citation/trend aggregation
├── models/
│   └── schemas.py               # Pydantic schemas
├── ui/
│   ├── library_view.py          # 📚 Library Dashboard
│   ├── chat_view.py             # 🤖 Chat Assistant
│   └── analytics_view.py        # 📈 Trend Insights    
|                   
├── docs/
│   └── architecture.md          # System architecture + Mermaid diagram
├── requirements.txt
└── README.md
```

## Limitations

- FAISS is in-memory — uploaded papers are lost on restart
- Single-user Streamlit (no auth, no multi-tenant isolation)
- Free-tier Groq has a 100k token-per-day cap on the 70B model; the system surfaces 429 errors clearly but doesn't auto-fall-back to a smaller model
- Semantic Scholar's public API rate-limits aggressively (~100 req / 5 min shared); a free API key bumps this to ~1 req/sec
- Scanned PDFs are detected but not OCR'd — text-only PDFs are required

## Acknowledgements

- LangChain for the agent runtime
- Groq for fast inference
- Semantic Scholar for open research metadata
- Tavily for web-search fallback
>>>>>>> c329d43 (changes in parser and mcp tools)
