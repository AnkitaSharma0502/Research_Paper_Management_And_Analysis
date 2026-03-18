import streamlit as st
import os
from dotenv import load_dotenv

from core.indexer import ResearchIndexer
from core.rag_pipeline import ResearchRAG
from analytics.trends import TrendAnalyzer
from ui import library_view, chat_view, analytics_view

#  CONFIGURATION

load_dotenv()

st.set_page_config(
    page_title="AI Research Intelligence",
    page_icon="🔬",
    layout="wide",
)

# ------------------------------------------------------------------ #
#  SESSION STATE INITIALISATION
# ------------------------------------------------------------------ #
if "paper_store" not in st.session_state:
    st.session_state.paper_store = {}

if "indexer" not in st.session_state:
    st.session_state.indexer = ResearchIndexer()

if "rag_engine" not in st.session_state:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        st.error(
            "❌ GROQ_API_KEY is not set. "
            "Add it to your .env file and restart the app."
        )
        st.stop()

    st.session_state.rag_engine = ResearchRAG(
        api_key=groq_key,
        model_name="llama-3.3-70b-versatile",
    )

# ------------------------------------------------------------------ #
#  SIDEBAR
# ------------------------------------------------------------------ #
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📚 Library Dashboard",
            "🤖 Chat Assistant",
            "📈 Trend Insights",
        ],
    )

    st.markdown("---")

    n_papers = len(st.session_state.paper_store)
    indexed  = st.session_state.indexer.vector_store is not None

    st.metric("Papers in Library", n_papers)
    st.caption(f"Index status: {'✅ Ready' if indexed else '⚠️ Empty'}")

    st.markdown("---")
    st.caption("Powered by LangChain · Groq · FAISS · Streamlit")

# ------------------------------------------------------------------ #
#  PAGE ROUTING
# ------------------------------------------------------------------ #
if page == "📚 Library Dashboard":
    library_view.render(
        indexer     = st.session_state.indexer,
        paper_store = st.session_state.paper_store,
    )

elif page == "🤖 Chat Assistant":
    chat_view.render(
        rag_engine = st.session_state.rag_engine,
    )

elif page == "📈 Trend Insights":
    analyzer = TrendAnalyzer(list(st.session_state.paper_store.values()))
    analytics_view.render(
        analyzer   = analyzer,
        rag_engine = st.session_state.rag_engine,   # ← needed for topic categorization
    )