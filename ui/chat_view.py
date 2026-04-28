import streamlit as st
from tools.search_tool import ResearchSearchTool
from core.rag_pipeline import FALLBACK_SIGNAL


def _needs_web_search(answer: str) -> bool:
    # Only the exact sentinel enforced by the RAG prompt triggers fallback.
    # Fuzzy matches like "not discussed" wrongly hijacked grounded answers
    # that legitimately described what a paper didn't cover.
    return FALLBACK_SIGNAL.lower() in (answer or "").lower()


# ------------------------------------------------------------------ #
#  MAIN RENDER
# ------------------------------------------------------------------ #

def render(rag_engine, research_agent=None):
    """
    Renders the Research Chat Assistant.
    Covers Part VI Task 16 + Part IV Task 11:
      - Single paper Q&A
      - Full library Q&A
      - Cross-paper comparison
      - Web search fallback when local answer not found
      - Optional agent mode with MCP tools (Semantic Scholar + Tavily)
    """
    st.header("🤖 Research Chat Assistant")
    st.caption("Ask questions about your library. Falls back to live web search if needed.")

    # ------------------------------------------------------------------ #
    #  GUARDS
    # ------------------------------------------------------------------ #
    indexer     = st.session_state.get("indexer")
    paper_store = st.session_state.get("paper_store", {})

    if not indexer or indexer.vector_store is None:
        st.warning("⚠️ No papers indexed yet. Go to **Library** and upload papers first.")
        return

    # ------------------------------------------------------------------ #
    #  MODE SELECTOR
    # ------------------------------------------------------------------ #
    mode = st.radio(
        "Chat mode",
        ["🌐 Entire Library", "📄 Single Paper", "⚖️ Compare Papers"],
        horizontal=True,
    )

    selected_paper_id = None
    compare_ids       = []

    if mode == "📄 Single Paper":
        if not paper_store:
            st.info("No papers in library.")
            return
        paper_titles      = {p.paper_id: p.title for p in paper_store.values()}
        selected_paper_id = st.selectbox(
            "Select paper",
            options=list(paper_titles.keys()),
            format_func=lambda x: paper_titles.get(x, x),
        )

    elif mode == "⚖️ Compare Papers":
        if len(paper_store) < 2:
            st.info("Need at least 2 papers to compare.")
            return
        paper_titles = {p.paper_id: p.title for p in paper_store.values()}
        compare_ids  = st.multiselect(
            "Select papers to compare (choose 2 or more)",
            options=list(paper_titles.keys()),
            format_func=lambda x: paper_titles.get(x, x),
        )
        if len(compare_ids) < 2:
            st.info("Please select at least 2 papers.")
            return

    # ------------------------------------------------------------------ #
    #  MODE TOGGLES
    #  - Web fallback: Entire Library + Single Paper only.
    #    (Compare mode: Tavily can't answer cross-paper questions.)
    #  - Agent: Single Paper only.
    #    (Entire Library: tool-loops are token-heavy across many papers.
    #     Compare: agent may skip a paper, breaking guaranteed coverage.)
    # ------------------------------------------------------------------ #
    enable_web = False
    use_agent  = False

    if mode == "📄 Single Paper":
        col1, col2 = st.columns(2)
        with col1:
            enable_web = st.toggle(
                "🌍 Enable web search fallback",
                value=True,
                help="If the answer is not found in your library, the web is searched automatically.",
            )
        with col2:
            use_agent = st.toggle(
                "🤖 Use research agent (tools)",
                value=False,
                help="LLM decides when to call Semantic Scholar / Tavily / local library. "
                     "Required for metadata, related-work, and trend questions.",
                disabled=(research_agent is None),
            )
    elif mode == "🌐 Entire Library":
        enable_web = st.toggle(
            "🌍 Enable web search fallback",
            value=True,
            help="If the answer is not found in your library, the web is searched automatically.",
        )
    # Compare Papers: no toggles — pure local RAG only.

    # ------------------------------------------------------------------ #
    #  CHAT HISTORY
    # ------------------------------------------------------------------ #
    history_key = f"chat_history_{mode}_{selected_paper_id or 'library'}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    # Render previous messages
    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Re-render source expander if saved
            if msg.get("sources"):
                with st.expander("📎 Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f"- **{src.get('title', src.get('paper_id', 'Unknown'))}**"
                            f" — *{src.get('section_name', 'N/A')}*"
                            f" ({src.get('year', 'N/A')})"
                        )

    # ------------------------------------------------------------------ #
    #  NEW QUERY
    # ------------------------------------------------------------------ #
    query = st.chat_input("Ask a research question...")

    if query:
        # Show user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state[history_key].append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            try:
                vector_store = indexer.vector_store
                sources      = []
                tool_calls   = []
                used_web     = False
                used_agent   = False

                # ── Step 1: Agent path or plain RAG ───────────────────
                if use_agent and research_agent is not None and mode != "⚖️ Compare Papers":
                    with st.spinner("🤖 Agent reasoning with tools..."):
                        selected_title = (
                            paper_store[selected_paper_id].title
                            if selected_paper_id and selected_paper_id in paper_store
                            else None
                        )
                        result = research_agent.run(
                            query        = query,
                            vector_store = vector_store,
                            paper_id     = selected_paper_id,
                            paper_title  = selected_title,
                        )
                        answer     = result.get("answer", "")
                        sources    = result.get("sources", [])
                        tool_calls = result.get("tool_calls", [])
                        used_agent = True
                else:
                    with st.spinner("🔍 Searching local library..."):
                        if mode == "⚖️ Compare Papers":
                            result = rag_engine.compare_papers(
                                query        = query,
                                vector_store = vector_store,
                                paper_ids    = compare_ids,
                            )
                        else:
                            result = rag_engine.ask_question(
                                query        = query,
                                vector_store = vector_store,
                                paper_id     = selected_paper_id,
                            )

                        answer  = result.get("answer", "")
                        sources = result.get("sources", [])

                # ── Step 2: Web fallback if needed (non-agent path) ───
                # Skip fallback in Compare mode — Tavily can't answer cross-paper
                # comparison questions and just produces unrelated noise.
                if (not used_agent and enable_web and mode != "⚖️ Compare Papers"
                        and _needs_web_search(answer)):
                    with st.spinner("🌍 Not found locally. Searching the web..."):
                        try:
                            search_tool = ResearchSearchTool()
                            web_context = search_tool.search_papers(query)

                            web_prompt = f"""
The user asked: {query}

I searched the local research library but couldn't find a specific answer.
Based on the following live web search results, provide a comprehensive answer:

{web_context}

Give a clear, professional answer. Mention that this is from web search, not the local library.
"""
                            web_response = rag_engine.llm.invoke(web_prompt)
                            answer       = web_response.content
                            used_web     = True
                            sources      = []   # web results don't have paper metadata

                        except Exception as web_err:
                            # Web search failed — show original local answer
                            st.warning(f"Web search unavailable: {web_err}")

                # ── Step 3: Display answer ─────────────────────────────
                st.markdown(answer)

                # Source badge
                if used_web:
                    st.info("🌍 Source: Live Web Search")
                elif used_agent:
                    st.info("🤖 Source: Research Agent (tools)")
                elif sources:
                    st.success("📚 Source: Local Research Library")

                # Source details expander
                if sources and not used_web:
                    with st.expander("📎 Sources used", expanded=False):
                        seen = set()
                        for src in sources:
                            key = (src.get("paper_id"), src.get("section_name"))
                            if key not in seen:
                                seen.add(key)
                                st.markdown(
                                    f"- **{src.get('title', src.get('paper_id', 'Unknown'))}**"
                                    f" — *{src.get('section_name', 'N/A')}*"
                                    f" ({src.get('year', 'N/A')})"
                                )

                # Tool-calls expander (agent mode only)
                if tool_calls:
                    with st.expander(f"🔧 Research tools used ({len(tool_calls)})", expanded=False):
                        for i, tc in enumerate(tool_calls, 1):
                            st.markdown(
                                f"**{i}. `{tc['name']}`**\n\n"
                                f"  - **Input:** `{tc['input']}`\n"
                                f"  - **Output (truncated):** {tc['output']}"
                            )

                # Persist to history
                st.session_state[history_key].append({
                    "role":       "assistant",
                    "content":    answer,
                    "sources":    sources if not used_web else [],
                    "tool_calls": tool_calls,
                })

            except Exception as e:
                error_msg = f"❌ Error generating answer: {e}"
                st.error(error_msg)
                st.session_state[history_key].append({
                    "role":    "assistant",
                    "content": error_msg,
                    "sources": [],
                })

    # ------------------------------------------------------------------ #
    #  CLEAR CHAT
    # ------------------------------------------------------------------ #
    if st.session_state.get(history_key):
        if st.button("🗑️ Clear chat history"):
            st.session_state[history_key] = []
            st.rerun()