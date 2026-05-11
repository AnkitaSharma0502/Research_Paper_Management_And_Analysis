import streamlit as st

# ------------------------------------------------------------------ #
#  MAIN RENDER
# ------------------------------------------------------------------ #

def render(rag_engine, research_agent=None):
    """
    Renders the Research Chat Assistant.
Features:
- Single-paper Q&A
- Entire-library semantic search
- Cross-paper comparison
- Agentic tool orchestration
- Metadata and related-work discovery
    """
    st.header("Chat Assistant")
    st.caption("Ask questions about your research library using semantic retrieval and agentic tools.")

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
    #  # Optional agentic tool orchestration
    # Compare mode currently stays pure local RAG
    # to guarantee deterministic multi-paper coverage.
    
    # ------------------------------------------------------------------ #
    use_agent  = False

    if mode == "📄 Single Paper":
        use_agent = st.toggle(
        "🤖 Use research agent (tools)",
        value=True,
        help=(
            "LLM dynamically uses local retrieval, "
            "Semantic Scholar, Tavily, and related-work tools."
        ),
        disabled=(research_agent is None),
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

                # ── Step 3: Display answer ─────────────────────────────
                st.markdown(answer)

                # Source badge
                
                if used_agent:
                    st.info("🤖 Source: Research Agent (tools)")
                elif sources:
                    st.success("📚 Source: Local Research Library")

                # Source details expander
                if sources:
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
                    "sources":    sources,
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