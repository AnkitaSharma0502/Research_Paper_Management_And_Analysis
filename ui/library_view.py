import streamlit as st
import pandas as pd
from core.parser import PDFParser
import hashlib
import tempfile
import os


# ------------------------------------------------------------------ #
#  HELPERS
# ------------------------------------------------------------------ #


def _reading_progress(paper_store: dict) -> dict:
    """Returns counts of papers by reading status."""
    counts = {"to-read": 0, "reading": 0, "completed": 0}
    for p in paper_store.values():
        status = p.reading_status
        if status in counts:
            counts[status] += 1
    return counts


# ------------------------------------------------------------------ #
#  MAIN RENDER
# ------------------------------------------------------------------ #

def render(indexer, paper_store: dict):
    """
    Renders the Research Paper Library dashboard.
    """
    st.header("📚 Research Paper Library")

    # ------------------------------------------------------------------ #
    #  READING PROGRESS SUMMARY
    # ------------------------------------------------------------------ #
    if paper_store:
        progress = _reading_progress(paper_store)
        total    = len(paper_store)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📚 Total Papers", total)
        c2.metric("🔖 To-Read",      progress["to-read"])
        c3.metric("📖 Reading",      progress["reading"])
        c4.metric("✅ Completed",    progress["completed"])
        st.divider()

    # ------------------------------------------------------------------ #
    #  SECTION 1 — Upload & Index
    # ------------------------------------------------------------------ #
    with st.expander("➕ Upload New Research Papers", expanded=not bool(paper_store)):
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
        )

        if st.button("Process & Index Papers", type="primary"):
            if not uploaded_files:
                st.warning("Please select at least one PDF file first.")
            else:
                progress_bar = st.progress(0, text="Starting...")
                total_files  = len(uploaded_files)
                # Dedupe by content hash so renamed copies don't slip through
                # and same-named different papers don't collide.
                indexed_ids  = set(paper_store.keys())

                for i, uploaded_file in enumerate(uploaded_files):
                    progress_bar.progress(
                        i / total_files,
                        text=f"Parsing {uploaded_file.name}..."
                    )

                    pdf_bytes  = uploaded_file.getvalue()
                    content_id = hashlib.sha1(pdf_bytes).hexdigest()[:12]

                    if content_id in indexed_ids:
                        st.info(f"⏭️ Already indexed: {uploaded_file.name}")
                        continue

                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(pdf_bytes)
                            tmp_path = tmp.name

                        with PDFParser(tmp_path) as parser:
                            paper_obj = parser.parse(
                                paper_id=content_id,
                                llm=st.session_state.rag_engine.llm,
                            )

                        paper_store[paper_obj.paper_id] = paper_obj
                        indexed_ids.add(paper_obj.paper_id)
                        indexer.index_paper(paper_obj)
                        st.success(f"✅ Indexed: {paper_obj.title or uploaded_file.name}")

                    except Exception as e:
                        st.error(f"❌ Failed to process {uploaded_file.name}: {e}")
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.remove(tmp_path)

                    progress_bar.progress((i + 1) / total_files)

                progress_bar.empty()
                st.rerun()

    # ------------------------------------------------------------------ #
    #  SECTION 2 — Library Inventory (Editable)
    # ------------------------------------------------------------------ #
    st.subheader("Current Inventory")

    if not paper_store:
        st.info("Your library is empty. Upload some papers to get started!")
        return

    data = []
    for p_id, p in paper_store.items():
        data.append({
            "ID":      p.paper_id,
            "Title":   p.title,
            "Authors": ", ".join(p.authors) if p.authors else "Unknown",
            "Year":    p.year,
            "Venue":   p.venue or "",
            "Status":  p.reading_status,
        })

    df = pd.DataFrame(data)

    edited_df = st.data_editor(
        df,
        column_config={
            "Year": st.column_config.NumberColumn(
                "Year", format="%d", min_value=1900, max_value=2100
            ),
            "Venue":  st.column_config.TextColumn("Venue"),
            "Status": st.column_config.SelectboxColumn(
                "Reading Status",
                options=["to-read", "reading", "completed"],
                help="Track your reading progress for each paper.",
            ),
        },
        disabled=["ID", "Title", "Authors"],
        hide_index=True,
        use_container_width=True,
        key="library_editor",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Sync Metadata", use_container_width=True):
            for _, row in edited_df.iterrows():
                p_id = row["ID"]
                if p_id in paper_store:
                    year_val = row["Year"]
                    paper_store[p_id].year           = int(year_val) if pd.notna(year_val) else None
                    paper_store[p_id].venue          = row["Venue"] or None
                    paper_store[p_id].reading_status = row["Status"]
            # Rebuild FAISS so chunk metadata (year, venue) stays in sync
            indexer.clear_index()
            for paper in paper_store.values():
                indexer.index_paper(paper)
            st.success("✅ Metadata synced!")
            st.rerun()

    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.paper_store = {}
            indexer.clear_index()
            # Clear all derived caches so nothing from deleted papers lingers
            keys_to_remove = [
                k for k in st.session_state
                if k.startswith("summary_")
                or k.startswith("chat_history_")
                or k in ("paper_categories", "category_paper_ids")
            ]
            for k in keys_to_remove:
                st.session_state.pop(k, None)
            st.rerun()

    # ------------------------------------------------------------------ #
    #  SECTION 3 — Individual Paper Viewer
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader("🔍 Paper Viewer")

    paper_titles = {p.paper_id: p.title for p in paper_store.values()}
    selected_id  = st.selectbox(
        "Select a paper to view details",
        options=list(paper_titles.keys()),
        format_func=lambda x: paper_titles.get(x, x),
    )

    if not selected_id:
        return

    paper = paper_store[selected_id]

    st.markdown(f"### {paper.title}")

    # ──  Show page count if available ─────────────────────
 
    page_count = getattr(paper, "page_count", None)
    page_info  = f"  \n**Pages:** {page_count}" if page_count else ""

    st.markdown(
        f"**Authors:** {', '.join(paper.authors) if paper.authors else 'Unknown'}  \n"
        f"**Year:** {paper.year or 'N/A'}  \n"
        f"**Venue:** {paper.venue or 'N/A'}  \n"
        f"**Keywords:** {', '.join(paper.keywords) if paper.keywords else 'N/A'}"
        f"{page_info}"
    )

    # ── Abstract ──────────────────────────────────────────────────────
    with st.expander("📄 Abstract", expanded=True):
        st.write(paper.abstract or "No abstract available.")

    # ── AI Summary (generate on demand, cached per paper) ─────────────
    summary_key = f"summary_{selected_id}"
    with st.expander("✨ AI Summary"):
        cached = st.session_state.get(summary_key)
        if cached:
            st.markdown(cached)
            if st.button("🔄 Regenerate", key=f"regen_{selected_id}"):
                st.session_state.pop(summary_key, None)
                st.rerun()
        else:
            if st.button("Generate Summary", key=f"gen_{selected_id}", type="primary"):
                rag_engine = st.session_state.get("rag_engine")
                if rag_engine is None:
                    st.error("RAG engine not initialised.")
                else:
                    with st.spinner("Generating summary..."):
                        try:
                            summary = rag_engine.generate_summary(paper)
                            st.session_state[summary_key] = summary
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to generate summary: {e}")
            else:
                st.caption("Click to generate a Short + Structured summary using the LLM.")


    # ── References ───────────────────────────────────────────
    with st.expander("🔗 References"):
        if paper.raw_references:
            st.text_area(
                label            = "Raw References",
                value            = paper.raw_references,
                height           = 300,
                disabled         = True,
                label_visibility = "collapsed",
            )
        else:
            st.info("No references section found in this paper.")