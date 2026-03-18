import streamlit as st
import pandas as pd
from core.parser import PDFParser
import tempfile
import os
import re


# ------------------------------------------------------------------ #
#  HELPERS
# ------------------------------------------------------------------ #

def _smart_truncate(text: str, limit: int = 300) -> str:
    """
    Truncates text at the last complete sentence within the limit.
    Prevents mid-sentence cuts in the UI.

    How it works:
    - If text is already short enough, return as-is
    - Otherwise cut at `limit` characters
    - Walk backwards from cut point to find the last period
    - Cut there so we never end mid-sentence
    """
    if len(text) <= limit:
        return text

    truncated   = text[:limit]
    last_period = truncated.rfind('.')   # find last sentence ending

    if last_period > limit // 2:
        # Only cut at period if it's in the second half of the text
        # (avoids cutting too early if the first sentence is very long)
        return truncated[:last_period + 1] + " ..."

    return truncated + "..."


def _extract_doi(text: str):
    """Extracts a DOI from a reference string if present."""
    doi_match = re.search(
        r'10\.\d{4,9}/[-._;()/:A-Z0-9a-z]+',
        text
    )
    return doi_match.group() if doi_match else None


def _extract_url(text: str):
    """Extracts a plain URL from a reference string if present."""
    url_match = re.search(r'https?://\S+', text)
    return url_match.group().rstrip('.,)') if url_match else None


def _render_reference(ref: str, index: int):
    """
    Renders a single reference with a clickable DOI or URL link if found.
    """
    doi = _extract_doi(ref)
    url = _extract_url(ref)

    if doi:
        link    = f"https://doi.org/{doi}"
        display = ref.replace(doi, "").strip().rstrip('.,')
        st.markdown(f"{index}. {display} — [🔗 DOI]({link})")
    elif url:
        display = ref.replace(url, "").strip().rstrip('.,')
        st.markdown(f"{index}. {display} — [🔗 Link]({url})")
    else:
        st.markdown(f"{index}. {ref}")


def _reading_progress(paper_store: dict) -> dict:
    """Returns counts of papers by reading status."""
    counts = {"to-read": 0, "reading": 0, "completed": 0}
    for p in paper_store.values():
        status = p.reading_status
        if status in counts:
            counts[status] += 1
    return counts


# ------------------------------------------------------------------ #
#  SECTIONS TO SKIP IN THE SECTION VIEWER
# ------------------------------------------------------------------ #

SKIP_SECTIONS = {
    "header/metadata",
    "references",       # catches: "6. References", "References and Notes", etc.
    "bibliography",     # alternate name for references
    "works cited",      # another alternate name
    "abstract",         # shown in its own dedicated box above
    "introduction",     # often duplicates abstract content in short papers
}


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
                indexed_ids  = {p.paper_id for p in paper_store.values()}

                for i, uploaded_file in enumerate(uploaded_files):
                    progress_bar.progress(
                        i / total_files,
                        text=f"Parsing {uploaded_file.name}..."
                    )

                    if uploaded_file.name in indexed_ids:
                        st.info(f"⏭️ Already indexed: {uploaded_file.name}")
                        continue

                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        with PDFParser(tmp_path) as parser:
                            paper_obj = parser.parse(
                                paper_id=uploaded_file.name,
                                llm=st.session_state.rag_engine.llm,
                            )

                        paper_store[paper_obj.paper_id] = paper_obj
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

    # st.info(
    #     "💡 Edit **Year**, **Venue**, and **Status** inline. "
    #     "Click **Sync Metadata** to save."
    # )

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
        if st.button("💾 Sync Library Metadata", use_container_width=True):
            for _, row in edited_df.iterrows():
                p_id = row["ID"]
                if p_id in paper_store:
                    paper_store[p_id].year           = int(row["Year"]) if row["Year"] else None
                    paper_store[p_id].venue          = row["Venue"] or None
                    paper_store[p_id].reading_status = row["Status"]
            st.success("✅ Metadata synced!")
            st.rerun()

    with col2:
        if st.button("🗑️ Clear All Papers", use_container_width=True):
            st.session_state.paper_store = {}
            indexer.clear_index()
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

    # ── Sections with Show More / Show Less ───────────────────────────
    with st.expander("📑 Sections"):
        rendered = 0

        for idx, section in enumerate(paper.sections):

            # --- SUBSTRING matching instead of exact match ──
            #  check if any skip-word appears INSIDE the section name
            #      → catches all variants regardless of numbering or suffix
            if any(skip in section.section_name.lower() for skip in SKIP_SECTIONS):
                continue

            rendered += 1
            st.markdown(f"**{section.section_name}**")

            content    = section.content
            is_long    = len(content) > 300
            toggle_key = f"expand_{selected_id}_{idx}_{section.section_name}"

            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = False

            if is_long and not st.session_state[toggle_key]:
                # Show truncated version with "Show more" button
                st.write(_smart_truncate(content, 300))
                if st.button(
                    "Show more ▼",
                    key=f"btn_more_{selected_id}_{idx}_{section.section_name}",
                ):
                    st.session_state[toggle_key] = True
                    st.rerun()
            else:
                # Show full content with "Show less" button
                st.write(content)
                if is_long and st.button(
                    "Show less ▲",
                    key=f"btn_less_{selected_id}_{idx}_{section.section_name}",
                ):
                    st.session_state[toggle_key] = False
                    st.rerun()

            st.divider()

        if rendered == 0:
            st.info("No additional sections found beyond Abstract.")

    # ── References with clickable DOI/URL links ───────────────────────
    with st.expander("🔗 References"):
        if paper.references:
            st.caption(f"{len(paper.references)} references extracted")
            for idx, ref in enumerate(paper.references, 1):
                _render_reference(ref, idx)

            # Raw reference text toggle
            if paper.raw_references:
                if st.toggle("Show raw reference text", key=f"raw_ref_{selected_id}"):
                    st.text_area(
                        label            = "Raw",
                        value            = paper.raw_references,
                        height           = 200,
                        disabled         = True,
                        label_visibility = "collapsed",
                    )

        elif paper.raw_references:
            st.caption(
                "References shown as raw text "
                "(format not recognized for individual parsing)"
            )
            st.text_area(
                label            = "Raw References",
                value            = paper.raw_references,
                height           = 300,
                disabled         = True,
                label_visibility = "collapsed",
            )
        else:
            st.info("No references section found in this paper.")