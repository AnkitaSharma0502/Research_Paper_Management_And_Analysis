import streamlit as st
import pandas as pd
from analytics.trends import TrendAnalyzer


# ------------------------------------------------------------------ #
#  LLM CATEGORIZATION  
# ------------------------------------------------------------------ #

def _assign_categories(papers, rag_engine) -> dict:
    """
    Uses LLM to assign a free-form category to each paper based on abstract.
    Results cached in st.session_state to avoid re-running on every rerun.
    Returns:
        { paper_id: "category string", ... }
    """
    cached_ids  = set(st.session_state.get("category_paper_ids", []))
    current_ids = set(p.paper_id for p in papers)

    if cached_ids == current_ids and st.session_state.get("paper_categories"):
        return st.session_state.paper_categories

    categories = {}
    progress   = st.progress(0, text="Analyzing paper topics...")
    total      = len(papers)

    for i, paper in enumerate(papers):
        try:
            prompt = f"""Read this research paper abstract and assign ONE short category label.
Be specific but concise (2-4 words max).
Examples: "Machine Learning", "Political Theory", "Algebraic Topology",
          "Victorian Literature", "Quantum Computing", "Public Health"

Abstract:
{paper.abstract[:1000]}

Return ONLY the category name, nothing else. No explanation."""

            response = rag_engine.llm.invoke(prompt)
            categories[paper.paper_id] = response.content.strip().strip('"').strip("'")

        except Exception:
            categories[paper.paper_id] = "Uncategorized"

        progress.progress((i + 1) / total, text=f"Categorizing: {paper.title[:40]}...")

    progress.empty()

    st.session_state.paper_categories   = categories
    st.session_state.category_paper_ids = list(current_ids)

    return categories


# ------------------------------------------------------------------ #
#  MAIN RENDER
# ------------------------------------------------------------------ #

def render(analyzer: TrendAnalyzer, rag_engine):
    """
    Renders the Trend & Citation Analytics panel.
    """
    st.header("📈 Research Trend & Citation Analytics")

    if not analyzer.papers:
        st.info(" Upload and index papers to see trend analytics.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Library Overview",
        "🏷️ Research Topics",
        "🚀 Emerging Trends",
        "🔗 Citation Network",
    ])

    # ================================================================== #
    #  TAB 1 — LIBRARY OVERVIEW
    # ================================================================== #
    with tab1:
        st.subheader("📊 Library Overview")

        stats = analyzer.get_library_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📚 Total Papers",  stats["total"])
        c2.metric("📅 Year Range",    stats["year_range"])
        c3.metric("🏛️ Unique Venues", stats["unique_venues"])
        c4.metric("✍️ Unique Authors", stats["authors_count"])

        st.divider()

        st.subheader("Papers Published Per Year")
        year_counts = analyzer.get_papers_per_year()

        if year_counts:
            year_df = pd.DataFrame(
                list(year_counts.items()),
                columns=["Year", "Papers"]
            ).sort_values("Year")
            st.bar_chart(year_df.set_index("Year"))
        else:
            st.info("No year data available. Edit years in the Library tab.")

        st.divider()

        st.subheader("Venue Breakdown")
        venue_data = analyzer.get_venue_table()

        if venue_data:
            venue_df = pd.DataFrame(venue_data)
            st.dataframe(venue_df, use_container_width=True, hide_index=True)
        else:
            st.info("No venue data found. Edit venues in the Library tab.")

    # ================================================================== #
    #  TAB 2 — RESEARCH TOPICS
    # ================================================================== #
    with tab2:
        st.subheader("🏷️ Research Topics")
        st.caption("LLM reads each abstract and assigns a category automatically.")

        col1, col2 = st.columns([2, 1])
        with col1:
            analyze_clicked = st.button(
                "🔍 Analyze Topics",
                type="primary",
                help="Runs once and caches results. Re-run if you add new papers.",
            )
        with col2:
            if st.button("🔄 Reset Categories"):
                st.session_state.pop("paper_categories",   None)
                st.session_state.pop("category_paper_ids", None)
                st.rerun()

        if analyze_clicked:
            with st.spinner("Running LLM categorization..."):
                st.session_state.pop("paper_categories",   None)
                st.session_state.pop("category_paper_ids", None)
                _assign_categories(analyzer.papers, rag_engine)
            st.success("✅ Topics analyzed!")

        categories = st.session_state.get("paper_categories")

        if not categories:
            st.info("Click **Analyze Topics** to categorize your papers using AI.")

        else:
            distribution = analyzer.get_category_distribution(categories)

            if not distribution:
                st.info("No category data to display.")
            else:
                st.subheader("Category Distribution")
                chart_df = pd.DataFrame([
                    {"Category": d["category"], "Papers": d["count"]}
                    for d in distribution
                ]).set_index("Category")
                st.bar_chart(chart_df)

                st.divider()

                st.subheader("Category × Year")
                cat_by_year = analyzer.get_category_by_year(categories)

                if cat_by_year:
                    cat_year_df = pd.DataFrame(cat_by_year).T.fillna(0).astype(int)
                    cat_year_df.index.name = "Year"
                    st.dataframe(cat_year_df, use_container_width=True)
                else:
                    st.info("No year data available for this chart.")

                st.divider()

                st.subheader("Papers by Category")
                for item in distribution:
                    with st.expander(
                        f"**{item['category']}** — "
                        f"{item['count']} paper(s) | {item['year_range']}"
                    ):
                        for title in item["papers"]:
                            st.markdown(f"- {title}")

    # ================================================================== #
    #  TAB 3 — EMERGING TRENDS
    # ================================================================== #
    with tab3:
        st.subheader("🚀 Emerging Trends")
        st.caption("Newest papers in your library, sorted by publication year.")

        year_filter = st.radio(
            "Show papers from:",
            ["Last 1 year", "Last 2 years", "Last 3 years", "All time"],
            horizontal=True,
        )

        limit_map = {
            "Last 1 year":  1,
            "Last 2 years": 2,
            "Last 3 years": 3,
            "All time":     None,
        }
        limit      = limit_map[year_filter]
        categories = st.session_state.get("paper_categories")

        newest = analyzer.get_newest_papers(
            limit_years=limit,
            categories=categories,
        )

        if not newest:
            st.info("No papers found for the selected time range.")
        else:
            st.markdown(f"**{len(newest)} papers found**")

            display_cols = ["title", "year", "venue", "authors"]
            if categories:
                display_cols.insert(2, "category")

            newest_df = pd.DataFrame(newest)[display_cols]
            newest_df.columns = [c.capitalize() for c in display_cols]
            st.dataframe(newest_df, use_container_width=True, hide_index=True)

    # ================================================================== #
    #  TAB 4 — CITATION NETWORK
    # ================================================================== #
    with tab4:
        st.subheader("🔗 Citation Network")

        citation_graph = analyzer.get_citation_graph()

        if not citation_graph:
            st.info("No citation data found. Citations are extracted from PDF reference sections.")

        else:
            # ── Summary counts table ───────────────────────────────────
            # Shows how many references each paper has and how many times
            # it is cited by other papers IN your library.
            # We intentionally do NOT show raw reference titles here —
            # that information belongs in the Library tab's References section.
            st.subheader("Paper Connections")
            st.caption(
                "**References** = papers this work cites. "
                "**Cited By** = other papers in your library that cite this work."
            )

            rows = []
            for title, edges in citation_graph.items():
                refs     = edges.get("references", [])
                cited_by = edges.get("cited_by",   [])
                rows.append({
                    "Paper":      title,
                    "References": len(refs),     # count only — no raw titles
                    "Cited By":   len(cited_by), # count only
                })

            graph_df = pd.DataFrame(rows).sort_values("References", ascending=False)

            st.dataframe(
                graph_df[["Paper", "References", "Cited By"]],
                use_container_width=True,
                hide_index=True,
            )

            # ── Cited-by connections (library papers only) ─────────────
            # Only show this section if any paper in the library
            # is cited by another paper in the library.
            # This is always a small, meaningful set.
            internal_citations = {
                title: edges["cited_by"]
                for title, edges in citation_graph.items()
                if edges.get("cited_by")
            }
            if internal_citations:
                st.divider()
                st.subheader("Internal Citations")
                st.caption("Papers in your library that cite other papers in your library.")

    # Show as a simple dataframe — no expanders, no scrolling walls of text
    # The count is what matters; full detail is visible in the Library tab
                rows = [
                    {"Paper": title, "Cited By (in library)": len(cited_by_list)}
                    for title, cited_by_list in internal_citations.items()
                ]
                st.dataframe(
                    pd.DataFrame(rows).sort_values("Cited By (in library)", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

            # ── Most referenced external papers ────────────────────────
            st.divider()
            st.subheader("📌 Most Referenced External Papers")
            st.caption("Papers outside your library cited most often across all your papers.")

            external = analyzer.get_most_referenced_external(top_n=10)

            if external:
                ext_df = pd.DataFrame(external)
                ext_df.columns = ["Title", "Cited By (count)"]
                st.dataframe(ext_df, use_container_width=True, hide_index=True)
            else:
                st.info("No external references found yet.")