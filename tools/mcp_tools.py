import os
import requests
from langchain.tools import tool
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
#  Lazy Tavily initialisation
# ──────────────────────────────────────────────────────────────────────────────

def _get_tavily():
    """Returns a TavilySearch instance. Raises clear error if key missing."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "TAVILY_API_KEY is not set. Add it to your .env file."
        )
    # ✅ Correct class: TavilySearch (not TavilySearchResults)
    from langchain_tavily import TavilySearch
    return TavilySearch(max_results=5)


# ──────────────────────────────────────────────────────────────────────────────
#  Semantic Scholar helper
# ──────────────────────────────────────────────────────────────────────────────

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"


def _semantic_scholar_search(title: str) -> Optional[dict]:
    try:
        params = {
            "query":  title,
            "limit":  1,
            "fields": "title,year,venue,citationCount,authors,externalIds",
        }
        resp = requests.get(
            f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
            params=params, timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("data", [])
        return results[0] if results else None
    except Exception:
        return None


def _semantic_scholar_related(paper_id: str, limit: int = 5) -> list:
    try:
        resp = requests.get(
            f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}/references",
            params={"fields": "title,year,venue,citationCount", "limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception:
        return []


def _parse_tavily_results(results) -> str:
    """Parses TavilySearch results (may be JSON string or list)."""
    import json
    if isinstance(results, str):
        try:
            parsed = json.loads(results)
            if isinstance(parsed, dict):
                items = parsed.get("results", [])
            else:
                items = parsed
        except Exception:
            return results
    elif isinstance(results, list):
        items = results
    else:
        return str(results)

    snippets = "\n\n".join(
        f"- {r.get('title','')}: {r.get('content','')[:200]}"
        for r in items[:3]
    )
    return f"[Web search results]\n{snippets}"


# ──────────────────────────────────────────────────────────────────────────────
#  TOOL 1 — Paper Metadata Lookup
# ──────────────────────────────────────────────────────────────────────────────

@tool
def paper_metadata_lookup(query: str) -> str:
    """
    Look up a research paper's metadata: year, venue, citation count, authors.
    Input: Paper title or DOI.
    """
    # Primary: Semantic Scholar
    paper = _semantic_scholar_search(query)
    if paper:
        authors = ", ".join(a.get("name", "") for a in paper.get("authors", []))
        return (
            f"Title         : {paper.get('title', 'N/A')}\n"
            f"Year          : {paper.get('year', 'N/A')}\n"
            f"Venue         : {paper.get('venue', 'N/A')}\n"
            f"Citation Count: {paper.get('citationCount', 'N/A')}\n"
            f"Authors       : {authors or 'N/A'}\n"
            f"Source        : Semantic Scholar"
        )

    # Fallback: Tavily
    try:
        tavily  = _get_tavily()
        results = tavily.invoke({"query": f"research paper metadata year venue citation count: {query}"})
        return _parse_tavily_results(results)
    except Exception as e:
        return f"Error retrieving metadata: {e}"


# ──────────────────────────────────────────────────────────────────────────────
#  TOOL 2 — Related Work Discovery
# ──────────────────────────────────────────────────────────────────────────────

@tool
def related_work_discovery(paper_title: str) -> str:
    """
    Find semantically related papers and citation-based neighbours.
    Input: Paper title or Semantic Scholar paper ID.
    """
    paper = _semantic_scholar_search(paper_title)
    if paper:
        paper_id   = paper.get("paperId", "")
        references = _semantic_scholar_related(paper_id, limit=5)
        if references:
            lines = [f"Related papers for: '{paper.get('title')}'\n"]
            for ref in references:
                cited = ref.get("citedPaper", {})
                lines.append(
                    f"  • {cited.get('title', 'N/A')} "
                    f"({cited.get('year', '?')}) — "
                    f"{cited.get('citationCount', '?')} citations"
                )
            return "\n".join(lines)

    # Fallback: Tavily
    try:
        tavily  = _get_tavily()
        results = tavily.invoke({"query": f"research papers related to or citing: {paper_title}"})
        return _parse_tavily_results(results)
    except Exception as e:
        return f"Error discovering related work: {e}"


# ──────────────────────────────────────────────────────────────────────────────
#  TOOL 3 — Trend Analytics
# ──────────────────────────────────────────────────────────────────────────────

@tool
def trend_analytics_tool(topic: str) -> str:
    """
    Identify publication frequency and emerging subtopics for a research area.
    Input: Research topic or keyword.
    """
    output_lines = [f"📊 Trend Analytics for: '{topic}'\n"]

    # Semantic Scholar: year-wise counts
    try:
        params = {"query": topic, "limit": 50, "fields": "title,year,venue"}
        resp   = requests.get(
            f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
            params=params, timeout=10,
        )
        resp.raise_for_status()
        papers = resp.json().get("data", [])

        if papers:
            from collections import Counter
            year_counts  = Counter(p["year"]  for p in papers if p.get("year"))
            venue_counts = Counter(p["venue"] for p in papers if p.get("venue"))

            output_lines.append("── Publication Frequency (by year) ──")
            for year in sorted(year_counts.keys(), reverse=True)[:6]:
                bar = "█" * year_counts[year]
                output_lines.append(f"  {year}: {bar} ({year_counts[year]} papers)")

            output_lines.append("\n── Top Venues ──")
            for venue, count in venue_counts.most_common(5):
                output_lines.append(f"  • {venue}: {count} papers")
        else:
            output_lines.append("No papers found on Semantic Scholar for this topic.")

    except Exception as e:
        output_lines.append(f"Semantic Scholar error: {e}")

    # Tavily: qualitative emerging trends
    try:
        tavily  = _get_tavily()
        results = tavily.invoke({
            "query": f"emerging research trends and new directions in {topic} 2024 2025"
        })
        output_lines.append("\n── Emerging Trend Signals (web) ──")
        output_lines.append(_parse_tavily_results(results))
    except Exception as e:
        output_lines.append(f"Tavily error: {e}")

    return "\n".join(output_lines)


# ──────────────────────────────────────────────────────────────────────────────
#  Tool list
# ──────────────────────────────────────────────────────────────────────────────

mcp_tools_list = [
    paper_metadata_lookup,
    related_work_discovery,
    trend_analytics_tool,
]