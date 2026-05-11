import os
import time
import requests
from langchain_core.tools import tool
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
    #  Correct class: TavilySearch (not TavilySearchResults)
    from langchain_tavily import TavilySearch
    return TavilySearch(max_results=5)

@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for recent or external research information.
    Use ONLY when local library and Semantic Scholar are insufficient.
    """
    try:
        tavily = _get_tavily()
        results = tavily.invoke({
            "query": f"research paper academic study: {query}"
        })
        return _parse_tavily_results(results)

    except Exception as e:
        return f"Web search failed: {e}"
# ──────────────────────────────────────────────────────────────────────────────
#  Semantic Scholar helper
# ──────────────────────────────────────────────────────────────────────────────

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"

# Shared cache: normalized title → paperId, populated by whichever tool finds it first.
# Lets related_work_discovery skip the search step if paper_metadata_lookup already ran.
_paper_id_cache: dict = {}

def _normalize_title(title: str) -> str:
    return title.lower().strip()


def _semantic_scholar_search(title: str, retries: int = 4) -> Optional[dict]:
    """
    Searches Semantic Scholar with retry on 429/5xx AND on empty results.
    The free tier sometimes returns 200 OK with an empty data list instead of
    a proper 429 when rate-limited, so empty responses must also be retried.
    """
    params = {
        "query":  title,
        "limit":  1,
        "fields": "title,year,venue,citationCount,authors,externalIds",
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
                params=params, timeout=10,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < retries:
                    time.sleep(2 ** attempt)   # 1s, 2s, 4s, 8s
                    continue
                return None
            resp.raise_for_status()
            results = resp.json().get("data", [])
            if results:
                paper = results[0]
                # Cache the paperId so other tools can skip searching
                _paper_id_cache[_normalize_title(title)] = paper.get("paperId")
                return paper
            # Empty data can mean rate-limiting on a 200 OK — retry with backoff
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return None
        except requests.RequestException:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


def _semantic_scholar_full(paper_id: str, retries: int = 3) -> Optional[dict]:
    """Fetch a paper's metadata plus references AND citations in one call."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}",
                params={
                    "fields": (
                        "title,year,citationCount,"
                        "references.title,references.year,references.citationCount,references.paperId,references.externalIds,"
                        "citations.title,citations.year,citations.citationCount,citations.paperId,citations.externalIds"
                    ),
                },
                timeout=10,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


def _paper_url(paper: dict) -> str:
    """Build the best available URL for a paper: ArXiv preferred, else Semantic Scholar."""
    arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    paper_id = paper.get("paperId")
    if paper_id:
        return f"https://www.semanticscholar.org/paper/{paper_id}"
    return ""

import re

def _smart_truncate(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text

    sentences = re.split(r'(?<=[.!?])\s+', text)

    output = ""

    for sentence in sentences:
        if len(output) + len(sentence) <= limit:
            output += sentence + " "
        else:
            break

    # fallback if no full sentence fits
    if not output:
        truncated = text[:limit]
        last_space = truncated.rfind(" ")

        if last_space != -1:
            truncated = truncated[:last_space]

        return truncated + "..."

    return output.strip() + "..."

def _parse_tavily_results(results) -> str:
    import json

    if isinstance(results, str):
        try:
            results = json.loads(results)
        except Exception:
            return results

    if isinstance(results, dict):
        items = results.get("results", [])
    elif isinstance(results, list):
        items = results
    else:
        return str(results)

    if not items:
        return "[Web search returned no results]"

    snippets = "\n\n".join(
        f"- {r.get('title', '')}: "
        f"{_smart_truncate(r.get('content', ''))}"
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
        url = _paper_url(paper)
        return (
            f"Title         : {paper.get('title', 'N/A')}\n"
            f"Year          : {paper.get('year', 'N/A')}\n"
            f"Venue         : {paper.get('venue', 'N/A')}\n"
            f"Citation Count: {paper.get('citationCount', 'N/A')}\n"
            f"Authors       : {authors or 'N/A'}\n"
            f"Link          : {url or 'N/A'}\n"
            f"Source        : Semantic Scholar"
        )

    # Fallback: Tavily — search for the paper itself, not articles about metadata.
    try:
        tavily  = _get_tavily()
        results = tavily.invoke({"query": f'"{query}" research paper'})
        parsed  = _parse_tavily_results(results)
        return (
            "Semantic Scholar had no record of this paper. "
            "Web search results below — these may NOT contain a citation count; "
            "if not present, say so plainly instead of guessing.\n\n" + parsed
        )
    except Exception as e:
        return (
            f"No metadata could be retrieved for '{query}'. "
            f"Semantic Scholar returned no match and web search failed ({e}). "
            f"Tell the user the citation count is unavailable."
        )


# ──────────────────────────────────────────────────────────────────────────────
#  TOOL 2 — Related Work Discovery
# ──────────────────────────────────────────────────────────────────────────────

@tool
def related_work_discovery(paper_title: str) -> str:
    """
    Find related papers in two directions: (a) papers this one cites
    (its bibliography), and (b) papers that cite this one (follow-up work).
    Useful for "what should I read alongside this paper" and "what builds on this".
    Input: Paper title or Semantic Scholar paper ID.
    """
    # Use cached paperId if paper_metadata_lookup already found this paper,
    # saving one round-trip to Semantic Scholar.
    cached_id = _paper_id_cache.get(_normalize_title(paper_title))
    if cached_id:
        paper = {"paperId": cached_id, "title": paper_title}
    else:
        paper = _semantic_scholar_search(paper_title)

    if paper:
        full = _semantic_scholar_full(paper.get("paperId", ""))
        if full:
            lines = [f"Related work for: '{full.get('title')}'\n"]

            cites = (full.get("citations") or [])[:5]
            if cites:
                lines.append("── Cited by (newer work):")
                for c in cites:
                    url = _paper_url(c)
                    link = f" | {url}" if url else ""
                    lines.append(
                        f"  • {c.get('title', 'N/A')} "
                        f"({c.get('year', '?')}) — "
                        f"{c.get('citationCount', '?')} citations{link}"
                    )

            refs = (full.get("references") or [])[:5]
            if refs:
                lines.append("\n── This paper's references (background):")
                for r in refs:
                    url = _paper_url(r)
                    link = f" | {url}" if url else ""
                    lines.append(
                        f"  • {r.get('title', 'N/A')} "
                        f"({r.get('year', '?')}) — "
                        f"{r.get('citationCount', '?')} citations{link}"
                    )

            if cites or refs:
                return "\n".join(lines)

    # Fallback: Tavily
    try:
        tavily  = _get_tavily()
        results = tavily.invoke({"query": f'papers related to "{paper_title}"'})
        return (
            "Semantic Scholar had no record of this paper, so citation-based "
            "neighbours are unavailable. Web results below may suggest related "
            "work but cannot be confirmed as actual citations:\n\n"
            + _parse_tavily_results(results)
        )
    except Exception as e:
        return (
            f"No related work could be retrieved for '{paper_title}' "
            f"(Semantic Scholar miss; web search error: {e})."
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Tool list
# ──────────────────────────────────────────────────────────────────────────────

mcp_tools_list = [
    paper_metadata_lookup,
    related_work_discovery,
    web_search_tool
]