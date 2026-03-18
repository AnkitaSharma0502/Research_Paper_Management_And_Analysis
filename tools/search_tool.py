import os
from typing import List, Dict


class ResearchSearchTool:
    """
    Web search wrapper using TavilySearch (correct class for langchain-tavily >= 0.1.0).
    """

    def __init__(self, max_results: int = 5):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "TAVILY_API_KEY is not set. "
                "Please add it to your .env file:\n  TAVILY_API_KEY=your_key_here"
            )
        # ✅ Correct class: TavilySearch (not TavilySearchResults)
        from langchain_tavily import TavilySearch
        self.tool = TavilySearch(max_results=max_results)
        self.max_results = max_results

    # ------------------------------------------------------------------ #
    #  SEARCH METHODS
    # ------------------------------------------------------------------ #

    def search(self, query: str) -> str:
        """General web search."""
        try:
            results = self.tool.invoke({"query": query})
            return self._format_results(results)
        except Exception as e:
            return f"Search failed: {e}"

    def search_papers(self, query: str) -> str:
        """Academic-focused search."""
        try:
            results = self.tool.invoke({"query": f"research paper academic study: {query}"})
            return self._format_results(results, label="📄 Academic Search")
        except Exception as e:
            return f"Academic search failed: {e}"

    def search_trends(self, topic: str) -> str:
        """Searches for recent trends in a research area."""
        try:
            results = self.tool.invoke({
                "query": f"emerging research trends recent advances 2024 2025 in: {topic}"
            })
            return self._format_results(results, label="📈 Trend Search")
        except Exception as e:
            return f"Trend search failed: {e}"

    def search_citations(self, paper_title: str) -> str:
        """Searches for papers citing or related to a given paper."""
        try:
            results = self.tool.invoke({
                "query": f"papers citing or related to research paper: {paper_title}"
            })
            return self._format_results(results, label="🔗 Citation Search")
        except Exception as e:
            return f"Citation search failed: {e}"

    # ------------------------------------------------------------------ #
    #  FORMATTER
    # ------------------------------------------------------------------ #

    def _format_results(
        self,
        results,
        label: str = "🔍 Search Results",
    ) -> str:
        """Formats raw Tavily results into a clean LLM-readable string."""
        if not results:
            return "No results found."

        # TavilySearch returns a JSON string — parse it if needed
        if isinstance(results, str):
            import json
            try:
                results = json.loads(results)
                # TavilySearch wraps results under "results" key
                if isinstance(results, dict):
                    results = results.get("results", [])
            except Exception:
                return results  # return raw string if unparseable

        if not isinstance(results, list):
            return str(results)

        lines = [f"{label}\n" + "─" * 50]
        for i, r in enumerate(results, 1):
            title   = r.get("title",   "No title")
            url     = r.get("url",     "N/A")
            content = r.get("content", "").strip()
            if len(content) > 400:
                content = content[:400] + "..."

            lines.append(
                f"\n[{i}] {title}\n"
                f"    Source : {url}\n"
                f"    Snippet: {content}"
            )

        return "\n".join(lines)