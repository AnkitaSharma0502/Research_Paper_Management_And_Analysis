import os


class ResearchSearchTool:
    """
    Tavily web search wrapper used by the non-agent fallback path in chat_view.

    When the RAG pipeline emits FALLBACK_SIGNAL because local context is
    insufficient, chat_view calls search_papers() to fetch web context and
    re-prompts the LLM with the results.
    """

    def __init__(self, max_results: int = 5):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "TAVILY_API_KEY is not set. Please add it to your .env file."
            )
        from langchain_tavily import TavilySearch
        self.tool = TavilySearch(max_results=max_results)
        self.max_results = max_results

    def search_papers(self, query: str) -> str:
        """Academic-focused web search."""
        try:
            results = self.tool.invoke({"query": f"research paper academic study: {query}"})
            return self._format_results(results, label="Academic Search")
        except Exception as e:
            return f"Academic search failed: {e}"

    def _format_results(self, results, label: str = "Search Results") -> str:
        """Formats raw Tavily results into a clean LLM-readable string."""
        if not results:
            return "No results found."

        if isinstance(results, str):
            import json
            try:
                results = json.loads(results)
                if isinstance(results, dict):
                    results = results.get("results", [])
            except Exception:
                return results

        if not isinstance(results, list):
            return str(results)

        lines = [f"{label}\n" + "-" * 50]
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
