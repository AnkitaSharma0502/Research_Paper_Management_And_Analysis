from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents import create_agent

from tools.mcp_tools import mcp_tools_list


AGENT_SYSTEM_PROMPT = """You are a research assistant with access to tools.

Decide which tool fits the user's question:
- local_library_search: questions about papers the user has uploaded (datasets, methods, results, contributions of a specific paper in their library).
- paper_metadata_lookup: a paper's year, venue, citation count, or authors — especially when the paper is NOT in the local library.
- related_work_discovery: finding papers related to or cited by a given paper.

Rules:
- Always try local_library_search first when the question references "this paper", "the paper", or anything that sounds like the user's own library.
- For metadata about a specific external paper, prefer paper_metadata_lookup over web search.
- When citing local-library results, mention the paper title and section.
- Be concise and grounded. Do not invent results.
- Call each tool at most ONCE per query. If the tool result is unhelpful, report that honestly — do NOT retry the same tool with the same input.
- If a tool explicitly says metadata/citations are unavailable, tell the user exactly that — do NOT pad the answer with generic instructions like "go to Google Scholar" or "use Crossref". Short, honest answers beat fluff.
- When a tool returns URLs or links, ALWAYS include them in your answer so the user can click through to the paper.
"""


class ResearchAgent:
    """
    LangChain (v1) tool-calling agent that routes queries between the local
    FAISS library and external research tools (Semantic Scholar, Tavily).
    """

    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0,
        )

    @staticmethod
    def _wrap_with_dedup(tools: list) -> list:
        """
        Wrap each tool so that calling it twice with identical arguments
        returns the cached result instead of hitting the API again.
        This prevents the agent from looping on the same tool call when
        the first response is slow or initially unhelpful.
        """
        wrapped = []
        for t in tools:
            cache: dict = {}
            original_func = t.func

            def make_cached(func, tool_cache):
                def cached_func(**kwargs):
                    key = str(sorted(kwargs.items()))
                    if key in tool_cache:
                        return tool_cache[key]
                    result = func(**kwargs)
                    # Cache everything — success or failure — so the agent never
                    # calls the same tool+input more than once per query.
                    # Retrying a failed call won't help (same rate-limit window);
                    # the shared _paper_id_cache in mcp_tools handles cross-tool reuse.
                    tool_cache[key] = result
                    return result
                return cached_func

            wrapped.append(StructuredTool(
                name=t.name,
                description=t.description,
                args_schema=t.args_schema,
                func=make_cached(original_func, cache),
            ))
        return wrapped

    def _build_local_search_tool(self, vector_store, paper_id: Optional[str]):
        """Wraps the FAISS retriever as a tool. Closes over vector_store + paper_id."""
        search_kwargs: dict = {"k": 5}
        if paper_id:
            search_kwargs["filter"] = {"paper_id": paper_id}
            search_kwargs["fetch_k"] = 50

        captured_sources: List[dict] = []

        @tool
        def local_library_search(query: str) -> str:
            """Search the user's local research library for relevant paper sections.
            Input: a natural-language query. Returns the most relevant chunks with
            their paper title and section name."""
            docs = vector_store.similarity_search(query, **search_kwargs)
            if not docs:
                return "No matching content found in the local library."
            captured_sources.extend(doc.metadata for doc in docs)
            return "\n\n".join(
                f"[{d.metadata.get('title', 'Unknown')} "
                f"— {d.metadata.get('section_name', 'Section')}]\n{d.page_content}"
                for d in docs
            )

        return local_library_search, captured_sources

    def run(
        self,
        query: str,
        vector_store,
        paper_id: Optional[str] = None,
        paper_title: Optional[str] = None,
    ) -> Dict:
        """
        Executes the agent on a single query.

        Args:
            query:       Natural-language question.
            vector_store: FAISS vector store.
            paper_id:    If set, restricts local_library_search to this paper.
            paper_title: If set (single-paper mode), prepended to the prompt so the
                         agent can resolve "this paper" when calling external tools
                         like paper_metadata_lookup.

        Returns:
            {answer, sources, tool_calls}
        """
        local_search, captured_sources = self._build_local_search_tool(
            vector_store, paper_id
        )
        tools = [local_search] + self._wrap_with_dedup(mcp_tools_list)

        if paper_title:
            user_content = (
                f'Selected paper title: "{paper_title}"\n'
                f'When the user says "this paper", they mean the paper above. '
                f'Use that exact title for any external tool calls '
                f'(paper_metadata_lookup, related_work_discovery).\n\n'
                f'User question: {query}'
            )
        else:
            user_content = query

        agent = create_agent(self.llm, tools, system_prompt=AGENT_SYSTEM_PROMPT)
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_content}],
        })

        messages = result.get("messages", [])
        tool_calls = self._extract_tool_calls(messages)
        answer = self._extract_final_answer(messages)

        return {
            "answer":     answer,
            "sources":    captured_sources,
            "tool_calls": tool_calls,
        }

    @staticmethod
    def _extract_tool_calls(messages) -> List[Dict[str, str]]:
        """Pairs AIMessage tool_calls with their corresponding ToolMessage outputs."""
        pending: Dict[str, Dict[str, str]] = {}
        ordered: List[str] = []

        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get("id") or f"{tc.get('name')}-{len(ordered)}"
                    pending[tc_id] = {
                        "name":   tc.get("name", "unknown"),
                        "input":  str(tc.get("args", "")),
                        "output": "",
                    }
                    ordered.append(tc_id)
            elif isinstance(msg, ToolMessage):
                tc_id = getattr(msg, "tool_call_id", None)
                if tc_id and tc_id in pending:
                    pending[tc_id]["output"] = str(msg.content)[:500]

        return [pending[tid] for tid in ordered]

    @staticmethod
    def _extract_final_answer(messages) -> str:
        """Final answer is the last AIMessage with no pending tool_calls."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                return msg.content or ""
        return ""
