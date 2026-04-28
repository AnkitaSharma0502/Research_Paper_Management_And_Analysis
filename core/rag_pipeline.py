from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from models.schemas import ResearchPaper

# ──────────────────────────────────────────────────────────────────────────────
#  FALLBACK SIGNAL PHRASE
# ──────────────────────────────────────────────────────────────────────────────

FALLBACK_SIGNAL = "I don't have enough information in the provided papers to answer this."


class ResearchRAG:
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        """
        Initialises the Groq LLM used for summarisation and Q&A.
        temperature=0 keeps answers deterministic and grounded.
        """
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0,
        )

    # ------------------------------------------------------------------ #
    #  HELPERS
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_docs(docs) -> str:
        """Formats retrieved chunks with their section label for the prompt."""
        return "\n\n".join(
            f"[{doc.metadata.get('title', 'Unknown')} "
            f"— {doc.metadata.get('section_name', 'Section')}]\n{doc.page_content}"
            for doc in docs
        )

    # ------------------------------------------------------------------ #
    #  SUMMARISATION
    # ------------------------------------------------------------------ #

    def generate_summary(self, paper: ResearchPaper) -> str:
        """
        Generates a two-part structured summary:
          1. Short Summary  (5-6 bullets)
          2. Structured Summary  (Problem / Approach / Contributions /
                                  Results / Limitations)

        Uses abstract + key sections to avoid hallucination.
        """
        key_sections  = ["abstract", "introduction", "conclusion",
                         "results", "methodology", "method"]
        context_parts = []

        for section in paper.sections:
            if any(k in section.section_name.lower() for k in key_sections):
                context_parts.append(
                    f"[{section.section_name}]\n{section.content[:1500]}"
                )

        if not context_parts:
            context_parts.append(f"[Abstract]\n{paper.abstract[:3000]}")

        context = "\n\n".join(context_parts)

        summary_prompt = ChatPromptTemplate.from_template("""
You are a research assistant. Generate a summary based ONLY on the provided
paper content. Do NOT invent results or details not present in the text.


PAPER TITLE: 
{title}

CONTENT:
{context}


Please provide:

## Short Summary
(3-4 concise bullet points capturing the paper's essence)

## Structured Summary

**Problem Statement:**
What specific problem or gap does this paper address?

**Proposed Approach:**
What method, model, or framework is proposed?

**Key Contributions:**
List the main contributions.

**Results:**
What were the key findings or metrics?

**Limitations:**
What limitations or future work are mentioned?

Maintain a neutral, academic tone throughout.
""")

        chain = summary_prompt | self.llm | StrOutputParser()
        return chain.invoke({"title": paper.title, "context": context})

    # ------------------------------------------------------------------ #
    #  SINGLE-PAPER / LIBRARY Q&A  (RAG)
    # ------------------------------------------------------------------ #

    def ask_question(
        self,
        query: str,
        vector_store,
        paper_id: Optional[str] = None,
    ) -> Dict:
        """
        RAG pipeline for answering questions.

        Args:
            query:        Natural language question.
            vector_store: FAISS vector store instance.
            paper_id:     If provided, restricts retrieval to that paper only.
                          If None, searches the entire library.
        """
        search_kwargs: dict = {"k": 5}
        if paper_id:
            search_kwargs["filter"] = {"paper_id": paper_id}

        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

        qa_prompt = ChatPromptTemplate.from_template("""
You are a technical research assistant.

Answer the question using ONLY the context provided below.
If the answer is not present in the context, respond with exactly:
"{fallback_signal}"

Always mention the paper title and section name when referencing information.


CONTEXT:
{{context}}


QUESTION: 
{{question}}

Provide a clear, professional, and well-structured answer.
""".format(fallback_signal=FALLBACK_SIGNAL))

        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: self._format_docs(retriever.invoke(x["question"]))
            )
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke({"question": query})
        docs   = retriever.invoke(query)

        return {
            "answer":  answer,
            "sources": [doc.metadata for doc in docs],
        }

    # ------------------------------------------------------------------ #
    #  CROSS-PAPER COMPARATIVE Q&A
    # ------------------------------------------------------------------ #

    def compare_papers(
        self,
        query: str,
        vector_store,
        paper_ids: List[str],
    ) -> Dict:
        """
        Answers comparative questions across multiple papers.

        Retrieves the top-4 chunks from each specified paper (excluding
        Reference/Bibliography sections to avoid citation confusion) and
        synthesises a structured comparison in a single LLM call.

        If retrieval finds nothing useful, the prompt emits FALLBACK_SIGNAL,
        but chat_view INTENTIONALLY does not trigger web fallback in compare
        mode — Tavily can't answer cross-paper questions.

        Args:
            query:        Comparison question (e.g. "Compare the methods used").
            vector_store: FAISS vector store instance.
            paper_ids:    List of paper_ids to compare.
        """
        # Over-fetch then drop reference/bibliography chunks — those contain
        # citations to other papers that the LLM confuses with comparison
        # candidates, ballooning a 2-paper compare into 4+ table rows.
        all_docs = []
        for pid in paper_ids:
            docs = vector_store.similarity_search(
                query,
                k=6,
                filter={"paper_id": pid},
            )
            docs = [
                d for d in docs
                if not any(
                    w in d.metadata.get("section_name", "").lower()
                    for w in ("reference", "bibliograph", "works cited")
                )
            ][:4]
            all_docs.extend(docs)

        if not all_docs:
            return {
                "answer":  FALLBACK_SIGNAL,
                "sources": [],
            }

        context = self._format_docs(all_docs)

        # Pin the comparison set so the LLM can't drift into citing papers
        # it saw inside the retrieved chunks.
        titles = sorted({
            d.metadata.get("title", "Unknown") for d in all_docs
        })
        titles_block = "\n".join(f"- {t}" for t in titles)

        compare_prompt = ChatPromptTemplate.from_template("""
You are an experienced research analyst comparing EXACTLY these {n_papers} papers and no others:
{titles_block}

Do not introduce any paper outside this list. Any paper titles appearing
inside the CONTEXT below as citations or references are NOT papers to
compare — they are only background for the listed papers.

Answer the comparison question using ONLY the context provided below.

If the comparison question cannot be answered from the provided context,
respond with exactly:
"{fallback_signal}"

Otherwise, structure your response as:
1. A brief section for each of the {n_papers} listed papers summarising its relevant approach.
2. A side-by-side comparison table with EXACTLY {n_papers} rows — two per listed paper.
3. An overall comparative conclusion.


CONTEXT:
{{context}}

COMPARISON QUESTION:
{{question}}
""".format(
            fallback_signal=FALLBACK_SIGNAL,
            n_papers=len(titles),
            titles_block=titles_block,
        ))

        chain  = compare_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query})

        return {
            "answer":  answer,
            "sources": [doc.metadata for doc in all_docs],
        }