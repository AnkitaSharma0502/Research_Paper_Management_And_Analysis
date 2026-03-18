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
          1. Short Summary  (4-5 bullets)
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

        Retrieves the top-3 chunks from each specified paper, then
        synthesises a structured comparison in a single LLM call.
        the prompt instructs the LLM to emit FALLBACK_SIGNAL when the
        context doesn't contain a good answer. chat_view.py checks for this
        signal and triggers Tavily web search, exactly as it already does
        for ask_question().

        Args:
            query:        Comparison question (e.g. "Compare the methods used").
            vector_store: FAISS vector store instance.
            paper_ids:    List of paper_ids to compare.
        """
        # Retrieve top-3 chunks from each selected paper
        all_docs = []
        for pid in paper_ids:
            docs = vector_store.similarity_search(
                query,
                k=3,
                filter={"paper_id": pid},
            )
            all_docs.extend(docs)

        if not all_docs:
            return {
                "answer":  FALLBACK_SIGNAL,
                "sources": [],
            }

        context = self._format_docs(all_docs)

        compare_prompt = ChatPromptTemplate.from_template("""
You are an experienced research analyst tasked with comparing multiple research papers.

Answer the comparison question using ONLY the context provided below.

If the comparison question cannot be answered from the provided context
(e.g. the question is about something not covered in these papers),
respond with exactly:
"{fallback_signal}"

Otherwise, structure your response as:
1. A brief section for each paper summarising its relevant approach.
2. A side-by-side comparison table (if applicable).
3. An overall comparative conclusion.


CONTEXT:
{{context}}

COMPARISON QUESTION: 
{{question}}
""".format(fallback_signal=FALLBACK_SIGNAL))

        chain  = compare_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query})

        return {
            "answer":  answer,
            "sources": [doc.metadata for doc in all_docs],
        }