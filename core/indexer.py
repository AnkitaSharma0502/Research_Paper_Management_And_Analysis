import os
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from models.schemas import ResearchPaper, PaperSection
from typing import List, Optional


# Sections short enough to keep whole — never chunk these
SHORT_SECTIONS = {"abstract", "conclusion", "future work", "acknowledgements"}


class ResearchIndexer:
    def __init__(self):
        """
        Initialises the embedding model based on EMBEDDING_PROVIDER env var.

        
        """
        self.embeddings   = self._init_embeddings()
        self.vector_store: Optional[FAISS] = None

    # ------------------------------------------------------------------ #
    #  EMBEDDING INITIALISATION
    # ------------------------------------------------------------------ #

    def _init_embeddings(self):
        provider   = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        if provider == "huggingface_api":
            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if not hf_token:
                raise EnvironmentError(
                    "HUGGINGFACEHUB_API_TOKEN is not set.\n"
                  
                )
            from langchain_huggingface import HuggingFaceEndpointEmbeddings
            print(" Using HuggingFace Inference API for embeddings.")
            return HuggingFaceEndpointEmbeddings(
                model=model_name,
                huggingfacehub_api_token=hf_token,
            )
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            print(" Using local HuggingFace model for embeddings.")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
            )

    # ------------------------------------------------------------------ #
    #  CONTENT-AWARE CHUNKING
    # ------------------------------------------------------------------ #

    def _make_doc(self, text: str, paper: ResearchPaper, section_name: str) -> Document:
        """Helper — wraps a text chunk as a LangChain Document with full metadata."""
        return Document(
            page_content=text,
            metadata={
                "paper_id":     paper.paper_id,
                "title":        paper.title,
                "section_name": section_name,
                "year":         paper.year     or "Unknown",
                "venue":        paper.venue    or "Unknown",
                "authors":      ", ".join(paper.authors),
                "keywords":     ", ".join(paper.keywords),
            },
        )

    def _chunk_section(
        self,
        section: PaperSection,
        paper: ResearchPaper,
    ) -> List[Document]:
        """
        Applies the right chunking strategy based on section type and length.

        Strategy:
          SHORT sections (abstract, conclusion, etc.) or content ≤ 600 chars
            → Keep as ONE chunk — never split
          REFERENCES section
            → Split per individual reference entry
          All other sections
            → Paragraph-aware splitting:
               tries \n\n first, then \n, then sentence boundary ". "
               chunk_size=800 to keep ideas complete
        """
        name    = section.section_name.lower()
        content = section.content.strip()

        if not content:
            return []

        # ── 1. Short sections or small content → keep whole ───────────
        if name in SHORT_SECTIONS or len(content) <= 600:
            return [self._make_doc(content, paper, section.section_name)]

        # ── 2. References → one chunk per reference entry ─────────────
        if "reference" in name:
            import re
            entries = re.findall(r'(?:\[\d+\]|\d+\.)\s+.+', content)
            if entries:
                return [
                    self._make_doc(entry.strip(), paper, section.section_name)
                    for entry in entries
                    if entry.strip()
                ]
            # Fallback if references aren't numbered
            return [self._make_doc(content, paper, section.section_name)]

        # ── 3. Long sections → paragraph-aware splitting ──────────────
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=[
                "\n\n",   # paragraph boundary  ← try first
                "\n",     # line boundary
                ". ",     # sentence boundary
                " ",      # word boundary       ← last resort
                "",       # character           ← never ideally
            ],
        )

        chunks = splitter.split_text(content)
        return [
            self._make_doc(chunk, paper, section.section_name)
            for chunk in chunks
            if chunk.strip()
        ]

    def create_chunks(self, paper: ResearchPaper) -> List[Document]:
        """
        Main chunking entry point.
        Applies content-aware chunking per section.
        Falls back to full_text if no sections available.
        """
        all_chunks: List[Document] = []

        # ── Section-level chunking (preferred) ────────────────────────
        if paper.sections:
            for section in paper.sections:
                all_chunks.extend(self._chunk_section(section, paper))

        # ── Fallback: full_text ────────────────────────────────────────
        elif paper.full_text:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            for chunk in splitter.split_text(paper.full_text):
                if chunk.strip():
                    all_chunks.append(
                        self._make_doc(chunk, paper, "Full Text")
                    )

        return all_chunks

    # ------------------------------------------------------------------ #
    #  INDEXING
    # ------------------------------------------------------------------ #

    def index_paper(self, paper: ResearchPaper) -> None:
        """Generates embeddings for one paper and adds them to FAISS."""
        chunks = self.create_chunks(paper)
        if not chunks:
            print(f"  No chunks generated for: {paper.title}")
            return

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

        print(f" Indexed '{paper.title}' ({len(chunks)} chunks)")

    def clear_index(self) -> None:
        """Resets the vector store."""
        self.vector_store = None
        print("  Index cleared.")