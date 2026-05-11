# Research Intelligence System — Complete Study Guide

A consolidated reference for the Gen AI / RAG concepts, project decisions, and interview defenses developed across this project. Read end-to-end as a textbook, or jump to specific parts via the table of contents.

---

## Table of Contents

- [Part 1 — Gen AI Fundamentals (10 lessons)](#part-1)
- [Part 2 — Beginner Interview Foundations](#part-2)
- [Part 3 — Deep Technical Concepts](#part-3) (embeddings, pooling, normalization, softmax, chunking)
- [Part 4 — Build From Scratch (17-day developer journal)](#part-4)
- [Part 5 — File-by-File Interview Defense Playbooks](#part-5)
- [Part 6 — How MCP Tools Actually Work](#part-6)
- [Part 7 — Cross-cutting Patterns & Design Philosophy](#part-7)
- [Part 8 — Common Interview Questions with Model Answers](#part-8)
- [Part 9 — Project Audit & Critical Analysis](#part-9)
- [Part 10 — Final Cheat Sheets](#part-10)

---

<a name="part-1"></a>

# Part 1 — Gen AI Fundamentals: 10-Lesson Curriculum

> *Use this project as the textbook. Each lesson grounds one core concept in the actual code.*

## Lesson 1 — Why GenAI? Why RAG?

**The core problem:** LLMs are trained on data up to a cutoff date. They don't know your private PDFs, company documents, or papers published last week. They also *hallucinate* — confidently making up facts when they don't know.

**The pattern that solves this: Retrieval-Augmented Generation (RAG).**

> Instead of asking the LLM to *remember* an answer, you give it the relevant text *in the prompt* and ask it to *read* and respond.

**Where this lives in the project:** `core/rag_pipeline.py:114` — `ask_question`. The prompt: *"Answer the question using ONLY the context provided below."* That single sentence is the whole RAG ethos.

**Pitfall:** RAG quality depends 70% on *retrieval quality*, 30% on the LLM. A good LLM with bad chunks beats a bad LLM only marginally. Spend your engineering on retrieval.

## Lesson 2 — Embeddings: turning meaning into geometry

**Mental model:** an embedding is a high-dimensional point. Two pieces of text that *mean similar things* are close points in this space, even if they share no words.

> *"The cat sat on the mat"* and *"a feline rested on the rug"* live near each other. *"The cat sat on the mat"* and *"calculate quarterly revenue"* live far apart.

This is what makes search work without keyword matching.

**In this project:** every chunk is converted into a 384-dimensional vector by `all-MiniLM-L6-v2`. When you ask a question, your question is *also* embedded into the same space. Then we ask: *"which chunks are nearest to my question?"*

**Where:** `core/indexer.py:32` — `_init_embeddings`. The model is `sentence-transformers/all-MiniLM-L6-v2` running on CPU.

**Pitfall:** the embedding model is a fixed translator. If your model is bad at academic text, your retrieval is permanently bad. MiniLM is a reasonable default.

## Lesson 3 — Chunking: the most underrated step

**Problem:** an LLM can't process a 30-page PDF in one shot. And even if it could, you don't want to embed an entire paper as a single point — that's like averaging a whole book into one number.

**Fix:** break the document into small, semantically coherent pieces.

> Naive chunking (every 500 chars) splits sentences mid-thought, separates a question from its answer, and breaks code blocks. **Content-aware chunking** respects the document's structure.

**In this project:**
- Short sections (Abstract, Conclusion) → kept whole
- References → one chunk per `[N]` entry
- Long sections (Methods, Results) → split at paragraph boundaries with overlap

**Where:** `core/indexer.py:72` — `_chunk_section`.

**Pitfall:** chunk too small → context shatters. Chunk too big → retrieval becomes coarse, and you waste tokens. 800-char + 150-overlap is reasonable for academic prose.

## Lesson 4 — Vector search: nearest neighbors at scale

**Problem:** you have 10,000 chunk-vectors. Someone asks a question. How do you find the nearest 5 without comparing to all 10,000?

**Answer: a vector database.** It uses approximate nearest-neighbor (ANN) algorithms (HNSW, IVF, etc.) to give you "the 5 closest vectors" in milliseconds.

**In this project: FAISS** (Facebook AI Similarity Search). In-memory, single-process. Production systems use Pinecone, Weaviate, Chroma, or Postgres+pgvector — but the core API is the same: `add_documents()` and `similarity_search()`.

**Where:** `core/indexer.py:159` — `index_paper`. Then `core/rag_pipeline.py:133` — `as_retriever(search_kwargs={"k": 5, "filter": {"paper_id": ...}})`.

**Note the metadata filter** — every chunk carries metadata (`paper_id`, `title`, `section`, `year`). So we can ask for *"top 5 chunks **from this specific paper**"*. That's how Single-Paper mode works.

**Pitfall:** "similar" by cosine distance doesn't always mean "useful". A query about *"results"* might return chunks from a Limitations section. Real systems add reranking on top.

## Lesson 5 — The RAG loop, end to end

```
1. INDEX TIME (offline)
   PDF → chunks → embeddings → FAISS

2. QUERY TIME (online)
   user_question → embedding → FAISS → top-K chunks
                                        ↓
                            ┌───────────┴──────────┐
                            ↓                      ↓
                      format chunks          user_question
                      as context                    │
                            ↓                      ↓
                            └──→  LLM PROMPT  ←────┘
                                       ↓
                                    answer
```

**Look at the whole loop:** `core/rag_pipeline.py:155-164`. Read those lines slowly. That's the whole pipeline:

```python
rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: self._format_docs(retriever.invoke(x["question"]))
    )
    | qa_prompt | self.llm | StrOutputParser()
)
```

The `|` operator is LangChain's chain composition. **Memorize this idiom.** You'll write it dozens of times.

## Lesson 6 — Prompt engineering: contracts, not magic words

A good prompt has:
1. **A role.** *"You are a technical research assistant."*
2. **A grounding rule.** *"Use ONLY the context below."*
3. **A failure clause.** *"If the answer isn't there, respond with exactly: '...'"*
4. **A format spec.** *"Always mention the paper title and section name."*

**Where:** `core/rag_pipeline.py:135` — the QA prompt. All four ingredients are there.

**The sentinel pattern:** notice the project uses an exact phrase — `FALLBACK_SIGNAL = "I don't have enough information..."` — that the LLM emits when it gives up. The Streamlit code watches for this string and triggers a web search. **The LLM's output isn't just text, it's a control signal.**

**Pitfall:** LLMs drift. A sentinel like "I don't know" gets paraphrased to "I'm unsure". Use long, specific sentences for sentinels.

## Lesson 7 — LangChain: less framework, more glue

LangChain is glue code. It composes LLM calls, retrievers, prompts, and parsers using `|` operators.

Two patterns matter:
- **Chain composition:** `prompt | llm | parser`
- **Agent construction:** `create_agent(llm, tools, system_prompt=...)`

**Pitfall:** LangChain has a sprawling API and changes fast. Pick a small subset (chains, agents, retrievers) and stick with it.

## Lesson 8 — Agents: when the LLM decides what to do

So far, **you** wrote the code that called the retriever. With agents, **the LLM** decides which tool to call.

**The agent loop:**
```
1. User asks a question
2. LLM sees: question + list of available tools (with descriptions)
3. LLM emits a structured "tool call": {tool: paper_metadata_lookup, args: {...}}
4. Your code executes that tool, gets the result
5. Result is added back into the conversation
6. LLM sees the result, decides: "another tool?" or "done, here's the final answer"
7. Loop until done
```

**The tools in this project** (`tools/mcp_tools.py:245`):
- `paper_metadata_lookup` — for citation counts, venues
- `related_work_discovery` — for finding cited/related papers
- `trend_analytics_tool` — for publication frequency / emerging topics
- `local_library_search` — built dynamically in `core/agent.py:41`

**Why this is a big deal:** the LLM picks the right tool based on the question. *"What dataset does this paper use?"* → `local_library_search`. *"What's the citation count of BERT?"* → `paper_metadata_lookup`. **You wrote zero routing logic.** The LLM does it from tool descriptions alone.

**Pitfall:** agents are 5–10× more expensive in tokens than plain RAG. Use them for tool-shaped queries.

## Lesson 9 — Fallback strategies: where prototypes die

| Failure | Mitigation | Where |
|---|---|---|
| Semantic Scholar 429 | Retry with backoff | `tools/mcp_tools.py:30` |
| Tool returns nothing | Honest "no record" message, not fluff | `tools/mcp_tools.py:131` |
| LLM gives up on context | FALLBACK_SIGNAL → Tavily web search | `ui/chat_view.py` |
| User uploads same PDF renamed | Dedupe by content hash, not filename | `ui/library_view.py` |
| User leaves Year blank | `pd.notna()` before `int(...)` | same file |

**Lesson:** every API call can fail. Every LLM output can be malformed. Plan for it from day one.

## Lesson 10 — Observability: how do you know it's working?

Three levels of observability everyone needs:

1. **Direct probe** — does the API return real data?
2. **Component probe** — does each module work in isolation?
3. **End-to-end UI probe** — does the user-facing flow do what users expect?

**Where:** the **🔧 Research tools used** expander in `ui/chat_view.py` is observability surfaced to users. They can see exactly which tools the agent called.

> Build the inspector before you build the feature.

## The portable cheat sheet

Every GenAI app you'll ever build has these layers:

| Layer | Concept | This project |
|---|---|---|
| 1. Ingest | Parse / clean / extract structure | `parser.py` |
| 2. Chunk | Split into semantically coherent units with metadata | `indexer.py` |
| 3. Embed | Map to vector space | MiniLM in `indexer.py` |
| 4. Index | Store vectors in a queryable DB | FAISS in `indexer.py` |
| 5. Retrieve | Top-K nearest, optionally filtered | `vector_store.as_retriever()` |
| 6. Prompt | Compose context + question + grounding rules | `rag_pipeline.py` |
| 7. Generate | LLM call, structured output if needed | Groq via LangChain |
| 8. Orchestrate | Chain or agent — who decides what to call? | `rag_pipeline.py` vs `agent.py` |
| 9. Fallback | What happens when stages fail? | `mcp_tools.py` retry; `chat_view.py` sentinel |
| 10. Observe | Surface tool calls, sources, latencies | Sources expander, tool-calls expander |

---

<a name="part-2"></a>

# Part 2 — Beginner Interview Foundations

## What recruiters actually look for

Before tech, learn this. Interviewers grade you on five things, in this order:

| Rank | What they grade | What it sounds like |
|---|---|---|
| 1 | **Clarity of thinking** | Can you take a fuzzy question and give a structured answer in 90 seconds? |
| 2 | **Trade-off awareness** | When you make a choice, can you name what you're giving up? |
| 3 | **Connection to real work** | "I had this exact problem in my project — here's what I did." |
| 4 | **Honest gaps** | "I don't know X but I'd find out by Y." |
| 5 | **Raw correctness** | Does your answer have factual errors? |

**Notice: correctness is #5, not #1.** A junior who says *"I'm not 100% sure but I think transformers use self-attention to weigh token relationships"* beats a senior who freezes when asked to apply concepts.

## The 10 concepts you must own

### 1. What is an LLM, really?

> A model that predicts the next *token* (word fragment) given the previous tokens. That's all. Everything else (reasoning, conversation, code generation) is emergent behavior from doing this very well at massive scale.

### 2. What's a token?

> A subword piece. "Transformer" might be one token; "Transformers" might be ["Transform", "ers"]. Models have a fixed vocabulary (~30K–100K tokens) and use Byte-Pair Encoding (BPE) or similar.

### 3. What's the context window?

> The maximum number of tokens an LLM can process in one call. Llama 3.3-70B has 128K tokens (~96K words, roughly 200 pages of text).

### 4. What's a transformer (the architecture)?

> A neural network architecture built on **self-attention**. Each token in the input "looks at" every other token to figure out what it should pay attention to.

> Memorize the intuition: *"attention lets the model decide which other words matter for understanding this word."*

### 5. Difference between embedding models and chat models?

> **Embedding models** (MiniLM, mpnet, OpenAI ada) take text → output a single fixed-size vector. Trained for *similarity*. **Chat models** (GPT, Llama, Claude) take text → output more text. Trained for *generation*.

### 6. What's RAG?

> Retrieval-Augmented Generation. **R**etrieve → **A**ugment (prompt) → **G**enerate.

### 7. What's prompt engineering?

> Designing the input to an LLM to reliably get the output you want. Includes: system messages, few-shot examples, output format specs, temperature/top-p tuning, and chain-of-thought prompting.

> Senior framing: prompts are *contracts*, not magic words. A good prompt has a role, grounding rules, a failure clause, and a format spec.

### 8. Fine-tuning vs RAG?

| | RAG | Fine-tune |
|---|---|---|
| Update knowledge | ✅ instant | ❌ retrain |
| Per-user data | ✅ trivial | ❌ infeasible |
| Tone/style change | ❌ needs prompts | ✅ baked in |
| Cost | $$ per query | $$$ upfront |

### 9. What's an agent?

> An LLM that uses *tools*. It can call functions (search the web, query a database, run code), see the results, and decide what to do next. The "decision" is just the LLM emitting structured JSON.

### 10. What's a hallucination?

> When an LLM generates something that *sounds* plausible but is factually wrong. Mitigation: RAG (provide truth in context), grounding instructions, citation requirements, output validation, lower temperature.

## The STAR-T framework for project storytelling

| Letter | Means | Example for this project |
|---|---|---|
| **S**ituation | What problem? | "Researchers can't quickly find relevant info across hundreds of PDFs." |
| **T**ask | What was your role / scope? | "Build an end-to-end research assistant that ingests PDFs and answers questions." |
| **A**ction | What did *you* do, technically? | "Designed a layout-aware PDF parser, content-aware chunker, RAG pipeline with FAISS, and a tool-calling agent." |
| **R**esult | What worked? | "End-to-end system handling single-paper, library-wide, and cross-paper Q&A. 41 tests, ~85% spec coverage." |
| **T**rade-off | What did you choose to *not* do, and why? | "Skipped persistent storage and multi-tenant isolation — prototype, not production." |

**The T is what separates senior from junior.** Every project has trade-offs. Naming them shows you *chose* rather than stumbled.

## Six "tell me about a time" questions to drill

### 1. *"Tell me about a tough bug you solved."*

> *"In my research-paper system, the citation network kept showing zero edges between papers. I traced it to the matching logic: it was comparing paper titles to raw reference strings using exact string equality. Reference strings look like '[1] Vaswani A. Attention Is All You Need 2017' — they never literally equal a stored title. I replaced exact match with normalized substring containment: lowercase both sides, strip punctuation, then check if the normalized title appears inside the normalized reference. Added a length guard for titles under 8 chars."*

### 2. *"Tell me about a trade-off you made."*

> *"I had two retrieval paths: plain RAG (cheap) and a tool-calling agent (expensive but flexible). Initially I considered making the agent the default, but agents are 3–5× more expensive in tokens. So I made the agent an opt-in toggle, and only enable it in Single Paper mode. Compare mode I deliberately keep on plain RAG because the agent could decide to skip a paper, breaking the comparison guarantee."*

### 3. *"Tell me about something you'd do differently."*

> *"I'd write tests before features, not after. I added regression tests at the end, and the moment I did, I discovered three bugs — one in the citation graph, one in the title extractor, one in chunking. They'd been there silently for weeks. The lesson: testing isn't a checkbox at the end, it's a feedback loop during development."*

### 4. *"Walk me through your system architecture."*

> *"PDF goes through a parser that produces a structured ResearchPaper object. That gets chunked content-aware — short sections kept whole, references split per entry, long sections split with overlap. Chunks are embedded with MiniLM and stored in FAISS with metadata. At query time, two retrieval paths: plain RAG (FAISS retrieve → format → LLM) and an agent path (LangChain create_agent with four tools). UI is Streamlit with three views: library dashboard, chat assistant, trend analytics. Failures degrade gracefully — Tavily fallback when Semantic Scholar 429s, retry-with-backoff, honest miss messages."*

### 5. *"What's the most interesting thing you learned?"*

> *"That the LLM is the easy part. I spent 80% of the engineering on retrieval quality — chunking strategy, metadata propagation, fuzzy matching, reference filtering in compare mode. The LLM is essentially a commodity now. The differentiation is in how you prepare and present the data to it."*

### 6. *"What would you build next?"*

> *"Two things. First, persistent storage — right now everything's lost on restart. Second, an evaluation harness using RAGAS — quantitative measurements of retrieval recall and answer faithfulness. Without metrics, every tuning decision is vibes-based."*

## Red flags vs senior signals

| ❌ Red flag | ✅ Replace with |
|---|---|
| "I just used LangChain because it's popular" | "LangChain v1 has `create_agent` which gives me native tool calling, exactly what I needed for the MCP integration." |
| "RAG is when you give the model documents" | "RAG retrieves relevant chunks at query time and inserts them into the prompt as grounded context. The retrieval quality, not the LLM, is the dominant factor in answer quality." |
| Long pause when asked a question | "Let me think — I want to break this into three parts." |
| "I don't know" with no follow-up | "I haven't worked with that specifically, but my mental model would be... and I'd validate by..." |

| ✅ Senior signal | What it sounds like |
|---|---|
| Naming a trade-off without being asked | "I went with FAISS, but the trade-off is no persistence — for production I'd switch to Pinecone." |
| Connecting to your project | "I actually hit this exact issue when..." |
| Acknowledging gaps gracefully | "I don't have hands-on experience with reranking, but I understand the mechanism — second-stage cross-encoder over the top-K results." |
| Asking clarifying questions | "Before I answer — are you asking about retrieval quality or response quality?" |

---

<a name="part-3"></a>

# Part 3 — Deep Technical Concepts

## Why 384 dimensions for embeddings?

The 384 is **a property of the model**, not a tunable.

`all-MiniLM-L6-v2` literally means:
- **L6** = 6 transformer layers (BERT-base has 12)
- **Hidden size** = 384 (BERT-base has 768)

The output vector dim = the model's internal hidden size. It's a smaller model **knowledge-distilled** from a larger one to be fast on CPU.

| Model | Dim | Speed | Quality |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ fastest | Good |
| `all-mpnet-base-v2` | 768 | ⚡⚡ | Better |
| `BAAI/bge-large-en-v1.5` | 1024 | ⚡ | Excellent |
| OpenAI `text-embedding-3-small` | 1536 | API | Excellent |
| OpenAI `text-embedding-3-large` | 3072 | API | Best |

**More dimensions buy you:** more expressive (finer semantic distinctions), better retrieval quality at the top of leaderboards.

**More dimensions cost you:**
- Memory (10K chunks × 384 dims × 4 bytes ≈ 15 MB; same chunks at 3072 dims ≈ 120 MB)
- Search latency (FAISS query time scales with dimension)
- The **curse of dimensionality** — in very high-dim spaces (>1000), Euclidean distances start to converge — *everything* looks roughly equidistant
- Diminishing returns. 384 → 768 buys 3-5%; 1024 → 3072 buys <1%

**How to choose:** define a retrieval task → benchmark candidates on Recall@5, MRR, NDCG → pick the smallest model that's "good enough."

## Mean vs Max vs CLS pooling

The transformer outputs **one vector per token**. We need **one vector per document**. Pooling collapses them.

**Mean pooling** — average all token vectors:
```
chunk_vector = (token_1_vec + token_2_vec + ... + token_N_vec) / N
```

**Max pooling** — element-wise maximum across tokens.

**CLS pooling** — use the special `[CLS]` token's vector.

**How is the choice made? It's not — it's locked in at training time.** Sentence-transformers like MiniLM are trained with mean pooling. If you switch at inference, you're using a vector the model wasn't optimized for; quality drops 5-15%.

| | Mean | Max | CLS |
|---|---|---|---|
| What it captures | Average semantic content | Salient features | Whatever model learned |
| Robustness to length | Good | Decent | Length-independent |
| Common in practice | ✅ Sentence-BERT family | Less common | Original BERT classification |

## Why L2 normalize?

A vector `v = [a, b, c]` has length `||v|| = sqrt(a² + b² + c²)`.

L2-normalize = divide by length → unit vector (length 1).

**Why it matters for similarity:**

Cosine similarity:
```
cos(u, v) = (u · v) / (||u|| × ||v||)
```

If both are L2-normalized: `cos(u, v) = u · v` — just a dot product. **No division, no sqrt** — hardware-friendly and ~2× faster.

Other benefits:
- **Stability** — vectors of different magnitudes can swamp similarity scores
- **Geometric interpretation** — all embeddings live on a unit sphere

**Sentence-transformers does this for you.** Output is already L2-normalized.

## What is softmax? How is it used?

**Softmax** = function that turns numbers into probabilities (sum to 1).

```
softmax([2.0, 1.0, 0.1]) = [0.66, 0.24, 0.10]
```

Formula: `softmax(x_i) = exp(x_i) / sum_j(exp(x_j))`

**In our project (we don't write it directly, but it runs millions of times):**

### A. Inside the embedding model (attention)

Every transformer layer uses **scaled dot-product attention**:
```
attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V
```

The softmax converts arbitrary scores into a probability distribution: *"I'll pay 60% attention to token A, 30% to token B, 10% to others."* This is THE central operation of transformers.

### B. Inside the LLM (next-token sampling)

When Groq Llama generates text:
1. Model outputs logit vector (raw scores) over vocabulary
2. **Softmax** converts to probabilities
3. Sample (or argmax with temperature=0)
4. Append the chosen token; repeat

When you set `temperature=0`, you're effectively asking for argmax (always pick the highest-probability token) — but softmax still runs to determine which is highest.

## RecursiveCharacterTextSplitter — how chunks and overlaps are selected

```python
RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

### The algorithm

1. Try to split on the FIRST separator (`\n\n` — paragraph break)
2. For each piece, if it's still > chunk_size, try the NEXT separator (`\n` — line break)
3. Recurse until chunks fit
4. **Merge step**: combine small chunks back together until they're ~chunk_size, with overlap

### Walking a concrete example

Input: 2500-char Methods section with paragraphs.

**Step 1:** Try `\n\n` (paragraph break). Splits into 4 paragraphs:
- P1: 600 chars; P2: 1100 chars (too big!); P3: 400 chars; P4: 400 chars

**Step 2:** Recurse on P2 with `["\n", ". ", " ", ""]`. Try `". "`. Splits into sentences (300 + 400 + 400 chars).

**Step 3 — Merge step (where overlap happens):**
- Chunk 1: P1 alone = 600 chars
- Chunk 2: last 150 chars of chunk 1 (overlap) + S1 (300) + S2 (400) = 850 chars
- Chunk 3: last 150 of chunk 2 + S3 + P3 + P4...

Each chunk has up to `chunk_overlap` characters of the previous chunk at its start.

### Why this is smart

Tries **semantically meaningful boundaries first**. If a paragraph fits, you get a clean paragraph chunk. Only when paragraphs are too big does it descend to weaker boundaries.

### Trade-offs of overlap size

- **More overlap (300/800):** very high boundary safety, lots of redundant text
- **Less overlap (50/800):** efficient, but boundary-spanning answers can be missed
- **No overlap:** ~30% of boundary-spanning answers fail in practice

`150 / 800 ≈ 19%` is a reasonable middle.

---

<a name="part-4"></a>

# Part 4 — Build From Scratch (17-Day Developer Journal)

The way the project would actually have been built: iteratively, each step solving one new problem.

## Day 1: "What am I even modeling?"

> Code is a verb. Data is a noun. Get the nouns right first.

Define `models/schemas.py` with `PaperSection`, `ResearchPaper`, `Citation` Pydantic models. Use `Field(default_factory=list)` for mutable defaults (Python gotcha: shared list across instances).

**Decision logged:** Pydantic over plain `dataclass` because PDF data is messy and runtime validation will catch bugs at the boundary.

## Day 2: "Read a PDF and prove it works"

```python
import fitz  # PyMuPDF
def parse_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "".join(page.get_text() for page in doc)
```

Run on a paper. Get text. Wall of mush — no idea where Abstract starts.

**Mental note:** I need `page.get_text("dict")` to get font sizes and positions if I want to detect the title.

## Day 3: "Show it in the browser"

Minimal Streamlit app: `st.file_uploader` → write to tempfile → parse → display.

**Critical lesson:** at this point the system is useless, but every layer touches every other. Add features by *replacing* parts, not by adding sideways.

## Day 4: "Make it searchable" — embeddings + FAISS

```python
self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_text(text)
docs = [Document(page_content=c, metadata={"paper_id": paper_id}) for c in chunks]
self.vector_store = FAISS.from_documents(docs, self.embeddings)
```

**Bug noticed:** every Streamlit interaction rebuilds the indexer, which re-downloads MiniLM. Catastrophic.

## Day 5: "Make it stop rebuilding" — `st.session_state`

```python
if "indexer" not in st.session_state:
    st.session_state.indexer = Indexer()
```

First run creates the Indexer once. Subsequent runs find it in session_state and skip construction.

**Decision logged:** session_state for per-user state, NOT `@st.cache_resource` (which is multi-session and shares data between users — privacy issue).

## Day 6: "Real Q&A — call the LLM"

```python
self.llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0)
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
CONTEXT: {context}
QUESTION: {question}
""")
chain = prompt | self.llm | StrOutputParser()
return chain.invoke({"context": context, "question": query})
```

The `|` operator composes Runnables. `chain.invoke()` runs the whole pipeline.

## Day 7: "Hallucinations are killing me" — sentinel pattern

User asks for an F1 score. LLM confidently invents one. Fix: add grounding instructions and a sentinel.

```python
FALLBACK_SIGNAL = "I don't have enough information in the provided papers to answer this."
```

Template trick — using `.format()` for one placeholder while preserving others for LangChain:

```python
prompt = ChatPromptTemplate.from_template("""
If not present, respond with exactly: "{fallback_signal}"
CONTEXT: {{context}}
QUESTION: {{question}}
""".format(fallback_signal=FALLBACK_SIGNAL))
```

Double-braced placeholders (`{{context}}`) survive `.format()` and become single-braced for LangChain to fill at chain-invoke time.

## Day 8: "The parser is dumb — actually find sections"

Switch from `page.get_text()` to `page.get_text("dict")`. Now I have font sizes, bold flags, positions.

Build `PageModel`: layout-aware view over the PDF.

```python
@dataclass
class Line:
    text: str; page: int; x0: float; y0: float; size: float; bold: bool
```

Compute the `body_size` (modal font size of long lines). Anything bigger is potentially a heading.

## Day 9: "Score lines as title candidates"

```python
score = 0.35 * size_score + 0.25 * len_score + 0.20 * top_score + 0.20 * clean_score
```

Wrap returns in `Extracted(value, confidence, source)` so callers can decide when to escalate.

## Day 10: "When heuristics fail, ask the LLM"

```python
def llm_refine(fields, first_page, llm, threshold=0.6):
    needs = [k for k, ex in fields.items() if ex.confidence < threshold]
    if not needs: return fields  # cheap path
    # batch all low-confidence fields into one JSON-mode prompt
```

Pattern: **cheap-first, then expensive.** Heuristics free; LLM only when stuck.

## Day 11: "Smart chunking — different content types deserve different splits"

- Short sections (Abstract, Conclusion ≤600 chars) → kept whole
- References → one chunk per `[N]` entry
- Long sections → recursive splitter

## Day 12: "Compare two papers"

Per-paper retrieval. For each paper_id, do a separate `similarity_search` with `filter={"paper_id": pid}`. Drop reference-section chunks. Pin titles in the prompt: *"compare EXACTLY these N papers."*

## Day 13: "Citation graph — does this paper cite that one?"

Naive exact match returns zero. Fix: normalize titles, check substring containment.

```python
def _normalize_title(s):
    s = s.lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()
```

## Day 14: "Tools for the LLM — building the agent"

```python
@tool
def paper_metadata_lookup(query: str) -> str:
    """Look up a research paper's metadata: year, venue, citation count, authors."""
    ...
```

Build agent with `create_agent(llm, tools, system_prompt=...)`. Inject paper title into user message so "this paper" resolves correctly.

## Day 15: "The agent loop in action — a worked example"

Walk the message history: HumanMessage → AIMessage(tool_calls) → ToolMessage(result) → AIMessage(answer).

Pair AIMessage tool_calls with ToolMessage results by `tool_call_id`.

## Day 16: "Resilience — what happens when APIs fail?"

Add retry-with-backoff for 429/5xx. Distinguish transient (retry) from permanent (don't retry) errors.

```python
for attempt in range(retries + 1):
    try:
        resp = requests.get(...)
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None
        ...
```

Add honest-miss messages with embedded LLM instructions.

## Day 17: "Polish — tests, README, docs"

Tests, README, architecture diagram, evaluation document. Cleanup of dead code.

## The meta-lessons

- **Define data first.** The schema is the contract.
- **Build end-to-end at MVP scale, then iterate.**
- **When you add a feature, find what's *wrong* without it.**
- **Keep the cheap path cheap.** Heuristics first, LLM only when stuck.
- **Be honest in your data structures.** `Extracted(value, confidence, source)` lets later code make smart decisions.
- **Test the bugs you fixed.**
- **Document the *why*, not the *what*.**

---

<a name="part-5"></a>

# Part 5 — File-by-File Interview Defense Playbooks

## 5.1 `core/parser.py`

### 30-second elevator pitch

> *"This file converts a raw PDF into a structured `ResearchPaper` object. The challenge is PDFs are messy. So instead of one big extraction function, I built a pipeline: a `PageModel` that gives me layout-aware lines with font and position info, then independent heuristic extractors for each field that score candidates and emit a confidence score, then an LLM-based rescue path that only fires for low-confidence fields. The pattern is **cheap-first, then expensive**."*

### Architecture map

| Region | Responsibility |
|---|---|
| Imports & constants | Known headings, venue tokens, validation thresholds |
| Data structures | `Line`, `Extracted`, `HeadingAnchor` |
| Text utilities | `_normalize`, `_clean_text` |
| `PageModel` | Structure-aware view over PDF |
| Validators | `_validate_title`, `_validate_year`, etc. |
| Heading detection & section split | Find headings, divide content |
| Title / Author / Year / Venue / Abstract extractors | Field-specific |
| Reference extractor | Regex-driven `[N]` splitting |
| LLM refiner | JSON-mode rescue |
| `PDFParser` (main) | Orchestrates everything |

### Key defenses

**Why hardcoded heading lists instead of ML?**
> *"For a class-scale project, ~40 known headings cover 90%+ of academic papers. Training a classifier needs labeled data and maintenance. ML wins when the long tail matters; for academic papers, the head of the distribution covers most cases."*

**Why `@dataclass` for `Line` instead of Pydantic?**
> *"`Line` is hot-path. ~5000 instances per paper. Pydantic adds runtime validation overhead — fine for boundary types, wasteful for internal types that are already trusted."*

**Why `Extracted(value, confidence, source)`?**
> *"Every extractor returns this wrapper. Confidence lets downstream code make conditional decisions: 'if confidence < 0.6, escalate to LLM.' Source is debugging metadata — when something looks wrong, I can see *which path* produced it."*

**Why NFKC normalization?**
> *"NFKC also folds compatibility characters like 'ﬁ' (ligature) into 'fi'. For text mining, that's what we want — we don't care about typography. NFC preserves stylistic forms; NFKC normalizes them away."*

**Why character-weighted average for font size?**
> *"Trade-off. Max would over-weight a single large character. Character-weighted average gives me the line's 'effective' font size dominated by the bulk of its content."*

**Why context-aware year extraction?**
> *"A paper might mention many years. I want the *publication* year. The trick is **context-aware scoring**: I look 40 chars before/after each year. Negative context like 'received' or '©' demotes; positive context like 'Proceedings of...' promotes. The year with the highest score wins."*

### The bug stories to tell

**Story 1 — Author bio bleeding into Conclusion:**
> *"A user asked 'what does this paper conclude?' and got an answer about the author's mailing address and ORCID — because the parser merged the bio block into the conclusion section's content. I traced it to the section split: the last anchor's content extends to end-of-document. I added `_trim_author_bio` that detects bio markers (mailing address, ORCID, e-mail) and trims them off — but only if they appear in the latter 60%, so corresponding-author footnotes on page 1 don't get accidentally cut."*

**Story 2 — Venue banner as title:**
> *"PDFs often have 'Journal of X' at the top of page 1 in a large bold font. Without a venue-banner reject, those win the title contest. Added `_validate_title` to reject anything starting with VENUE_TOKENS regex."*

### Closing line for the section

> *"The thing I'm proudest of in this file is the discipline of confidence scores. By tagging every extracted field with a confidence and a source, I can build downstream features (like the LLM rescue) that respond to uncertainty instead of papering over it. That's the difference between a parser that 'usually works' and one that 'works and tells you when it doesn't.'"*

---

## 5.2 `core/agent.py`

### 30-second elevator pitch

> *"This file wraps a LangChain v1 tool-calling agent. The class exposes one main method — `run(query, vector_store, paper_id, paper_title)` — that builds a fresh agent per call, binds four tools (one for local FAISS search, three MCP tools), and walks the resulting message history to extract the final answer plus a trace of which tools were called. Three design choices to call out: I build the local search tool as a closure so it can capture the vector store and paper_id without polluting the LLM's argument schema; I inject the selected paper's title into the user message so 'this paper' resolves correctly; and I keep RAG and the agent as separate paths in the chat UI rather than making the agent the default."*

### The system prompt structure

```python
AGENT_SYSTEM_PROMPT = """You are a research assistant with access to tools.

Decide which tool fits the user's question:
- local_library_search: questions about uploaded papers (datasets, methods, results).
- paper_metadata_lookup: year, venue, citation count for external papers.
- related_work_discovery: papers related to or cited by a given paper.
- trend_analytics_tool: publication frequency, trends.

Rules:
- Always try local_library_search first when the question references "this paper".
- For metadata about a specific external paper, prefer paper_metadata_lookup over web search.
- When citing local-library results, mention the paper title and section.
- Be concise and grounded. Do not invent results.
- If a tool says metadata is unavailable, tell the user exactly that — do NOT pad
  with generic instructions like "go to Google Scholar". Short, honest answers beat fluff.
"""
```

**Defense:** Tool descriptions belong on the tools (what they do). The system prompt is *how to choose between them and how to behave*. Mixing them leads to bloated tool descriptions or system prompts that drift.

### The closure pattern (tool state)

```python
def _build_local_search_tool(self, vector_store, paper_id):
    search_kwargs = {"k": 5}
    if paper_id:
        search_kwargs["filter"] = {"paper_id": paper_id}

    captured_sources = []  # captured by closure

    @tool
    def local_library_search(query: str) -> str:
        """Search the user's local research library for relevant paper sections."""
        docs = vector_store.similarity_search(query, **search_kwargs)
        captured_sources.extend(d.metadata for d in docs)
        return "\n\n".join(...)

    return local_library_search, captured_sources
```

**Why a closure:** the tool's signature is fixed (`def x(query: str)`). I can't pass `vector_store` as an argument because the LLM only knows to fill `query`. So I capture it via closure.

**Why fresh per call:** `paper_id` and `vector_store` change per query. Building globally would lock in stale state.

**Why `captured_sources` is mutable:** the tool can't return both formatted string (for the LLM) AND source metadata (for the UI). Side-channel via closure.

### Message extraction

```python
@staticmethod
def _extract_tool_calls(messages) -> List[Dict[str, str]]:
    pending: Dict[str, Dict[str, str]] = {}
    ordered: List[str] = []

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id") or f"{tc.get('name')}-{len(ordered)}"
                pending[tc_id] = {"name": ..., "input": ..., "output": ""}
                ordered.append(tc_id)
        elif isinstance(msg, ToolMessage):
            tc_id = getattr(msg, "tool_call_id", None)
            if tc_id and tc_id in pending:
                pending[tc_id]["output"] = str(msg.content)[:500]
    return [pending[tid] for tid in ordered]
```

**Why pair by ID, not position:** an AIMessage can issue multiple parallel tool calls. Pairing by `tool_call_id` is robust to that.

**Why truncate to 500 chars:** for **UI display only**. The LLM gets the full output (uncapped); the user sees a preview.

### Closing line

> *"Agents are a lossy compression of decision-making. Plain RAG is deterministic. Agents add an LLM-driven control flow, which is more flexible but less reproducible. So I treat agents as the *expensive but flexible* path, not as a default replacement for RAG."*

---

## 5.3 `tools/mcp_tools.py`

### 30-second elevator pitch

> *"This file defines the three MCP tools the LangChain agent can call. Each follows the same pattern: try Semantic Scholar first because it returns structured data — citation counts, paper IDs, venue strings; if Semantic Scholar misses or rate-limits, fall through to Tavily web search with a query specifically targeted to find the paper itself; if everything fails, return an honest 'unavailable' message that instructs the LLM not to fabricate."*

### The retry-with-backoff function

```python
def _semantic_scholar_search(title, retries=2):
    params = {"query": title, "limit": 1, "fields": "title,year,venue,citationCount,authors"}
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                return None
            resp.raise_for_status()
            return resp.json().get("data", [None])[0]
        except requests.RequestException:
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None
    return None
```

**Distinguishing transient (429/5xx) from permanent failures:** 4xx like 404 (paper not found) is permanent — retrying is wasteful. 429 (rate limit) and 5xx (server error) are transient.

**Why linear backoff (1.5s, 3s):** simpler than exponential; adequate for 2 retries. Production with more aggressive retries would use exponential + jitter.

**Why return None, not raise:** caller's fallback path handles None cleanly.

### The honest-miss preamble

```python
return (
    "Semantic Scholar had no record of this paper. "
    "Web search results below — these may NOT contain a citation count; "
    "if not present, say so plainly instead of guessing.\n\n" + parsed
)
```

**The tool's return value is part of the LLM's prompt.** When something fails, don't just report — *instruct* the LLM how to handle it.

### Cross-cutting patterns

1. **Structured-source primary, web-source fallback** — Semantic Scholar for facts, Tavily for narrative
2. **Embed instructions in tool returns** — return values are prompts
3. **Distinguish transient vs permanent failures** — only retry on 429/5xx
4. **Independent failure paths within a tool** — each source in its own try/except
5. **Honest-miss preamble** — tell the LLM the data is degraded
6. **Lazy initialization of optional dependencies** — `_get_tavily()` is a function, not a module-level import

---

## 5.4 `core/rag_pipeline.py`

### 30-second elevator pitch

> *"This file owns the non-agent retrieval path. One class — `ResearchRAG` — exposes three methods: summarize one paper, ask a question, and compare multiple papers. All three use Groq Llama 3.3-70B at temperature 0 for grounded determinism, all three include a FALLBACK_SIGNAL contract for graceful 'I don't know' handling. The interesting design choice is that compare_papers diverges from the standard RAG shape — instead of one similarity_search, it loops per-paper, drops reference-section chunks, and pins the comparison set in the prompt."*

### `_format_docs` — small but loaded

```python
@staticmethod
def _format_docs(docs) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('title', 'Unknown')} "
        f"— {doc.metadata.get('section_name', 'Section')}]\n{doc.page_content}"
        for doc in docs
    )
```

**Why prefix each chunk with `[Title — Section]`:** the LLM needs to know which paper and section each chunk came from, so it can cite sources properly and not conflate content across papers.

### The canonical RAG chain (`ask_question`)

```python
rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: self._format_docs(retriever.invoke(x["question"]))
    )
    | qa_prompt | self.llm | StrOutputParser()
)
answer = rag_chain.invoke({"question": query})
docs   = retriever.invoke(query)  # second invoke for source metadata
```

**The four stages:**
1. `RunnablePassthrough.assign` — input is `{question}`, output is `{question, context}`
2. `qa_prompt` — fills the template
3. `self.llm` — sends to Groq
4. `StrOutputParser` — extracts `.content` string

**The double-invoke pattern:** I retrieve twice — once inside the chain (for context), once outside (for source metadata to display). It's wasteful (~50ms duplicate work). Clean fix is a side-channel via closure, like in the agent's `local_library_search`.

### `compare_papers` — the contrarian method

Three innovations over standard RAG:

**1. Per-paper similarity_search:**
```python
for pid in paper_ids:
    docs = vector_store.similarity_search(query, k=6, filter={"paper_id": pid})
```
Forces equal representation across papers.

**2. Reference-section filtering:**
```python
docs = [d for d in docs if not any(
    w in d.metadata.get("section_name", "").lower()
    for w in ("reference", "bibliograph", "works cited")
)][:4]
```
References to other papers were confusing the LLM into adding extra rows to comparison tables.

**3. Title-pinning in the prompt:**
> *"You are an experienced research analyst comparing EXACTLY these {n_papers} papers and no others: {titles_block}"*

**Defense in depth** — even with reference filtering, the LLM occasionally drifted; explicit instruction prevents it.

### Closing line

> *"What I find most interesting about this file is that it shows the limits of plain RAG honestly. `ask_question` is 50 lines. `compare_papers` is 90 lines because compare is genuinely harder — you can't just 'do RAG harder.' You need new strategies: per-paper retrieval, reference filtering, prompt pinning. Once I built it, I realized I'd graduated from 'I'm using RAG' to 'I'm engineering retrieval strategies.'"*

---

<a name="part-6"></a>

# Part 6 — How MCP Tools Actually Work

## How an LLM "sees" a tool

When you bind tools to an LLM and invoke it, here's what gets sent to Groq:

```json
{
  "model": "llama-3.3-70b-versatile",
  "messages": [
    {"role": "system", "content": "You are a research assistant..."},
    {"role": "user", "content": "What's the citation count of the Attention paper?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "paper_metadata_lookup",
        "description": "Look up a research paper's metadata: year, venue, citation count, authors.",
        "parameters": {
          "type": "object",
          "properties": {"query": {"type": "string"}},
          "required": ["query"]
        }
      }
    }
  ]
}
```

The `tools` array tells the model what's available. The model considers the user's question alongside this list. If the question matches a tool's description well, it generates a **structured tool call** instead of regular text.

## What the LLM emits

Instead of normal text:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "id": "call_abc123",
    "type": "function",
    "function": {
      "name": "paper_metadata_lookup",
      "arguments": "{\"query\": \"Attention Is All You Need\"}"
    }
  }]
}
```

Three things to note:
1. **`content` is null** — no free text
2. **`tool_calls` is structured** — name, arguments (JSON string), unique `id`
3. **The `id` matters** — when you return the tool's result later, you reference this id

The LLM was **fine-tuned during training** to emit this format. **Smaller models (Llama-3.1-8B) often fail** — they emit tool calls as raw text inside `content`. That's why we couldn't fall back to 8B for the agent path.

## The runtime loop

```python
response = llm.invoke(messages, tools=tools)

if response.tool_calls:
    for tc in response.tool_calls:
        tool_func = name_to_function[tc.name]
        result = tool_func(**tc.args)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(result),
        })
    response = llm.invoke(messages, tools=tools)

return response.content
```

## What `@tool` decorator does

When you write:
```python
@tool
def paper_metadata_lookup(query: str) -> str:
    """Look up a research paper's metadata."""
    ...
```

The decorator wraps your function in a LangChain `Tool` that:
- Reads function name → tool name
- Reads docstring → description (this is what the LLM sees)
- Reads type hints → JSON schema for arguments
- Adds `invoke()` method conforming to the protocol

**Three implications:**
1. **The docstring IS the prompt for that tool.** Vague docstring → wrong tool selection.
2. **Type hints become the contract.** Validation before your function runs.
3. **Multiple decorators stack.** You can combine `@tool` with `@retry` or `@cache`.

## Tool design checklist

| ✅ Good | ❌ Bad |
|---|---|
| Description names *when* to use it | Description names *how* it works internally |
| Argument names suggest semantic role | Generic names (`x`, `data`, `q`) |
| Returns a string the LLM can read directly | Returns raw JSON / Python objects |
| On failure, returns explanatory text | On failure, raises exceptions |
| Single clear responsibility | Mega-tools that do many things |
| Idempotent (safe to call again) | Has side effects that compound |

---

<a name="part-7"></a>

# Part 7 — Cross-cutting Patterns & Design Philosophy

## The unifying principle

> **Make the cheap, predictable path the default; have explicit escape hatches for failure.**

- Heuristic parsing first; LLM only when stuck
- Mean pooling because the model trained that way
- L2-normalize at index time because cosine = dot product is faster
- Plain RAG as default; agent only when the toggle is on
- Semantic Scholar primary; Tavily fallback; honest failure last

## The patterns to name in interviews

### Pattern 1: Cheap-first, then expensive
> *"Heuristics free; LLM only when heuristics fail. Keeps cost bounded."*

### Pattern 2: Confidence-tagged returns
> *"Every extractor returns `(value, confidence, source)`. Lets downstream code make smart decisions."*

### Pattern 3: Validate at every boundary
> *"PDF metadata, heuristic outputs, LLM outputs — all run through the same validators."*

### Pattern 4: Independent extractors over a shared model
> *"Each field has its own extractor. They don't depend on each other's *output*. Failure isolation."*

### Pattern 5: Voting / aggregation across noisy signals
> *"`_detect_layout` votes across 5 pages. `_score_title` aggregates four features. When any signal is noisy, aggregate."*

### Pattern 6: Soft degradation
> *"Layout scanned → return stub. References can't be parsed → store raw text. LLM rescue fails → keep low-confidence fields. Always return *something* useful."*

### Pattern 7: Closures for tool state
> *"When a tool needs config the LLM doesn't know about, capture via closure. Don't make it a tool argument; don't make it global."*

### Pattern 8: Embed instructions in tool returns
> *"The tool's return value is part of the LLM's prompt. When something fails, *instruct* the LLM how to handle it."*

### Pattern 9: Pair-by-ID, not position
> *"For any structured async/parallel pattern, pair by stable identifiers, not array indices."*

### Pattern 10: Honest-miss preamble
> *"When you fall through to a degraded source, *tell the LLM about it* in the response."*

---

<a name="part-8"></a>

# Part 8 — Common Interview Questions with Model Answers

## RAG fundamentals

**Q: What's RAG?**
> *"A pattern where instead of asking the LLM to recall an answer — risking hallucination — you retrieve relevant text from your data and feed it into the prompt as context. The LLM's job becomes 'read this and answer,' which is much more reliable than 'remember and answer.' In my project, I FAISS-retrieve the top-5 most semantically similar chunks from the relevant paper and stuff them into the prompt with grounding instructions. The trade-off is retrieval quality becomes the bottleneck — bad chunks → bad answers, regardless of LLM."*

**Q: RAG vs fine-tuning?**
> *"RAG when knowledge is **dynamic** (docs change, you add new papers daily) or **per-user** (each tenant has different data). Fine-tuning when knowledge is **stable** and you need the model to *behave* differently (style, format). They're complementary."*

**Q: What's the biggest source of bad RAG answers?**
> *"Almost always retrieval, not the LLM. If the right chunk doesn't make it into context, no amount of clever prompting helps. Investing in better chunking, hybrid search (BM25 + vector), and reranking pays more than upgrading the LLM."*

**Q: What's hybrid search?**
> *"Combining lexical (BM25, keyword matching) and semantic (vector) retrieval. Vector search nails *concepts* but misses rare terms. BM25 nails exact tokens but misses paraphrasing. Production systems run both, fuse with RRF (Reciprocal Rank Fusion), and rerank."*

## Vector store

**Q: Why FAISS instead of Pinecone?**
> *"For my project, scale is small (a few hundred chunks per session) and FAISS is in-process — zero infra. For >100K vectors with concurrent users, you want a persistent vector DB. FAISS is a *library*, not a *database* — that distinction matters."*

**Q: Cosine vs Euclidean vs dot product?**
> *"With L2-normalized vectors (which sentence-transformers produce by default), cosine similarity = dot product = a monotonic function of Euclidean distance. The choice doesn't matter when normalized. With unnormalized vectors, cosine is robust to magnitude (good for text); Euclidean isn't."*

**Q: How would you handle 10 million documents?**
> *"Switch FAISS index type from flat (exhaustive) to IVF or HNSW (approximate). Add chunk metadata into the DB for filtered search. Probably move to a real vector DB at this scale. Pre-filter aggressively (by date, paper_id) before vector search."*

## Chunking & embeddings

**Q: How do you choose chunk size?**
> *"Trade-off between context preservation and retrieval precision. Small chunks → precise hits but lose surrounding context. Large chunks → preserve context but waste embedding capacity. 200–800 tokens is the typical sweet spot for prose. Test it empirically with a held-out QA set."*

**Q: What if a single answer needs information from 5 chunks?**
> *"That's the *multi-hop retrieval* problem. Naive top-K won't work. Solutions: increase K (often enough), add a reranker, use parent-document retrieval, or use an agent that does multiple retrieval calls."*

**Q: How would you update embeddings when the model changes?**
> *"Re-embed everything. Embeddings are **not portable** across models. Some teams maintain side-by-side indexes during migrations."*

## LLM behavior

**Q: How do you reduce hallucinations in RAG?**
> *"Four levers in priority order: (1) *better retrieval* (right chunk in context), (2) *grounding instructions* in the prompt, (3) *temperature 0* for deterministic outputs, (4) *citation requirements*. The biggest one is retrieval, by a large margin."*

**Q: What's prompt injection? How to defend?**
> *"Untrusted text in context tries to override instructions. Defense layers: (1) sanitize inputs, (2) keep system instructions in system message (clearer trust boundary), (3) for tool-calling agents, never let user text become a tool argument without validation, (4) output filtering."*

**Q: Why temperature 0?**
> *"Determinism. For RAG, you want the same context to produce the same answer every time. Temperature > 0 introduces randomness — useful for creative writing, terrible for retrieval-grounded Q&A."*

## Agents & tool calling

**Q: RAG vs agent — when do you use which?**
> *"RAG when you know the data lives in your library and the question maps to one search. Agent when the question might require *multiple steps* or *external tools* the LLM should choose between."*

**Q: How does the LLM "decide" which tool to call?**
> *"It doesn't decide *intelligently* — it generates structured JSON output that the runtime parses. Modern LLMs have been fine-tuned to emit a special tool-call format when the prompt advertises tools. The decision is 'does this query match any tool description well enough that the model autocompletes a tool call?' Small models often fail."*

**Q: Cost difference between RAG and agents?**
> *"Agents are 3–10× more expensive per query. Each tool call is another round-trip: prompt → tool_call → execute → re-prompt → final_answer. For high-volume systems, route only tool-shaped queries to the agent."*

## Evaluation

**Q: How do you evaluate a RAG system?**
> *"Three dimensions: (1) **retrieval** — Recall@K, MRR (does the right chunk appear in top-K?), (2) **faithfulness** — does the answer follow from the retrieved context? (3) **answer quality** — does it actually answer the question? Tools like RAGAS automate (2) and (3) using an LLM-as-judge."*

**Q: What metrics matter in production?**
> *"Latency (P50, P95, P99), token spend per query, retrieval recall@K, user feedback (thumbs-up/down). The ML-research metrics (BLEU, ROUGE) are mostly useless for chat-style RAG."*

## System design

**Q: Design a multi-tenant RAG system.**
> *"Per-tenant namespace in the vector store. Per-tenant API keys for LLM. Strict authorization on document upload. Audit logs for compliance. Caching by `(tenant_id, query_hash)`. Same retrieval/LLM stack, just isolated."*

**Q: How would you handle 1000 QPS?**
> *"Cache aggressively. RAG answers are highly cacheable. Tiered cache: in-memory (per-pod) → Redis (shared) → recompute. Streaming responses. Vector search at this rate needs a managed DB and probably IVF/HNSW indexes."*

**Q: What breaks first as the system scales?**
> *"Usually the LLM rate limit before anything else. Then memory (vector store grows). Then indexing throughput. Embedding throughput rarely bottlenecks unless API-based."*

## Project-specific

**Q: Walk me through what happens when a user asks 'compare paper A and B'.**
> *"Mode = Compare → `compare_papers()` in rag_pipeline.py. For each paper_id, run a separate `vector_store.similarity_search(query, k=6, filter={paper_id})`. Drop chunks from References sections. Take top 3 per paper. Concatenate into context with title labels. Build a prompt that pins the title set ('compare EXACTLY these N papers, no others'). Single LLM call. Return answer + sources."*

**Q: Why fuzzy matching for the citation graph instead of exact?**
> *"Reference strings look like *'[1] Vaswani A. Attention Is All You Need 2017'* — they never exactly equal a stored title. Exact equality returned 0 cross-paper edges. Normalize both sides (lowercase, strip punctuation), then check substring containment. Length guard rejects matches on titles under 8 chars."*

**Q: Why is the agent disabled in Compare mode?**
> *"Two reasons: (1) the agent picks tools dynamically and might decide it doesn't need to query one of the papers — that breaks the comparison guarantee. (2) Compare mode requires per-paper retrieval which is hard to express as a single tool call."*

---

<a name="part-9"></a>

# Part 9 — Project Audit & Critical Analysis

## Spec scorecard against [GA02 PDF]

| Task | Status | Notes |
|---|---|---|
| 1. Schema | ✅ | Pydantic 2; `Citation` populated |
| 2. Section-level PDF parsing | ✅✅ | Layout-aware, scoring-based, LLM-refined — above expectation |
| 3. Metadata enrichment | ✅ | Title/authors/year/venue/keywords/abstract |
| 4. Intelligent chunking | ✅✅ | Content-aware: short = whole, references = per-entry, long = recursive |
| 5. FAISS vector store | ✅ | With metadata on every chunk |
| 6. Semantic paper discovery | ⚠️ | Works via similarity, but no dedicated discovery UI |
| 7. Auto summarization | ✅ | Short + Structured, on-demand button, cached |
| 8. RAG Q&A | ✅ | Single-paper + library, k=5, source attribution |
| 9. Cross-paper compare | ✅✅ | Per-paper retrieval with reference filtering |
| 10. Citation graph | ✅ | Fuzzy match + auto-populated Citation objects |
| 11. MCP tool integration | ✅✅ | Real `create_agent`, three tools, fallback chain — above expectation |
| 12. Keyword/topic aggregation | ✅ | Year/venue/category breakdowns |
| 13. Emerging trend identification | ⚠️ | "Newest papers" filter only — no rapidly-increasing-keywords |
| 14. Library dashboard | ✅ | With reading-status filters |
| 15. Interactive paper viewer | ✅ | Abstract + AI Summary + References |
| 16. Chat assistant | ✅✅ | 3 modes, source expander, agent expander |
| 17. Citation/trend visualization | ⚠️ | Bar charts ✅ but citation network is tables-only |
| 18. Scenario-based evaluation | ✅ | EVALUATION.md |
| 19. Quality assessment | ✅ | EVALUATION.md |
| 20. Final deliverables (README, arch diagram) | ✅ | docs/architecture.md, README.md |

**Score: ~17/20 fully done, 3 partial, 0 missing → ~85% rubric coverage with extra credit on Tasks 2/4/9/11.**

## Strengths (above-standard)

1. **PDF parser is impressive.** ~810 lines of careful layout detection, font-scoring extractors, and LLM-refined low-confidence fields.
2. **Agent integration is real.** Not a fake "tool description in prompt" — actual `create_agent` with native tool-calling.
3. **Compare-mode prompt design.** Pinning the title set prevents reference-bleed.
4. **Graceful degradation.** Tavily fallback, 429-retry, honest-miss messages, NaN-safe Year sync, fuzzy-match recovery.
5. **Citation graph fuzzy matching.** Solves a real bug elegantly.

## Weaknesses (visible to a reviewer)

1. **No quantitative evaluation.** Qualitative observations only.
2. **Emerging-trends is shallow.** Just newest-papers filter.
3. **Parser is monolithic.** 810 lines in one file.
4. **Hardcoded model name.** Should be a config constant.
5. **No fallback when Groq is rate-limited.** Need graceful "switch to 8b" path.

## Production-readiness

| Aspect | Verdict |
|---|---|
| Error handling | 🟡 Okay |
| Configuration | 🔴 Hardcoded model names |
| Persistence | 🔴 Everything lost on restart |
| Concurrency | 🔴 Single-user Streamlit |
| Logging | 🟡 `print()` statements |
| Tests | ✅ 41 regression tests |
| Security | 🟡 `dangerous_deserialization=True`; no input validation on PDFs |
| Cost control | 🔴 LLM calls on every chat message; no answer caching |

This is a **prototype-quality** project, appropriate for a class assignment. Not production. Don't claim otherwise on a resume.

## Top 5 highest-leverage improvements

1. **Persistence** — `save_index` + `load_index`, JSON-serialize `paper_store`
2. **Configuration** — externalize model name and hyperparameters into `config.py`
3. **Logging** — replace `print()` with structured `logging`
4. **Caching** — RAG answers cached by `(query, paper_id)` in session state
5. **Quantitative evaluation harness** — RAGAS over a curated ground-truth set

---

<a name="part-10"></a>

# Part 10 — Final Cheat Sheets

## Cheat sheet 1: The 10 layers of any GenAI app

| Layer | Concept | This project |
|---|---|---|
| 1. Ingest | Parse / clean / extract structure | `parser.py` |
| 2. Chunk | Split into semantically coherent units | `indexer.py` |
| 3. Embed | Map to vector space | MiniLM in `indexer.py` |
| 4. Index | Store vectors in queryable DB | FAISS in `indexer.py` |
| 5. Retrieve | Top-K nearest, optionally filtered | `as_retriever()` |
| 6. Prompt | Compose context + question + grounding | `rag_pipeline.py` |
| 7. Generate | LLM call | Groq via LangChain |
| 8. Orchestrate | Chain or agent | `rag_pipeline.py` vs `agent.py` |
| 9. Fallback | What happens when stages fail | `mcp_tools.py` retry; sentinel |
| 10. Observe | Surface tool calls, sources | Sources/tool-calls expanders |

## Cheat sheet 2: Every meaningful trade-off in this project

| Choice | Why this | Alternative rejected |
|---|---|---|
| Pydantic | Runtime validation on PDF data | dataclass — wanted boundary safety |
| PyMuPDF | Layout-aware (font + position) | PyPDF2 — can't get fonts |
| MiniLM (local) | Free, fast, no API dep | OpenAI — cost + lock-in |
| FAISS | Zero-infra in-process | Pinecone — overkill for prototype |
| LangChain v1 | Native `create_agent` | Build agent manually — reinventing |
| Groq Llama 3.3 | Fast, free tier, native tool calling | OpenAI GPT-4 — cost |
| Streamlit | Fastest to UI | Gradio / FastAPI+React |
| Tavily | First-class LangChain integration | SerpAPI / Brave |
| Semantic Scholar | Free, structured academic metadata | Google Scholar — no public API |
| Temperature 0 | Determinism, grounding | Higher temp — randomness |
| k=5 retrieval | Balance: enough context, not diluted | Lower → miss; higher → noise |
| 800-char chunks | Fits paragraph density | Smaller → shatter; bigger → coarse |
| 150 overlap (~19%) | Boundary safety | More → redundancy; less → loss |
| Mean pooling | Model trained with it | Max/CLS — not what model expects |
| L2 normalize | Cosine = dot product, faster | No norm — magnitude swamping |
| Heuristic-first parser | Cheap, debuggable | ML model — needs labeled data |
| Confidence scoring | Lets downstream decide escalation | Boolean success — loses nuance |
| Per-call agent build | `paper_id` changes per query | Cached agent — stale state |
| Closures for tool state | LLM-invisible config | Globals — breaks isolation |
| FALLBACK_SIGNAL | Predictable detection | Fuzzy match — false positives |
| Compare: drop refs | LLM was citing other papers | Keep all chunks — drift |
| Compare: pin titles | Belt-and-suspenders defense | Just retrieval filter — drift |
| Fuzzy citation match | Real refs differ from titles | Exact match — 0 hits |
| SHA-1 paper_id | Content-addressed dedupe | filename — collisions |
| Retry on 429/5xx only | Permanent errors don't benefit | Retry-all — wastes budget |
| Linear backoff | Simple; 2 retries enough | Exponential — overkill at scale |

## Cheat sheet 3: The interview close lines

For each file, your closing pitch:

**parser.py:**
> *"The thing I'm proudest of in this file is the discipline of confidence scores. By tagging every extracted field with a confidence and a source, I can build downstream features (like the LLM rescue) that respond to uncertainty instead of papering over it. That's the difference between a parser that 'usually works' and one that 'works and tells you when it doesn't.'"*

**agent.py:**
> *"What I find most interesting about agents architecturally is that they're a lossy compression of decision-making. Plain RAG is deterministic. Agents add an LLM-driven control flow — more flexible but less reproducible. So I treat agents as the *expensive but flexible* path, not as a default replacement for RAG."*

**mcp_tools.py:**
> *"What I find most interesting about this file is how much of it is *prompt engineering* disguised as code. The docstrings are the LLM's tool descriptions. The return-value preambles are LLM behavioral instructions. Every string is a prompt, every error message is a behavior contract."*

**rag_pipeline.py:**
> *"This file shows the limits of plain RAG honestly. `ask_question` is 50 lines. `compare_papers` is 90 lines because compare is genuinely harder — you can't just 'do RAG harder.' You need new strategies. That mindset shift — that retrieval is a design space with real trade-offs, not a single library call — is what I learned most from this project."*

## Cheat sheet 4: Two-week prep plan

| Day | Focus |
|---|---|
| 1 | Read all 10 concepts in Part 1. Write a 60-sec explanation for each. |
| 2 | Re-read your own code: `parser.py`, `indexer.py`, `rag_pipeline.py`. |
| 3 | Same for `agent.py`, `mcp_tools.py`. Trace data flow on paper. |
| 4 | Write out your STAR-T project answer. Time: under 90 seconds. |
| 5 | Practice the six "tell me about a time" questions out loud. |
| 6 | Drill the technical questions in Part 8 — out loud, no notes. |
| 7 | Review and rest. |
| 8 | Mock interview. Get blunt feedback. |
| 9 | Identify weakest answers. Re-drill those. |
| 10 | Read 2–3 papers/blogs on what you're shaky on. |
| 11 | Mock interview round 2. |
| 12 | Review hesitant areas; drill those. |
| 13 | Final mock. Time everything. |
| 14 | Light review only. Sleep. |

## Cheat sheet 5: Three closing pieces of advice

1. **Be honest about scope.** Don't oversell. *"A class assignment that I extended into a portfolio piece"* is fine.
2. **Have one strong opinion about Gen AI.** Mine: *"The LLM is the easy part — engineering effort should go into retrieval quality and evaluation infrastructure, not into prompt tweaking."*
3. **Practice talking, not reading.** Knowledge in your head sounds different when it leaves your mouth. Record yourself. Iterate.

---

# Appendix A — Three Mermaid Diagrams

## A.1 The full system

```
PDF upload
  → PDFParser (PyMuPDF + heuristics + LLM refine)
  → ResearchIndexer (HuggingFace embeddings + FAISS)
  → ResearchPaper objects in session state

Chat query
  → ResearchRAG (plain LangChain) ─OR─ ResearchAgent (tool-calling)
  → Groq Llama 3.3-70B
  → answer + sources + tool-call trace
```

## A.2 The agent loop

```
User message → LLM
                 ↓
            tool_calls?
              /     \
            yes      no
            /         \
       execute        return
       tool             content
       ↓               (final answer)
       ToolMessage
       ↓
       LLM (loop)
```

## A.3 PDF parsing pipeline

```
PDF bytes
  ↓
fitz.open()
  ↓
PageModel:
  _build_lines (text + font + position)
  _detect_layout (single/two-column/scanned)
  _body_size (modal font)
  _repeating_lines (header/footer)
  ↓
extract_title, extract_authors, extract_year, extract_venue
  ↓
_detect_anchors + _split_sections + _trim_author_bio
  ↓
extract_abstract, extract_keywords, extract_references
  ↓
llm_refine (only if confidence < 0.6)
  ↓
ResearchPaper
```

---

# Appendix B — Quick-Reference Concept Glossary

| Term | One-line definition |
|---|---|
| **LLM** | Model that predicts the next token given previous tokens |
| **Token** | Subword piece (model's atomic unit) |
| **Context window** | Max tokens the LLM can process in one call |
| **Embedding** | High-dimensional vector representing semantic meaning |
| **Cosine similarity** | Angle-based similarity metric, equals dot product when L2-normalized |
| **L2 normalization** | Dividing a vector by its length to make it unit-length |
| **Mean pooling** | Averaging token vectors to get a single document vector |
| **Softmax** | Converts arbitrary numbers to probabilities (sum to 1) |
| **Self-attention** | Mechanism letting each token "look at" every other token |
| **Transformer** | Architecture stacked from self-attention layers |
| **Chunking** | Breaking documents into smaller pieces for embedding |
| **Overlap** | Shared characters between adjacent chunks for boundary safety |
| **FAISS** | Facebook's nearest-neighbor library |
| **Vector store** | Database optimized for similarity search on vectors |
| **RAG** | Retrieval-Augmented Generation: retrieve context, then generate |
| **Retriever** | Component that fetches top-K relevant documents for a query |
| **Hybrid search** | Combining lexical (BM25) + semantic (vector) retrieval |
| **Reranking** | Second-stage scoring to refine top-K results |
| **Hallucination** | LLM-generated content that's plausible but factually wrong |
| **Grounding** | Forcing LLM answers to cite/use provided context |
| **Sentinel** | Exact phrase the LLM emits as a control signal |
| **Prompt engineering** | Designing prompts to reliably elicit desired outputs |
| **Temperature** | LLM sampling randomness; 0 = deterministic, higher = creative |
| **Few-shot** | Providing examples in the prompt before the actual query |
| **Chain-of-thought** | Asking the LLM to reason step-by-step |
| **Agent** | LLM that can call tools and decide what to do next |
| **Tool calling** | LLM emitting structured JSON to invoke external functions |
| **Function calling** | OpenAI's older name for tool calling |
| **MCP** | Model Context Protocol — Anthropic's spec for tool servers |
| **Streaming** | Token-by-token output as the LLM generates |
| **Fine-tuning** | Updating model weights with new data |
| **Knowledge distillation** | Training a smaller "student" model from a larger "teacher" |
| **Pydantic** | Python library for typed data validation |
| **Closure** | Function capturing variables from its enclosing scope |
| **Idempotent** | Safe to call multiple times with same effect |
| **Backoff** | Increasing delay between retries |
| **Jitter** | Random variation added to backoff to prevent thundering herd |
| **Circuit breaker** | After N failures, stop trying for a cool-down period |

---

*End of study guide.*
