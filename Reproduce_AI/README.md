# ReproduceAI

**AI-powered Research Paper Analysis Assistant — now with Corrective RAG**

ReproduceAI helps students, researchers, and machine learning engineers understand research papers and generate a structured roadmap for reproducing them — through a conversational, ChatGPT-style interface with built-in analysis tools.

Instead of functioning as another "Chat with PDF" application, ReproduceAI focuses on extracting important implementation details from research papers, organizing them into actionable outputs, and letting you explore the paper conversationally — and, unlike plain RAG, it self-corrects when the retriever comes back empty-handed.

## Screenshots

**Upload & Analysis**

![Upload and analyze a paper](images/crag1.jpeg)

**CRAG in action — answering from a corrected retrieval**

![CRAG evaluating and correcting retrieval](images/crag2.jpeg)

## Why ReproduceAI?

Reading research papers is only the first step.

Actually reproducing a paper requires identifying:

- datasets
- architectures
- optimizers
- learning rates
- batch sizes
- evaluation metrics
- implementation details

These details are often scattered throughout the paper, making reproduction time-consuming — and standard RAG has no way of knowing when its own retrieval has failed to surface them.

ReproduceAI automates this process by combining **Corrective Retrieval-Augmented Generation (CRAG)** with structured metadata extraction, then presents everything through a single conversation — one-click tool buttons for the structured outputs, and free-form chat for anything else.

## Why Corrective RAG, not plain RAG?

Most RAG apps blindly trust whatever the retriever returns. If the retriever pulls irrelevant chunks — which happens constantly with dense, appendix-heavy research papers — the LLM either hallucinates or confidently answers from the wrong context.

ReproduceAI's chat layer adds a **retrieval evaluator** in front of the generator, based on the CRAG paper (Yan et al., 2024). Every retrieval is scored and routed into one of three actions:

| Action | What happens |
|---|---|
| ✅ **Correct** | Retrieved chunks are individually re-scored, irrelevant ones are filtered out, and only the useful strips are passed to the generator. |
| ⚠️ **Ambiguous** | The retrieved chunks are combined with a live web search (scoped to the paper's title) so the answer draws on both sources. |
| ❌ **Incorrect** | The retrieved chunks are discarded entirely and a web search is used instead, so the model isn't forced to answer from irrelevant context. |

This means ReproduceAI can still give a grounded answer even when a detail is buried deep in an appendix or wasn't captured well by the vector store — instead of silently making something up.

## Features

Upload a research paper and start a conversation with it:

| Feature | Description |
|---|---|
| 📄 Paper Overview | Title, authors, conference, publication year, research domain, and a one-line description |
| 📊 Structured Metadata | Dataset, architecture, optimizer, learning rate, batch size, epochs, loss function, hardware, metrics — extracted field-by-field, validated with Pydantic, with page citations |
| 📝 Paper Summary | A concise summary, key bullet points, and the main contribution |
| 🛠 Implementation Plan | Step-by-step implementation roadmap, required libraries, project structure, training and evaluation workflow |
| ⚠ Risk Analysis | Missing implementation details, reproduction challenges, and recommendations before implementation |
| 💬 Ask Anything (Corrective RAG) | Free-form Q&A about the paper — the retrieval evaluator checks context quality first, then answers from refined paper context, a live web search, or both |

All five structured outputs are generated once during analysis and cached — clicking a tool button just displays the cached result instantly, with no additional API call. Only free-form chat questions call the LLM after the initial analysis, and each one runs through the CRAG evaluator before answering.

## How It Works

```
Upload Paper
     │
     ▼
Pipeline runs once
     │
     ▼
Cached report: { overview, metadata, summary, plan, risk }
     │
     ▼
Conversation begins
     │
     ├── Click a tool button → display cached result (no LLM call)
     │
     └── Type a question → Retrieve → Evaluate → Correct/Refine or Web Search → Answer
```

This keeps the app fast and minimizes API usage: a single analysis pass generates all five structured outputs up front, and the chat box is the only feature that makes further LLM calls — each one routed through the corrective layer.

## Architecture

```
                    Research Paper (PDF)
                              │
                              ▼
                     PyMuPDFLoader
                              │
                              ▼
            RecursiveCharacterTextSplitter
                              │
                              ▼
             HuggingFace Embedding Model
         (sentence-transformers/all-MiniLM-L6-v2)
                              │
                              ▼
                         ChromaDB
                              │
                              ▼
                         Retriever
                              │
        ┌──────────┬──────────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼          ▼
    Overview   Metadata    Summary     Planner      Risk
                              │
                              ▼
                      Conversation Layer
                              │
                              ▼
                Free-form Chat  ──▶  Retrieval Evaluator
                                          │
                        ┌─────────────────┼─────────────────┐
                        ▼                 ▼                 ▼
                    Correct           Ambiguous          Incorrect
                        │                 │                 │
                  Refine chunks    Refine + Web       Web Search only
                  (filter noise)      Search          (paper title-scoped)
                        │                 │                 │
                        └────────┬────────┴────────┬────────┘
                                 ▼                 
                              Generator → Answer
```

The retriever and LLM instance are reused across the entire session — the same vector store powers the structured tools and the chat feature. The only addition for chat is the evaluator sitting between retrieval and generation.

## Design Philosophy

Instead of using the traditional approach

```
Paper
   │
   ▼
Large Language Model
   │
   ▼
Everything
```

ReproduceAI follows a metadata-driven, self-correcting RAG pipeline.

```
Paper
   │
   ▼
Retriever
   │
   ▼
Retrieval Evaluator ──▶ Correct / Ambiguous / Incorrect
   │
   ▼
Structured Metadata / Refined Knowledge
   │
   ▼
Planner
   │
   ▼
Risk Analysis
```

This approach

- reduces hallucinations
- improves scalability
- keeps responses grounded in retrieved context — and knows when that context isn't good enough
- falls back to live web knowledge instead of guessing, when the paper genuinely doesn't answer the question
- enables downstream modules (planner, risk analysis, chat) to reuse structured metadata and the same retriever

## Tech Stack

**Programming Language**
Python

**Frameworks**
- LangChain
- Streamlit (chat-based UI via `st.chat_message` / `st.chat_input`)

**Large Language Model**
Groq — `openai/gpt-oss-20b`

**Web Search (CRAG fallback)**
Tavily

**Embeddings**
`sentence-transformers/all-MiniLM-L6-v2`

**Vector Database**
ChromaDB

**PDF Processing**
PyMuPDF

**Validation**
Pydantic

## Project Structure

```
ReproduceAI/
│
├── app.py
├── main.py
├── core.py
├── features.py
├── prompts.py
│
├── notebooks/
│   ├── reproduceai_v0.ipynb
│   └── README.md
│
├── data/
│   └── paper.pdf
│
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

## File Description

### `app.py`
Streamlit chat interface responsible for
- file upload and triggering analysis
- rendering the conversation (`st.session_state.messages`)
- tool buttons that inject cached analysis results into the conversation
- a chat input box that routes free-form questions through Corrective RAG

### `main.py`
Sequential pipeline that runs the full analysis once and hands back the retriever and LLM for reuse by chat.

```
Load PDF
   ↓
Split Documents
   ↓
Create Vector Store
   ↓
Initialize Retriever
   ↓
Generate Overview
   ↓
Extract Metadata
   ↓
Generate Summary
   ↓
Generate Implementation Plan
   ↓
Generate Risk Analysis
   ↓
Return report, retriever, llm
```

### `core.py`
Infrastructure layer responsible for
- PDF loading
- document chunking
- embedding generation
- vector database creation (isolated per-paper collection)
- retriever creation
- LLM initialization
- Tavily client initialization (used by CRAG's web-search fallback)

### `prompts.py`
Contains all prompt templates used throughout the application, including the chat prompt used for free-form Q&A and the retrieval evaluator prompt that scores retrieved chunks as Correct, Ambiguous, or Incorrect.

Keeping prompts separate makes experimentation and prompt engineering easier.

### `features.py`
Business logic layer containing
- overview generation
- metadata extraction
- paper summarization
- implementation planning
- risk analysis
- **Corrective RAG**: `evaluate_retrieval` (scores retrieval quality), `refine_documents` (filters out low-relevance chunks), `web_search_context` (title-scoped Tavily fallback), and `chat_with_paper` — the CRAG-based Q&A function powering the conversation

## Local Installation

Clone the repository

```
git clone https://github.com/<your-username>/ReproduceAI.git
cd ReproduceAI
```

Create a virtual environment

```
python -m venv venv
```

Activate the environment

Windows
```
venv\Scripts\activate
```

Linux / macOS
```
source venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

Create a `.env` file

```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

`TAVILY_API_KEY` powers the CRAG web-search fallback (Ambiguous/Incorrect actions). Get a free key at [tavily.com](https://tavily.com).

## Running the Application

Run the Streamlit chat application

```
streamlit run app.py
```

Run the pipeline without the UI

```
python main.py
```

## Using the Chat Interface

1. Upload a research paper PDF and click **Analyze Paper**.
2. Once analysis completes, the assistant confirms readiness in the conversation.
3. Click any of the five tool buttons (📄 Overview, 📊 Metadata, 📝 Summary, 🛠 Plan, ⚠ Risk) to instantly inject that result into the conversation — no additional API call.
4. Type any question in the chat box — e.g. *"Explain the architecture"*, *"What confidence thresholds were used?"*, *"What are the limitations?"* — and get an answer that's been checked and, if needed, corrected before it reaches you.

## Future Improvements

- LangGraph multi-agent workflow — each tool button becomes a specialized agent (Overview Agent, Metadata Agent, Planner Agent, etc.) behind a Supervisor node, without changing the chat interface
- Hybrid Retrieval (Semantic + BM25)
- Cross-Encoder Reranking
- Multi-paper comparison
- Research report export (PDF)
- GitHub repository recommendation
- Starter code generation
- Persistent chat history across sessions
- Replace the LLM-based retrieval evaluator with a fine-tuned lightweight model (as in the original CRAG paper), rather than prompting a general-purpose LLM to self-score

## Current Limitations

- Supports one research paper at a time.
- Extraction quality depends on the information available in the paper.
- Some implementation details may not be explicitly reported by the authors.
- The retrieval evaluator is an LLM prompted to self-score, not a dedicated fine-tuned model (as used in the original CRAG paper) — so its judgment of Correct/Ambiguous/Incorrect can occasionally be wrong.
- Web search fallback depends on Tavily's index; very recent or niche follow-up work may not surface.

## About

Built by Shaurya, a B.Tech AI & ML student, as a hands-on project to explore Retrieval-Augmented Generation (RAG), Corrective RAG, structured information extraction, and conversational AI application design.

## License

This project is licensed under the MIT License.
