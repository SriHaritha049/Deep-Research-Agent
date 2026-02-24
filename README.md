# Deep Research Agent

An AI-powered research assistant that autonomously searches the web, synthesizes findings, and generates verified research reports — powered by LangGraph, FastAPI, and React.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     React Frontend                         │
│                  (Chat Interface @ :3000)                   │
└──────────────────────────┬─────────────────────────────────┘
                           │ SSE Stream
┌──────────────────────────▼─────────────────────────────────┐
│                   FastAPI Backend (:8000)                   │
│         /chat/stream  /history  /chat/messages              │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│                  LangGraph State Machine                    │
│                                                            │
│  ┌─────────────────┐    ┌─────────┐    ┌──────────────┐   │
│  │ Conversational   │───▶│ Planner │───▶│ Researchers  │   │
│  │ Agent (Router)   │    └─────────┘    │ (Parallel)   │   │
│  └────────┬─────────┘                   └──────┬───────┘   │
│           │ (can answer                        │           │
│           │  from context)              ┌──────▼───────┐   │
│           │                             │ Synthesizer  │   │
│           ▼                             └──────┬───────┘   │
│     Direct Response                     ┌──────▼───────┐   │
│                                         │   Verifier   │   │
│                                         └──────┬───────┘   │
│                                                │           │
│                                    ┌───────────▼────────┐  │
│                                    │ Pass? ──▶ Done     │  │
│                                    │ Fail? ──▶ Re-search│  │
│                                    └────────────────────┘  │
└────────────────────────────────────────────────────────────┘
         │                    │                    │
    ┌────▼────┐        ┌─────▼──────┐      ┌─────▼──────┐
    │ Ollama  │        │  Tavily    │      │  ChromaDB  │
    │ LLM     │        │  Web Search│      │  Memory    │
    └─────────┘        └────────────┘      └────────────┘
```

## Features

- **Agentic Research Pipeline** — Autonomous planning, parallel web research, synthesis, and verification with gap re-research
- **Conversational Memory** — Follow-up questions answered from context without re-searching
- **Sliding Window Summarization** — Older messages compressed into meta-summaries to prevent unbounded context growth
- **Long-term Memory (ChromaDB)** — Key facts stored as embeddings across sessions; relevant past research recalled automatically
- **Token Counting (tiktoken)** — Exact token counts before every LLM call with automatic context trimming
- **Streaming Updates** — Real-time pipeline progress via server-sent events
- **Research History** — Browse, reload, and delete past research sessions
- **PDF Export** — Download any research report as a formatted PDF
- **Model Abstraction** — Swap between Ollama, OpenAI, or Anthropic with environment variables

## Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| Frontend    | React 19, React Markdown            |
| Backend     | FastAPI, Uvicorn                    |
| Orchestration | LangGraph (state machine)         |
| LLM         | Ollama (llama3.1:8b) / OpenAI / Anthropic |
| Web Search  | Tavily API                          |
| Memory      | ChromaDB (persistent embeddings)    |
| Token Mgmt  | tiktoken                            |
| Database    | SQLite                              |
| PDF         | FPDF                                |
| Containers  | Docker, Docker Compose              |

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.ai) with `llama3.1:8b` pulled
- [Tavily API key](https://tavily.com)

### Without Docker

1. **Clone and configure**
   ```bash
   git clone https://github.com/SriHaritha049/deep-research-agent.git
   cd deep-research-agent
   cp .env.example .env   # Add your TAVILY_API_KEY
   ```

2. **Install backend dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama and pull the model**
   ```bash
   ollama pull llama3.1:8b
   ```

4. **Run the backend**
   ```bash
   uvicorn api:api --reload --port 8000
   ```

5. **Install and run the frontend**
   ```bash
   cd research-frontend
   npm install
   npm start
   ```

6. Open http://localhost:3000

### With Docker

```bash
# Set your Tavily API key
echo "TAVILY_API_KEY=your_key_here" > .env

# Build and start all services
docker compose up --build

# Pull the model into the Ollama container (first time only)
docker compose exec ollama ollama pull llama3.1:8b
```

## How the Pipeline Works

1. **Conversational Agent** — Routes the query: can it be answered from existing context, or does it need new research?
2. **Planner** — Breaks the question into 3-4 focused sub-topics for web search
3. **Researchers** — Search the web in parallel (via Tavily), one per sub-topic, and summarize findings
4. **Synthesizer** — Combines all research into a coherent report with references; generates a summary for future context
5. **Verifier** — Checks the report for coverage, consistency, and gaps
6. **Gap Re-research** — If the verifier finds gaps, those are researched and the report is regenerated (max 2 loops)

## Configuration

| Environment Variable    | Default           | Description                              |
|------------------------|-------------------|------------------------------------------|
| `LLM_PROVIDER`         | `ollama`          | LLM backend: `ollama`, `openai`, `anthropic` |
| `LLM_MODEL`            | `llama3.1:8b`     | Model name to use                        |
| `MAX_CONTEXT_TOKENS`   | `4096`            | Max tokens for context window            |
| `MAX_SUMMARY_TOKENS`   | `200`             | Max tokens for summaries                 |
| `CHROMA_PERSIST_DIR`   | `./chroma_memory` | ChromaDB persistence directory           |
| `TAVILY_API_KEY`       | —                 | Required. Tavily API key for web search  |

## Screenshots

*Coming soon*
