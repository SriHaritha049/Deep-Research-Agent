import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration — change these to swap models
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama, openai, anthropic
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

# Token limits
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "4096"))
MAX_SUMMARY_TOKENS = int(os.getenv("MAX_SUMMARY_TOKENS", "200"))

# Memory
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_memory")

# Sliding window
SLIDING_WINDOW_KEEP = 6  # Keep last N messages as-is
SLIDING_WINDOW_THRESHOLD = 8  # Trigger summarization after N messages


def get_llm():
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=LLM_MODEL)
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=LLM_MODEL)
    elif LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=LLM_MODEL)
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")
