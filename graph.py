from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.types import Send
import operator
import json
import logging
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from config import get_llm, MAX_CONTEXT_TOKENS, SLIDING_WINDOW_KEEP, SLIDING_WINDOW_THRESHOLD
from token_utils import count_tokens, truncate_to_tokens, trim_messages_to_fit
from memory import recall_relevant_facts, store_research_memory

logger = logging.getLogger(__name__)

tavily = TavilyClient()

llm = get_llm()


class ResearchState(TypedDict):
    query: str
    messages: list[dict]
    conversation_summary: str
    current_response: str
    needs_research: bool
    sub_topics: list[str]
    research_results: Annotated[list[str], operator.add]
    sources: Annotated[list[dict], operator.add]
    report: str
    verification_status: str
    gaps: list[str]
    loop_count: int


class ResearcherInput(TypedDict):
    sub_topic: str


def sliding_window_summarize(messages: list[dict]) -> tuple[str, list[dict]]:
    """Compress older messages into a meta-summary when conversation gets long.

    Returns (meta_summary, recent_messages).
    If under threshold, returns ("", messages).
    """
    if len(messages) <= SLIDING_WINDOW_THRESHOLD:
        return "", messages

    older = messages[:-SLIDING_WINDOW_KEEP]
    recent = messages[-SLIDING_WINDOW_KEEP:]

    # Build text from older messages
    older_text = "\n".join(
        f"{m['role']}: {m.get('summary') or m['content'][:500]}"
        for m in older
    )

    summary_response = llm.invoke(
        f"Summarize this conversation history into a concise paragraph. "
        f"Capture the key topics discussed, questions asked, and main findings.\n\n"
        f"{older_text}\n\n"
        f"Summary:"
    )

    meta_summary = summary_response.content.strip()
    logger.info(f"Sliding window: compressed {len(older)} older messages into summary ({count_tokens(meta_summary)} tokens)")
    return meta_summary, recent


def build_summary_context(messages: list[dict], meta_summary: str = "") -> str:
    """Build context using summaries for research reports, full text for short messages."""
    parts = []
    if meta_summary:
        parts.append(f"[Earlier conversation summary] {meta_summary}")
    if not messages:
        return "\n".join(parts)
    for m in messages[-10:]:
        role = m["role"]
        if m.get("summary"):
            parts.append(f"{role}: [Research Report Summary] {m['summary']}")
        else:
            parts.append(f"{role}: {m['content']}")
    return "\n".join(parts)


def build_full_context(messages: list[dict], meta_summary: str = "") -> str:
    """Build context using full content for response generation."""
    parts = []
    if meta_summary:
        parts.append(f"[Earlier conversation summary] {meta_summary}")
    if not messages:
        return "\n".join(parts)
    for m in messages[-6:]:
        content = m["content"][:2000]
        parts.append(f"{m['role']}: {content}")
    return "\n".join(parts)


def conversational_agent(state: ResearchState) -> dict:
    query = state["query"]
    messages = state.get("messages", [])

    if not messages or len(messages) <= 1:
        # Check long-term memory for relevant past research
        past_facts = recall_relevant_facts(query)
        memory_context = ""
        if past_facts:
            memory_context = "From your past research: " + "; ".join(past_facts)
            logger.info(f"Injected {len(past_facts)} facts from long-term memory")

        return {
            "needs_research": True,
            "current_response": "",
            "conversation_summary": memory_context
        }

    # Apply sliding window summarization
    meta_summary, recent_messages = sliding_window_summarize(messages)

    # Trim messages to fit token budget
    recent_messages = trim_messages_to_fit(recent_messages, MAX_CONTEXT_TOKENS)

    # Router uses SUMMARIES — fast and cheap
    summary_context = build_summary_context(recent_messages, meta_summary)

    # Check long-term memory
    past_facts = recall_relevant_facts(query)
    if past_facts:
        summary_context += "\n\nFrom your past research: " + "; ".join(past_facts)

    # Count tokens before LLM call
    prompt_text = (
        f"You are a research assistant. Based on this conversation:\n\n"
        f"{summary_context}\n\n"
        f"Can you answer this new question using ONLY the information above?\n"
        f"Question: {query}\n\n"
        f"Reply ONLY 'yes' or 'no'. Nothing else."
    )
    logger.info(f"Router prompt tokens: {count_tokens(prompt_text)}")

    decision = llm.invoke(prompt_text)

    answer = decision.content.strip().lower()
    print(f"ROUTER DECISION: '{answer}'")
    print(f"SUMMARY CONTEXT LENGTH: {len(summary_context)}")

    if answer.startswith("yes"):
        # Response generation uses FULL content
        full_context = build_full_context(recent_messages, meta_summary)
        response_prompt = (
            f"You are a helpful research assistant. Based on this conversation:\n\n"
            f"{full_context}\n\n"
            f"Answer this question: {query}\n\n"
            f"Be concise. Only use information from the conversation above."
        )
        logger.info(f"Response prompt tokens: {count_tokens(response_prompt)}")

        response = llm.invoke(response_prompt)
        return {"needs_research": False, "current_response": response.content, "conversation_summary": ""}

    return {"needs_research": True, "current_response": "", "conversation_summary": ""}


def planner(state: ResearchState) -> dict:
    query = state["query"]
    messages = state.get("messages", [])

    # Apply sliding window + token trimming
    meta_summary, recent_messages = sliding_window_summarize(messages)
    recent_messages = trim_messages_to_fit(recent_messages, MAX_CONTEXT_TOKENS)
    summary_context = build_summary_context(recent_messages, meta_summary)

    prompt = (
        f"Break this research question into 3-4 specific sub-topics to investigate.\n"
        f"Rules: Return ONLY the sub-topics, one per line. No numbering, no bullets, "
        f"no explanation, no introductory text. Each sub-topic should be a clear search query.\n\n"
    )
    if summary_context:
        prompt += f"Previous conversation for context:\n{summary_context}\n\n"
    prompt += f"Question: {query}"

    logger.info(f"Planner prompt tokens: {count_tokens(prompt)}")
    response = llm.invoke(prompt)

    # Clean up LLM output — strip bullets, numbers, asterisks, junk lines
    topics = []
    for line in response.content.strip().split("\n"):
        cleaned = line.strip().lstrip("*-•1234567890. ")
        if not cleaned or len(cleaned) < 5:
            continue
        if cleaned.lower().startswith(("here", "sure", "based")):
            continue
        topics.append(cleaned)

    # Limit to max 4 sub-topics
    topics = topics[:4]

    logger.info(f"Planner produced {len(topics)} sub-topics: {topics}")
    return {"sub_topics": topics}


def researcher(state: ResearcherInput) -> dict:
    topic = state["sub_topic"]

    try:
        search_results = tavily.search(query=topic, max_results=3)

        sources = []
        sources_text_parts = []
        for result in search_results["results"]:
            sources.append({
                "title": result["title"],
                "url": result["url"]
            })
            sources_text_parts.append(f"Title: {result['title']}\nContent: {result['content']}\nURL: {result['url']}")

        sources_text = "\n\n".join(sources_text_parts)

        # Token-aware: truncate sources if too long
        sources_text = truncate_to_tokens(sources_text, MAX_CONTEXT_TOKENS // 2)

        research_prompt = (
            f"Based on these search results, provide key findings about: {topic}\n\n"
            f"Search Results:\n{sources_text}\n\n"
            f"Summarize the key findings in 2-3 sentences. Cite the sources by mentioning the source name."
        )
        logger.info(f"Researcher prompt tokens: {count_tokens(research_prompt)}")

        response = llm.invoke(research_prompt)

        return {
            "research_results": [response.content],
            "sources": sources
        }
    except Exception as e:
        return {
            "research_results": [f"Research on '{topic}' failed due to connection error."],
            "sources": []
        }


def synthesizer(state: ResearchState) -> dict:
    query = state["query"]
    research = state["research_results"]
    sources = state["sources"]
    messages = state.get("messages", [])

    meta_summary, recent_messages = sliding_window_summarize(messages)
    recent_messages = trim_messages_to_fit(recent_messages, MAX_CONTEXT_TOKENS)
    summary_context = build_summary_context(recent_messages, meta_summary)

    prompt = (
        f"Combine all the research findings below into one coherent, well-structured report "
        f"that answers the original question. Remove repetition and connect ideas.\n"
        f"Do NOT include a references section.\n\n"
    )
    if summary_context:
        prompt += f"Previous conversation for context:\n{summary_context}\n\n"
    prompt += (
        f"Original question: {query}\n\n"
        f"Research findings:\n{research}"
    )

    logger.info(f"Synthesizer prompt tokens: {count_tokens(prompt)}")
    response = llm.invoke(prompt)

    sources_list = "\n".join([f"- [{s['title']}]({s['url']})" for s in sources])
    report = response.content + f"\n\n**References:**\n{sources_list}"

    # Generate summary of this report
    summary_response = llm.invoke(
        f"Summarize this research report in 2-3 sentences. Include the key topics, findings, and any specific data points mentioned.\n\n"
        f"Report:\n{response.content}\n\n"
        f"Summary:"
    )

    summary = summary_response.content.strip()

    # Store in long-term memory
    store_research_memory(report, query, state.get("thread_id", "unknown"))

    return {
        "report": report,
        "conversation_summary": summary
    }


def verifier(state: ResearchState) -> dict:
    query = state["query"]
    report = state["report"]
    sub_topics = state["sub_topics"]
    loop_count = state.get("loop_count", 0)
    messages = state.get("messages", [])

    meta_summary, recent_messages = sliding_window_summarize(messages)
    recent_messages = trim_messages_to_fit(recent_messages, MAX_CONTEXT_TOKENS)
    summary_context = build_summary_context(recent_messages, meta_summary)

    prompt = (
        f"You are a research report verifier. Check if this report adequately answers the original question.\n\n"
    )
    if summary_context:
        prompt += f"Previous conversation for context:\n{summary_context}\n\n"
    prompt += (
        f"Original question: {query}\n\n"
        f"Sub-topics that should be covered: {sub_topics}\n\n"
        f"Report:\n{report}\n\n"
        f"Check:\n"
        f"1. Does the report answer the original question?\n"
        f"2. Is every sub-topic addressed in the report?\n"
        f"3. Are there any contradicting statements?\n"
        f"4. Are there important gaps or missing information?\n\n"
        f"Return ONLY a JSON object in this exact format, nothing else:\n"
        f'{{"status": "pass", "gaps": []}}\n'
        f'or\n'
        f'{{"status": "fail", "gaps": ["specific gap 1", "specific gap 2"]}}'
    )

    logger.info(f"Verifier prompt tokens: {count_tokens(prompt)}")
    response = llm.invoke(prompt)

    try:
        result = json.loads(response.content.strip())
        status = result.get("status", "pass")
        gaps = result.get("gaps", [])
    except json.JSONDecodeError:
        content = response.content.strip().lower()
        if "fail" in content:
            status = "fail"
            gaps = ["Report needs improvement"]
        else:
            status = "pass"
            gaps = []

    return {
        "verification_status": status,
        "gaps": gaps,
        "loop_count": loop_count + 1
    }


def respond(state: ResearchState) -> dict:
    return {}


def route_after_agent(state: ResearchState):
    if state["needs_research"]:
        return "planner"
    return "respond"

def route_to_researchers(state: ResearchState):
    return [Send("researcher", {"sub_topic": t}) for t in state["sub_topics"]]

def route_after_verification(state: ResearchState):
    if state["verification_status"] == "pass" or state["loop_count"] >= 2:
        return "end"
    return "research_gaps"

def route_gaps_to_researchers(state: ResearchState):
    return [Send("researcher", {"sub_topic": g}) for g in state["gaps"]]


# Build graph
graph = StateGraph(ResearchState)

graph.add_node("conversational_agent", conversational_agent)
graph.add_node("planner", planner)
graph.add_node("researcher", researcher)
graph.add_node("synthesizer", synthesizer)
graph.add_node("verifier", verifier)
graph.add_node("respond", respond)
graph.add_node("research_gaps", lambda state: state)

graph.set_entry_point("conversational_agent")
graph.add_conditional_edges("conversational_agent", route_after_agent, {
    "planner": "planner",
    "respond": "respond"
})
graph.add_conditional_edges("planner", route_to_researchers, ["researcher"])
graph.add_edge("researcher", "synthesizer")
graph.add_edge("synthesizer", "verifier")
graph.add_conditional_edges("verifier", route_after_verification, {
    "end": END,
    "research_gaps": "research_gaps"
})
graph.add_conditional_edges("research_gaps", route_gaps_to_researchers, ["researcher"])
graph.add_edge("respond", END)

conn = sqlite3.connect("research_history.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = graph.compile(checkpointer=memory)


if __name__ == "__main__":
    import uuid
    thread_id = str(uuid.uuid4())
    result = app.invoke({
        "query": "What are the latest advances in electric vehicle battery technology?",
        "messages": [],
        "conversation_summary": "",
        "current_response": "",
        "needs_research": False,
        "sub_topics": [],
        "research_results": [],
        "sources": [],
        "report": "",
        "verification_status": "",
        "gaps": [],
        "loop_count": 0
    },
    config={"configurable": {"thread_id": thread_id}}
    )

    print("=== Sub-Topics ===")
    for t in result["sub_topics"]:
        print(f"- {t}")

    print("\n=== Final Report ===")
    print(result["report"])
