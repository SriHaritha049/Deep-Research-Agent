from fastapi import FastAPI
from pydantic import BaseModel
from graph import app as research_graph
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json as json_lib
import uuid
import sqlite3
from datetime import datetime
from fastapi.responses import FileResponse
from fpdf import FPDF
import os
from typing import Optional


api = FastAPI(title="Deep Research Agent")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# History database
history_db = sqlite3.connect("research_history.db", check_same_thread=False)
history_db.execute("""
    CREATE TABLE IF NOT EXISTS research_sessions (
        thread_id TEXT PRIMARY KEY,
        query TEXT,
        sub_topics TEXT,
        sources TEXT,
        report TEXT,
        created_at TIMESTAMP
    )
""")
history_db.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT,
        role TEXT,
        content TEXT,
        message_type TEXT,
        created_at TIMESTAMP
    )
""")
history_db.commit()

class ResearchRequest(BaseModel):
    query: str

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


def save_to_history(thread_id, result):
    history_db.execute(
        "INSERT OR REPLACE INTO research_sessions (thread_id, query, sub_topics, sources, report, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            thread_id,
            result["query"],
            json_lib.dumps(result["sub_topics"]),
            json_lib.dumps(result["sources"]),
            result["report"],
            datetime.now().isoformat()
        )
    )
    history_db.commit()

def save_message(thread_id, role, content, message_type="chat"):
    history_db.execute(
        "INSERT INTO messages (thread_id, role, content, message_type, created_at) VALUES (?, ?, ?, ?, ?)",
        (thread_id, role, content, message_type, datetime.now().isoformat())
    )
    history_db.commit()

def load_messages(thread_id):
    cursor = history_db.execute(
        "SELECT role, content, message_type FROM messages WHERE thread_id = ? ORDER BY created_at",
        (thread_id,)
    )
    messages = []
    for row in cursor.fetchall():
        messages.append({"role": row[0], "content": row[1], "message_type": row[2]})
    return messages


@api.post("/chat/stream")
def chat_stream(request: ChatRequest):
    if request.thread_id is None:
        thread_id = str(uuid.uuid4())
    else:
        thread_id = request.thread_id

    # Save user message
    save_message(thread_id, "user", request.message)

    # Load conversation history
    messages = load_messages(thread_id)

    final_result = {}
    all_sources = []
    all_research_results = []

    def generate():
        nonlocal final_result, all_sources, all_research_results

        for event in research_graph.stream(
            {
                "query": request.message,
                "messages": messages,
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
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="updates"
        ):
            for node_name, node_output in event.items():
                if node_output is None:
                    continue
                if "sources" in node_output:
                    all_sources.extend(node_output["sources"])
                if "research_results" in node_output:
                    all_research_results.extend(node_output["research_results"])
                final_result.update(node_output)
                yield json_lib.dumps({"step": node_name, "data": node_output}) + "\n"

        # Determine what to save
        if final_result.get("needs_research") and final_result.get("report"):
            # Research path — save report as assistant message
            seen_urls = set()
            unique_sources = []
            for s in all_sources:
                if s["url"] not in seen_urls:
                    seen_urls.add(s["url"])
                    unique_sources.append(s)

            save_message(thread_id, "assistant", final_result["report"], "research_report")

            final_result["query"] = request.message
            final_result["sources"] = unique_sources
            final_result["research_results"] = all_research_results
            if "sub_topics" in final_result:
                save_to_history(thread_id, final_result)
        elif final_result.get("current_response"):
            # Chat path — save direct response
            save_message(thread_id, "assistant", final_result["current_response"], "chat")

        yield json_lib.dumps({"step": "done", "data": {"thread_id": thread_id}}) + "\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@api.get("/chat/messages/{thread_id}")
def get_messages(thread_id: str):
    messages = load_messages(thread_id)
    return {"messages": messages}

@api.get("/history")
def get_history():
    cursor = history_db.execute(
        "SELECT thread_id, query, created_at FROM research_sessions ORDER BY created_at DESC"
    )
    sessions = []
    for row in cursor.fetchall():
        sessions.append({
            "thread_id": row[0],
            "query": row[1],
            "created_at": row[2]
        })
    return sessions

@api.get("/history/{thread_id}")
def get_session(thread_id: str):
    cursor = history_db.execute(
        "SELECT thread_id, query, sub_topics, sources, report, created_at FROM research_sessions WHERE thread_id = ?",
        (thread_id,)
    )
    row = cursor.fetchone()
    if not row:
        return {"error": "Session not found"}
    return {
        "thread_id": row[0],
        "query": row[1],
        "sub_topics": json_lib.loads(row[2]),
        "sources": json_lib.loads(row[3]),
        "report": row[4],
        "created_at": row[5]
    }


@api.get("/history/{thread_id}/pdf")
def download_pdf(thread_id: str):
    cursor = history_db.execute(
        "SELECT query, sub_topics, sources, report FROM research_sessions WHERE thread_id = ?",
        (thread_id,)
    )
    row = cursor.fetchone()
    if not row:
        return {"error": "Session not found"}
    
    query = row[0]
    sub_topics = json_lib.loads(row[1])
    sources = json_lib.loads(row[2])
    report = row[3]
    
    clean_report = report.replace("**", "").replace("*", "").replace("#", "")
    
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    def write_text(text, size=11, style="", align="L"):
        pdf.set_x(20)
        pdf.set_font("Helvetica", style, size)
        safe_text = text.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 7, safe_text, align=align)
    
    write_text("Deep Research Agent Report", size=18, style="B", align="C")
    pdf.ln(8)
    
    write_text("Research Question:", size=12, style="B")
    write_text(query)
    pdf.ln(6)
    
    write_text("Sub-topics Investigated:", size=12, style="B")
    for topic in sub_topics:
        write_text(f"- {topic}")
    pdf.ln(6)
    
    write_text("Report:", size=12, style="B")
    report_parts = clean_report.split("References:")
    main_report = report_parts[0]
    
    for paragraph in main_report.split("\n"):
        paragraph = paragraph.strip()
        if paragraph:
            write_text(paragraph)
            pdf.ln(3)
    
    if len(sources) > 0:
        pdf.ln(4)
        write_text("References:", size=12, style="B")
        for i, s in enumerate(sources):
            title = s.get("title", "Unknown")
            url = s.get("url", "")
            pdf.set_x(20)
            pdf.set_font("Helvetica", "", 9)
            safe_title = title.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 6, f"{i+1}. {safe_title}")
            if url:
                pdf.set_x(25)
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(0, 0, 255)
                display_url = url if len(url) < 120 else url[:117] + "..."
                safe_url = display_url.encode("latin-1", errors="replace").decode("latin-1")
                pdf.multi_cell(0, 5, safe_url)
                pdf.set_text_color(0, 0, 0)
    
    filepath = f"report_{thread_id[:8]}.pdf"
    pdf.output(filepath)
    
    return FileResponse(filepath, filename="research_report.pdf", media_type="application/pdf")

@api.delete("/history/{thread_id}")
def delete_session(thread_id: str):
    history_db.execute("DELETE FROM research_sessions WHERE thread_id = ?", (thread_id,))
    history_db.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
    history_db.commit()
    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)