import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [currentThreadId, setCurrentThreadId] = useState(null);
  const [steps, setSteps] = useState([]);
  const [showSteps, setShowSteps] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const fetchHistory = async () => {
    try {
      const res = await fetch("http://localhost:8000/history");
      const data = await res.json();
      setHistory(data);
    } catch (e) {
      console.error("Failed to load history:", e);
    }
  };

  const loadSession = async (threadId) => {
    try {
      const res = await fetch(`http://localhost:8000/chat/messages/${threadId}`);
      const data = await res.json();
      setMessages(data.messages || []);
      setCurrentThreadId(threadId);
      setSteps([]);
      setShowSteps(false);
    } catch (e) {
      const res = await fetch(`http://localhost:8000/history/${threadId}`);
      const data = await res.json();
      setMessages([
        { role: "user", content: data.query },
        { role: "assistant", content: data.report, message_type: "research_report" }
      ]);
      setCurrentThreadId(threadId);
      setSteps([]);
      setShowSteps(false);
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setCurrentThreadId(null);
    setSteps([]);
    setShowSteps(false);
    setInput("");
  };

  const deleteSession = async (threadId) => {
    try {
      await fetch(`http://localhost:8000/history/${threadId}`, {
        method: "DELETE",
      });
      if (currentThreadId === threadId) {
        handleNewChat();
      }
      fetchHistory();
    } catch (e) {
      console.error("Failed to delete:", e);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setLoading(true);
    setSteps([]);
    setShowSteps(false);

    try {
      const response = await fetch("http://localhost:8000/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage,
          thread_id: currentThreadId,
        }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantContent = "";
      let messageType = "chat";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split("\n").filter((line) => line.trim());

        for (const line of lines) {
          try {
            const parsed = JSON.parse(line);

            if (parsed.step === "conversational_agent") {
              if (parsed.data?.needs_research) {
                messageType = "research_report";
                setShowSteps(true);
              }
              if (parsed.data?.current_response) {
                assistantContent = parsed.data.current_response;
                messageType = "chat";
              }
              setSteps((prev) => [...prev, parsed]);
            } else if (parsed.step === "respond") {
              setSteps((prev) => [...prev, parsed]);
            } else if (parsed.step === "done") {
              if (parsed.data.thread_id) {
                setCurrentThreadId(parsed.data.thread_id);
              }
            } else if (parsed.step === "synthesizer" && parsed.data?.report) {
              assistantContent = parsed.data.report;
              setSteps((prev) => [...prev, parsed]);
            } else {
              setSteps((prev) => [...prev, parsed]);
            }
          } catch (e) {}
        }
      }

      if (assistantContent) {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: assistantContent, message_type: messageType },
        ]);
      }
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Something went wrong. Please try again." },
      ]);
    }

    setLoading(false);
    fetchHistory();
  };

  const getStepLabel = (step) => {
    switch (step) {
      case "conversational_agent": return "🧠 Thinking";
      case "respond": return "💬 Responding";
      case "planner": return "📋 Planning";
      case "researcher": return "🔍 Researching";
      case "synthesizer": return "📝 Synthesizing";
      case "verifier": return "✅ Verifying";
      case "research_gaps": return "🔄 Re-researching gaps";
      default: return step;
    }
  };

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      {/* Sidebar */}
      <div
        style={{
          width: "280px",
          backgroundColor: "#1a1a2e",
          color: "#eee",
          padding: "20px",
          overflowY: "auto",
          borderRight: "1px solid #333",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <button
          onClick={handleNewChat}
          style={{
            padding: "12px",
            marginBottom: "20px",
            backgroundColor: "#2563eb",
            color: "white",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            fontSize: "14px",
            fontWeight: "bold",
          }}
        >
          + New Research
        </button>
        <h3 style={{ marginBottom: "15px", color: "#fff", fontSize: "14px" }}>
          📚 History
        </h3>
        {history.length === 0 && (
          <p style={{ color: "#888", fontSize: "13px" }}>No past research yet</p>
        )}
        {history.map((session) => (
          <div
            key={session.thread_id}
            style={{
              padding: "10px",
              marginBottom: "6px",
              backgroundColor:
                currentThreadId === session.thread_id ? "#1a1a4e" : "#16213e",
              borderRadius: "8px",
              cursor: "pointer",
              fontSize: "13px",
              lineHeight: "1.4",
              borderLeft:
                currentThreadId === session.thread_id
                  ? "3px solid #2563eb"
                  : "3px solid transparent",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "flex-start",
            }}
            onMouseOver={(e) =>
              (e.currentTarget.style.backgroundColor = "#1a1a4e")
            }
            onMouseOut={(e) => {
              if (currentThreadId !== session.thread_id) {
                e.currentTarget.style.backgroundColor = "#16213e";
              }
            }}
          >
            <div
              onClick={() => loadSession(session.thread_id)}
              style={{ flex: 1 }}
            >
              <div style={{ fontWeight: "bold", marginBottom: "3px" }}>
                {session.query.length > 50
                  ? session.query.substring(0, 50) + "..."
                  : session.query}
              </div>
              <div style={{ color: "#888", fontSize: "11px" }}>
                {new Date(session.created_at).toLocaleDateString()}
              </div>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                deleteSession(session.thread_id);
              }}
              style={{
                background: "none",
                border: "none",
                color: "#666",
                cursor: "pointer",
                fontSize: "14px",
                padding: "2px 6px",
                borderRadius: "4px",
              }}
              onMouseOver={(e) => (e.currentTarget.style.color = "#ff4444")}
              onMouseOut={(e) => (e.currentTarget.style.color = "#666")}
              title="Delete"
            >
              🗑️
            </button>
          </div>
        ))}
      </div>

      {/* Main Chat Area */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          height: "100vh",
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "15px 30px",
            borderBottom: "1px solid #eee",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <h2 style={{ margin: 0 }}>🔬 Deep Research Agent</h2>
          {currentThreadId && (
            <button
              onClick={() => {
                window.open(
                  `http://localhost:8000/history/${currentThreadId}/pdf`
                );
              }}
              style={{
                padding: "8px 16px",
                backgroundColor: "#10b981",
                color: "white",
                border: "none",
                borderRadius: "6px",
                cursor: "pointer",
                fontSize: "13px",
              }}
            >
              📥 Download PDF
            </button>
          )}
        </div>

        {/* Messages */}
        <div
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "20px 30px",
          }}
        >
          {messages.length === 0 && !loading && (
            <div
              style={{
                textAlign: "center",
                marginTop: "100px",
                color: "#999",
              }}
            >
              <p style={{ fontSize: "18px" }}>
                Ask a research question to get started
              </p>
              <p style={{ fontSize: "14px" }}>
                The agent will search the web, synthesize findings, and verify
                the report
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div
              key={i}
              style={{
                marginBottom: "20px",
                display: "flex",
                justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
              }}
            >
              <div
                style={{
                  maxWidth: msg.role === "user" ? "60%" : "85%",
                  padding: "12px 16px",
                  borderRadius:
                    msg.role === "user"
                      ? "16px 16px 4px 16px"
                      : "16px 16px 16px 4px",
                  backgroundColor:
                    msg.role === "user" ? "#2563eb" : "#f0f0f0",
                  color: msg.role === "user" ? "white" : "black",
                  lineHeight: "1.6",
                }}
              >
                {msg.role === "assistant" &&
                msg.message_type === "research_report" ? (
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                ) : (
                  <span>{msg.content}</span>
                )}
              </div>
            </div>
          ))}

          {/* Pipeline Steps - collapsible */}
          {steps.length > 0 && (
            <div style={{ marginBottom: "20px" }}>
              <div
                onClick={() => setShowSteps(!showSteps)}
                style={{
                  cursor: "pointer",
                  padding: "10px 14px",
                  backgroundColor: "#f0f4ff",
                  borderRadius: "8px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  fontSize: "14px",
                }}
              >
                <strong>
                  {loading ? "⏳ Agent is thinking..." : "✅ Research complete"}
                </strong>
                <span style={{ fontSize: "12px", color: "#666" }}>
                  {showSteps ? "▲ Hide" : "▼ Show"} ({steps.length} steps)
                </span>
              </div>

              {showSteps && (
                <div style={{ marginTop: "6px" }}>
                  {steps.map((step, i) => (
                    <div
                      key={i}
                      style={{
                        padding: "6px 10px",
                        margin: "3px 0",
                        backgroundColor: "#f8f9fa",
                        borderRadius: "6px",
                        fontSize: "13px",
                        borderLeft: "3px solid #2563eb",
                      }}
                    >
                      <strong>{getStepLabel(step.step)}</strong>
                      {step.step === "planner" && step.data?.sub_topics && (
                        <ul style={{ margin: "4px 0", paddingLeft: "20px" }}>
                          {step.data.sub_topics.map((t, j) => (
                            <li key={j}>{t}</li>
                          ))}
                        </ul>
                      )}
                      {step.step === "researcher" &&
                        step.data?.research_results && (
                          <p style={{ margin: "4px 0", color: "#555", fontSize: "12px" }}>
                            {step.data.research_results[0]?.substring(0, 120)}...
                          </p>
                        )}
                      {step.step === "verifier" &&
                        step.data?.verification_status && (
                          <span style={{ marginLeft: "8px", fontSize: "12px" }}>
                            {step.data.verification_status.toUpperCase()}
                          </span>
                        )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div
          style={{
            padding: "15px 30px",
            borderTop: "1px solid #eee",
            display: "flex",
            gap: "10px",
          }}
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder={
              messages.length === 0
                ? "Ask a research question..."
                : "Ask a follow-up question..."
            }
            disabled={loading}
            style={{
              flex: 1,
              padding: "12px",
              fontSize: "15px",
              border: "1px solid #ccc",
              borderRadius: "8px",
            }}
          />
          <button
            onClick={handleSend}
            disabled={loading}
            style={{
              padding: "12px 24px",
              fontSize: "15px",
              backgroundColor: loading ? "#ccc" : "#2563eb",
              color: "white",
              border: "none",
              borderRadius: "8px",
              cursor: loading ? "not-allowed" : "pointer",
            }}
          >
            {loading ? "..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;