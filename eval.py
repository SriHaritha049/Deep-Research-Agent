from langsmith import Client
from langsmith.evaluation import evaluate
from graph import app as research_graph
from dotenv import load_dotenv
load_dotenv()

client = Client()

# This function runs your pipeline for each test case
def run_research(inputs: dict) -> dict:
    print(f"Running research for: {inputs['query']}")
    result = research_graph.invoke({
        "query": inputs["query"],
        "sub_topics": [],
        "research_results": [],
        "sources": [],
        "report": "",
        "verification_status": "",
        "gaps": [],
        "loop_count": 0
    })
    print(f"Research complete for: {inputs['query']}")
    return {
        "report": result["report"],
        "sub_topics": result["sub_topics"],
        "sources": result["sources"],
        "verification_status": result["verification_status"]
    }
# Evaluator 1: Topic Coverage - does the report cover expected topics?
def topic_coverage(run, example) -> dict:
    report = run.outputs["report"].lower()
    expected = example.outputs["expected_topics"]
    
    covered = 0
    missing = []
    for topic in expected:
        if topic.lower() in report:
            covered += 1
        else:
            missing.append(topic)
    
    score = covered / len(expected)
    return {
        "key": "topic_coverage",
        "score": score,
        "comment": f"Covered {covered}/{len(expected)}. Missing: {missing}" if missing else "All topics covered"
    }

# Evaluator 2: Has Sources - does the report include references?
def has_sources(run, example) -> dict:
    sources = run.outputs.get("sources", [])
    has_urls = any("url" in s for s in sources)
    return {
        "key": "has_sources",
        "score": 1.0 if has_urls else 0.0,
        "comment": f"{len(sources)} sources found" if has_urls else "No sources found"
    }

# Evaluator 3: Report Length - is the report substantial enough?
def report_length(run, example) -> dict:
    report = run.outputs["report"]
    word_count = len(report.split())
    # Score: 0 if under 100 words, 1 if over 300, linear between
    score = min(max((word_count - 100) / 200, 0), 1)
    return {
        "key": "report_length",
        "score": score,
        "comment": f"{word_count} words"
    }

# Evaluator 4: Verification Passed
def verification_passed(run, example) -> dict:
    status = run.outputs.get("verification_status", "")
    return {
        "key": "verification_passed",
        "score": 1.0 if status == "pass" else 0.0,
        "comment": f"Status: {status}"
    }

# Run evaluation
results = evaluate(
    run_research,
    data="research-agent-eval",
    evaluators=[topic_coverage, has_sources, report_length, verification_passed],
    experiment_prefix="baseline-ollama-3b"
)

print("Evaluation complete! Check LangSmith for results.")