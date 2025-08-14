import os
import re
import requests
from difflib import get_close_matches
from flask import Flask, request, jsonify, render_template

# ---------- Config (env vars) ----------
USE_MONGO = os.getenv("USE_MONGO", "false").lower() in {"1", "true", "yes", "on"}

# Point this at your local Ollama or your friend's host (VPN/LAN/public)
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/chat")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-r1:8b")

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "5000"))
DEBUG = os.getenv("FLASK_DEBUG", "true").lower() in {"1", "true", "yes", "on"}

# ---------- Optional Mongo (only if enabled) ----------
if USE_MONGO:
    from pymongo import MongoClient
    MONGO_URI = os.getenv(
        "MONGO_URI",
        "mongodb+srv://rnd:rnduser@cluster0.5lkna8o.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    )
    mongo_client = MongoClient(MONGO_URI)
    courses_collection = mongo_client["BCIS_Courses"]["courses"]

# ---------- RAG (PDFs) ----------
from multi_rag import MultiRAG
rag = MultiRAG(pdf_folder="docs", db_path="faiss_index_pdf")

# ---------- Flask ----------
app = Flask(__name__)

# ---------- Helpers ----------
def clean_ollama_response(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()

def extract_last_paragraph(text: str) -> str:
    paragraphs = [p.strip() for p in (text or "").split('\n') if p.strip()]
    return paragraphs[-1] if paragraphs else (text or "").strip()

KNOWN_MAJORS = [
    "Software Development", "Computer Science",
    "Data Science", "Networks and Cybersecurity", "Digital Services"
]

def detect_major(text: str):
    t = (text or "").lower()
    for major in KNOWN_MAJORS:
        if major.lower() in t:
            return major
    return None

SMALL_TALK_RESPONSES = {
    "hello": "Hi there! How can I help you with your study plan?",
    "hi": "Hello! How can I assist you today?",
    "hey": "Hey! Looking for a study plan or need help?",
    "thanks": "You're welcome!",
    "thank you": "Happy to help!",
    "who are you": "I'm a helpful assistant trained to guide you through your course planning and study queries.",
    "who is the best rnd client": "The legend Matthew! 😎",
}

def match_small_talk(input_text: str):
    if not input_text:
        return None
    matches = get_close_matches(input_text.lower(), SMALL_TALK_RESPONSES.keys(), n=1, cutoff=0.8)
    if matches:
        return SMALL_TALK_RESPONSES[matches[0]]
    return None

def build_study_plan_from_mongo(user_input: str):
    """Return a formatted study plan string from Mongo, or None/str error if not available."""
    if not USE_MONGO:
        return None

    detected_major = detect_major(user_input) or "Software Development"
    match_year = re.search(r"year\s*([1-3])", user_input.lower())
    filter_year = int(match_year.group(1)) if match_year else None

    query = {"majors": {"$in": [detected_major]}}
    if filter_year:
        query["year"] = filter_year

    results = list(courses_collection.find(query))
    if not results and filter_year:
        query.pop("year", None)
        results = list(courses_collection.find(query))

    if not results:
        return f"Sorry, I couldn’t find courses in MongoDB for {detected_major}."

    plan = {}
    for course in results:
        year = course.get('year')
        code = course.get('code')
        title = course.get('title')
        sems = course.get('semester', [])
        label = f"{code}: {title}"
        for sem in sems:
            year_key = f"Year {year}"
            sem_key = f"Semester {sem}"
            plan.setdefault(year_key, {}).setdefault(sem_key, set()).add(label)

    output = f"📘 Study Plan for {detected_major}"
    if filter_year:
        output += f" (Year {filter_year})"
    output += ":\n\n"

    for year in sorted(plan.keys()):
        output += f"{year}:\n"
        for sem in sorted(plan[year].keys()):
            courses = sorted(plan[year][sem])
            output += f"  {sem}: " + ", ".join(courses) + "\n"
        output += "\n"

    return output.strip()

# ---------- Routes ----------
@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return {"ok": True, "mongo_enabled": USE_MONGO}

@app.post("/chat")
def chat():
    user_input = request.json.get('message', '').strip()
    print("💬 User:", user_input)

    if not user_input:
        return jsonify({"response": "No message received."}), 400

    # 1) Small talk shortcut
    small = match_small_talk(user_input)
    if small:
        return jsonify({"response": small})

    # 2) Study plan via Mongo (only if enabled and user asked)
    wants_plan = bool(re.search(r"study\s*plan|course\s*list", user_input.lower()))
    if USE_MONGO and wants_plan:
        try:
            plan = build_study_plan_from_mongo(user_input)
            if plan and not plan.startswith("Sorry"):
                return jsonify({"response": plan})
            # If Mongo had no data, fall through to PDF RAG
        except Exception as e:
            print("Mongo error:", e)

    # 3) PDF-based Q&A via RAG (default + fallback)
    context = rag.retrieve_relevant_context(user_input)
    if not context:
        return jsonify({"response": "No relevant information found in the documents."})

    prompt = f"""You are a helpful assistant. Use the provided context to answer concisely and clearly. Do not guess. Only use what's found in the context.

Context:
{context}

User Question: {user_input}
Answer:"""

    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=60
        )
        if not resp.ok:
            print("Ollama HTTP error:", resp.status_code, resp.text[:200])
            return jsonify({"response": "LLM call failed (non-200). Please try again."}), 502

        full = resp.json()
        raw_answer = (full.get("message") or {}).get("content", "") or ""
        cleaned = clean_ollama_response(raw_answer)
        concise = extract_last_paragraph(cleaned) or "I couldn't find a clear answer in the documents."
        return jsonify({"response": concise})
    except requests.Timeout:
        return jsonify({"response": "LLM call timed out. Please try again."}), 504
    except Exception as e:
        print("LLM error:", str(e))
        return jsonify({"response": "An error occurred connecting to Ollama."}), 502

# ---------- Runner ----------
if __name__ == "__main__":
    print(f"Starting Flask on http://{HOST}:{PORT}  (debug={DEBUG}) | USE_MONGO={USE_MONGO}")
    app.run(host=HOST, port=PORT, debug=DEBUG)