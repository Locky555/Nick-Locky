import os, re, requests
from difflib import get_close_matches
from flask import Flask, request, jsonify, render_template
from multi_rag import MultiRAG, MAJORS, COURSE_CODE_RE

# ---------- Config ----------
COURSE_DOCS_DIR = os.getenv("COURSE_DOCS_DIR", "docs")                  # JSONL lives here (catalog_master.jsonl)
FAISS_JSONL_DIR = os.getenv("FAISS_JSONL_DIR", "faiss_index_jsonl_v2")  # change to force rebuild
PDF_DOCS_DIR    = os.getenv("PDF_DOCS_DIR", "docs_pdf")                 # course PDFs here (e.g., COMP500.pdf)
FAISS_PDF_DIR   = os.getenv("FAISS_PDF_DIR", "faiss_index_pdfs_v2")     # change to force rebuild

OLLAMA_API_URL  = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/chat")
MODEL_NAME      = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:8b")

# ---------- Flask ----------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ---------- RAG (JSONL + PDFs) ----------
rag = MultiRAG(
    jsonl_folder=COURSE_DOCS_DIR,
    db_path_jsonl=FAISS_JSONL_DIR,
    pdf_folder=PDF_DOCS_DIR,
    db_path_pdf=FAISS_PDF_DIR,
)

# ---------- Small talk ----------
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

# ---------- Helpers ----------
def detect_major(text: str):
    for major in MAJORS:
        if major.lower() in text.lower():
            return major
    return None

def join_sems(sems):
    return ", ".join(sems) if sems else "—"

def prereq_line(rec):
    pre = rec.get("prerequisites") or []
    if not pre:
        return "There are no prerequisites for this course."
    return "The prerequisites for this course are: " + ", ".join(pre) + "."

NEG_PHRASES = (
    "not specified", "not mentioned", "unspecified", "unknown",
    "likely", "typically might", "probably", "may include"
)

def drop_unsure_lines(text: str) -> str:
    """Remove hedgy/negative lines to keep answers clean."""
    lines = [ln for ln in (text or "").splitlines()]
    kept = []
    for ln in lines:
        low = ln.strip().lower()
        if any(neg in low for neg in NEG_PHRASES):
            continue
        if not ln.strip():
            continue
        kept.append(ln)
    return "\n".join(kept).strip()

def ollama_answer(prompt: str, max_chars: int = 900) -> str:
    def compact(text: str) -> str:
        import re
        text = re.sub(r"\s+\n", "\n", text or "").strip()
        # Prefer bullets if present
        lines = text.splitlines()
        bullets = [ln for ln in lines if ln.strip().startswith(("-", "•", "*"))]
        if bullets:
            out, total = [], 0
            for b in bullets:
                if total + len(b) + 1 > max_chars: break
                out.append(b); total += len(b) + 1
            if out: return "\n".join(out).strip()
        # Else trim by sentences
        parts = re.split(r"(?<=[.!?])\s+", text)
        out, total = [], 0
        for s in parts:
            if total + len(s) + 1 > max_chars: break
            out.append(s); total += len(s) + 1
        return " ".join(out).strip()

    r = requests.post(
        OLLAMA_API_URL,
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": 220}
        },
        timeout=60
    )
    r.raise_for_status()
    txt = (r.json().get("message", {}) or {}).get("content", "") or ""
    return compact(txt)

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return "BCIS Course Advisor is running", 200

@app.route("/debug/pdfindex")
def debug_pdfindex():
    return jsonify(rag.debug_pdf_index() or [])

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    msg = user_input.lower()
    print("💬 User:", user_input)

    # ===== 0) Small talk shortcut =====
    small = match_small_talk(user_input)
    if small:
        return jsonify({"response": small})

    # Detect a course code upfront
    code_match = COURSE_CODE_RE.search(user_input)
    course_code = code_match.group(0).upper().replace(" ", "") if code_match else None

    # ===== 1) Deterministic lists (never PDFs here) =====
    if re.search(r"(study\s*plan|course\s*list|list\s+courses?|show\s+all|\bcourses?\b|\bpapers\b|\bsubjects\b)", msg):
        major = detect_major(user_input) or "Software Development"
        yr_match = re.search(r"year\s*([1-3])", msg)
        yr = int(yr_match.group(1)) if yr_match else None

        rows = rag.filter_courses(major=major, year=yr)
        if not rows:
            return jsonify({"response": f"No courses found for {major}" + (f" Year {yr}" if yr else "")})

        by_year = {}
        for r in rows:
            by_year.setdefault(r["year"], []).append(r)
        single_year_view = (yr is not None) or (len(by_year) == 1)

        lines = [f"{major} courses" + (f" — Year {yr}" if yr else "")]
        for y in sorted(by_year):
            if not single_year_view:
                lines.append(f"\nYear {y}:")
            else:
                lines.append("")
            for r in sorted(by_year[y], key=lambda x: x["code"]):
                sems = join_sems(r.get("semesters"))
                core = r.get("core_type") or "Major/Elective"
                note = r.get("note") or ""
                extra = f" — {note}" if note else ""
                lines.append(f"- {r['code']}: {r['title']} • {sems} • {core}{extra}")

        return jsonify({"response": "\n".join(lines)})

    # ===== 2) Prereqs/semester lookup (catalog first; fallback PDFs) =====
    if course_code and re.search(r"(prereq|pre-req|prerequisite|semester|offered|available)", msg):
        rec = rag.get_course_by_code(course_code)
        if rec:
            title = rec.get("title","")
            sems = join_sems(rec.get("semesters"))
            year = rec.get("year","?")
            header = f"{course_code} — {title} (Year {year}, {sems})"
            return jsonify({"response": f"{header}\n{prereq_line(rec)}"})

        # Fallback: PDFs
        ctx = rag.retrieve_pdf_context(course_code, question=user_input, k=10, max_chars=1200)
        prompt = f"""Extract (if present) the semester(s) when it is offered and the prerequisites.
Write 1–2 short lines, conversational, with no headings. Omit anything not explicitly in the context.
End with a single bracketed citation like [Source: FILENAME.pdf].
Context:
{ctx}
Answer:"""
        try:
            ans = ollama_answer(prompt, max_chars=300)
            ans = drop_unsure_lines(ans)
        except Exception:
            ans = "No details found in my sources."
        return jsonify({"response": f"{course_code}\n{ans}"})

    # ===== 3) “What is COMPxxx?” — PDF summary (paraphrased) =====
    if course_code and re.search(r"(what\s+is|tell\s+me\s+about|summary|summarise|summarize|overview|description|hours|workload|assessment|learning\s+outcomes|syllabus|recommend)", msg):
        # Header from catalog
        header_lines = []
        rec = rag.get_course_by_code(course_code)
        if rec:
            title = rec.get("title","")
            sems = join_sems(rec.get("semesters"))
            year = rec.get("year","?")
            header_lines.append(f"{course_code} — {title} (Year {year}, {sems})")
            header_lines.append(prereq_line(rec))

        # Pull description + workload + assessment + materials
        ctx = rag.summarize_course_pdf(course_code, user_input, max_chars=1600)

        prompt = f"""You are writing a concise, paraphrased course overview from an official outline.
Use ONLY the context. Do NOT copy sentences verbatim; rephrase in your own words.

Output format:
- First: 2–4 short sentences describing what students will learn and how the course runs.
- Then add up to THREE bullets ONLY IF the context explicitly contains them:
  • Total learning hours or weekly commitment.
  • Assessment components (e.g., tests/assignments/projects) — include percentages if shown.
  • Delivery/materials (e.g., lectures, labs, tutorials, group project, online activities).
- If an item is not in the context, omit it entirely (no “not specified”).

Hard limits:
- Max ~120 words total.
- No section headings.
- End with a single citation like [Source: FILENAME.pdf].

Context:
{ctx}

Answer:"""

        try:
            ans = ollama_answer(prompt, max_chars=700)
            ans = drop_unsure_lines(ans)
        except Exception:
            ans = "No summary available from the provided sources."

        body = ("\n".join(header_lines) + ("\n" if header_lines else "") + ans).strip()
        return jsonify({"response": body})

    # ===== 4) General JSONL Q&A (no PDFs) =====
    detected_major = detect_major(user_input)
    yr_match = re.search(r"year\s*([1-3])", msg)
    filter_year = int(yr_match.group(1)) if yr_match else None

    ctx = rag.retrieve_relevant_context(
        user_input, k=18, max_chars=3200,
        major=detected_major, year=filter_year
    )

    prompt = f"""Use ONLY the provided context. If the answer is not in the context, reply exactly: "Sorry I did not understand that."
Answer in a concise, conversational way (no headings). Use the provided [Source: ...] note if applicable.
Context:
{ctx}

User: {user_input}
Answer:"""

    try:
        ans = ollama_answer(prompt, max_chars=900)
        ans = drop_unsure_lines(ans)
    except Exception:
        ans = "An error occurred connecting to Ollama."

    return jsonify({"response": ans})

if __name__ == "__main__":
    app.run(debug=True)
