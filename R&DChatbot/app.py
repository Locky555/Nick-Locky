# app.py
import os, re, requests
from flask import Flask, request, jsonify, render_template
from multi_rag import MultiRAG, MAJORS, COURSE_CODE_RE

# ---------- Config ----------
COURSE_DOCS_DIR = os.getenv("COURSE_DOCS_DIR", "docs")                  # JSONL lives here
FAISS_JSONL_DIR = os.getenv("FAISS_JSONL_DIR", "faiss_index_jsonl_v2")  # fresh name => fresh build
PDF_DOCS_DIR    = os.getenv("PDF_DOCS_DIR", "docs_pdf")                 # course PDFs here
FAISS_PDF_DIR   = os.getenv("FAISS_PDF_DIR", "faiss_index_pdfs_v1")

OLLAMA_API_URL  = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/chat")
MODEL_NAME      = os.getenv("OLLAMA_MODEL_NAME", "deepseek-v2:16b")

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
    """Remove hedgy/negative lines like 'not specified' to keep answers clean."""
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

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    msg = user_input.lower()
    print("💬 User:", user_input)

    # ===== Detect a course code upfront =====
    code_match = COURSE_CODE_RE.search(user_input)
    course_code = code_match.group(0).upper() if code_match else None

    # ===== 1) Deterministic lists (never PDFs here) =====
    if re.search(r"(study\s*plan|course\s*list|list\s+courses?|show\s+all|\bcourses?\b|\bpapers\b|\bsubjects\b)", msg):
        major = detect_major(user_input) or "Software Development"
        yr_match = re.search(r"year\s*([1-3])", msg)
        yr = int(yr_match.group(1)) if yr_match else None

        rows = rag.filter_courses(major=major, year=yr)
        if not rows:
            return jsonify({"response": f"No courses found for {major}" + (f" Year {yr}" if yr else "")})

        by_year = {}
        for r in rows: by_year.setdefault(r["year"], []).append(r)
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

    # ===== 2) Conversational prereqs/semester lookup (catalog first) =====
    if course_code and re.search(r"(prereq|pre-req|prerequisite|semester|offered|available)", msg):
        rec = rag.get_course_by_code(course_code)
        if rec:
            title = rec.get("title","")
            sems = join_sems(rec.get("semesters"))
            year = rec.get("year","?")
            header = f"{course_code} — {title} (Year {year}, {sems})"
            return jsonify({"response": f"{header}\n{prereq_line(rec)}"})

        # Try PDFs if not in catalog
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

    # ===== 3) Course PDF summary/workload/etc. (conversational) =====
    if course_code and re.search(r"(what\s+is|tell\s+me\s+about|summary|summarise|summarize|hours|workload|assessment|overview|learning\s+outcomes|syllabus|recommend)", msg):
        # Build a light header from catalog if available
        header_lines = []
        rec = rag.get_course_by_code(course_code)
        if rec:
            title = rec.get("title","")
            sems = join_sems(rec.get("semesters"))
            year = rec.get("year","?")
            header_lines.append(f"{course_code} — {title} (Year {year}, {sems})")
            header_lines.append(prereq_line(rec))

        # Pull relevant PDF chunks for this course
        ctx = rag.retrieve_pdf_context(course_code, question=user_input, k=10, max_chars=1600)

        # Ask for a short, conversational summary + optional bullets only if present
        prompt = f"""Using ONLY the context, write a brief conversational summary of {course_code}:
- 1–3 short sentences describing what the course covers.
- Then include up to two bullets ONLY IF explicitly present:
  • one bullet for "Total learning hours: N" or "Approx. weekly commitment: N hours/week"
  • one bullet for "Assessment: …"
- If a detail is not in the context, omit it entirely (do NOT say "not specified").
- No section headings. Keep it light and natural.
- Add a single citation at the end of the paragraph or the last bullet like [Source: FILENAME.pdf].

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
