# app.py
import os
import re
import json
import requests
from difflib import get_close_matches
from pathlib import Path
from typing import List, Optional

from flask import Flask, request, jsonify, render_template

# =========================
# Config
# =========================
OLLAMA_API_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v2:16b")

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "5000"))
DEBUG = os.getenv("FLASK_DEBUG", "true").lower() in {"1", "true", "yes", "on"}

# =========================
# LangChain / FAISS
# =========================
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# =============================================================================
# MultiRAG: JSONL (catalog facts) + PDFs (narrative/details)
# =============================================================================
MAJORS = [
    "Software Development",
    "Computer Science",
    "Data Science",
    "Networks & Cybersecurity",
    "Digital Services",
]

# Canonical minor IDs and aliases (extend if you add more minors later)
MINOR_ALIASES = {
    "AI": [
        "ai", "a.i", "a.i.", "artificial intelligence", "artificial-intelligence",
        "ai minor", "a.i minor", "artificial intelligence minor"
    ],
}
MINOR_DISPLAY = {"AI": "Artificial Intelligence"}  # canonical -> pretty name

COURSE_PREFIXES = r"(COMP|MATH|STAT|INFS|ENEL|ENSE)"
COURSE_CODE_RE = re.compile(rf"\b{COURSE_PREFIXES}\s*\d{{3}}\b", re.I)

DESC_HINTS = (
    "course description", "description", "paper description",
    "overview", "synopsis", "aim", "course aim", "paper aim", "purpose"
)
WORKLOAD_HINTS = (
    "learning hours", "workload", "weekly hours", "total hours",
    "prescribed learning hours", "time commitment", "150 hours", "120 hours"
)
ASSESS_HINTS = (
    "assessment", "assessments", "assessment structure",
    "weighting", "%", "percentage", "exam", "test", "assignment",
    "project", "report", "presentation", "lab", "quiz"
)
MATERIAL_HINTS = (
    "delivery", "learning and teaching", "materials",
    "lectures", "tutorials", "labs", "laboratories", "workshops",
    "online activities", "studio", "group work", "team project"
)
TOPIC_HINTS = (
    "topics", "content", "learning outcomes", "students will learn",
    "covers", "includes", "focuses on"
)

def _contains_any(text: str, phrases) -> bool:
    low = (text or "").lower()
    return any(p in low for p in phrases)


class MultiRAG:
    """
    Hybrid retriever:
      - JSONL catalog (deterministic facts)
      - PDF store (narrative details)
    """
    def __init__(
        self,
        jsonl_folder: str = "docs",
        db_path_jsonl: str = "faiss_index_jsonl_v1",
        pdf_folder: str = "docs_pdf",
        db_path_pdf: str = "faiss_index_pdfs_v1",
    ):
        self.jsonl_folder = Path(jsonl_folder)
        self.db_path_jsonl = Path(db_path_jsonl)
        self.pdf_folder = Path(pdf_folder)
        self.db_path_pdf = Path(db_path_pdf)

        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # JSONL catalog index
        self.catalog = self._load_catalog_jsonl()
        self.jsonl_store = None
        self._build_jsonl_store()

        # PDF index (optional)
        self.pdf_store = None
        self.pdf_meta_index = []
        self._pdf_all_docs: List[Document] = []  # keep all chunks for fallback
        if self.pdf_folder.exists():
            self._build_pdf_store()

    # ------------- JSONL CATALOG -------------
    def _load_catalog_jsonl(self) -> list:
        if not self.jsonl_folder.exists():
            raise FileNotFoundError(f"JSONL folder not found: {self.jsonl_folder.resolve()}")

        files = []
        master = self.jsonl_folder / "catalog_master.jsonl"
        if master.exists():
            files = [master]
        else:
            files = sorted(self.jsonl_folder.glob("*.jsonl"))
            if not files:
                raise FileNotFoundError(f"No .jsonl files found in {self.jsonl_folder.resolve()}")

        recs, seen = [], set()
        for p in files:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    code = (r.get("code") or "").upper().replace(" ", "")
                    if code and code not in seen:
                        # normalize
                        r["code"] = code
                        if "prerequisites" not in r:
                            r["prerequisites"] = r.get("prereqs", []) or []
                        if "semesters" not in r:
                            r["semesters"] = r.get("semester", []) or []
                        # keep minors if present
                        r["minors"] = r.get("minors", [])
                        recs.append(r)
                        seen.add(code)

        print(f"📚 Loaded {len(recs)} JSONL course records from: "
              f"{', '.join(p.name for p in files)}")
        return recs

    def _jsonl_records_to_documents(self) -> List[Document]:
        docs: List[Document] = []
        for r in self.catalog:
            sems = r.get("semesters", [])
            prereqs = r.get("prerequisites", [])
            minors = r.get("minors", [])
            content = (
                f"{r.get('code','')} — {r.get('title','')}\n"
                f"Year: {r.get('year','')}  •  Semesters: {', '.join(sems) if sems else '—'}\n"
                f"Prerequisites: {r.get('prereq_text') or (', '.join(prereqs) if prereqs else 'None')}\n"
                f"Core: {r.get('core_type','Major/Elective')}\n"
                f"Majors: {', '.join(r.get('majors', []))}\n"
                f"{'Minors: ' + ', '.join(minors) if minors else ''}\n"
                f"{('Note: ' + r.get('note','')) if r.get('note') else ''}"
            ).strip()
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": "catalog_master.jsonl",
                    "kind": "course",
                    "code": r.get("code"),
                    "year": r.get("year"),
                    "majors": r.get("majors", []),
                    "minors": minors,
                    "core_type": r.get("core_type", "Major/Elective"),
                    "semesters": sems,
                    "prerequisites": prereqs,
                    "title": r.get("title", "")
                }
            ))
        return docs

    def _build_jsonl_store(self):
        print(f"🔧 Building JSONL FAISS in: {self.db_path_jsonl.resolve()}")
        documents = self._jsonl_records_to_documents()
        self.jsonl_store = FAISS.from_documents(documents, self.embeddings)
        self.db_path_jsonl.mkdir(parents=True, exist_ok=True)
        self.jsonl_store.save_local(str(self.db_path_jsonl))
        print("✅ JSONL FAISS built.")

    def filter_courses(self, major: Optional[str] = None,
                       year: Optional[int] = None,
                       semester: Optional[str] = None,
                       core_type: Optional[str] = None,
                       minor: Optional[str] = None) -> list:
        recs = self.catalog
        if major:
            mlow = major.lower()
            recs = [r for r in recs if any((str(x) or "").lower() == mlow for x in r.get("majors", []))]
        if minor:
            # minor is canonical (e.g., "AI")
            recs = [r for r in recs if minor in (r.get("minors") or [])]
        if year is not None:
            try:
                y = int(year)
                recs = [r for r in recs if r.get("year") == y]
            except ValueError:
                pass
        if semester:
            recs = [r for r in recs if semester in (r.get("semesters") or [])]
        if core_type:
            recs = [r for r in recs if r.get("core_type") == core_type]
        return sorted(recs, key=lambda r: (r.get("year", 99), r.get("code", "")))

    def get_course_by_code(self, code: str) -> Optional[dict]:
        code = (code or "").upper().replace(" ", "")
        for r in self.catalog:
            if r.get("code") == code:
                return r
        return None

    def retrieve_relevant_context(self, query: str,
                                  k: int = 14, max_chars: int = 3200,
                                  major: Optional[str] = None,
                                  year: Optional[int] = None) -> str:
        results = self.jsonl_store.similarity_search_with_score(query, k=k)
        docs = [d for d, _ in results]

        if major:
            mlow = major.lower()
            by_major = [d for d in docs if any((str(m) or "").lower() == mlow for m in d.metadata.get("majors", []))]
            if by_major:
                docs = by_major
        if year is not None:
            try:
                y = int(year)
                by_year = [d for d in docs if d.metadata.get("year") == y]
                if by_year:
                    docs = by_year
            except ValueError:
                pass

        used, out, size = set(), [], 0
        for d in docs:
            src = d.metadata.get("source")
            key = (src, d.metadata.get("code"))
            frag = (d.page_content or "").strip()
            if not frag or key in used:
                continue
            piece = frag + f"\n[Source: {src}]\n"
            if size + len(piece) > max_chars:
                break
            out.append(piece)
            size += len(piece)
            used.add(key)
        return "\n---\n".join(out) if out else "(no high-confidence matches)\n"

    # ------------- PDF STORE -------------
    def _infer_code_from_filename(self, name: str) -> Optional[str]:
        m = COURSE_CODE_RE.search(name or "")
        return m.group(0).upper().replace(" ", "") if m else None

    def _infer_code_from_text(self, text: str) -> Optional[str]:
        m = COURSE_CODE_RE.search(text or "")
        return m.group(0).upper().replace(" ", "") if m else None

    def _tag_section(self, text: str) -> str:
        t = (text or "").lower()
        if _contains_any(t, DESC_HINTS):     return "description"
        if _contains_any(t, WORKLOAD_HINTS): return "workload"
        if _contains_any(t, ASSESS_HINTS):   return "assessment"
        if _contains_any(t, MATERIAL_HINTS): return "materials"
        if _contains_any(t, TOPIC_HINTS):    return "topics"
        return "general"

    def _build_pdf_store(self):
        print(f"🔧 Building PDF FAISS in: {self.db_path_pdf.resolve()}")
        docs: List[Document] = []
        self.pdf_meta_index = []

        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=140)
        pdfs = sorted(self.pdf_folder.glob("*.pdf"))
        if not pdfs:
            print("⚠️ No PDFs found. Skipping PDF FAISS.")
            return

        for pdf in pdfs:
            loader = PyPDFLoader(str(pdf))
            pages = loader.load()

            filename_code = self._infer_code_from_filename(pdf.name)
            first_text = "\n".join([p.page_content for p in pages[:2]]) if pages else ""
            text_code = self._infer_code_from_text(first_text)
            course_code = filename_code or text_code

            chunks = splitter.split_documents(pages)
            for ch in chunks:
                ch.metadata = ch.metadata or {}
                ch.metadata["source"] = pdf.name
                ch.metadata["section"] = self._tag_section(ch.page_content)
                if course_code:
                    ch.metadata["course_code"] = course_code
                docs.append(ch)

            self.pdf_meta_index.append({
                "file": pdf.name,
                "detected_code": course_code,
                "has_code_from_filename": bool(filename_code),
                "has_code_from_text": bool(text_code),
                "pages": len(pages)
            })

        self._pdf_all_docs = docs  # keep for metadata fallback
        self.pdf_store = FAISS.from_documents(docs, self.embeddings)
        self.db_path_pdf.mkdir(parents=True, exist_ok=True)
        self.pdf_store.save_local(str(self.db_path_pdf))
        print(f"✅ PDF FAISS built with {len(docs)} chunks across {len(pdfs)} files.")

    def debug_pdf_index(self) -> List[dict]:
        return getattr(self, "pdf_meta_index", [])

    def retrieve_pdf_context(
        self,
        course_code: str,
        question: str = "",
        prefer_description: bool = True,
        prefer_sections: tuple = (),
        k: int = 10,
        max_chars: int = 1600
    ) -> str:
        if self.pdf_store is None:
            return "(no PDFs indexed)"

        code = (course_code or "").upper().replace(" ", "")
        q = (question or "").strip()

        # Expand query to bias toward useful sections
        if prefer_description:
            q = f"{code} description overview aim synopsis learning outcomes content topics workload hours {q}".strip()
        else:
            q = f"{code} {q}".strip()

        results = self.pdf_store.similarity_search_with_score(q, k=max(30, k * 3))
        candidates = [d for d, _ in results]

        # Fallback: if similarity returns nothing, pull by metadata/filename match
        if not candidates:
            all_docs = getattr(self, "_pdf_all_docs", [])
            candidates = [d for d in all_docs
                          if (d.metadata.get("course_code") or "").upper() == code
                          or code.lower() in (d.metadata.get("source") or "").lower()]

        # Strong filter to the right course
        strict = [d for d in candidates if (d.metadata.get("course_code") or "").upper() == code]
        if not strict:
            strict = [d for d in candidates if code.lower() in (d.metadata.get("source") or "").lower()]
        pool = strict if strict else candidates

        # Section boosting
        if not prefer_sections and prefer_description:
            prefer_sections = ("description", "workload", "topics", "materials", "assessment")

        if prefer_sections:
            order = {sec: i for i, sec in enumerate(prefer_sections)}
            def sec_key(d):
                sec = d.metadata.get("section") or "general"
                return (0 if sec in order else 1, order.get(sec, 999))
            pool = sorted(pool, key=sec_key)

        # Build compact context
        used, out, size = set(), [], 0
        for d in pool:
            src = d.metadata.get("source")
            key = (src, hash(d.page_content))
            frag = (d.page_content or "").strip()
            if not frag or key in used:
                continue
            piece = frag + f"\n[Source: {src}]\n"
            if size + len(piece) > max_chars:
                break
            out.append(piece)
            size += len(piece)
            used.add(key)
            if len(out) >= k:
                break

        return "\n---\n".join(out) if out else "(no high-confidence matches)\n"

    def summarize_course_pdf(self, course_code: str, question: str = "", max_chars: int = 1600) -> str:
        return self.retrieve_pdf_context(
            course_code=course_code,
            question=question,
            prefer_description=True,
            prefer_sections=("description", "workload", "topics", "materials", "assessment"),
            k=12,
            max_chars=max_chars
        )


# =============================================================================
# Flask app
# =============================================================================
app = Flask(__name__)
rag = MultiRAG(
    jsonl_folder="docs",
    db_path_jsonl="faiss_index_jsonl_v1",
    pdf_folder="docs_pdf",
    db_path_pdf="faiss_index_pdfs_v1"
)

# =========================
# Helpers
# =========================
def clean_ollama_response(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()

def extract_last_paragraph(text: str) -> str:
    paragraphs = [p.strip() for p in (text or "").split('\n') if p.strip()]
    return paragraphs[-1] if paragraphs else (text or "").strip()

def detect_major(text: str) -> Optional[str]:
    t = (text or "").lower()
    for m in MAJORS:
        if m.lower() in t:
            return m
    return None

def detect_minor(text: str) -> Optional[str]:
    """Return canonical minor key (e.g. 'AI') if any alias is present."""
    t = (text or "").lower()
    for canon, aliases in MINOR_ALIASES.items():
        for a in aliases:
            if a in t:
                return canon
    return None

SMALL_TALK = {
    "hello": "Hi there! How can I help you with your study plan?",
    "hi": "Hello! How can I assist you today?",
    "hey": "Hey! Looking for a study plan or need help?",
    "thanks": "You're welcome!",
    "thank you": "Happy to help!",
}
def match_small_talk(input_text: str):
    if not input_text:
        return None
    matches = get_close_matches(input_text.lower(), SMALL_TALK.keys(), n=1, cutoff=0.8)
    return SMALL_TALK[matches[0]] if matches else None

# --- intent gates ---
def is_course_related(text: str) -> bool:
    t = text or ""

    # allow "<major> year N" OR "<minor> year N"
    if (detect_major(t) or detect_minor(t)) and re.search(r"\byear\s*[1-3]\b", t, re.I):
        return True

    # accept explicit "minor" queries
    if "minor" in t and (detect_minor(t) or re.search(r"\bai\b|\bartificial intelligence\b|\ba\.i\b", t, re.I)):
        return True

    # accept course codes or course-ish keywords
    return bool(
        re.search(rf"\b{COURSE_PREFIXES}\s*\d{{3}}\b", t, re.I) or
        re.search(r"\bcourse|paper|prereq|prerequisite|semester|major|minor|credits?|points?|catalog|study\s*plan|course\s*list|list\s+courses", t, re.I)
    )

CATALOG_INTENT_PATTERNS = (
    r"\bpre[-\s]?req", r"\bprereq", r"\bprerequisite",
    r"\bmajor(s)?\b", r"\bminor(s)?\b",
    r"\bsemester(s)?\b", r"\bwhen\b.*\boffer(ed|s)\b",
    r"\byear\s*[1-3]\b", r"\bcore\b", r"\bpoints?\b",
    r"\bcounts?\s+for\b|\bavailable\s+to\b|\bwhich\s+major\b",
    r"\bcode\b|\bcourse\s+list\b|\bcatalog\b"
)
def wants_catalog_facts(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in CATALOG_INTENT_PATTERNS)

def wants_course_list(text: str) -> bool:
    t = (text or "").lower()
    if re.search(r"study\s*plan|course\s*list|list\s+courses", t):
        return True
    # Implicit: a known major/minor + a year mention
    return (detect_major(t) or detect_minor(t)) and bool(re.search(r"\byear\s*[1-3]\b", t))

def extract_course_code(text: str) -> Optional[str]:
    m = re.search(rf"\b{COURSE_PREFIXES}\s*([0-9]{{3}})\b", text or "", re.I)
    return (m.group(1).upper() + m.group(2)) if m else None

def format_prereqs(rec: dict) -> str:
    txt = rec.get("prereq_text")
    if txt:
        return txt
    arr = rec.get("prerequisites") or []
    return "None" if not arr else ", ".join(arr)

def answer_from_catalog(query: str) -> Optional[str]:
    code = extract_course_code(query)
    if not code:
        return None
    rec = rag.get_course_by_code(code)
    if not rec:
        return f"I couldn't find {code} in the catalog."

    t = query.lower()
    lines = [f"{code} — {rec.get('title','')}"]

    if "prereq" in t or "pre req" in t:
        lines.append(f"Prerequisites: {format_prereqs(rec)}")

    if "semester" in t or "when" in t or "offer" in t:
        sems = rec.get("semesters") or []
        lines.append("Semesters: " + (", ".join(sems) if sems else "—"))

    if "major" in t or "minor" in t or "count" in t:
        majors = rec.get("majors") or []
        lines.append("Counts toward majors: " + (", ".join(majors) if majors else "—"))

    if re.search(r"\byear\b", t):
        lines.append(f"Year: {rec.get('year', '—')}")

    if "core" in t:
        lines.append(f"Core type: {rec.get('core_type','Major/Elective')}")

    if len(lines) == 1:  # default fact bundle (no prereqs unless asked)
        sems = ", ".join(rec.get("semesters") or []) or "—"
        lines += [
            f"Year: {rec.get('year','—')}",
            f"Semesters: {sems}",
            "Majors: " + (", ".join(rec.get("majors") or []) or "—"),
        ]

    if rec.get("note"):
        lines.append(f"Note: {rec['note']}")
    return "\n".join(lines)

def build_study_plan_from_json(user_input: str) -> str:
    # Detect year and whether it's a major or minor request
    ymatch = re.search(r"\byear\s*([1-3])\b", (user_input or "").lower())
    yfilter = int(ymatch.group(1)) if ymatch else None

    minor_key = detect_minor(user_input)
    major_name = detect_major(user_input)

    # Choose filter priority: explicit minor beats major
    using_minor = bool(minor_key)
    using_major = not using_minor and bool(major_name)

    if using_minor:
        # filter by minor (canonical key)
        recs = [r for r in rag.catalog if minor_key in (r.get("minors") or [])]
        header_name = f"{MINOR_DISPLAY.get(minor_key, minor_key)} Minor"
    elif using_major:
        # filter by major (name match)
        mlow = major_name.lower()
        recs = [r for r in rag.catalog if any((str(x) or "").lower() == mlow for x in r.get("majors", []))]
        header_name = major_name
    else:
        # default: treat as generic course list
        recs = rag.catalog[:]
        header_name = "Courses"

    if yfilter is not None:
        recs = [r for r in recs if r.get("year") == yfilter]

    if not recs:
        who = header_name + (f" (Year {yfilter})" if yfilter else "")
        return f"Sorry, I couldn’t find courses for {who}."

    # Aggregate semesters per course (each course once)
    sem_order = {"S1": 1, "S2": 2, "SS": 3, "—": 99}
    courses = {}  # code -> {title, sems:set[str]}
    for r in recs:
        code = (r.get("code") or "").upper()
        title = r.get("title") or ""
        sems = r.get("semesters") or []
        if not sems:
            sems = ["—"]
        entry = courses.setdefault(code, {"title": title, "sems": set()})
        for s in sems:
            entry["sems"].add(s)

    # Build output
    header = f"📘 Study Plan for {header_name}" + (f" (Year {yfilter})" if yfilter else "")
    lines = [header, ""]
    for code in sorted(courses.keys()):
        title = courses[code]["title"]
        sems_sorted = sorted(courses[code]["sems"], key=lambda s: sem_order.get(s, 50))
        sem_str = f" ({', '.join(sems_sorted)})" if sems_sorted else ""
        lines.append(f"• {code}: {title}{sem_str}")

    return "\n".join(lines).strip()

# ---- hours/credits extraction for controlled summaries ----
def _extract_hours_credits(text: str) -> dict:
    info = {}
    h = re.search(r"\b(\d{2,3})\s*(?:total\s+)?(?:learning\s+)?hours\b", text, re.I)
    if not h:
        h = re.search(r"\blearning\s+hours\s*[:\-]?\s*(\d{2,3})\b", text, re.I)
    if h:
        info["hours"] = h.group(1)

    c = re.search(r"\b(\d{1,2})\s*-\s*point\b|\b(\d{1,2})\s*points\b|\b(\d{1,2})\s*credits?\b", text, re.I)
    if c:
        credits = next((g for g in c.groups() if g), None)
        if credits:
            info["credits"] = credits
    return info

def summarize_pdf_with_ai(course_code: str, user_question: str) -> str:
    """
    Build PDF context then ask Ollama for ONE compact paragraph:
    - include: course code and title
    - include (if found): credits and/or learning hours
    - include: core content/topics and learning outcomes
    - exclude: dates/availability/prereqs/assessment breakdowns
    """
    ctx = rag.summarize_course_pdf(course_code=course_code, question=user_question, max_chars=1600)
    if not ctx or ctx.startswith("(no") or "no high-confidence" in ctx.lower():
        return f"I couldn’t find detailed PDF info for {course_code}. Make sure its PDF is in docs_pdf/ and restart to rebuild the index."

    rec = rag.get_course_by_code(course_code) or {}
    code_title = f"{course_code} — {rec.get('title','')}".strip(" —")

    facts = _extract_hours_credits(ctx)
    credits_line = f"{facts.get('credits')} credits" if facts.get("credits") else ""
    hours_line = f"~{facts.get('hours')} learning hours" if facts.get("hours") else ""
    fact_lines = [f"Course: {code_title}"]
    if credits_line:
        fact_lines.append(f"Credits: {credits_line}")
    if hours_line:
        fact_lines.append(f"Hours: {hours_line}")
    facts_block = "\n".join(fact_lines)

    prompt = f"""You are a precise editor.
Using ONLY the context and the Facts below, write ONE compact paragraph (2–3 sentences, ≤ 90 words).
Include: the course code and title; include the credits and/or learning hours ONLY if they appear in Facts.
Summarize the main content/topics and the learning outcomes.
Do NOT include dates, availability/semesters, prerequisites/corequisites, or assessment breakdowns. No bullet points.

Facts:
{facts_block}

Context:
{ctx}

Answer:"""

    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=120
        )
        if not resp.ok:
            return "LLM call failed while summarizing the PDF."
        full = resp.json()
        raw = (full.get("message") or {}).get("content", "") or ""
        paragraph = extract_last_paragraph(clean_ollama_response(raw))
        return paragraph or "I couldn't summarise the course from the document."
    except Exception as e:
        print("LLM error (PDF summary):", str(e))
        return "An error occurred connecting to the local model."


# =========================
# Flask routes
# =========================
@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return {"ok": True}

# Optional: debug the PDF index
@app.get("/debug/pdfs")
def debug_pdfs():
    return jsonify(rag.debug_pdf_index())


@app.post("/chat")
def chat():
    user_input = request.json.get('message', '').strip()
    print("💬 User:", user_input)

    if not user_input:
        return jsonify({"response": "No message received."}), 400

    # 0) Small talk
    small = match_small_talk(user_input)
    if small:
        return jsonify({"response": small})

    # 1) Hard gate for non-course queries (allows major/minor + year)
    if not is_course_related(user_input):
        return jsonify({"response": "Sorry I didn’t understand that."})

    # 2) Course list / study plan (major OR minor + year)
    if wants_course_list(user_input):
        return jsonify({"response": build_study_plan_from_json(user_input)})

    # 3) Catalog facts (JSON: prereqs, semesters, majors, core, year)
    if wants_catalog_facts(user_input):
        fact = answer_from_catalog(user_input)
        if fact:
            return jsonify({"response": fact})

    # 4) Course PDF summary (AI)
    code = extract_course_code(user_input)
    if code:
        return jsonify({"response": summarize_pdf_with_ai(code, user_input)})

    # 5) Last resort: JSONL similarity
    ctx = rag.retrieve_relevant_context(user_input)
    if not ctx or ctx.startswith("(no"):
        return jsonify({"response": "Sorry, I didn’t find enough info to answer that."})

    prompt = f"""Use ONLY the context to answer concisely.
Context:
{ctx}

Question: {user_input}
Answer:"""
    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=120
        )
        if not resp.ok:
            return jsonify({"response": "LLM call failed (non-200). Please try again."}), 502
        raw = (resp.json().get("message") or {}).get("content", "") or ""
        return jsonify({"response": extract_last_paragraph(clean_ollama_response(raw)) or "I couldn't find a clear answer in the documents."})
    except Exception as e:
        print("LLM error:", str(e))
        return jsonify({"response": "An error occurred connecting to Ollama."}), 502


# =========================
# Runner
# =========================
if __name__ == "__main__":
    print(f"Starting Flask on http://{HOST}:{PORT}  (debug={DEBUG}) | JSONL=docs/ | PDFs=docs_pdf/")
    app.run(host=HOST, port=PORT, debug=DEBUG)
