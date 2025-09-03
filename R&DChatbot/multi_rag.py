import os, json, re
from pathlib import Path
from typing import List, Optional

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

MAJORS = [
    "Software Development",
    "Computer Science",
    "Data Science",
    "Networks & Cybersecurity",
    "Digital Services",
]

COURSE_CODE_RE = re.compile(r"\b(?:COMP|MATH|STAT|INFS|ENEL|ENSE)\s*\d{3}\b", re.I)

# -------- Section Hints for PDF chunk tagging --------
DESC_HINTS = (
    "course description", "description", "paper description",
    "overview", "synopsis", "aim", "course aim", "paper aim", "purpose"
)
WORKLOAD_HINTS = (
    "learning hours", "workload", "weekly hours", "total hours",
    "prescribed learning hours", "time commitment"
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
      - JSONL catalog (deterministic: lists, prereqs, semesters)
      - PDF store (per-course summaries/workload/etc., paraphrased)
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

        # PDF index (optional; only if PDFs exist)
        self.pdf_store = None
        self.pdf_meta_index = []
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
                        # Normalize + keep extras
                        r["code"] = code
                        if "prerequisites" not in r:
                            r["prerequisites"] = r.get("prereqs", []) or []
                        if "semesters" not in r:
                            r["semesters"] = r.get("semester", []) or []
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
            content = (
                f"{r.get('code','')} — {r.get('title','')}\n"
                f"Year: {r.get('year','')}  •  Semesters: {', '.join(sems) if sems else '—'}\n"
                f"Prerequisites: {', '.join(prereqs) if prereqs else 'None listed'}\n"
                f"Core: {r.get('core_type','Major/Elective')}\n"
                f"Majors: {', '.join(r.get('majors', []))}\n"
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
                       core_type: Optional[str] = None) -> list:
        recs = self.catalog
        if major:
            mlow = major.lower()
            recs = [r for r in recs if any(str(x).lower() == mlow for x in r.get("majors", []))]
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
            by_major = [d for d in docs if any(str(m).lower() == mlow for m in d.metadata.get("majors", []))]
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
                docs.extend([ch])

            self.pdf_meta_index.append({
                "file": pdf.name,
                "detected_code": course_code,
                "has_code_from_filename": bool(filename_code),
                "has_code_from_text": bool(text_code),
                "pages": len(pages)
            })

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
            q = f"{code} description overview aim synopsis {q}".strip()
        else:
            q = f"{code} {q}".strip()

        results = self.pdf_store.similarity_search_with_score(q, k=max(30, k * 3))
        candidates = [d for d, _ in results]

        # Strong filter to the right course
        strict = [d for d in candidates if (d.metadata.get("course_code") or "").upper() == code]
        if not strict:
            strict = [d for d in candidates if code.lower() in (d.metadata.get("source") or "").lower()]
        pool = strict if strict else candidates

        # Section boosting
        if not prefer_sections and prefer_description:
            prefer_sections = ("description", "workload", "assessment", "materials")

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
            prefer_sections=("description", "workload", "assessment", "materials", "topics"),
            k=12,
            max_chars=max_chars
        )
