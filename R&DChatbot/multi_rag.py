# ---------- multi_rag.py (RAG utilities) ----------
import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

class MultiRAG:
    def __init__(self, pdf_folder="docs", db_path="faiss_index_pdf"):
        self.pdf_folder = Path(pdf_folder)
        self.db_path = Path(db_path)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.build_or_load_vectorstore()

    def build_or_load_vectorstore(self):
        """
        Loads PDFs, extracts text with metadata (source/page), chunks, embeds, and builds FAISS.
        Rebuilds each run to keep behavior consistent with your original code.
        """
        print("📁 Loading PDFs...")
        documents = []
        pdfs = [p for p in self.pdf_folder.glob("*.pdf")]
        for pdf_path in pdfs:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            for i, doc in enumerate(pages):
                # Ensure metadata for citation
                doc.metadata = doc.metadata or {}
                doc.metadata["source"] = pdf_path.name
                # Some loaders already put 'page' in metadata; normalize as 1-based page index
                if "page" not in doc.metadata:
                    doc.metadata["page"] = i + 1
                else:
                    try:
                        # Make it 1-based if it looks 0-based
                        doc.metadata["page"] = int(doc.metadata["page"]) + 1
                    except Exception:
                        doc.metadata["page"] = i + 1
                documents.append(doc)

        print(f"📄 {len(documents)} pages loaded. Splitting and embedding...")

        # Slightly larger chunks improve coherence for course descriptors
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=160)
        chunks = splitter.split_documents(documents)

        # Keep metadata on chunks
        for c in chunks:
            c.metadata = c.metadata or {}
            c.metadata.setdefault("source", c.metadata.get("source", "unknown.pdf"))
            c.metadata.setdefault("page", c.metadata.get("page", "?"))

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.db_path))
        print("✅ FAISS index built from PDF files.")

    def _build_context(self, docs, max_chars=3200):
        """
        Deduplicate by (source,page) and build a compact, cited context.
        Each piece ends with a note like: [Source: file.pdf p.X]
        """
        used = set()
        out = []
        size = 0
        for d in docs:
            src = d.metadata.get("source")
            pg = d.metadata.get("page")
            key = (src, pg)
            if key in used:
                continue
            fragment = (d.page_content or "").strip()
            if not fragment:
                continue
            note = f"\n[Source: {src} p.{pg}]\n"
            piece = fragment + note
            if size + len(piece) > max_chars:
                break
            out.append(piece)
            size += len(piece)
            used.add(key)
        if not out:
            return "(no high-confidence matches)\n"
        return "\n---\n".join(out)

    def retrieve_relevant_context(self, query, k=14, max_chars=3200):
        """
        Uses similarity_search_with_score to rank chunks.
        FAISS typically returns LOWER distances for better matches.
        We keep a conservative distance threshold; tune as needed.
        """
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        # Filter by distance (lower is better). Start with 0.8 as a safe default threshold.
        # If you find good answers are being dropped, increase this to ~1.0.
        MAX_DISTANCE = 0.8
        filtered = []
        for doc, dist in results:
            try:
                # Keep good matches; print/debug if needed
                if dist <= MAX_DISTANCE:
                    filtered.append(doc)
            except Exception:
                # If score is weird, keep doc to avoid false negatives
                filtered.append(doc)

        # Fallback: if filtering removes everything, use the top original docs
        if not filtered:
            filtered = [d for d, _ in results]

        return self._build_context(filtered, max_chars=max_chars)
