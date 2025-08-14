import os
import json   # ✅ ADDED: for saving/loading manifest
import hashlib  # ✅ ADDED: for hashing file contents
from pathlib import Path  # ✅ ADDED: easier path handling
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

class MultiRAG:
    def __init__(self, pdf_folder="docs", db_path="faiss_index_pdf"):
        # ✅ CHANGED: use Path instead of strings for easier file ops
        self.pdf_folder = Path(pdf_folder)
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)  # ✅ ADDED: ensure folder exists

        # ✅ ADDED: explicit file paths for FAISS and manifest tracking
        self.index_file = self.db_path / "index.faiss"
        self.store_file = self.db_path / "index.pkl"
        self.manifest_file = self.db_path / "manifest.json"

        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self._build_or_update_vectorstore()  # ✅ CHANGED: now handles smart updating

    # ---------- Helpers ----------
    def _hash_file(self, path: Path, chunk_size=1024 * 1024) -> str:
        """✅ ADDED: Create MD5 hash to detect file changes."""
        h = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def _scan_pdfs(self):
        """✅ ADDED: Scan PDFs and return name + hash + mtime."""
        out = {}
        for p in sorted(self.pdf_folder.glob("*.pdf")):
            out[p.name] = {
                "hash": self._hash_file(p),
                "mtime": int(p.stat().st_mtime),
            }
        return out

    def _load_manifest(self):
        """✅ ADDED: Load saved PDF manifest."""
        if self.manifest_file.exists():
            try:
                return json.loads(self.manifest_file.read_text())
            except Exception:
                return {}
        return {}

    def _save_manifest(self, manifest: dict):
        """✅ ADDED: Save current PDF manifest."""
        self.manifest_file.write_text(json.dumps(manifest, indent=2))

    def _faiss_exists(self) -> bool:
        """✅ ADDED: Check if FAISS index files exist."""
        return self.index_file.exists() and self.store_file.exists()

    # ---------- Core ----------
    def _build_or_update_vectorstore(self):
        """✅ CHANGED: Decides whether to rebuild or update FAISS."""
        current = self._scan_pdfs()
        previous = self._load_manifest()

        prev_files = set(previous.keys())
        cur_files  = set(current.keys())

        # ✅ ADDED: Detect new, removed, or modified PDFs
        new_files = sorted(cur_files - prev_files)
        removed_files = sorted(prev_files - cur_files)
        modified_files = sorted(
            f for f in (cur_files & prev_files)
            if current[f]["hash"] != previous[f]["hash"]
        )

        # ✅ ADDED: If any file removed/modified or index missing → full rebuild
        need_full_rebuild = (
            not self._faiss_exists() or
            len(modified_files) > 0 or
            len(removed_files) > 0 or
            (not previous and len(current) > 0 and not self._faiss_exists())
        )

        if need_full_rebuild:
            print("🔄 Full rebuild of FAISS index (new install or files changed/removed).")
            self._build_full_index(current)
        elif len(new_files) > 0:
            print(f"➕ Incremental add for {len(new_files)} new file(s): {', '.join(new_files)}")
            self._incremental_add(new_files, current)
        else:
            print("📂 No changes detected. Loading existing FAISS index...")
            self.vectorstore = FAISS.load_local(
                str(self.db_path), self.embeddings, allow_dangerous_deserialization=True
            )

    def _build_full_index(self, manifest_now: dict):
        """✅ CHANGED: Rebuilds FAISS index completely."""
        documents = []
        print("📁 Loading PDFs for full rebuild...")
        for fname in sorted(manifest_now.keys()):
            path = self.pdf_folder / fname
            loader = PyPDFLoader(str(path))
            documents.extend(loader.load())

        print(f"📄 {len(documents)} pages loaded. Splitting and embedding...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(str(self.db_path))
        self._save_manifest(manifest_now)  # ✅ ADDED: Save manifest after rebuild
        print("✅ Full FAISS index rebuilt and manifest updated.")

    def _incremental_add(self, new_files, manifest_now: dict):
        """✅ ADDED: Adds only new PDFs to existing FAISS index."""
        self.vectorstore = FAISS.load_local(
            str(self.db_path), self.embeddings, allow_dangerous_deserialization=True
        )

        new_docs = []
        for fname in new_files:
            path = self.pdf_folder / fname
            loader = PyPDFLoader(str(path))
            new_docs.extend(loader.load())

        print(f"📄 {len(new_docs)} pages from new PDFs. Splitting and embedding...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        new_chunks = splitter.split_documents(new_docs)

        self.vectorstore.add_documents(new_chunks)
        self.vectorstore.save_local(str(self.db_path))

        # ✅ ADDED: Update manifest with new files
        prev = self._load_manifest()
        prev.update({f: manifest_now[f] for f in new_files})
        self._save_manifest(prev)
        print("✅ Added new files to FAISS index and updated manifest.")

    # ---------- Public ----------
    def retrieve_relevant_context(self, query, k=20):
        # (No change — same as before)
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n".join(doc.page_content for doc in docs)
