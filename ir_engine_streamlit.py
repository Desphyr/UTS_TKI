import json
import logging
import math
import re
import os
import atexit
import sqlite3

from collections import defaultdict
from typing import Dict, List, Tuple, Set

import streamlit as st
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    import seaborn as sns
    import matplotlib.pyplot as plt
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False

# ==================== KONFIGURASI ====================
INDEX_DB_FILE = "ir_index.sqlite"

# Legacy JSON files (migration only)
DOCUMENTS_FILE = "documents.json"
INVERTED_INDEX_FILE = "inverted_index.json"
DOC_VECTORS_FILE = "doc_vectors.json"

MAX_CACHED_DOCUMENTS = 50  # Maksimal dokumen yang disimpan


INDONESIAN_STOPWORDS = {
    'yang', 'dan', 'di', 'dari', 'ke', 'adalah', 'ini', 'itu', 'atau', 'tidak',
    'untuk', 'dengan', 'pada', 'oleh', 'telah', 'akan', 'sudah', 'juga', 'dapat',
    'dalam', 'ada', 'karena', 'namun', 'hanya', 'seperti', 'saat', 'ketika',
    'lalu', 'maka', 'jadi', 'sebagai', 'setelah', 'sebelum', 'selama', 'sejak',
    'sampai', 'hingga', 'melalui', 'terhadap', 'tanpa', 'bagian', 'selain',
    'daripada', 'a', 'an', 'the', 'to', 'at', 'in', 'on', 'by', 'as', 'of', 'is'
}


# ==================== PREPROCESSING ====================
def case_folding(text: str) -> str:
    return text.lower()

def remove_punctuation(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def stopword_removal(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in INDONESIAN_STOPWORDS]

def stemming(tokens: List[str]) -> List[str]:
    if not SASTRAWI_AVAILABLE:
        return tokens
    try:
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return [stemmer.stem(t) for t in tokens]
    except Exception as e:
        logging.warning("Error stemming: %s", e)
        return tokens

def preprocess_text(text: str) -> List[str]:
    text = case_folding(text)
    text = remove_punctuation(text)
    tokens = text.split()
    tokens = stopword_removal(tokens)
    tokens = stemming(tokens)
    return [t for t in tokens if t]


# ==================== INVERTED INDEX ====================
class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.documents: Dict[int, str] = {}
        self.doc_vectors: Dict[int, Dict[str, float]] = {}
        self.vocabulary: Set[str] = set()
        self.doc_id_counter = 0

    def add_document(self, doc_id: int, doc_content: str) -> None:
        # Enforce limit maksimal dokumen (FIFO - buang dokumen tertua)
        if len(self.documents) >= MAX_CACHED_DOCUMENTS and doc_id not in self.documents:
            oldest_id = min(self.documents.keys())
            del self.documents[oldest_id]
            for term in list(self.index.keys()):
                if oldest_id in self.index[term]:
                    del self.index[term][oldest_id]
                if not self.index[term]:
                    del self.index[term]
            if oldest_id in self.doc_vectors:
                del self.doc_vectors[oldest_id]
        
        self.documents[doc_id] = doc_content
        tokens = preprocess_text(doc_content)
        tf_dict = defaultdict(int)
        for token in tokens:
            tf_dict[token] += 1
        for token, freq in tf_dict.items():
            self.index[token][doc_id] = freq
            self.vocabulary.add(token)

    def calculate_idf(self) -> Dict[str, float]:
        N = len(self.documents)
        idf = {}
        for term in self.vocabulary:
            df = len(self.index[term])
            idf[term] = math.log10(N / df) if df > 0 else 0
        return idf

    def calculate_tfidf_vectors(self) -> Dict[int, Dict[str, float]]:
        idf = self.calculate_idf()
        self.doc_vectors = {}
        for doc_id, doc_content in self.documents.items():
            tokens = preprocess_text(doc_content)
            tf_dict = defaultdict(int)
            for token in tokens:
                tf_dict[token] += 1
            doc_vector = {}
            for term in self.vocabulary:
                if term in tf_dict:
                    tf = 1 + math.log10(tf_dict[term])
                    doc_vector[term] = tf * idf[term]
                else:
                    doc_vector[term] = 0
            self.doc_vectors[doc_id] = doc_vector
        return self.doc_vectors

    def get_inverted_index_dict(self) -> Dict:
        return {term: dict(docs) for term, docs in self.index.items()}

    def _init_db(self, conn: sqlite3.Connection) -> None:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY,
                content TEXT NOT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS postings (
                term TEXT NOT NULL,
                doc_id INTEGER NOT NULL,
                tf INTEGER NOT NULL,
                PRIMARY KEY (term, doc_id)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                doc_id INTEGER NOT NULL,
                term TEXT NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY (doc_id, term)
            );
        """)
        conn.commit()

    def _maybe_migrate_legacy_json_once(self, conn: sqlite3.Connection) -> None:
        """Jika DB belum ada tapi JSON legacy ada, import sekali."""
        if os.path.exists(INDEX_DB_FILE):
            return
        if not os.path.exists(DOCUMENTS_FILE):
            return

        self._init_db(conn)

        with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
            self.documents = {int(k): v for k, v in json.load(f).items()}

        if os.path.exists(INVERTED_INDEX_FILE):
            with open(INVERTED_INDEX_FILE, 'r', encoding='utf-8') as f:
                index_dict = json.load(f)
            self.index = defaultdict(lambda: defaultdict(int))
            self.vocabulary = set()
            for term, docs in index_dict.items():
                for k, v in docs.items():
                    doc_id = int(k)
                    self.index[term][doc_id] = int(v)
                self.vocabulary.add(term)
        else:
            self.index = defaultdict(lambda: defaultdict(int))
            self.vocabulary = set()
            for doc_id, doc_content in self.documents.items():
                self.add_document(doc_id, doc_content)

        if os.path.exists(DOC_VECTORS_FILE):
            with open(DOC_VECTORS_FILE, 'r', encoding='utf-8') as f:
                vectors = json.load(f)
            self.doc_vectors = {int(k): v for k, v in vectors.items()}
        else:
            self.calculate_tfidf_vectors()

        self.save_to_disk()

    def save_to_disk(self):
        """Simpan ke disk menggunakan SQLite (tanpa JSON runtime)."""
        conn = sqlite3.connect(INDEX_DB_FILE)
        try:
            self._init_db(conn)
            cur = conn.cursor()
            cur.execute("DELETE FROM documents")
            cur.execute("DELETE FROM postings")
            cur.execute("DELETE FROM vectors")

            cur.executemany(
                "INSERT INTO documents (doc_id, content) VALUES (?, ?)",
                [(int(doc_id), content) for doc_id, content in self.documents.items()],
            )

            postings_rows = []
            for term, docs in self.index.items():
                for doc_id, tf in docs.items():
                    postings_rows.append((term, int(doc_id), int(tf)))
            if postings_rows:
                cur.executemany(
                    "INSERT INTO postings (term, doc_id, tf) VALUES (?, ?, ?)",
                    postings_rows,
                )

            vectors_rows = []
            for doc_id, vec in self.doc_vectors.items():
                for term, value in vec.items():
                    vectors_rows.append((int(doc_id), term, float(value)))
            if vectors_rows:
                cur.executemany(
                    "INSERT INTO vectors (doc_id, term, value) VALUES (?, ?, ?)",
                    vectors_rows,
                )

            conn.commit()
        finally:
            conn.close()


    def clear_index(self):
        """Kosongkan semua dokumen dan index"""
        self.documents.clear()
        self.index.clear()
        self.doc_vectors.clear()
        self.vocabulary.clear()
        self.doc_id_counter = 0

    def load_from_disk(self):
        """Load dari disk menggunakan SQLite.

        Migration: jika DB belum ada tapi JSON legacy ada, lakukan import sekali.
        Setelah itu aplikasi tidak lagi bergantung pada JSON.
        """
        db_exists = os.path.exists(INDEX_DB_FILE)
        conn = sqlite3.connect(INDEX_DB_FILE)
        try:
            self._init_db(conn)

            if not db_exists:
                self._maybe_migrate_legacy_json_once(conn)

            cur = conn.cursor()
            cur.execute("SELECT doc_id, content FROM documents")
            rows = cur.fetchall()
            if not rows:
                return

            all_docs = {int(doc_id): content for doc_id, content in rows}

            # enforce max cached docs (FIFO): ambil 50 dokumen terakhir
            if len(all_docs) > MAX_CACHED_DOCUMENTS:
                sorted_ids = sorted(all_docs.keys(), reverse=True)[:MAX_CACHED_DOCUMENTS]
                self.documents = {doc_id: all_docs[doc_id] for doc_id in sorted_ids}
            else:
                self.documents = all_docs

            self.index = defaultdict(lambda: defaultdict(int))
            self.vocabulary = set()
            self.doc_vectors = {}

            # postings filter by loaded docs
            loaded_doc_ids = set(self.documents.keys())
            cur.execute("SELECT term, doc_id, tf FROM postings")
            postings_rows = cur.fetchall()
            for term, doc_id, tf in postings_rows:
                doc_id = int(doc_id)
                if doc_id in loaded_doc_ids:
                    self.index[term][doc_id] = int(tf)
                    self.vocabulary.add(term)

            # vectors filter by loaded docs
            cur.execute("SELECT doc_id, term, value FROM vectors")
            vec_rows = cur.fetchall()
            for doc_id, term, value in vec_rows:
                doc_id = int(doc_id)
                if doc_id in loaded_doc_ids:
                    self.doc_vectors.setdefault(doc_id, {})[term] = float(value)

            # Hitung ulang vectors jika kosong
            if self.documents and (not self.doc_vectors):
                self.calculate_tfidf_vectors()

            # Persist agar DB konsisten dengan max cache
            if len(self.documents) != len(all_docs):
                self.save_to_disk()
        finally:
            conn.close()



# ==================== SIMILARITY ====================
def normalize_vector(vector: Dict[str, float]) -> Dict[str, float]:
    norm = math.sqrt(sum(v ** 2 for v in vector.values()))
    if norm == 0:
        return vector
    return {k: v / norm for k, v in vector.items()}

def cosine_similarity_custom(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    n1 = normalize_vector(vec1)
    n2 = normalize_vector(vec2)
    return sum(n1.get(t, 0) * n2.get(t, 0) for t in set(n1) | set(n2))

def retrieve_documents(query: str, ir_index: InvertedIndex, top_k: int = 10) -> List[Tuple[int, str, float]]:
    query_tokens = preprocess_text(query)
    if not query_tokens:
        return []
    
    # Hitung term frequency dan unique terms
    query_tf = defaultdict(int)
    for token in query_tokens:
        query_tf[token] += 1
    
    idf = ir_index.calculate_idf()
    query_vector = {}
    
    # Build query vector hanya untuk terms yang ada di query
    for term in query_tf:
        if term in ir_index.vocabulary:
            tf = 1 + math.log10(query_tf[term])
            query_vector[term] = tf * idf[term]
    
    # Normalisasi query vector
    query_vector = normalize_vector(query_vector)
    
    scores = []
    for doc_id, doc_vector in ir_index.doc_vectors.items():
        # Normalisasi doc vector
        normalized_doc = normalize_vector(doc_vector)
        # Hitung similarity hanya untuk terms yang ada
        sim = sum(query_vector.get(t, 0) * normalized_doc.get(t, 0) for t in query_vector)
        if sim > 0:  # Hanya include hasil dengan score > 0
            scores.append((doc_id, ir_index.documents[doc_id], sim))
    
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]


# ==================== SESSION STATE ====================
@st.cache_resource
def get_index():
    idx = InvertedIndex()
    idx.load_from_disk()
    # Pastikan vocabulary dan vectors ter-rebuild
    if idx.documents and not idx.vocabulary:
        for doc_id, doc_content in idx.documents.items():
            tokens = preprocess_text(doc_content)
            tf_dict = defaultdict(int)
            for token in tokens:
                tf_dict[token] += 1
            for token, freq in tf_dict.items():
                idx.index[token][doc_id] = freq
                idx.vocabulary.add(token)
    if idx.documents:
        idx.calculate_tfidf_vectors()
    return idx

# Clear cache on session end
def clear_cache_on_exit():
    try:
        ir_index = get_index()
        ir_index.documents.clear()
        ir_index.index.clear()
        ir_index.doc_vectors.clear()
        ir_index.vocabulary.clear()
        # Jangan hapus file persist normal (SQLite). Migration sudah menjaga runtime.
    except:
        pass



# ==================== STREAMLIT UI ====================
st.set_page_config(
    page_title="IR Engine — Vector Space Model",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Information Retrieval Engine")
st.caption("Vector Space Model · TF-IDF · Cosine Similarity · Bahasa Indonesia")

if not SASTRAWI_AVAILABLE:
    st.warning("⚠️ Sastrawi tidak ditemukan — stemming dinonaktifkan. Install: `pip install PySastrawi`")

ir_index = get_index()

# Register cleanup pada exit
atexit.register(clear_cache_on_exit)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("📊 Statistik Index")
    n_docs = len(ir_index.documents)
    n_vocab = len(ir_index.vocabulary)
    total_postings = sum(len(d) for d in ir_index.index.values())
    avg_terms = round(total_postings / n_docs, 1) if n_docs else 0

    col1, col2 = st.columns(2)
    col1.metric("Dokumen", n_docs)
    col2.metric("Vocabulary", n_vocab)
    col1.metric("Postings", total_postings)
    col2.metric("Avg terms/dok", avg_terms)

    st.divider()
    st.caption(f"💾 Cache: Max **{MAX_CACHED_DOCUMENTS}** dokumen/sesi")
    st.caption("IR Engine v1.0 · Streamlit UI")

# ==================== TABS ====================
tab_search, tab_add, tab_excel, tab_index, tab_heatmap = st.tabs([
    "🔍 Pencarian", "➕ Tambah Dokumen", "📂 Import Excel", "📋 Inverted Index", "🗺️ Heatmap"
])


# ==================== TAB: PENCARIAN ====================
with tab_search:
    st.subheader("Pencarian Dokumen")

    col_q, col_k = st.columns([4, 1])
    with col_q:
        query = st.text_input("Query", placeholder="Masukkan kata kunci pencarian…", label_visibility="collapsed")
    with col_k:
        top_k = st.number_input("Top-K", min_value=1, max_value=50, value=10, label_visibility="collapsed")

    if st.button("🔍 Cari", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Query tidak boleh kosong.")
        elif n_docs == 0:
            st.warning("Belum ada dokumen. Tambahkan dokumen dulu di tab **Tambah Dokumen**.")
        else:
            with st.spinner("Menghitung cosine similarity…"):
                results = retrieve_documents(query, ir_index, top_k=top_k)
                relevant = [(doc_id, content, score) for doc_id, content, score in results if score > 0]

            if not relevant:
                st.error("❌ Tidak ada dokumen relevan ditemukan.")
                
                # DEBUG: Tampilkan informasi debugging
                with st.expander("🔧 Debug Info"):
                    q_tokens = preprocess_text(query)
                    st.write(f"**Raw query:** `{query}`")
                    st.write(f"**Query tokens setelah preprocessing:** {q_tokens if q_tokens else '(kosong)'}")
                    
                    # Cek term matching
                    if q_tokens:
                        matching_terms = [t for t in q_tokens if t in ir_index.vocabulary]
                        st.write(f"**Term di vocabulary:** {matching_terms if matching_terms else '(tidak ada)'}")
                        st.write(f"**Vocabulary size:** {len(ir_index.vocabulary)}")
                        st.write(f"**Total dokumen:** {n_docs}")
                        
                        # PENTING: Tampilkan sample vocabulary untuk diagnostik
                        st.write("**📌 Sample vocabulary (20 terms pertama):**")
                        sample_vocab = sorted(ir_index.vocabulary)[:20]
                        st.write(f"`{' | '.join(sample_vocab)}`")
                        
                        # Tampilkan sample dokumen
                        if ir_index.documents:
                            first_doc_id = min(ir_index.documents.keys())
                            first_content = ir_index.documents[first_doc_id][:200]
                            st.write(f"**📄 Sample dokumen (No. {first_doc_id + 1}):**")
                            st.write(f"`{first_content}…`")
                        
                        # Tampilkan similarity scores untuk semua dokumen
                        all_results = retrieve_documents(query, ir_index, top_k=n_docs)
                        if all_results:
                            st.write("**Similarity scores (semua dokumen):**")
                            for doc_id, content, score in all_results[:10]:
                                preview = (content[:100] + "…") if len(content) > 100 else content
                                st.write(f"  - No. {doc_id + 1}: {score:.6f} | {preview}")
            else:
                st.success(f"Ditemukan **{len(relevant)}** dokumen relevan")

                # Token hasil preprocessing query
                q_tokens = preprocess_text(query)
                st.caption(f"Token query setelah preprocessing: `{' | '.join(q_tokens)}`")

                st.divider()
                for rank, (doc_id, content, score) in enumerate(relevant, 1):
                    # Debug: tampilkan term query yang ikut membentuk skor (agar jelas kenapa similarity bisa lebih besar)
                    # Ini membantu validasi khususnya kasus kata seperti "modern" hanya muncul 1 kali.
                    # Hanya tampil jika query lebih dari 0 term setelah preprocessing.

                    with st.container():
                        c1, c2 = st.columns([6, 1])
                        with c1:
                            st.markdown(f"**#{rank} — Dokumen {doc_id + 1}** (Similarity: {score:.4f})")
                            preview = content if len(content) <= 300 else content[:300] + "…"
                            
                            # Highlight query terms dalam preview (tanpa emote)
                            # Catatan: preview adalah teks mentah; agar konsisten dengan query yang sudah dipreprocess,
                            # kita tokenisasi preview lalu membandingkan token hasil preprocessing.
                            def highlight_preview(raw_text: str, query_tokens_pre: List[str]) -> str:
                                if not raw_text:
                                    return raw_text
                                if not query_tokens_pre:
                                    return raw_text

                                query_token_set = set(query_tokens_pre)

                                # split token dengan regex agar kata/non-kata tetap terbaca
                                # (gunakan \W+ yang benar, bukan \\W+ literal)
                                parts = re.split(r"(\W+)", raw_text)

                                out_parts = []
                                for p in parts:
                                    if not p:
                                        continue
                                    # hanya highlight untuk segmen yang berupa "kata" (mengandung huruf/angka)
                                    if re.fullmatch(r"\w+", p, flags=re.UNICODE):
                                        # preprocess token p dengan pipeline yang sama seperti query
                                        pp = preprocess_text(p)
                                        # preprocess_text bisa menghasilkan beberapa token; highlight bila ada yang match
                                        if any(t in query_token_set for t in pp):
                                            out_parts.append(f"**{p}**")
                                        else:
                                            out_parts.append(p)

                                    else:
                                        out_parts.append(p)
                                return "".join(out_parts)

                            # Highlight kata yang mirip dengan keyword query (tebal) — tanpa emote
                            st.markdown(highlight_preview(preview, q_tokens))

                            # Tambahan: jika dokumen hasil memiliki score sangat tinggi (mis. 1.0),
                            # tetap pastikan kata yang mirip ditampilkan tebal. (highlight_preview sudah konsisten)

                        with c2:
                            st.metric("Skor", f"{score:.4f}")
                        st.divider()


# ==================== TAB: TAMBAH DOKUMEN ====================
with tab_add:
    st.subheader("Tambah Dokumen ke Index")

    doc_content = st.text_area(
        "Isi dokumen",
        placeholder="Tempel atau ketik isi dokumen di sini…",
        height=150
    )

    if st.button("➕ Tambahkan ke Index", type="primary"):
        if not doc_content.strip():
            st.warning("Isi dokumen tidak boleh kosong.")
        else:
            doc_id = len(ir_index.documents)
            ir_index.add_document(doc_id, doc_content.strip())
            ir_index.calculate_tfidf_vectors()
            ir_index.save_to_disk()
            st.success(f"✅ Dokumen **{doc_id + 1}** berhasil ditambahkan!")
            tokens = preprocess_text(doc_content)
            st.caption(f"Token hasil preprocessing: `{' | '.join(tokens)}`")
            st.rerun()

    if ir_index.documents:
        st.divider()
        st.subheader("Daftar Dokumen Tersimpan")
        # Hanya tampilkan 50 dokumen yang aktif
        active_docs = dict(sorted(ir_index.documents.items()))
        rows = [{"No": doc_id + 1, "Preview": (content[:120] + "…") if len(content) > 120 else content}
                for doc_id, content in list(active_docs.items())[-50:]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Menampilkan {len(rows)} dari {len(ir_index.documents)} dokumen aktif")


# ==================== TAB: IMPORT EXCEL ====================
with tab_excel:
    st.subheader("Import Dokumen dari Excel")
    st.info("Upload file Excel hasil preprocessing. Bisa import dari beberapa sheet sekaligus dalam 1 file.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Kosongkan Index", type="secondary", use_container_width=True):
            ir_index.clear_index()
            ir_index.save_to_disk()
            st.success("✅ Index kosong! Semua dokumen dan postings dihapus.")
            st.rerun()
    with col2:
        st.write("")  # Spacing

    uploaded = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

    if uploaded:
        try:
            xl = pd.ExcelFile(uploaded)
            st.write(f"**{len(xl.sheet_names)} sheet** ditemukan: {', '.join(str(s) for s in xl.sheet_names)}")
            
            # Auto-detect yang sheet paling cocok untuk dokumen
            preferred_sheets = [s for s in xl.sheet_names if 'stemming' in s.lower() or 'asli' in s.lower()]
            default_sheet_list = preferred_sheets if preferred_sheets else xl.sheet_names
            
            # Multi-select sheets
            selected_sheets = st.multiselect(
                "Pilih sheet yang ingin di-import",
                xl.sheet_names,
                default=default_sheet_list if len(default_sheet_list) <= 5 else default_sheet_list[:1]
            )
            
            if selected_sheets:
                # Pilih kolom (sama untuk semua sheet)
                first_df = pd.read_excel(uploaded, sheet_name=selected_sheets[0], skiprows=1)
                col_options = first_df.columns.tolist()
                # Auto-detect kolom: cari "Teks Asli" atau "asli" atau "original"
                default_col = next((c for c in col_options if any(kw in c.lower() for kw in ['asli', 'original', 'teks'])), col_options[0] if col_options else None)
                selected_col = st.selectbox("Pilih kolom dokumen", col_options, index=col_options.index(default_col) if default_col else 0)
                
                # Preview
                all_docs = []
                for sheet_name in selected_sheets:
                    df = pd.read_excel(uploaded, sheet_name=sheet_name, skiprows=1)
                    if selected_col in df.columns:
                        docs = df[selected_col].dropna().astype(str).tolist()
                        all_docs.extend(docs)
                
                st.write(f"**Total {len(all_docs)} dokumen** dari {len(selected_sheets)} sheet. Preview 5 dokumen pertama:")
                st.dataframe(pd.DataFrame({"Dokumen": all_docs[:5]}), hide_index=True, use_container_width=True)
                
                if st.button("📥 Import Semua ke Index", type="primary"):
                    with st.spinner(f"Mengindex {len(all_docs)} dokumen dari {len(selected_sheets)} sheet…"):
                        # Clear index terlebih dahulu
                        ir_index.clear_index()
                        # Import dokumen baru (max 50)
                        # Pakai doc_id berurutan mulai dari 0 agar konsisten dengan urutan di dokumen.
                        docs_to_import = all_docs[:MAX_CACHED_DOCUMENTS]
                        for i, content in enumerate(docs_to_import):
                            clean = content.replace(' | ', ' ').strip()
                            ir_index.add_document(i, clean)

                        ir_index.calculate_tfidf_vectors()
                        ir_index.save_to_disk()
                    st.success(f"✅ {len(docs_to_import)} dari {len(all_docs)} dokumen berhasil diimport ke Index (max 50 per sesi)!")
                    st.rerun()
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")


# ==================== TAB: INVERTED INDEX ====================
with tab_index:
    st.subheader("Inverted Index")

    if not ir_index.vocabulary:
        st.info("Belum ada dokumen diindex.")
    else:
        search_term = st.text_input("🔎 Filter term", placeholder="Cari term tertentu…")
        terms = sorted(ir_index.vocabulary)
        if search_term:
            terms = [t for t in terms if search_term.lower() in t]

        st.caption(f"Menampilkan {len(terms)} dari {len(ir_index.vocabulary)} terms")

        rows = []
        for term in terms[:200]:
            postings = ir_index.index[term]
            rows.append({
                "Term": term,
                "DF (jumlah dokumen)": len(postings),
                "Posting List (No: tf)": ", ".join(f"No.{d+1}:{f}" for d, f in sorted(postings.items()))
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if len(terms) > 200:
            st.caption(f"… {len(terms) - 200} term lainnya tidak ditampilkan. Gunakan filter.")


# ==================== TAB: HEATMAP ====================
with tab_heatmap:
    st.subheader("Heatmap Cosine Similarity Antar Dokumen")

    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn tidak tersedia. Install: `pip install scikit-learn`")
    elif n_docs < 2:
        st.info("Butuh minimal 2 dokumen untuk menampilkan heatmap.")
    else:
        max_show = st.slider("Jumlah dokumen yang ditampilkan", min_value=2, max_value=min(50, n_docs), value=min(15, n_docs))

        if st.button("🗺️ Generate Heatmap", type="primary"):
            with st.spinner("Menghitung cosine similarity…"):
                doc_ids = sorted(ir_index.doc_vectors.keys())[:max_show]
                vectors = []
                for doc_id in doc_ids:
                    vec = ir_index.doc_vectors[doc_id]
                    all_terms = sorted(ir_index.vocabulary)
                    vectors.append([vec.get(t, 0) for t in all_terms])

                sim_matrix = []
                for v1 in vectors:
                    row = []
                    for v2 in vectors:
                        norm1 = math.sqrt(sum(x**2 for x in v1))
                        norm2 = math.sqrt(sum(x**2 for x in v2))
                        dot = sum(a * b for a, b in zip(v1, v2))
                        sim = dot / (norm1 * norm2) if norm1 and norm2 else 0
                        row.append(round(sim, 4))
                    sim_matrix.append(row)

                labels = [f"No.{i+1}" for i in doc_ids]
                df_sim = pd.DataFrame(sim_matrix, index=labels, columns=labels)

                fig, ax = plt.subplots(figsize=(max(8, max_show * 0.6), max(6, max_show * 0.5)))
                sns.heatmap(
                    df_sim, annot=True, fmt=".2f", cmap="Blues",
                    annot_kws={"size": max(7, 11 - max_show // 5)},
                    ax=ax
                )
                ax.set_title(f"Cosine Similarity Heatmap ({max_show} dokumen)", fontsize=13)
                ax.set_xlabel("Dokumen ID")
                ax.set_ylabel("Dokumen ID")
                plt.tight_layout()
                st.pyplot(fig)

                # Export
                csv = df_sim.to_csv().encode('utf-8')
                st.download_button("⬇️ Download CSV", csv, "similarity_matrix.csv", "text/csv")