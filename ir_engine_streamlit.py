import json
import logging
import math
import re
import os
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
DOCUMENTS_FILE = "documents.json"
INVERTED_INDEX_FILE = "inverted_index.json"
DOC_VECTORS_FILE = "doc_vectors.json"

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

    def add_document(self, doc_id: int, doc_content: str) -> None:
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

    def save_to_disk(self):
        with open(DOCUMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        with open(INVERTED_INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.get_inverted_index_dict(), f, ensure_ascii=False, indent=2)
        with open(DOC_VECTORS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.doc_vectors, f, ensure_ascii=False, indent=2)

    def load_from_disk(self):
        if not os.path.exists(DOCUMENTS_FILE):
            return
        with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
            self.documents = {int(k): v for k, v in json.load(f).items()}
        if os.path.exists(INVERTED_INDEX_FILE):
            with open(INVERTED_INDEX_FILE, 'r', encoding='utf-8') as f:
                index_dict = json.load(f)
                for term, docs in index_dict.items():
                    self.index[term] = defaultdict(int, {int(k): v for k, v in docs.items()})
                    self.vocabulary.add(term)
        else:
            for doc_id, doc_content in self.documents.items():
                self.add_document(doc_id, doc_content)
        if os.path.exists(DOC_VECTORS_FILE):
            with open(DOC_VECTORS_FILE, 'r', encoding='utf-8') as f:
                vectors = json.load(f)
                self.doc_vectors = {int(k): v for k, v in vectors.items()}
        else:
            if self.documents:
                self.calculate_tfidf_vectors()
        try:
            if not os.path.exists(INVERTED_INDEX_FILE) or not os.path.exists(DOC_VECTORS_FILE):
                self.save_to_disk()
        except Exception:
            pass


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
    query_tf = defaultdict(int)
    for token in query_tokens:
        query_tf[token] += 1
    idf = ir_index.calculate_idf()
    query_vector = {}
    for term in ir_index.vocabulary:
        if term in query_tf:
            tf = 1 + math.log10(query_tf[term])
            query_vector[term] = tf * idf[term]
        else:
            query_vector[term] = 0
    scores = []
    for doc_id, doc_vector in ir_index.doc_vectors.items():
        sim = cosine_similarity_custom(query_vector, doc_vector)
        scores.append((doc_id, ir_index.documents[doc_id], sim))
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]


# ==================== SESSION STATE ====================
@st.cache_resource
def get_index():
    idx = InvertedIndex()
    idx.load_from_disk()
    if idx.documents:
        idx.calculate_tfidf_vectors()
    return idx


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
                st.error("Tidak ada dokumen relevan ditemukan.")
            else:
                st.success(f"Ditemukan **{len(relevant)}** dokumen relevan")

                # Token hasil preprocessing query
                q_tokens = preprocess_text(query)
                st.caption(f"Token query setelah preprocessing: `{' | '.join(q_tokens)}`")

                st.divider()
                for rank, (doc_id, content, score) in enumerate(relevant, 1):
                    with st.container():
                        c1, c2 = st.columns([6, 1])
                        with c1:
                            st.markdown(f"**#{rank} — Dokumen {doc_id}**")
                            preview = content if len(content) <= 300 else content[:300] + "…"
                            st.write(preview)
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
            st.success(f"✅ Dokumen **{doc_id}** berhasil ditambahkan!")
            tokens = preprocess_text(doc_content)
            st.caption(f"Token hasil preprocessing: `{' | '.join(tokens)}`")
            st.rerun()

    if ir_index.documents:
        st.divider()
        st.subheader("Daftar Dokumen Tersimpan")
        rows = [{"ID": doc_id, "Preview": (content[:120] + "…") if len(content) > 120 else content}
                for doc_id, content in ir_index.documents.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ==================== TAB: IMPORT EXCEL ====================
with tab_excel:
    st.subheader("Import Dokumen dari Excel")
    st.info("Upload file Excel hasil preprocessing. Kolom yang dibaca: **Token Setelah Stemming (Preprocessing Output)**")

    uploaded = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

    if uploaded:
        try:
            xl = pd.ExcelFile(uploaded)
            sheet = st.selectbox("Pilih sheet", xl.sheet_names)
            df_raw = pd.read_excel(uploaded, sheet_name=sheet, skiprows=1)

            col_options = df_raw.columns.tolist()
            default_col = next((c for c in col_options if 'stemming' in c.lower() or 'token' in c.lower()), col_options[0])
            selected_col = st.selectbox("Pilih kolom dokumen", col_options, index=col_options.index(default_col))

            docs_preview = df_raw[selected_col].dropna().astype(str).tolist()
            st.write(f"**{len(docs_preview)} dokumen** ditemukan. Preview:")
            st.dataframe(pd.DataFrame({"Dokumen": docs_preview[:5]}), hide_index=True, use_container_width=True)

            if st.button("📥 Import ke Index", type="primary"):
                with st.spinner(f"Mengindex {len(docs_preview)} dokumen…"):
                    start_id = len(ir_index.documents)
                    for i, content in enumerate(docs_preview):
                        clean = content.replace(' | ', ' ').strip()
                        ir_index.add_document(start_id + i, clean)
                    ir_index.calculate_tfidf_vectors()
                    ir_index.save_to_disk()
                st.success(f"✅ {len(docs_preview)} dokumen berhasil diimport!")
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
                "Posting List (doc_id: tf)": ", ".join(f"D{d}:{f}" for d, f in sorted(postings.items()))
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

                labels = [f"D{i}" for i in doc_ids]
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