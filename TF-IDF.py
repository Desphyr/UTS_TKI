from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None

try:
    faiss = importlib.import_module("faiss")
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    faiss = None

try:
    torch = importlib.import_module("torch")
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    torch = None

try:
    transformers_module = importlib.import_module("transformers")
    AutoModel = getattr(transformers_module, "AutoModel")
    AutoTokenizer = getattr(transformers_module, "AutoTokenizer")
except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None

try:
    sentence_module = importlib.import_module("sentence_transformers")
    CrossEncoder = getattr(sentence_module, "CrossEncoder")
    SentenceTransformer = getattr(sentence_module, "SentenceTransformer")
except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover
    CrossEncoder = None
    SentenceTransformer = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("green-industry-search")

DEFAULT_BERT_MODEL = "indobenchmark/indobert-base-p1"
FALLBACK_BERT_MODEL = "indobenchmark/indobert-base-p1"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_DOCUMENTS = [
    "Industri hijau adalah model produksi yang menekan emisi karbon dan memaksimalkan efisiensi energi.",
    "Energi terbarukan seperti tenaga surya, angin, dan biogas mendukung transisi industri menuju ekonomi rendah karbon.",
    "Program daur ulang limbah industri membantu mengurangi sampah dan meningkatkan pemanfaatan material kembali.",
    "Audit energi di pabrik dapat mengidentifikasi peluang penghematan listrik dan bahan bakar.",
    "Sertifikasi green industry mendorong perusahaan untuk mengadopsi praktik ramah lingkungan dan efisiensi sumber daya.",
    "Penggunaan teknologi sensor pintar mempercepat monitoring kualitas udara, air, dan konsumsi energi.",
    "Ekonomi sirkular mengutamakan desain produk yang tahan lama, mudah diperbaiki, dan dapat didaur ulang.",
    "Penerapan bahan baku lokal dan pengurangan limbah berkontribusi pada ketahanan industri hijau.",
]
SUPPORTED_EXTENSIONS = {".txt", ".json", ".csv", ".xlsx", ".xls", ".md"}


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def can_run_search(documents: Optional[List[Any]], query: str) -> bool:
    if not documents:
        return False
    if not isinstance(documents, list):
        documents = list(documents or [])
    normalized_query = (query or "").strip()
    if not normalized_query:
        return False
    return any(str(item).strip() for item in documents if item is not None)


def build_score_badge(score: float) -> str:
    score_value = float(score)
    if score_value < 0.4:
        bg_color, text_color = "#f8d7da", "#842029"
    elif score_value < 0.7:
        bg_color, text_color = "#fff3cd", "#6c4f00"
    else:
        bg_color, text_color = "#d1e7dd", "#0f5132"
    return (
        f"<span style='display:inline-block;font-weight:700;padding:4px 12px;border-radius:999px;"
        f"background-color:{bg_color};color:{text_color};font-size:0.95rem;'>Skor: {score_value:.4f}</span>"
    )


def _streamlit_css() -> str:
    return """
    <style>
    .reportview-container, .main {
        background: linear-gradient(180deg, #f5f8fb 0%, #e9f3ea 40%, #ffffff 100%);
    }
    .stApp {
        background: linear-gradient(180deg, #fbf9ff 0%, #eef7f1 55%, #ffffff 100%);
    }
    .result-card {
        border: 1px solid rgba(34, 111, 52, 0.12);
        border-radius: 22px;
        padding: 20px 24px;
        margin-bottom: 20px;
        background: rgba(255, 255, 255, 0.95);
        box-shadow: 0 16px 40px rgba(34, 111, 52, 0.08);
    }
    .result-source {
        font-size: 1.18rem;
        font-weight: 700;
        margin-bottom: 6px;
        color: #1b4332;
    }
    .result-snippet {
        color: #344054;
        line-height: 1.7;
        margin-top: 12px;
        margin-bottom: 12px;
    }
    strong {
        font-weight: 800;
        color: #0f5132;
        background: rgba(119, 221, 119, 0.14);
        padding: 0 3px;
        border-radius: 4px;
    }
    .stButton>button {
        border-radius: 999px;
        background-color: #2f855a;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #276749;
    }
    .stSidebar .css-1d391kg {
        border-radius: 18px;
        padding: 18px;
        background: rgba(255,255,255,0.9);
    }
    .sidebar .stTextInput>div>div>input {
        border-radius: 999px;
    }
    .modern-search-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        min-height: 44px;
        padding: 0 14px;
        border-radius: 999px;
        background: linear-gradient(90deg, #ebf8f0 0%, #dff4e5 100%);
        color: #1f5d3b;
        font-weight: 700;
        border: 1px solid rgba(47, 133, 90, 0.16);
        box-shadow: 0 8px 18px rgba(47, 133, 90, 0.09);
    }
    .modern-search-wrapper {
        margin-top: 6px;
        margin-bottom: 8px;
    }
    div[data-testid="stTextInput"] > div > div > input {
        min-height: 46px;
        border-radius: 999px;
        padding: 0 16px;
        box-shadow: inset 0 0 0 1px rgba(47, 133, 90, 0.16);
    }
    div[data-testid="stTextInput"] > div > div > div:last-child {
        display: none !important;
    }
    </style>
    """


def load_documents_from_file(file_path: str) -> List[str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(item) for item in payload]
        if isinstance(payload, dict):
            if "documents" in payload:
                return [str(item) for item in payload["documents"]]
            return [str(v) for v in payload.values()]

    if suffix == ".csv":
        if pd is None:
            raise ImportError("pandas belum terinstal")
        df = pd.read_csv(path)
        for column in ["text", "content", "dokumen", "isi", "document"]:
            if column in df.columns:
                return [str(value) for value in df[column].dropna().tolist()]
        return [str(value) for value in df.iloc[:, 0].dropna().tolist()]

    if suffix in {".xlsx", ".xls"}:
        if pd is None:
            raise ImportError("pandas/openpyxl belum terinstal")

        try:
            excel_file = pd.ExcelFile(path)
            sheet_names = excel_file.sheet_names
        except Exception:
            sheet_names = []

        preferred_columns = [
            "Teks Asli",
            "teks asli",
            "Text",
            "text",
            "content",
            "isi",
            "document",
            "Token Setelah Stemming (Preprocessing Output)",
            "Token Setelah Stopword",
            "Preprocessing Output",
        ]

        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(path, sheet_name=sheet_name, skiprows=1)
            except Exception:
                continue
            if df.empty:
                continue

            for column in preferred_columns:
                if column in df.columns:
                    values = [str(value).strip() for value in df[column].dropna().astype(str).tolist() if str(value).strip()]
                    if values and any(len(value.split()) >= 2 for value in values):
                        return values

            for column in df.columns:
                column_name = str(column).strip()
                if column_name.lower() in {"dokumen", "document", "no", "id", "index", "nomor"}:
                    continue
                values = [str(value).strip() for value in df[column].dropna().astype(str).tolist() if str(value).strip()]
                if values and any(len(value.split()) >= 2 for value in values):
                    return values

        if sheet_names:
            try:
                df = pd.read_excel(path, sheet_name=sheet_names[0], skiprows=1)
                if not df.empty and len(df.columns) > 0:
                    first_col = df.columns[0]
                    return [str(value) for value in df[first_col].dropna().astype(str).tolist() if str(value).strip()]
            except Exception:
                return []

        return []

    if suffix in {".txt", ".md"}:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    raise ValueError(f"Format file tidak didukung: {suffix}")


def load_documents_from_path(path: str) -> List[Dict[str, str]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path tidak ditemukan: {path}")

    if target.is_file():
        file_docs = load_documents_from_file(str(target))
        return [{"text": text, "source": target.name} for text in file_docs if text]

    entries: List[Dict[str, str]] = []
    for file_path in sorted(target.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            file_docs = load_documents_from_file(str(file_path))
        except Exception as exc:  # pragma: no cover
            logger.warning("Lewati file %s: %s", file_path, exc)
            continue
        for index, text in enumerate(file_docs):
            if not text:
                continue
            source_name = file_path.name if len(file_docs) == 1 else f"{file_path.name}#baris-{index + 1}"
            entries.append({"text": str(text), "source": source_name})
    return entries


class HybridGreenIndustrySearch:
    def __init__(
        self,
        documents: Optional[List[Any]] = None,
        use_bert: bool = True,
        use_reranking: bool = True,
        bert_model_name: str = DEFAULT_BERT_MODEL,
        reranker_model_name: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        self.document_entries = self._normalize_documents(documents or DEFAULT_DOCUMENTS)
        self.raw_documents = [entry["text"] for entry in self.document_entries]
        self.document_sources = [entry["source"] for entry in self.document_entries]
        self.normalized_documents = [normalize_text(doc) for doc in self.raw_documents]
        self.use_bert = use_bert
        self.use_reranking = use_reranking
        self.bert_model_name = bert_model_name
        self.reranker_model_name = reranker_model_name

        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.dense_embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        self.sentence_model = None
        self.tokenizer = None
        self.model = None
        self.cross_encoder = None

        self._build_lexical_index()
        self._build_dense_index()

    def _normalize_documents(self, documents: List[Any]) -> List[Dict[str, str]]:
        entries: List[Dict[str, str]] = []
        for idx, item in enumerate(documents):
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or item.get("dokumen") or "")
                source = str(item.get("source") or item.get("file") or f"dokumen-{idx + 1}")
            else:
                text = str(item)
                source = f"dokumen-{idx + 1}"
            if text.strip():
                entries.append({"text": text.strip(), "source": source})
        return entries

    def _build_lexical_index(self) -> None:
        if TfidfVectorizer is None:
            logger.warning("scikit-learn tidak tersedia, mode lexical dinonaktifkan")
            return
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.normalized_documents)

    def _build_dense_index(self) -> None:
        if not self.use_bert:
            return
        self._load_bert_model()
        if self.dense_embeddings is None:
            return
        if faiss is None:
            logger.warning("faiss tidak tersedia, indeks dense dilewati")
            return
        dim = self.dense_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.dense_embeddings.astype("float32"))

    def _load_bert_model(self) -> None:
        if SentenceTransformer is not None:
            try:
                self.sentence_model = SentenceTransformer(self.bert_model_name)  # type: ignore
                self.dense_embeddings = self._encode_texts(self.normalized_documents)
                return
            except Exception as exc:
                logger.warning("Gagal memuat sentence-transformers model %s: %s", self.bert_model_name, exc)

        if AutoTokenizer is not None and AutoModel is not None and torch is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)  # type: ignore
                self.model = AutoModel.from_pretrained(self.bert_model_name)  # type: ignore
                self.model.eval()
                self.dense_embeddings = self._encode_texts(self.normalized_documents)
                return
            except Exception as exc:
                logger.warning("Gagal memuat transformers model %s: %s", self.bert_model_name, exc)

        try:
            if SentenceTransformer is not None:
                self.sentence_model = SentenceTransformer(FALLBACK_BERT_MODEL)  # type: ignore
                self.dense_embeddings = self._encode_texts(self.normalized_documents)
                return
        except Exception as exc:
            logger.warning("Model embedding tidak tersedia, fallback ke mode lexical saja: %s", exc)
            self.dense_embeddings = None

        if self.use_reranking and CrossEncoder is not None:
            try:
                self.cross_encoder = CrossEncoder(self.reranker_model_name)  # type: ignore
            except Exception as exc:
                logger.warning("Gagal memuat cross-encoder: %s", exc)

    def _encode_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        if not texts:
            return None
        if self.sentence_model is not None:
            try:
                return self.sentence_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            except Exception as exc:
                logger.warning("Gagal encode dengan sentence-transformers: %s", exc)
                return None
        if self.model is not None and self.tokenizer is not None and torch is not None:
            try:
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=256,
                )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                summed = (embeddings * mask).sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / lengths
                return pooled.cpu().numpy()
            except Exception as exc:
                logger.warning("Gagal encode dengan transformers: %s", exc)
                return None
        return None

    def _encode_query(self, query: str) -> Optional[np.ndarray]:
        if self.dense_embeddings is None:
            return None
        if self.sentence_model is not None:
            try:
                return self.sentence_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
            except Exception:
                return None
        if self.model is not None and self.tokenizer is not None and torch is not None:
            try:
                inputs = self.tokenizer([query], padding=True, truncation=True, return_tensors="pt", max_length=256)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                summed = (embeddings * mask).sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / lengths
                return pooled[0].cpu().numpy()
            except Exception:
                return None
        return None

    def _lexical_scores(self, query: str) -> np.ndarray:
        if self.tfidf_matrix is None or self.tfidf_vectorizer is None or cosine_similarity is None:
            return np.zeros(len(self.raw_documents))
        query_vector = self.tfidf_vectorizer.transform([normalize_text(query)])
        scores = cosine_similarity(self.tfidf_matrix, query_vector).ravel()
        return scores

    def _dense_scores(self, query: str) -> np.ndarray:
        if self.faiss_index is None:
            return np.zeros(len(self.raw_documents))
        query_vector = self._encode_query(query)
        if query_vector is None:
            return np.zeros(len(self.raw_documents))
        query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        distances, indices = self.faiss_index.search(query_vector, len(self.raw_documents))
        scores = np.zeros(len(self.raw_documents))
        for rank, idx in enumerate(indices[0]):
            if idx >= 0:
                scores[idx] = max(scores[idx], float(distances[0][rank]))
        return scores

    def _extract_query_terms(self, query: str) -> List[str]:
        terms = [term for term in re.findall(r"[A-Za-z0-9]+", normalize_text(query)) if len(term) >= 2]
        return list(dict.fromkeys(terms))

    def _combine_scores(self, query: str) -> np.ndarray:
        lexical_scores = self._lexical_scores(query)
        dense_scores = self._dense_scores(query)

        lexical_max = float(np.max(lexical_scores)) if len(lexical_scores) > 0 else 0.0
        dense_max = float(np.max(dense_scores)) if len(dense_scores) > 0 else 0.0

        if lexical_max > 0:
            lexical_norm = np.clip(lexical_scores, 0.0, None)
            lexical_norm = lexical_norm / lexical_max if lexical_max > 0 else lexical_norm
            if dense_max > 0:
                dense_norm = np.clip(dense_scores, 0.0, None)
                dense_norm = dense_norm / dense_max if dense_max > 0 else dense_norm
                return 0.65 * lexical_norm + 0.35 * dense_norm
            return lexical_norm

        if dense_max > 0:
            dense_norm = np.clip(dense_scores, 0.0, None)
            dense_norm = dense_norm / dense_max if dense_max > 0 else dense_norm
            return dense_norm

        return np.zeros(len(self.raw_documents))

    def _highlight_terms(self, text: str, query: str, html: bool = False) -> str:
        query_terms = self._extract_query_terms(query)
        if not query_terms or not text:
            return text

        pattern = re.compile(r"(?<!\\w)(" + "|".join(re.escape(term) for term in query_terms) + r")(?!\\w)", re.IGNORECASE)
        if html:
            return pattern.sub(lambda m: f"<strong>{m.group(0)}</strong>", text)
        return pattern.sub(lambda m: f"**{m.group(0)}**", text)

    def _build_snippet(self, text: str, query: str) -> str:
        query_terms = self._extract_query_terms(query)
        if not query_terms:
            return text[:220]
        lowered = text.lower()
        for term in query_terms:
            if term.lower() in lowered:
                idx = lowered.find(term.lower())
                start = max(0, idx - 70)
                end = min(len(text), idx + 140)
                snippet = text[start:end].strip()
                return snippet if snippet else text[:220]
        return text[:220]

    def _make_dedup_key(self, text: str) -> str:
        cleaned = normalize_text(text)
        cleaned = re.sub(r"[^a-z0-9]+", "", cleaned)
        return cleaned[:200] or cleaned

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        base_scores = self._combine_scores(query)
        candidate_ids = np.argsort(base_scores)[::-1][: max(10, top_k * 3)]
        candidates = []
        for doc_id in candidate_ids:
            score = float(base_scores[doc_id])
            if score is None:
                continue
            candidates.append((int(doc_id), score, self.raw_documents[int(doc_id)]))

        if self.use_reranking and self.cross_encoder is not None and len(candidates) > 1:
            pairs = [(query, doc_text) for _, _, doc_text in candidates]
            try:
                rerank_scores = self.cross_encoder.predict(pairs)
                reranked = []
                for (doc_id, base_score, doc_text), rerank_score in zip(candidates, rerank_scores):
                    final_score = float(rerank_score) * 0.7 + float(base_score) * 0.3
                    reranked.append((doc_id, final_score, doc_text))
                candidates = reranked
            except Exception as exc:
                logger.warning("Gagal reranking dokumen: %s", exc)

        candidates.sort(key=lambda item: item[1], reverse=True)
        results = []
        seen_keys = set()
        for doc_id, score, doc_text in candidates:
            source = self.document_sources[doc_id] if doc_id < len(self.document_sources) else "dokumen"
            dedup_key = self._make_dedup_key(doc_text)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            snippet_text = self._build_snippet(doc_text, query)
            highlighted_snippet = self._highlight_terms(snippet_text, query, html=True)
            highlighted_text = self._highlight_terms(doc_text, query, html=True)
            results.append(
                {
                    "doc_id": doc_id,
                    "score": round(float(score), 4),
                    "text": doc_text,
                    "source": source,
                    "snippet": snippet_text,
                    "highlighted_text": highlighted_text,
                    "highlighted_snippet": highlighted_snippet,
                }
            )
            if len(results) >= top_k:
                break
        return results



def run_streamlit() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"streamlit belum terinstal: {exc}")

    st.set_page_config(page_title="Industri Hijau Hybrid Search")
    st.markdown(_streamlit_css(), unsafe_allow_html=True)
    st.title("Pencarian Dokumen Industri Hijau")
    st.markdown("<p style='margin-bottom: 10px; color:#274c37;'>Menggunakan indoBERT , FAISS dan reranking cross-encoder untuk hasil yang lebih relevan.</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("📁 Unggah Dokumen")
        uploaded_file = st.file_uploader("Unggah dokumen", type=["csv", "xlsx"])
        st.markdown("---")
        st.markdown("**Tips:** gunakan file Excel dengan kolom teks berisi isi dokumen kata.")

    docs_input: List[Any] = []
    if uploaded_file is not None:
        temp_dir = Path.cwd() / "temp_uploads"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        temp_path.write_bytes(uploaded_file.getvalue())
        docs_input = load_documents_from_file(str(temp_path))

    search_result_count = 0
    if not docs_input:
        st.caption("")

    query = ""
    search_col, top_k_col, button_col = st.columns([6.2, 0.9, 1.0])
    with search_col:
        query = st.text_input(
            "Query",
            placeholder="Masukkan kata kunci pencarian…",
            label_visibility="collapsed",
            key="search_query",
            help=None,
            type="default",
        )
    with top_k_col:
        top_k = st.number_input("Top-K", min_value=1, max_value=50, value=5, label_visibility="collapsed", key="search_top_k")
    with button_col:
        search_clicked = st.button("Cari", type="primary", use_container_width=True, disabled=not docs_input or not query.strip())

    if not docs_input and query.strip():
        st.warning("Kamu belum upload dokumen")

    if search_clicked:
        if not can_run_search(docs_input, query):
            if not docs_input:
                st.warning("Kamu belum upload dokumen")
            else:
                st.warning("Masukkan kata kunci pencarian terlebih dahulu")
            return

        engine = HybridGreenIndustrySearch(documents=docs_input, use_bert=True, use_reranking=True)
        with st.spinner("Mencari dokumen..."):
            results = engine.search(query, top_k=top_k)
        search_result_count = len(results)
        if not results:
            st.info("Tidak ada hasil yang cocok.")
            return

        st.success(f"Ditemukan {len(results)} hasil")
        for item in results:  # type: ignore
            st.markdown(
                f"<div class='result-card'>"
                f"<div class='result-source'> {item['source']}</div>"
                f"{build_score_badge(float(item['score']))}"
                f"<div class='result-snippet'>{item['highlighted_snippet']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.divider()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid search untuk dokumen industri hijau")
    parser.add_argument("--file", type=str, default=None, help="Path ke file csv/xlsx/json/txt/md")
    parser.add_argument("--folder", type=str, default=None, help="Path ke folder yang berisi dokumen")
    parser.add_argument("--query", type=str, default=None, help="Kueri pencarian")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-bert", action="store_true", help="Nonaktifkan embedding dense")
    parser.add_argument("--no-rerank", action="store_true", help="Nonaktifkan cross-encoder reranking")
    parser.add_argument("--bert-model", type=str, default=DEFAULT_BERT_MODEL, help="Model embedding dense (default: IndoBERT)")
    parser.add_argument("--reranker-model", type=str, default=DEFAULT_RERANKER_MODEL, help="Model reranking cross-encoder")
    parser.add_argument("--streamlit", action="store_true", help="Jalankan UI Streamlit")
    args, extra_args = parser.parse_known_args()

    has_cli_flags = bool(
        args.file or args.folder or args.query or args.top_k != 5 or args.no_bert or args.no_rerank or args.streamlit
    )
    has_streamlit_runtime_flags = any(
        arg.startswith("--server") or arg.startswith("--browser") or arg.startswith("--theme") or arg.startswith("--client")
        for arg in extra_args
    )

    if args.streamlit or (not has_cli_flags and not has_streamlit_runtime_flags):
        run_streamlit()
        return

    documents = DEFAULT_DOCUMENTS
    if args.file:
        documents = load_documents_from_file(args.file)
    elif args.folder:
        documents = load_documents_from_path(args.folder)

    engine = HybridGreenIndustrySearch(
        documents=documents,
        use_bert=not args.no_bert,
        use_reranking=not args.no_rerank,
        bert_model_name=args.bert_model,
        reranker_model_name=args.reranker_model,
    )

    query = args.query or input("Masukkan kata kunci pencarian: ").strip()
    results = engine.search(query, top_k=args.top_k)
    if not results:
        print("Tidak ada hasil yang cocok.")
        return

    print(f"Hasil pencarian untuk: {query}")
    for item in results:  # type: ignore
        doc_num: int = item['doc_id'] + 1  # type: ignore
        print(f"[{doc_num}] score={item['score']}: {item['text']}")


if __name__ == "__main__":
    main()
