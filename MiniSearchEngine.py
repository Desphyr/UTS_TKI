
import argparse
import json
import logging
import math
import re
import os
import sqlite3
from pathlib import Path

from collections import defaultdict
from typing import Dict, List, Tuple, Set

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False

# ==================== KONFIGURASI ====================
EXCEL_PREPROCESSING = 'hasil_preprocessing.xlsx'
SHEET_NAME = 'Hasil Stemming'
COLUMN_NAME = 'Token Setelah Stemming (Preprocessing Output)'

OUTPUT_VSM_EXCEL = 'hasil_vector_space_model.xlsx'
OUTPUT_HEATMAP = 'similarity_heatmap.png'
INDEX_DB_FILE = "ir_index.sqlite"

# Legacy JSON files (migration only)
DOCUMENTS_FILE = "documents.json"
INVERTED_INDEX_FILE = "inverted_index.json"
DOC_VECTORS_FILE = "doc_vectors.json"


# Stop words bahasa Indonesia
INDONESIAN_STOPWORDS = {
    'yang', 'dan', 'di', 'dari', 'ke', 'adalah', 'ini', 'itu', 'atau', 'tidak',
    'untuk', 'dengan', 'pada', 'oleh', 'telah', 'akan', 'sudah', 'juga', 'dapat',
    'dalam', 'ada', 'karena', 'namun', 'hanya', 'seperti', 'saat', 'ketika',
    'lalu', 'maka', 'jadi', 'sebagai', 'setelah', 'sebelum', 'selama', 'sejak',
    'sampai', 'hingga', 'melalui', 'terhadap', 'tanpa', 'bagian', 'selain',
    'daripada', 'a', 'an', 'the', 'to', 'at', 'in', 'on', 'by', 'as', 'of', 'is'
}


# ==================== PREPROCESSING FUNCTIONS ====================
def case_folding(text: str) -> str:
    """Konversi teks ke huruf kecil"""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Hapus tanda baca"""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def stopword_removal(tokens: List[str]) -> List[str]:
    """Hapus stop words bahasa Indonesia"""
    return [token for token in tokens if token not in INDONESIAN_STOPWORDS]


def stemming(tokens: List[str]) -> List[str]:
    """Stemming menggunakan Sastrawi"""
    if not SASTRAWI_AVAILABLE:
        return tokens
    
    try:
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return [stemmer.stem(token) for token in tokens]
    except Exception as e:
        logging.warning("Error stemming: %s", e)
        return tokens


def preprocess_text(text: str) -> List[str]:
    """Pipeline preprocessing lengkap"""
    # 1. Case folding
    text = case_folding(text)
    # 2. Remove punctuation
    text = remove_punctuation(text)
    # 3. Tokenization
    tokens = text.split()
    # 4. Stop word removal
    tokens = stopword_removal(tokens)
    # 5. Stemming
    tokens = stemming(tokens)
    # Filter empty tokens
    tokens = [t for t in tokens if t]
    return tokens


# ==================== INVERTED INDEX CLASS ====================
class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.documents: Dict[int, str] = {}
        self.doc_vectors: Dict[int, Dict[str, float]] = {}
        self.vocabulary: Set[str] = set()
    
    def add_document(self, doc_id: int, doc_content: str) -> None:
        """Tambah dokumen ke index"""
        self.documents[doc_id] = doc_content
        tokens = preprocess_text(doc_content)
        
        # Hitung TF (term frequency)
        tf_dict = defaultdict(int)
        for token in tokens:
            tf_dict[token] += 1
        
        # Update inverted index
        for token, freq in tf_dict.items():
            self.index[token][doc_id] = freq
            self.vocabulary.add(token)
    
    def calculate_idf(self) -> Dict[str, float]:
        """Hitung IDF (inverse document frequency)"""
        N = len(self.documents)
        idf = {}
        
        for term in self.vocabulary:
            df = len(self.index[term])
            idf[term] = math.log10(N / df) if df > 0 else 0
        
        return idf
    
    def calculate_tfidf_vectors(self) -> Dict[int, Dict[str, float]]:
        """Hitung TF-IDF vectors dengan log frequency weighting"""
        idf = self.calculate_idf()
        self.doc_vectors = {}
        
        for doc_id, doc_content in self.documents.items():
            tokens = preprocess_text(doc_content)
            tf_dict = defaultdict(int)
            
            for token in tokens:
                tf_dict[token] += 1
            
            # TF dengan log frequency weighting: 1 + log10(tf)
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
        """Return inverted index sebagai dictionary"""
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
        """Jika DB kosong tapi JSON legacy ada, import sekali."""
        # DB file mungkin sudah tercipta oleh sqlite3.connect, jadi jangan check os.path.exists(INDEX_DB_FILE)
        if not os.path.exists(DOCUMENTS_FILE):
            return


        # Pastikan schema ada
        self._init_db(conn)

        # Load dari JSON legacy
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

        # Simpan ke SQLite
        self.save_to_disk()

    def save_to_disk(self) -> None:
        """Simpan ke disk menggunakan SQLite (tanpa JSON runtime)."""
        # Jika masih tidak ada dokumen, jangan bikin DB kosong
        if not self.documents:
            # tetap buat DB kosong agar load_from_disk tidak ambigu
            conn = sqlite3.connect(INDEX_DB_FILE)
            try:
                self._init_db(conn)
            finally:
                conn.close()
            return

        conn = sqlite3.connect(INDEX_DB_FILE)
        try:
            self._init_db(conn)
            cur = conn.cursor()
            cur.execute("DELETE FROM documents")
            cur.execute("DELETE FROM postings")
            cur.execute("DELETE FROM vectors")

            # documents
            cur.executemany(
                "INSERT INTO documents (doc_id, content) VALUES (?, ?)",
                [(int(doc_id), content) for doc_id, content in self.documents.items()],
            )

            # postings
            postings_rows = []
            for term, docs in self.index.items():
                for doc_id, tf in docs.items():
                    postings_rows.append((term, int(doc_id), int(tf)))
            if postings_rows:
                cur.executemany(
                    "INSERT INTO postings (term, doc_id, tf) VALUES (?, ?, ?)",
                    postings_rows,
                )

            # vectors
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

    def load_from_disk(self) -> None:
        """Load dari disk menggunakan SQLite.

        Migration: jika DB belum ada tapi JSON legacy ada, lakukan import sekali.
        Setelah itu aplikasi tidak lagi bergantung pada JSON.
        """
        # Pastikan database ada / siap
        db_exists = os.path.exists(INDEX_DB_FILE)
        conn = sqlite3.connect(INDEX_DB_FILE)
        try:
            self._init_db(conn)

            # Migration dilakukan jika DB kosong (tanpa mengandalkan keberadaan file sebelumnya)
            cur0 = conn.cursor()
            cur0.execute("SELECT COUNT(*) FROM documents")
            doc_count = cur0.fetchone()[0]

            if doc_count == 0 and (not self.documents):
                self._maybe_migrate_legacy_json_once(conn)


            cur = conn.cursor()
            cur.execute("SELECT doc_id, content FROM documents")
            rows = cur.fetchall()
            if not rows:
                return

            self.documents = {int(doc_id): content for doc_id, content in rows}
            self.index = defaultdict(lambda: defaultdict(int))
            self.vocabulary = set()
            self.doc_vectors = {}

            # postings
            cur.execute("SELECT term, doc_id, tf FROM postings")
            postings_rows = cur.fetchall()
            for term, doc_id, tf in postings_rows:
                doc_id = int(doc_id)
                self.index[term][doc_id] = int(tf)
                self.vocabulary.add(term)

            # vectors
            cur.execute("SELECT doc_id, term, value FROM vectors")
            vec_rows = cur.fetchall()
            for doc_id, term, value in vec_rows:
                doc_id = int(doc_id)
                self.doc_vectors.setdefault(doc_id, {})[term] = float(value)

            # Safety: jika vectors kosong, hitung ulang
            if self.documents and (not self.doc_vectors):
                self.calculate_tfidf_vectors()

        finally:
            conn.close()




def load_documents(excel_path: str = EXCEL_PREPROCESSING, sheet_name: str = SHEET_NAME):
    """Load dokumen dari Excel (backward compatibility)"""
    xl_file = pd.ExcelFile(excel_path)
    sheet_names = xl_file.sheet_names
    print("Available sheets:", sheet_names)

    if sheet_name in sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=1)
    else:
        print(f"Sheet '{sheet_name}' tidak ditemukan. Menggunakan sheet pertama...")
        df = pd.read_excel(excel_path, sheet_name=sheet_names[0], skiprows=1)
        df = df.dropna().reset_index(drop=True)

    if COLUMN_NAME not in df.columns:
        print(f"Column '{COLUMN_NAME}' not found. Available columns:", df.columns.tolist())
        raise ValueError(f"Missing column: {COLUMN_NAME}")

    docs = df[COLUMN_NAME].dropna().astype(str).tolist()
    print(f"Number of documents: {len(docs)}")

    # delimiter token hasil preprocessing dianggap sebagai spasi
    docs_clean = [doc.replace(' | ', ' ') for doc in docs]
    return docs, docs_clean


def build_tfidf(docs_clean):
    """Build TF-IDF menggunakan sklearn (legacy support)"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs_clean)
    return vectorizer, tfidf_matrix


def normalize_vector(vector: Dict[str, float]) -> Dict[str, float]:
    """Normalisasi vector"""
    norm = math.sqrt(sum(val ** 2 for val in vector.values()))
    if norm == 0:
        return vector
    return {k: v / norm for k, v in vector.items()}


def cosine_similarity_custom(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Hitung cosine similarity antara dua vector (custom implementation)"""
    vec1_norm = normalize_vector(vec1)
    vec2_norm = normalize_vector(vec2)
    
    dot_product = sum(
        vec1_norm.get(term, 0) * vec2_norm.get(term, 0)
        for term in set(vec1_norm.keys()) | set(vec2_norm.keys())
    )
    
    return dot_product


def export_vsm_and_heatmap(tfidf_matrix, out_excel=OUTPUT_VSM_EXCEL, out_heatmap=OUTPUT_HEATMAP, n_show=15):
    """Export hasil VSM dan heatmap"""
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    df_sim = pd.DataFrame(cosine_sim)

    # Export hasil VSM (cosine similarity) ke Excel
    with pd.ExcelWriter(out_excel, engine='openpyxl') as writer:
        df_sim.to_excel(writer, sheet_name='CosineSimilarity', index=False)
    print(f"Excel VSM saved as '{out_excel}'")

    # Heatmap
    plt.figure(figsize=(20, 20))
    n_docs = df_sim.shape[0]
    n_show = min(n_show, n_docs)
    sns.heatmap(
        df_sim.iloc[:n_show, :n_show],
        annot=True,
        fmt='.1f',
        annot_kws={"size": 12},
        cmap='Blues',
    )
    plt.title(f'Heatmap Cosine Similarity ({n_show} Dokumen Pertama)')

    plt.xlabel('Dokumen ID')
    plt.ylabel('Dokumen ID')
    plt.tight_layout()
    plt.savefig(out_heatmap, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved as '{out_heatmap}'")

    return df_sim


def retrieve_documents(query: str, ir_index: InvertedIndex, top_k: int = 10) -> List[Tuple[int, str, float]]:
    """Retrieve dokumen berdasarkan query (custom implementation)"""
    # Preprocess query
    query_tokens = preprocess_text(query)
    
    if not query_tokens:
        return []
    
    # Hitung TF query
    query_tf = defaultdict(int)
    for token in query_tokens:
        query_tf[token] += 1
    
    # Hitung query vector dengan log frequency weighting
    idf = ir_index.calculate_idf()
    query_vector = {}
    
    for term in ir_index.vocabulary:
        if term in query_tf:
            tf = 1 + math.log10(query_tf[term])
            query_vector[term] = tf * idf[term]
        else:
            query_vector[term] = 0
    
    # Hitung cosine similarity dengan semua dokumen
    scores = []
    for doc_id, doc_vector in ir_index.doc_vectors.items():
        similarity = cosine_similarity_custom(query_vector, doc_vector)
        scores.append((doc_id, ir_index.documents[doc_id], similarity))
    
    # Sort berdasarkan skor relevansi (descending)
    scores.sort(key=lambda x: x[2], reverse=True)
    
    return scores[:top_k]


def run_query(query_text: str, vectorizer: TfidfVectorizer, tfidf_matrix, docs, top_k: int = 10):
    """Menjalankan query (legacy support dengan sklearn)"""
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("Query text must be a non-empty string")

    # samakan delimiter query dengan dokumen
    query_clean = query_text.replace(' | ', ' ').strip()

    # vectorize query pakai TF-IDF model yg sama
    query_vec = vectorizer.transform([query_clean])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_k = max(1, min(int(top_k), len(scores)))
    top_idx = scores.argsort()[::-1][:top_k]

    print(f"\n=== Hasil Pencarian Query: '{query_text}' ===")
    print(f"Top {top_k} dokumen (berdasarkan cosine similarity):")

    for rank, doc_id in enumerate(top_idx, start=1):
        preview = docs[doc_id]
        if len(preview) > 200:
            preview = preview[:200] + '...'
        print(f"{rank}. Dokumen {doc_id} | skor={scores[doc_id]:.4f} | preview={preview}")

    return top_idx, scores


def main():
    """Main entry point dengan CLI interface"""
    parser = argparse.ArgumentParser(
        description='Information Retrieval Engine - Vector Space Model dengan TF-IDF + Custom Implementations'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['excel', 'json', 'interactive'],
        default='json',
        help='Mode operasi: excel (baca dari Excel), json (baca dari JSON), interactive (input manual)'
    )
    parser.add_argument(
        '--query',
        type=str,
        default=None,
        help='Teks query untuk dicari'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Jumlah dokumen teratas yang ditampilkan'
    )
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Jika --query tidak diberikan, jangan masuk mode interaktif'
    )

    # Use parse_known_args to ignore arguments passed by the Colab kernel
    args, unknown = parser.parse_known_args()

    print("=" * 60)
    print("Information Retrieval Engine - Vector Space Model")
    print("=" * 60)
    
    # ==================== MODE SELECTION ====================
    if args.mode == 'excel':
        print("\n[MODE: Excel Preprocessing]")
        try:
            docs, docs_clean = load_documents()
            vectorizer, tfidf_matrix = build_tfidf(docs_clean)
            
            # Tetap jalankan output existing (VSM dokumen & heatmap)
            df_sim = export_vsm_and_heatmap(tfidf_matrix)
            
            # Preview: mirip output awal script
            try:
                df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
                print("\nMatriks TF-IDF (5 Dokumen Pertama):")
                print(df_tfidf.head())
                print("\nMatriks VSM - Cosine Similarity (5 Dokumen Pertama):")
                print(df_sim.head())
            except Exception as e:
                print('Catatan: preview TF-IDF/VSM tidak berhasil karena:', str(e))
            
            # Query mode
            if args.query:
                run_query(args.query, vectorizer, tfidf_matrix, docs, top_k=args.top_k)
                return
            
            if args.no_interactive:
                return
            
            # Interactive mode
            while True:
                try:
                    q = input("\nMasukkan query (kosong untuk keluar): ")
                except EOFError:
                    break
                
                if not q or not q.strip():
                    break
                
                run_query(q, vectorizer, tfidf_matrix, docs, top_k=args.top_k)
        
        except FileNotFoundError:
            print("❌ File Excel tidak ditemukan. Gunakan --mode json atau --mode interactive")
    
    elif args.mode == 'json':
        print("\n[MODE: JSON Storage]")
        ir_index = InvertedIndex()
        ir_index.load_from_disk()
        
        if len(ir_index.documents) == 0:
            print("ℹ️  Belum ada dokumen tersimpan. Gunakan --mode interactive untuk menambah dokumen.")
        else:
            print(f"✅ Loaded {len(ir_index.documents)} documents")
            print(f"📝 Vocabulary size: {len(ir_index.vocabulary)}")
            
            # Recalculate vectors
            ir_index.calculate_tfidf_vectors()
            
            if args.query:
                print(f"\n🔍 Query: {args.query}")
                results = retrieve_documents(args.query, ir_index, top_k=args.top_k)
                
                if results:
                    print(f"✅ Found {len(results)} results\n")
                    for rank, (doc_id, content, score) in enumerate(results, 1):
                        preview = content[:100] if len(content) > 100 else content
                        print(f"{rank}. Doc {doc_id} | Score: {score:.4f}")
                        print(f"   Preview: {preview}...")
                        print()
                else:
                    print("❌ No results found")
                return
            
            if args.no_interactive:
                return
            
            # Interactive mode
            while True:
                try:
                    q = input("\n🔍 Enter query (empty to exit): ")
                except EOFError:
                    break
                
                if not q or not q.strip():
                    break
                
                results = retrieve_documents(q, ir_index, top_k=args.top_k)
                
                if results:
                    print(f"✅ Found {len(results)} results\n")
                    for rank, (doc_id, content, score) in enumerate(results, 1):
                        preview = content[:100] if len(content) > 100 else content
                        print(f"{rank}. Doc {doc_id} | Score: {score:.4f}")
                        print(f"   Preview: {preview}...")
                        print()
                else:
                    print("❌ No results found\n")
    
    elif args.mode == 'interactive':
        print("\n[MODE: Interactive - Manual Input]")
        ir_index = InvertedIndex()
        ir_index.load_from_disk()
        
        print("\nMenu:")
        print("  1. Add document")
        print("  2. Search")
        print("  3. View statistics")
        print("  4. Exit")
        
        while True:
            try:
                choice = input("\nPilih menu (1-4): ").strip()
            except EOFError:
                break
            
            if choice == '1':
                print("\n--- Tambah Dokumen ---")
                doc_content = input("Masukkan isi dokumen: ").strip()
                
                if doc_content:
                    doc_id = len(ir_index.documents)
                    ir_index.add_document(doc_id, doc_content)
                    ir_index.calculate_tfidf_vectors()
                    ir_index.save_to_disk()
                    print(f"✅ Dokumen {doc_id} ditambahkan")
                else:
                    print("❌ Isi dokumen tidak boleh kosong")
            
            elif choice == '2':
                query = input("\n🔍 Masukkan query: ").strip()
                
                if query:
                    results = retrieve_documents(query, ir_index, top_k=args.top_k)
                    
                    if results:
                        print(f"\n✅ Ditemukan {len(results)} hasil:\n")
                        for rank, (doc_id, content, score) in enumerate(results, 1):
                            preview = content[:100] if len(content) > 100 else content
                            print(f"{rank}. Doc {doc_id} | Score: {score:.4f}")
                            print(f"   {preview}...")
                            print()
                    else:
                        print("❌ Tidak ada hasil\n")
            
            elif choice == '3':
                print("\n--- Statistik Index ---")
                print(f"Total dokumen: {len(ir_index.documents)}")
                print(f"Total terms: {len(ir_index.vocabulary)}")
                total_postings = sum(len(docs) for docs in ir_index.index.values())
                print(f"Total postings: {total_postings}")
                
                if ir_index.documents:
                    avg = total_postings / len(ir_index.documents)
                    print(f"Rata-rata terms/doc: {avg:.1f}")
                
                print("\nTop 10 terms:")
                term_freq = sorted(
                    [(term, len(docs)) for term, docs in ir_index.index.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                
                for rank, (term, freq) in enumerate(term_freq, 1):
                    print(f"  {rank}. {term}: {freq} docs")
            
            elif choice == '4':
                break
            
            else:
                print("❌ Pilihan tidak valid")


if __name__ == '__main__':
    # Check dependencies
    if not SASTRAWI_AVAILABLE:
        print("⚠️  Warning: Sastrawi library not found.")
        print("   Install with: pip install Sastrawi")
        print("   Stemming will be skipped.\n")
    
    main()