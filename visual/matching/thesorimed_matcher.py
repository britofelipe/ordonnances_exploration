import sqlite3
import pandas as pd
import numpy as np
import re
import unicodedata
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Third-party for Fuzzy Matching
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from rapidfuzz import fuzz, process
except ImportError:
    print("Error: scikit-learn and rapidfuzz are required. Run: pip install scikit-learn rapidfuzz")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ThesorimedMatcher:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Thesorimed DB not found at: {self.db_path}")
        
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.names_processed = None
        
        # Load Data immediately
        self._load_data()
        self._build_index()

    def _normalize(self, text: str) -> str:
        """
        Normalize text: lowercase, remove accents, remove non-alphanumeric.
        """
        if not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Remove accents
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Collapse spaces
        return re.sub(r'\s+', ' ', text).strip()

    def _load_data(self):
        logging.info(f"Loading CIS table from {self.db_path}...")
        conn = sqlite3.connect(self.db_path)
        try:
            # Load specific columns: ID and Name
            query = "SELECT CIS_CODE_PK as cis, CIS_LIBELLE_SPECIALITE as name FROM CIS"
            self.df = pd.read_sql_query(query, conn)
            logging.info(f"Loaded {len(self.df)} drugs.")
        finally:
            conn.close()

    def _build_index(self):
        logging.info("Building TF-IDF Index...")
        # 1. Normalize Names
        self.df['name_norm'] = self.df['name'].apply(self._normalize)
        self.names_processed = self.df['name_norm'].tolist()

        # 2. TF-IDF Config
        # char n-grams (3,3) are very robust for OCR errors / short typos
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), min_df=1)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.names_processed)
        logging.info("TF-IDF Index Ready.")

    def match(self, query: str, top_k: int = 20, threshold: float = 60.0) -> List[Dict]:
        """
        Hybrid Match:
        1. Fast Retrieval: Use TF-IDF to get top_k * 3 candidates.
        2. Re-ranking: Use Levenshtein (RapidFuzz) on those candidates.
        3. Thresholding: Filter results below threshold.
        """
        if not query:
            return []
        
        q_norm = self._normalize(query)
        if not q_norm:
            return []

        # --- Step 1: TF-IDF Retrieval ---
        q_vec = self.vectorizer.transform([q_norm])
        
        # Calculate Cosine Similarity (dot product for normalized vectors)
        # Result is (1, N_docs)
        cosine_sims = (q_vec * self.tfidf_matrix.T).toarray()[0]
        
        # Get indices of top candidates (retrieve more than needed to allow re-ranking)
        retrieval_k = min(len(self.df), top_k * 5) 
        # partition is faster than sort
        top_indices = np.argpartition(cosine_sims, -retrieval_k)[-retrieval_k:]
        
        candidates = []
        for idx in top_indices:
            score_tfidf = cosine_sims[idx]
            if score_tfidf < 0.1: # Skip very bad matches early
                continue
                
            original_name = self.df.iloc[idx]['name']
            norm_name = self.names_processed[idx]
            cis_code = self.df.iloc[idx]['cis']
            
            # --- Step 2: RapidFuzz Re-ranking ---
            # Token Sort Ratio handles "Doliprane 1000mg" vs "1000mg Doliprane" well
            # But plain Ratio is better for exact spelling.
            # Weighted Ratio is a good balanced default.
            
            # We compare NORMALIZED strings for fairness
            fuzz_score = fuzz.WRatio(q_norm, norm_name)
            
            candidates.append({
                "cis": cis_code,
                "name": original_name,
                "score": fuzz_score,
                "score_tfidf": score_tfidf,
                "norm_name": norm_name
            })
            
        # Sort by Fuzzy Score primarily
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Filter and Truncate
        final_results = [
            c for c in candidates[:top_k] if c['score'] >= threshold
        ]
        
        return final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Thesorimed Fuzzy Matcher")
    parser.add_argument("--db", type=str, default="../../data/thesorimed/THESORIMED_SQ3", help="Path to DB")
    parser.add_argument("query", type=str, help="Drug name to search")
    args = parser.parse_args()

    # resolve path relative to script if needed
    db_path = Path(args.db)
    if not db_path.exists():
        # fallback for running from project root
        db_path = Path("data/thesorimed/THESORIMED_SQ3")

    matcher = ThesorimedMatcher(str(db_path))
    
    print(f"\nSearching for: '{args.query}'")
    results = matcher.match(args.query, top_k=5)
    
    if results:
        print(f"Found {len(results)} matches:")
        for r in results:
            print(f"[{r['score']:.1f}%] {r['name']} (CIS: {r['cis']})")
    else:
        print("No matches found.")
