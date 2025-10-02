import json
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union
import time
import re

class PlayRetriever:
    def __init__(self, data_dir: Path, use_hybrid: bool = False):
        """
        Initialize the play retrieval system.
        
        Args:
            data_dir: Path to data directory
            use_hybrid: If True, use hybrid (semantic + BM25) retrieval
        """
        self.data_dir = Path(data_dir)
        self.use_hybrid = use_hybrid
        
        print("Loading play corpus...")
        with open(self.data_dir / 'play_corpus.json') as f:
            self.corpus = json.load(f)
        
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(self.data_dir / 'play_index.faiss'))
        
        print("Loading metadata...")
        with open(self.data_dir / 'play_index_metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loading embedding model: {self.metadata['model_name']}...")
        self.model = SentenceTransformer(self.metadata['model_name'])
        
        if self.use_hybrid:
            print("Loading BM25 index...")
            with open(self.data_dir / 'bm25_index.pkl', 'rb') as f:
                bm25_data = pickle.load(f)
                self.bm25 = bm25_data['bm25']
            print(f"✓ Hybrid retriever ready: {self.metadata['num_plays']} plays indexed")
        else:
            self.bm25 = None
            print(f"✓ Retriever ready: {self.metadata['num_plays']} plays indexed")
    
    def query(self, query_text: str, k: int = 20) -> Dict:
        """
        Retrieve top-K plays for a given query.
        
        Args:
            query_text: Natural language query or structured situation text
            k: Number of plays to retrieve
        
        Returns:
            Dictionary with retrieved plays, latency, and metadata
        """
        if self.use_hybrid:
            return self._hybrid_query(query_text, k)
        else:
            return self._semantic_query(query_text, k)
    
    def _semantic_query(self, query_text: str, k: int = 20) -> Dict:
        """
        Pure semantic retrieval using FAISS.
        """
        embed_start = time.time()
        query_embedding = self.model.encode(
            [f"query: {query_text}"],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embed_latency = (time.time() - embed_start) * 1000
        
        search_start = time.time()
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        search_latency = (time.time() - search_start) * 1000
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            play = self.corpus[idx]
            similarity = 1 - dist
            results.append({
                'play_id': play['play_id'],
                'text': play['text'],
                'similarity': float(similarity),
                'play_data': play['play_data'],
                'reference_count': play['reference_count']
            })
        
        return {
            'query': query_text,
            'retrieved_plays': results,
            'latency_ms': {
                'embedding': embed_latency,
                'search': search_latency,
                'total': embed_latency + search_latency
            },
            'k': k
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        """
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _hybrid_query(self, query_text: str, k: int = 20, semantic_weight: float = 0.7) -> Dict:
        """
        Hybrid retrieval combining semantic (FAISS) and keyword (BM25) search.
        
        Args:
            query_text: Natural language query
            k: Number of plays to retrieve
            semantic_weight: Weight for semantic scores (1 - semantic_weight for BM25)
        
        Returns:
            Dictionary with retrieved plays after hybrid reranking
        """
        k_candidates = k * 2
        bm25_weight = 1 - semantic_weight
        
        embed_start = time.time()
        query_embedding = self.model.encode(
            [f"query: {query_text}"],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embed_latency = (time.time() - embed_start) * 1000
        
        semantic_start = time.time()
        distances, indices = self.index.search(query_embedding.astype('float32'), k_candidates)
        semantic_latency = (time.time() - semantic_start) * 1000
        
        bm25_start = time.time()
        query_tokens = self._tokenize(query_text)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_latency = (time.time() - bm25_start) * 1000
        
        bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        
        semantic_candidates = {}
        for idx, dist in zip(indices[0], distances[0]):
            semantic_sim = 1 - dist
            semantic_candidates[idx] = semantic_sim
        
        all_candidates = set(semantic_candidates.keys())
        bm25_top_indices = bm25_scores.argsort()[-k_candidates:][::-1]
        all_candidates.update(bm25_top_indices)
        
        hybrid_scores = {}
        for idx in all_candidates:
            semantic_score = semantic_candidates.get(idx, 0.0)
            bm25_score = bm25_scores_norm[idx]
            hybrid_score = semantic_weight * semantic_score + bm25_weight * bm25_score
            hybrid_scores[idx] = hybrid_score
        
        top_k_indices = sorted(hybrid_scores.keys(), key=lambda x: hybrid_scores[x], reverse=True)[:k]
        
        results = []
        for idx in top_k_indices:
            play = self.corpus[idx]
            results.append({
                'play_id': play['play_id'],
                'text': play['text'],
                'similarity': float(hybrid_scores[idx]),
                'play_data': play['play_data'],
                'reference_count': play['reference_count']
            })
        
        return {
            'query': query_text,
            'retrieved_plays': results,
            'latency_ms': {
                'embedding': embed_latency,
                'semantic_search': semantic_latency,
                'bm25_search': bm25_latency,
                'total': embed_latency + semantic_latency + bm25_latency
            },
            'k': k,
            'retrieval_mode': 'hybrid'
        }

def format_situation_for_query(situation: Dict, history: List[Dict] = None, teams: Dict = None) -> str:
    """
    Convert a structured situation dict into a natural language query.
    Matches document text format: teams + situation + history.
    """
    down = situation.get('down', '?')
    togo = situation.get('togo', '?')
    yardline = situation.get('yardline', '?')
    quarter = situation.get('quarter', '?')
    offense = situation.get('offense', '')
    defense = situation.get('defense', '')
    
    text = ""
    
    if teams:
        offense = teams.get('offense', offense)
        defense = teams.get('defense', defense)
    
    if offense and defense:
        text += f"{offense} offense vs {defense} defense. "
    
    text += f"Situation: {down} and {togo} at {yardline} yard line, quarter {quarter}. "
    
    if history and len(history) > 0:
        history_text = []
        for h in history[:5]:
            h_type = h.get('playtype', '')
            h_dir = h.get('direction', '')
            h_yards = h.get('yards', '0')
            if h_type:
                h_str = h_type
                if h_dir and h_dir != 'null':
                    h_str += f" {h_dir}"
                h_str += f" {h_yards}yd"
                history_text.append(h_str)
        if history_text:
            text += f"Recent plays: {', '.join(history_text)}. "
    
    return text.strip()

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "DATA"
    
    print("=" * 70)
    print("Play Retriever Test")
    print("=" * 70)
    print()
    
    retriever = PlayRetriever(data_dir)
    
    print("\n" + "=" * 70)
    print("Test Queries")
    print("=" * 70)
    
    test_queries = [
        "3rd and 5 at 25 yard line, pass short middle for touchdown",
        "1st and 10 at 50 yard line, run left for 8 yards",
        "4th and 1 at 10 yard line, quarterback sneak",
        "2nd and goal at 2 yard line, pass to tight end"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = retriever.query(query, k=5)
        
        print(f"Latency: {result['latency_ms']['total']:.2f}ms (embed: {result['latency_ms']['embedding']:.2f}ms, search: {result['latency_ms']['search']:.2f}ms)")
        print(f"\nTop 5 results:")
        for i, play in enumerate(result['retrieved_plays'], 1):
            print(f"  {i}. [{play['similarity']:.4f}] {play['text'][:80]}...")
