import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
import re
import time

def tokenize(text: str) -> list[str]:
    """
    Simple tokenization: lowercase, split on whitespace and punctuation.
    """
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def build_bm25_index(corpus_path: Path, output_dir: Path):
    """
    Build BM25 index for play corpus.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading play corpus...")
    with open(corpus_path) as f:
        corpus = json.load(f)
    
    print(f"Total plays: {len(corpus)}")
    
    print("\nTokenizing play texts...")
    start_time = time.time()
    tokenized_corpus = []
    play_ids = []
    
    for play in corpus:
        tokens = tokenize(play['text'])
        tokenized_corpus.append(tokens)
        play_ids.append(play['play_id'])
    
    tokenize_time = time.time() - start_time
    print(f"✓ Tokenized {len(tokenized_corpus)} plays in {tokenize_time:.2f}s")
    
    print("\nBuilding BM25 index...")
    start_time = time.time()
    bm25 = BM25Okapi(tokenized_corpus)
    build_time = time.time() - start_time
    
    print(f"✓ Built BM25 index in {build_time:.2f}s")
    
    bm25_data = {
        'bm25': bm25,
        'play_ids': play_ids,
        'num_plays': len(corpus),
        'tokenize_time': tokenize_time,
        'build_time': build_time
    }
    
    bm25_path = output_dir / 'bm25_index.pkl'
    print(f"\nSaving BM25 index to {bm25_path}...")
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25_data, f)
    
    bm25_size_mb = bm25_path.stat().st_size / 1024 / 1024
    print(f"✓ Saved ({bm25_size_mb:.2f} MB)")
    
    print("\nTesting BM25 retrieval...")
    test_query = "3rd and 5 at 25 yard line pass short middle for 10 yards touchdown"
    test_tokens = tokenize(test_query)
    
    start_time = time.time()
    scores = bm25.get_scores(test_tokens)
    top_k_indices = scores.argsort()[-5:][::-1]
    search_time = (time.time() - start_time) * 1000
    
    print(f"✓ Test search completed in {search_time:.2f}ms")
    print(f"\nTop 5 BM25 results for test query:")
    for i, idx in enumerate(top_k_indices):
        play = corpus[idx]
        print(f"  {i+1}. Score: {scores[idx]:.4f}")
        print(f"     Play ID: {play['play_id']}")
        print(f"     {play['text'][:100]}...")
    
    return bm25_data

if __name__ == "__main__":
    corpus_path = Path(__file__).parent.parent / "DATA" / "play_corpus.json"
    output_dir = Path(__file__).parent.parent / "DATA"
    
    print("=" * 70)
    print("BM25 Index Builder")
    print("=" * 70)
    
    bm25_data = build_bm25_index(corpus_path, output_dir)
    
    print("\n" + "=" * 70)
    print("BM25 index build complete!")
    print("=" * 70)
    print(f"Total plays indexed: {bm25_data['num_plays']}")
    print(f"Tokenize time: {bm25_data['tokenize_time']:.2f}s")
    print(f"Build time: {bm25_data['build_time']:.2f}s")
