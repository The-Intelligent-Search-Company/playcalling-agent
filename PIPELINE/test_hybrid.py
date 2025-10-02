from pathlib import Path
from query_pipeline import PlayRetriever

data_dir = Path(__file__).parent.parent / "DATA"

print("=" * 70)
print("Hybrid Retrieval Test")
print("=" * 70)

print("\n[1/2] Initializing semantic-only retriever...")
retriever_semantic = PlayRetriever(data_dir, use_hybrid=False)

print("\n[2/2] Initializing hybrid retriever...")
retriever_hybrid = PlayRetriever(data_dir, use_hybrid=True)

test_queries = [
    "3rd and 5 at 25 yard line, pass short middle for touchdown",
    "1st and 10 at 50 yard line, run left for 8 yards",
    "2nd and goal at 2 yard line, pass to tight end",
    "4th and 1 at 10 yard line, quarterback sneak"
]

print("\n" + "=" * 70)
print("Comparing Semantic vs Hybrid Retrieval")
print("=" * 70)

for i, query in enumerate(test_queries, 1):
    print(f"\n[Query {i}] {query}")
    print("-" * 70)
    
    result_semantic = retriever_semantic.query(query, k=10)
    result_hybrid = retriever_hybrid.query(query, k=10)
    
    print(f"\nSemantic-only:")
    print(f"  Latency: {result_semantic['latency_ms']['total']:.2f}ms")
    print(f"  Top 3 plays:")
    for j, play in enumerate(result_semantic['retrieved_plays'][:3], 1):
        print(f"    {j}. [{play['similarity']:.4f}] {play['play_id'][:20]}... | {play['text'][:60]}...")
    
    print(f"\nHybrid (0.7 semantic + 0.3 BM25):")
    print(f"  Latency: {result_hybrid['latency_ms']['total']:.2f}ms")
    print(f"  Top 3 plays:")
    for j, play in enumerate(result_hybrid['retrieved_plays'][:3], 1):
        print(f"    {j}. [{play['similarity']:.4f}] {play['play_id'][:20]}... | {play['text'][:60]}...")
    
    semantic_ids = [p['play_id'] for p in result_semantic['retrieved_plays'][:10]]
    hybrid_ids = [p['play_id'] for p in result_hybrid['retrieved_plays'][:10]]
    
    overlap = len(set(semantic_ids) & set(hybrid_ids))
    print(f"\n  Overlap in top-10: {overlap}/10 plays")
    print(f"  New plays from BM25: {10 - overlap}")

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
