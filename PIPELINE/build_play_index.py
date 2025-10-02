import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pickle
import time

def build_embeddings_and_index(corpus_path: Path, output_dir: Path):
    """
    Build E5-base-v2 embeddings and FAISS HNSW index for play corpus.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading play corpus...")
    with open(corpus_path) as f:
        corpus = json.load(f)
    
    print(f"Total plays: {len(corpus)}")
    
    print("\nLoading E5-base-v2 embedding model...")
    model = SentenceTransformer('intfloat/e5-base-v2')
    
    texts = []
    play_ids = []
    for play in corpus:
        texts.append("passage: " + play['text'])
        play_ids.append(play['play_id'])
    
    print(f"\nGenerating embeddings for {len(texts)} plays...")
    start_time = time.time()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    embed_time = time.time() - start_time
    
    print(f"✓ Generated embeddings in {embed_time:.2f}s")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Avg embedding time: {1000*embed_time/len(texts):.2f}ms per play")
    
    embeddings_path = output_dir / 'play_embeddings.npy'
    print(f"\nSaving embeddings to {embeddings_path}...")
    np.save(embeddings_path, embeddings)
    print(f"✓ Saved ({embeddings.nbytes / 1024 / 1024:.2f} MB)")
    
    print("\nBuilding FAISS HNSW index...")
    dimension = embeddings.shape[1]
    print(f"  Dimension: {dimension}")
    
    M = 32
    efConstruction = 200
    efSearch = 64
    
    print(f"  Parameters: M={M}, efConstruction={efConstruction}")
    
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = efConstruction
    
    print(f"  Adding {len(embeddings)} vectors...")
    start_time = time.time()
    index.add(embeddings.astype('float32'))
    build_time = time.time() - start_time
    
    print(f"✓ Built index in {build_time:.2f}s")
    
    index.hnsw.efSearch = efSearch
    print(f"  Set efSearch={efSearch} for retrieval")
    
    index_path = output_dir / 'play_index.faiss'
    print(f"\nSaving FAISS index to {index_path}...")
    faiss.write_index(index, str(index_path))
    
    index_size_mb = index_path.stat().st_size / 1024 / 1024
    print(f"✓ Saved ({index_size_mb:.2f} MB)")
    
    metadata = {
        'model_name': 'intfloat/e5-base-v2',
        'dimension': dimension,
        'num_plays': len(corpus),
        'play_ids': play_ids,
        'index_params': {
            'M': M,
            'efConstruction': efConstruction,
            'efSearch': efSearch
        },
        'build_time': build_time,
        'embedding_time': embed_time
    }
    
    metadata_path = output_dir / 'play_index_metadata.pkl'
    print(f"\nSaving metadata to {metadata_path}...")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved")
    
    print("\nTesting index retrieval...")
    test_query = "passage: 3rd and 5 at 25 yard line, pass short middle for 10 yards touchdown"
    test_embed = model.encode([test_query], convert_to_numpy=True, normalize_embeddings=True)
    
    start_time = time.time()
    distances, indices = index.search(test_embed.astype('float32'), 5)
    search_time = (time.time() - start_time) * 1000
    
    print(f"✓ Test search completed in {search_time:.2f}ms")
    print(f"\nTop 5 results for test query:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        play = corpus[idx]
        print(f"  {i+1}. Similarity: {1-dist:.4f}")
        print(f"     Play ID: {play['play_id']}")
        print(f"     {play['text'][:100]}...")
    
    return metadata

if __name__ == "__main__":
    corpus_path = Path(__file__).parent.parent / "DATA" / "play_corpus.json"
    output_dir = Path(__file__).parent.parent / "DATA"
    
    print("=" * 70)
    print("NFL Play Index Builder")
    print("=" * 70)
    
    metadata = build_embeddings_and_index(corpus_path, output_dir)
    
    print("\n" + "=" * 70)
    print("Index build complete!")
    print("=" * 70)
    print(f"Total plays indexed: {metadata['num_plays']}")
    print(f"Embedding model: {metadata['model_name']}")
    print(f"Dimension: {metadata['dimension']}")
    print(f"Build time: {metadata['build_time']:.2f}s")
