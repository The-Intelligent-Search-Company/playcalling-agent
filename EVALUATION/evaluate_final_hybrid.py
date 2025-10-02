import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "PIPELINE"))

import pandas as pd
import json
import numpy as np
import time
from collections import defaultdict
from query_pipeline import PlayRetriever, format_situation_for_query
from llm_agent import get_play_recommendation

def create_test_set(training_csv_path: str, test_size: int = 100, seed: int = 42):
    """
    Create test set from training_data.csv.
    """
    df = pd.read_csv(training_csv_path)
    
    unique_queries = df[['query', 'target_play_id']].drop_duplicates()
    
    np.random.seed(seed)
    test_indices = np.random.choice(len(unique_queries), size=min(test_size, len(unique_queries)), replace=False)
    test_df = unique_queries.iloc[test_indices].reset_index(drop=True)
    
    test_queries = []
    for idx, row in test_df.iterrows():
        query_dict = json.loads(row['query'])
        situation = query_dict['situation']
        history = query_dict.get('history', [])
        
        teams = {}
        if 'offense' in situation:
            teams['offense'] = situation['offense']
        if 'defense' in situation:
            teams['defense'] = situation['defense']
        
        query_text = format_situation_for_query(situation, history=history, teams=teams if teams else None)
        
        test_queries.append({
            'query_dict': query_dict,
            'query_text': query_text,
            'ground_truth_play_id': row['target_play_id']
        })
    
    return test_queries

def evaluate_retrieval_comprehensive(test_queries: list, retriever: PlayRetriever, k_values=[5, 10, 20, 50]):
    """
    Comprehensive retrieval evaluation with detailed metrics.
    """
    print(f"\n{'='*80}")
    print(f"RETRIEVAL EVALUATION - HYBRID (0.7 semantic + 0.3 BM25)")
    print(f"{'='*80}")
    print(f"\nCorpus size: {len(retriever.corpus)} plays")
    print(f"Test queries: {len(test_queries)}")
    print(f"\nEvaluating...")
    
    results = []
    latencies = []
    
    hits_at_k = {k: 0 for k in k_values}
    reciprocal_ranks = []
    
    for i, test_query in enumerate(test_queries):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(test_queries)}")
        
        query_text = test_query['query_text']
        ground_truth = test_query['ground_truth_play_id']
        
        result = retriever.query(query_text, k=max(k_values))
        latencies.append(result['latency_ms']['total'])
        
        retrieved_play_ids = [play['play_id'] for play in result['retrieved_plays']]
        
        rank = None
        if ground_truth in retrieved_play_ids:
            rank = retrieved_play_ids.index(ground_truth) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
        
        for k in k_values:
            if ground_truth in retrieved_play_ids[:k]:
                hits_at_k[k] += 1
        
        results.append({
            'query': query_text,
            'ground_truth': ground_truth,
            'retrieved_ids': retrieved_play_ids[:max(k_values)],
            'rank': rank,
            'latency_ms': result['latency_ms']['total']
        })
    
    metrics = {
        'hits_at_k': {k: hits_at_k[k] / len(test_queries) for k in k_values},
        'mrr': np.mean(reciprocal_ranks),
        'latency_ms': {
            'p50': float(np.percentile(latencies, 50)),
            'p90': float(np.percentile(latencies, 90)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
            'mean': float(np.mean(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies))
        }
    }
    
    print(f"\n{'='*80}")
    print("RETRIEVAL RESULTS")
    print(f"{'='*80}")
    print(f"\nHits@K (% of queries where ground-truth in top-K):")
    for k in k_values:
        print(f"  Hits@{k:2d}: {metrics['hits_at_k'][k]:6.1%} ({hits_at_k[k]}/{len(test_queries)})")
    
    print(f"\nMean Reciprocal Rank:")
    print(f"  MRR: {metrics['mrr']:.4f}")
    
    print(f"\nRetrieval Latency (ms):")
    print(f"  p50: {metrics['latency_ms']['p50']:7.2f}ms")
    print(f"  p90: {metrics['latency_ms']['p90']:7.2f}ms")
    print(f"  p95: {metrics['latency_ms']['p95']:7.2f}ms")
    print(f"  p99: {metrics['latency_ms']['p99']:7.2f}ms")
    print(f"  mean: {metrics['latency_ms']['mean']:6.2f}ms")
    
    return metrics, results

def evaluate_llm_play_calls(test_queries: list, retriever: PlayRetriever, k: int = 20, max_queries: int = 50):
    """
    Evaluate LLM play call generation with detailed metrics.
    """
    if max_queries:
        test_queries = test_queries[:max_queries]
    
    print(f"\n{'='*80}")
    print(f"LLM PLAY CALL EVALUATION")
    print(f"{'='*80}")
    print(f"\nTest queries: {len(test_queries)}")
    print(f"Retrieval: Top-{k} plays per query")
    print(f"\nEvaluating...")
    
    results = []
    latencies_total = []
    latencies_retrieval = []
    latencies_llm = []
    
    play_calls_generated = 0
    top_k_coverage = 0
    play_type_matches = 0
    errors = 0
    
    for i, test_query in enumerate(test_queries):
        print(f"  Progress: {i+1}/{len(test_queries)}", end='\r')
        
        query_text = test_query['query_text']
        ground_truth = test_query['ground_truth_play_id']
        
        try:
            result = get_play_recommendation(query_text, retriever, k=k, model="gpt-4o-mini")
            
            if 'error' in result:
                errors += 1
                print(f"\n  ⚠️  Error on query {i+1}: {result['error']}")
                continue
            
            play_calls_generated += 1
            
            recommended_maneuver = result['recommended_play_maneuver']
            
            latencies_total.append(result['latency_ms']['total'])
            latencies_retrieval.append(result['latency_ms']['retrieval'])
            latencies_llm.append(result['latency_ms']['llm'])
            
            retrieved_ids = [play['play_id'] for play in result['retrieved_plays']]
            if ground_truth in retrieved_ids:
                top_k_coverage += 1
            
            ground_truth_play = None
            for play in retriever.corpus:
                if play['play_id'] == ground_truth:
                    ground_truth_play = play
                    break
            
            if ground_truth_play and result['recommended_play']:
                gt_playtype = ground_truth_play['play_data']['play'].get('playtype', '').upper()
                rec_playtype = result['recommended_play']['play_data']['play'].get('playtype', '').upper()
                
                if gt_playtype and rec_playtype and gt_playtype == rec_playtype:
                    play_type_matches += 1
            
            results.append({
                'query': query_text,
                'ground_truth_play_id': ground_truth,
                'recommended_play_call': recommended_maneuver,
                'matched_play_id': result['recommended_play']['play_id'] if result['recommended_play'] else None,
                'in_top_k': (ground_truth in retrieved_ids),
                'latency_ms': result['latency_ms']
            })
        
        except Exception as e:
            errors += 1
            print(f"\n  ⚠️  Exception on query {i+1}: {str(e)}")
            continue
    
    print(f"\n")
    
    metrics = {
        'play_calls_generated': play_calls_generated,
        'top_k_coverage': top_k_coverage / len(results) if results else 0,
        'play_type_match': play_type_matches / len(results) if results else 0,
        'num_queries': len(results),
        'errors': errors,
        'latency_ms': {
            'total': {
                'p50': float(np.percentile(latencies_total, 50)) if latencies_total else 0,
                'p90': float(np.percentile(latencies_total, 90)) if latencies_total else 0,
                'p95': float(np.percentile(latencies_total, 95)) if latencies_total else 0,
                'p99': float(np.percentile(latencies_total, 99)) if latencies_total else 0,
                'mean': float(np.mean(latencies_total)) if latencies_total else 0
            },
            'retrieval': {
                'p50': float(np.percentile(latencies_retrieval, 50)) if latencies_retrieval else 0,
                'p90': float(np.percentile(latencies_retrieval, 90)) if latencies_retrieval else 0,
                'p95': float(np.percentile(latencies_retrieval, 95)) if latencies_retrieval else 0
            },
            'llm': {
                'p50': float(np.percentile(latencies_llm, 50)) if latencies_llm else 0,
                'p90': float(np.percentile(latencies_llm, 90)) if latencies_llm else 0,
                'p95': float(np.percentile(latencies_llm, 95)) if latencies_llm else 0
            }
        }
    }
    
    print(f"{'='*80}")
    print("LLM PLAY CALL RESULTS")
    print(f"{'='*80}")
    print(f"\nPlay Call Quality:")
    print(f"  Play calls generated: {play_calls_generated}/{len(test_queries)}")
    print(f"  Top-K coverage:       {metrics['top_k_coverage']:6.1%} (ground-truth in retrieved plays)")
    print(f"  Play type match:      {metrics['play_type_match']:6.1%} (same PASS/RUSH as ground-truth)")
    print(f"  Errors:               {errors}")
    
    print(f"\nEnd-to-End Latency (ms):")
    print(f"  p50: {metrics['latency_ms']['total']['p50']:7.2f}ms")
    print(f"  p90: {metrics['latency_ms']['total']['p90']:7.2f}ms")
    print(f"  p95: {metrics['latency_ms']['total']['p95']:7.2f}ms")
    print(f"  p99: {metrics['latency_ms']['total']['p99']:7.2f}ms")
    
    print(f"\nLatency Breakdown:")
    print(f"  Retrieval p50: {metrics['latency_ms']['retrieval']['p50']:7.2f}ms")
    print(f"  LLM p50:       {metrics['latency_ms']['llm']['p50']:7.2f}ms")
    
    return metrics, results

if __name__ == "__main__":
    import os
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        print("Please set: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    print("=" * 80)
    print("NFL PLAY RETRIEVAL AGENT - FINAL EVALUATION")
    print("Hybrid Retrieval (0.7 Semantic + 0.3 BM25) + LLM Play Call Generation")
    print("=" * 80)
    
    training_csv = Path(__file__).parent.parent.parent / "training_data.csv"
    data_dir = Path(__file__).parent.parent / "DATA"
    
    print("\n[1/3] Creating test set...")
    test_queries = create_test_set(training_csv, test_size=100, seed=42)
    print(f"✓ Created test set with {len(test_queries)} queries")
    
    print("\n[2/3] Initializing hybrid retriever...")
    retriever = PlayRetriever(data_dir, use_hybrid=True)
    
    retrieval_metrics, retrieval_results = evaluate_retrieval_comprehensive(
        test_queries, retriever, k_values=[5, 10, 20, 50]
    )
    
    print("\n[3/3] Evaluating LLM play call generation (50 queries)...")
    llm_metrics, llm_results = evaluate_llm_play_calls(
        test_queries, retriever, k=20, max_queries=50
    )
    
    evaluation_data = {
        'metadata': {
            'test_set_size': len(test_queries),
            'llm_test_size': len(llm_results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'corpus_size': len(retriever.corpus),
            'retrieval_mode': 'hybrid',
            'hybrid_weights': {'semantic': 0.7, 'bm25': 0.3}
        },
        'retrieval_metrics': retrieval_metrics,
        'llm_metrics': llm_metrics,
        'sample_retrieval_results': retrieval_results[:10],
        'sample_llm_results': llm_results[:10]
    }
    
    output_path = Path(__file__).parent / 'final_evaluation_results.json'
    print(f"\n\nSaving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    print("✓ Saved")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nKey Results:")
    print(f"  Hits@10:      {retrieval_metrics['hits_at_k'][10]:.1%}")
    print(f"  Hits@20:      {retrieval_metrics['hits_at_k'][20]:.1%}")
    print(f"  MRR:          {retrieval_metrics['mrr']:.4f}")
    print(f"  Retrieval p50: {retrieval_metrics['latency_ms']['p50']:.2f}ms")
    print(f"  End-to-end p50: {llm_metrics['latency_ms']['total']['p50']:.2f}ms")
    print(f"\nResults saved to: {output_path}")
