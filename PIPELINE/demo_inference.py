import os
from pathlib import Path
from query_pipeline import PlayRetriever
from llm_agent import get_play_recommendation, create_system_prompt, create_user_prompt

data_dir = Path(__file__).parent.parent / "DATA"

test_queries = [
    "3rd and 7 at 30 yard line, need first down",
    "2nd and goal at 2 yard line, looking for touchdown",
    "1st and 10 at midfield, establish run game",
]

print("=" * 80)
print("NFL Play LLM Agent - Demo Inference")
print("=" * 80)

print("\n[1/2] Loading semantic-only retriever...")
retriever_semantic = PlayRetriever(data_dir, use_hybrid=False)

print("\n[2/2] Loading hybrid retriever...")
retriever_hybrid = PlayRetriever(data_dir, use_hybrid=True)

for i, query in enumerate(test_queries, 1):
    print("\n" + "=" * 80)
    print(f"Query {i}: {query}")
    print("=" * 80)
    
    print("\n--- SEMANTIC-ONLY RETRIEVAL ---")
    result_semantic = get_play_recommendation(query, retriever_semantic, k=10, model="gpt-4o-mini")
    
    print("\nTop-5 Retrieved Plays:")
    for j, play in enumerate(result_semantic['retrieved_plays'][:5], 1):
        play_data = play['play_data']
        situation = play_data['context']['situation']
        play_info = play_data['play']
        print(f"  {j}. [{play['similarity']:.3f}] {play_info.get('playtype')} {play_info.get('direction', '')} for {play_info.get('yards')}yd")
        print(f"      Situation: {situation['down']} & {situation['togo']} at {situation['yardline']}, Q{situation['quarter']}")
        print(f"      ID: {play['play_id'][:20]}...")
    
    print(f"\n🤖 LLM REASONING (System Prompt):")
    print(create_system_prompt()[:200] + "...")
    
    print(f"\n💡 LLM Selected: Play #{result_semantic['retrieved_plays'].index(result_semantic['recommended_play']) + 1 if result_semantic['recommended_play'] in result_semantic['retrieved_plays'] else 'NOT IN TOP-K'}")
    print(f"   Play ID: {result_semantic['recommended_play_id']}")
    
    if result_semantic['recommended_play']:
        play_data = result_semantic['recommended_play']['play_data']
        play_info = play_data['play']
        print(f"   Details: {play_info.get('playtype')} {play_info.get('direction', '')} for {play_info.get('yards')}yd")
    
    print(f"\n⚡ Latency: Retrieval={result_semantic['latency_ms']['retrieval']:.0f}ms, LLM={result_semantic['latency_ms']['llm']:.0f}ms, Total={result_semantic['latency_ms']['total']:.0f}ms")
    
    print("\n--- HYBRID RETRIEVAL (0.7 semantic + 0.3 BM25) ---")
    result_hybrid = get_play_recommendation(query, retriever_hybrid, k=10, model="gpt-4o-mini")
    
    print("\nTop-5 Retrieved Plays:")
    for j, play in enumerate(result_hybrid['retrieved_plays'][:5], 1):
        play_data = play['play_data']
        situation = play_data['context']['situation']
        play_info = play_data['play']
        print(f"  {j}. [{play['similarity']:.3f}] {play_info.get('playtype')} {play_info.get('direction', '')} for {play_info.get('yards')}yd")
        print(f"      Situation: {situation['down']} & {situation['togo']} at {situation['yardline']}, Q{situation['quarter']}")
        print(f"      ID: {play['play_id'][:20]}...")
    
    print(f"\n💡 LLM Selected: Play #{result_hybrid['retrieved_plays'].index(result_hybrid['recommended_play']) + 1 if result_hybrid['recommended_play'] in result_hybrid['retrieved_plays'] else 'NOT IN TOP-K'}")
    print(f"   Play ID: {result_hybrid['recommended_play_id']}")
    
    if result_hybrid['recommended_play']:
        play_data = result_hybrid['recommended_play']['play_data']
        play_info = play_data['play']
        print(f"   Details: {play_info.get('playtype')} {play_info.get('direction', '')} for {play_info.get('yards')}yd")
    
    print(f"\n⚡ Latency: Retrieval={result_hybrid['latency_ms']['retrieval']:.0f}ms, LLM={result_hybrid['latency_ms']['llm']:.0f}ms, Total={result_hybrid['latency_ms']['total']:.0f}ms")
    
    if result_semantic['recommended_play_id'] != result_hybrid['recommended_play_id']:
        print(f"\n⚠️  Different recommendations! Semantic vs Hybrid changed the LLM's choice.")

print("\n" + "=" * 80)
print("Demo Complete")
print("=" * 80)
