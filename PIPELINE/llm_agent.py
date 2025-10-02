import os
import sys
import time
from pathlib import Path
from openai import OpenAI
from query_pipeline import PlayRetriever

def create_system_prompt():
    """
    System prompt for the LLM agent.
    The LLM acts as a pure reasoning layer - outputs a detailed play call.
    """
    return """You are an NFL play recommendation agent.

Your task: Analyze the provided historical play data and recommend a detailed play call for the given situation.

CRITICAL RULES:
1. Base your recommendation on patterns from the provided play data
2. Consider situational similarity (down, distance, field position)
3. Synthesize information from multiple plays to create an optimal recommendation
4. Output a detailed play call with formation, play type, direction, and target/gap
5. Do not reference specific players or teams from history
6. Do not provide explanations or alternatives

Output format: FORMATION - PLAYTYPE DIRECTION to TARGET/GAP
Examples:
- "SHOTGUN - PASS SHORT RIGHT to SLOT RECEIVER"
- "UNDER CENTER - RUSH LEFT TACKLE through A-GAP"
- "SHOTGUN NO HUDDLE - PASS DEEP MIDDLE to TIGHT END"
- "I-FORMATION - RUSH RIGHT END outside"
- "EMPTY SET - PASS SHORT LEFT to RUNNING BACK"
"""

def create_user_prompt(query: str, retrieved_plays: list) -> str:
    """
    User prompt with query and retrieved candidate plays.
    """
    prompt = f"Situation/Query: {query}\n\n"
    prompt += "Candidate plays (ranked by retrieval similarity):\n\n"
    
    for i, play in enumerate(retrieved_plays, 1):
        play_data = play['play_data']
        situation = play_data['context']['situation']
        play_info = play_data['play']
        
        prompt += f"Play {i}:\n"
        prompt += f"  play_id: {play['play_id']}\n"
        prompt += f"  Situation: {situation['down']} and {situation['togo']} at {situation['yardline']}, Q{situation['quarter']}\n"
        prompt += f"  Play: {play_info.get('playtype', 'UNKNOWN')}"
        
        direction = play_info.get('direction')
        if direction and direction != 'null':
            prompt += f" {direction}"
        
        yards = play_info.get('yards', '0')
        prompt += f" for {yards} yards\n"
        
        desc = play_info.get('description', '')
        if desc:
            prompt += f"  Description: {desc[:150]}\n"
        
        prompt += f"  Retrieval similarity: {play['similarity']:.4f}\n"
        prompt += "\n"
    
    prompt += "Recommend the best play call (FORMATION - PLAYTYPE DIRECTION to TARGET):"
    
    return prompt

def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 50, temperature: float = 0.3):
    """
    Call OpenAI API with the given prompts.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        llm_latency = (time.time() - start_time) * 1000
        
        content = response.choices[0].message.content.strip()
        
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        
        return {
            'play_maneuver': content,
            'llm_latency_ms': llm_latency
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'llm_latency_ms': (time.time() - start_time) * 1000
        }

def get_play_recommendation(query: str, retriever: PlayRetriever, k: int = 20, model: str = "gpt-4o-mini"):
    """
    Full pipeline: retrieve candidates and use LLM to select best play.
    """
    retrieval_result = retriever.query(query, k=k)
    
    system_prompt = create_system_prompt()
    user_prompt = create_user_prompt(query, retrieval_result['retrieved_plays'])
    
    llm_result = call_llm(system_prompt, user_prompt, model=model)
    
    if 'error' in llm_result:
        return {
            'query': query,
            'error': llm_result['error'],
            'latency_ms': {
                'retrieval': retrieval_result['latency_ms']['total'],
                'llm': llm_result['llm_latency_ms'],
                'total': retrieval_result['latency_ms']['total'] + llm_result['llm_latency_ms']
            }
        }
    
    recommended_play_maneuver = llm_result['play_maneuver']
    
    best_match = None
    best_score = 0
    
    for play in retrieval_result['retrieved_plays']:
        play_info = play['play_data']['play']
        playtype = (play_info.get('playtype') or '').upper()
        direction = (play_info.get('direction') or '').upper()
        formation = (play_info.get('formation') or '').upper()
        
        score = 0
        maneuver_upper = recommended_play_maneuver.upper()
        
        if playtype and playtype in maneuver_upper:
            score += 3
        if direction and direction != 'NULL' and direction in maneuver_upper:
            score += 2
        if formation and formation != 'NULL' and formation in maneuver_upper:
            score += 1
        
        if score > best_score:
            best_score = score
            best_match = play
    
    recommended_play = best_match
    
    return {
        'query': query,
        'recommended_play_maneuver': recommended_play_maneuver,
        'recommended_play': recommended_play,
        'retrieved_plays': retrieval_result['retrieved_plays'],
        'latency_ms': {
            'retrieval': retrieval_result['latency_ms']['total'],
            'llm': llm_result['llm_latency_ms'],
            'total': retrieval_result['latency_ms']['total'] + llm_result['llm_latency_ms']
        },
        'k': k
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "3rd and 5 at 25 yard line, pass short middle for touchdown"
    
    print("=" * 70)
    print("NFL Play LLM Agent")
    print("=" * 70)
    print()
    
    data_dir = Path(__file__).parent.parent / "DATA"
    retriever = PlayRetriever(data_dir, use_hybrid=True)
    
    print("\n" + "=" * 70)
    print(f"Query: {query}")
    print("=" * 70)
    
    result = get_play_recommendation(query, retriever, k=20, model="gpt-4o-mini")
    
    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
    else:
        print(f"\n🎯 Recommended Play Call:")
        print(f"   {result['recommended_play_maneuver']}")
        print()
        
        if result['recommended_play']:
            play_data = result['recommended_play']['play_data']
            situation = play_data['context']['situation']
            play_info = play_data['play']
            
            print(f"\n📊 Similar Historical Play (for reference):")
            print(f"   Situation: {situation['down']} and {situation['togo']} at {situation['yardline']}, Q{situation['quarter']}")
            print(f"   Teams: {situation['offense']} vs {situation['defense']}")
            
            formation = play_info.get('formation', 'UNKNOWN')
            playtype = play_info.get('playtype', 'UNKNOWN')
            direction = play_info.get('direction', '')
            yards = play_info.get('yards', '0')
            
            print(f"   Formation: {formation}")
            print(f"   Play: {playtype}", end='')
            if direction and direction != 'null':
                print(f" {direction}", end='')
            print(f" → {yards} yards")
            
            desc = play_info.get('description', '')
            if desc:
                print(f"   Description: {desc[:120]}...")
        else:
            print(f"\n⚠️  No matching play pattern found in top-{result['k']} retrieved plays")
        
    print(f"\n⚡ Performance:")
    print(f"   Retrieval: {result['latency_ms']['retrieval']:.2f}ms")
    print(f"   LLM: {result['latency_ms']['llm']:.2f}ms")
    print(f"   Total: {result['latency_ms']['total']:.2f}ms")
    
    print("\n" + "=" * 70)
