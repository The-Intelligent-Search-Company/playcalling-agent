#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, re
from typing import Any, Dict, List
import modal

app = modal.App("synthetic-inspirations")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "psycopg[binary]",
    "openai"
)

PG_URL = os.getenv("PG_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

SYSTEM_PROMPT = """You are an NFL play design expert. Your task: given ANY successful play, imagine a red-zone scenario where its core concept would work.

BE LIBERAL. Nearly all plays have SOME transferable element:
- Screens/RPOs → work in compressed spaces
- Play-action → freezes LBs near goal line
- Misdirection → harder to defend in tight field
- Power runs → goal-line situations
- Route combinations → can be shortened to red-zone depths

Your job: CREATE a plausible red-zone scenario where this play's concept applies.

Return JSON:
{
  "is_inspirational": true,
  "synthetic_redzone_scenario": {
    "situation": {"yardline": <1-20>, "down": <1-4>, "togo": <int>, "quarter": <int>, "offense": "TEAM", "defense": "TEAM"},
    "history": [{"playtype": "", "direction": "", "yards": ""}, ...]
  },
  "reasoning": "How this play's concept translates to red zone..."
}

Only return false if the play is a pure broken coverage deep shot with NO transferable elements.
"""

def is_snap(r: Dict[str,Any]) -> bool:
    if (r.get("isnoplay") or "").strip() == "1": return False
    pt = (r.get("playtype") or "").upper()
    return pt in {"RUSH","PASS","SCRAMBLE","SACK"}

def build_candidate_prompt(play: Dict[str, Any], history: List[Dict[str, Any]], examples: List[Dict[str, Any]]) -> str:
    prompt = "**PLAY TO EVALUATE:**\n\n"
    
    play_context = {
        "play": {
            "formation": play.get("formation", ""),
            "playtype": play.get("playtype", ""),
            "direction": play.get("rushdirection") or play.get("passtype", ""),
            "yards": play.get("yards", ""),
            "description": play.get("description", "")
        },
        "situation": {
            "yardline": play.get("yardlinefixed", ""),
            "down": play.get("down", ""),
            "togo": play.get("togo", ""),
            "quarter": play.get("quarter", ""),
            "offense": play.get("offenseteam", ""),
            "defense": play.get("defenseteam", "")
        },
        "history": [
            {
                "playtype": h.get("playtype", ""),
                "direction": h.get("rushdirection") or h.get("passtype", ""),
                "yards": h.get("yards", "")
            } for h in history[:5]
        ]
    }
    
    prompt += json.dumps(play_context, indent=2)
    prompt += "\n\nCould this play inspire a red-zone situation? Generate synthetic scenario if yes."
    
    return prompt

@app.function(image=image, timeout=600)
def evaluate_play(play_data: Dict[str, Any], history_data: List[Dict[str, Any]], example_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    import psycopg
    from psycopg.rows import dict_row
    from openai import OpenAI
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    prompt = build_candidate_prompt(play_data, history_data, example_pairs)
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        print(f"Play {play_data['id']}: is_inspirational={result.get('is_inspirational')}")
        if not result.get("is_inspirational"):
            print(f"  Reason: {result.get('reasoning', 'No reasoning provided')[:200]}")
        
        if result.get("is_inspirational") and result.get("synthetic_redzone_scenario"):
            with psycopg.connect(os.environ.get("PG_URL"), row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO nfl_synthetic_inspirations 
                        (synthetic_redzone_query, inspiration_play_id, reasoning)
                        VALUES (%s, %s, %s)
                    """, (
                        json.dumps(result["synthetic_redzone_scenario"]),
                        str(play_data["id"]),
                        result.get("reasoning", "")
                    ))
                conn.commit()
            
            return {
                "play_id": str(play_data["id"]),
                "is_inspirational": True,
                "scenario": result["synthetic_redzone_scenario"],
                "reasoning": result.get("reasoning", "")
            }
        else:
            return {
                "play_id": str(play_data["id"]),
                "is_inspirational": False
            }
    
    except Exception as e:
        return {
            "play_id": str(play_data["id"]),
            "error": str(e)
        }

@app.local_entrypoint()
def main():
    import psycopg
    from psycopg.rows import dict_row
    
    print("Fetching example inspiration pairs...")
    if not os.environ.get("PG_URL"):
        raise RuntimeError("PG_URL not set in environment")
    with psycopg.connect(os.environ.get("PG_URL"), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    i.play_id,
                    i.inspiration_play_id,
                    i.reasoning
                FROM nfl_inspirations i
                ORDER BY RANDOM()
                LIMIT 5;
            """)
            example_inspirations = [dict(r) for r in cur.fetchall()]
        
        print(f"Loading example pairs with full data...")
        example_pairs = []
        for ex in example_inspirations:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM nfl_plays WHERE id = %s", (str(ex["play_id"]),))
                target_row = cur.fetchone()
                if not target_row:
                    continue
                target = dict(target_row)
                
                cur.execute("SELECT * FROM nfl_plays WHERE id = %s", (str(ex["inspiration_play_id"]),))
                insp_row = cur.fetchone()
                if not insp_row:
                    continue
                inspiration = dict(insp_row)
            
            example_pairs.append({
                "query": {
                    "situation": {
                        "yardline": target.get("yardlinefixed"),
                        "down": target.get("down"),
                        "togo": target.get("togo"),
                        "quarter": target.get("quarter"),
                        "offense": target.get("offenseteam"),
                        "defense": target.get("defenseteam")
                    }
                },
                "document": {
                    "formation": inspiration.get("formation"),
                    "playtype": inspiration.get("playtype"),
                    "description": inspiration.get("description")
                },
                "reasoning": ex["reasoning"]
            })
        
        print(f"Fetching candidate plays (successful plays, 2023-2025, outside red zone)...")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT p.*
                FROM nfl_plays p
                WHERE CAST(NULLIF(p.seasonyear,'') AS INT) IN (2023, 2024, 2025)
                  AND CAST(NULLIF(p.yardlinefixed,'') AS INT) > 20
                  AND COALESCE(NULLIF(p.isnoplay,''),'0') = '0'
                  AND UPPER(COALESCE(p.playtype,'')) IN ('RUSH','PASS','SCRAMBLE')
                  AND CAST(NULLIF(p.yards,'') AS INT) >= 5
                  AND CAST(NULLIF(p.yards,'') AS INT) <= 40
                  AND p.id NOT IN (SELECT inspiration_play_id FROM nfl_inspirations)
                  AND p.id NOT IN (SELECT inspiration_play_id FROM nfl_synthetic_inspirations)
                ORDER BY RANDOM()
                LIMIT 10000;
            """)
            candidate_plays = [dict(r) for r in cur.fetchall()]
        
        print(f"Found {len(candidate_plays)} candidate plays to evaluate")
        
        print("Fetching game histories...")
        game_ids = list(set(p["gameid"] for p in candidate_plays if p.get("gameid")))
        game_cache = {}
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    p.id, p.gameid, p.quarter, p.minute, p.second, p.playtype,
                    p.rushdirection, p.passtype, p.yards, p.isnoplay
                FROM nfl_plays p
                WHERE p.gameid = ANY(%s)
                  AND CAST(NULLIF(p.seasonyear,'') AS INT) IN (2023, 2024, 2025)
                ORDER BY 
                    p.gameid,
                    CAST(NULLIF(p.quarter,'') AS INT) ASC,
                    (COALESCE(CAST(NULLIF(p.minute,'') AS INT),0)*60 + COALESCE(CAST(NULLIF(p.second,'') AS INT),0)) DESC;
            """, (game_ids,))
            
            for row in cur:
                gid = row["gameid"]
                if gid not in game_cache:
                    game_cache[gid] = []
                game_cache[gid].append(dict(row))
        
        print(f"Loaded {len(game_cache)} game histories")
    
    plays_batch = []
    histories_batch = []
    
    for play in candidate_plays:
        gid = play.get("gameid")
        if not gid or gid not in game_cache:
            continue
        
        game_plays = game_cache[gid]
        play_idx = None
        for i, p in enumerate(game_plays):
            if str(p["id"]) == str(play["id"]):
                play_idx = i
                break
        
        if play_idx is None:
            continue
        
        history = []
        for i in range(play_idx - 1, -1, -1):
            if is_snap(game_plays[i]):
                history.append(game_plays[i])
            if len(history) >= 5:
                break
        
        plays_batch.append(dict(play))
        histories_batch.append(history)
    
    print(f"Evaluating {len(plays_batch)} plays with histories...")
    
    num_containers = 100
    chunk_size = len(plays_batch) // num_containers + (1 if len(plays_batch) % num_containers else 0)
    
    play_chunks = [plays_batch[i:i+chunk_size] for i in range(0, len(plays_batch), chunk_size)]
    history_chunks = [histories_batch[i:i+chunk_size] for i in range(0, len(histories_batch), chunk_size)]
    
    print(f"Splitting into {len(play_chunks)} batches")
    
    inspirational_count = 0
    for chunk_idx, (play_chunk, hist_chunk) in enumerate(zip(play_chunks, history_chunks)):
        print(f"Processing chunk {chunk_idx+1}/{len(play_chunks)} ({len(play_chunk)} plays)...")
        
        results = []
        for play, hist in zip(play_chunk, hist_chunk):
            result = evaluate_play.remote(play, hist, example_pairs)
            results.append(result)
        
        for result in results:
            res = result
            if res.get("is_inspirational"):
                inspirational_count += 1
                print(f"✓ Inspirational: {res['play_id']} - {res.get('reasoning', '')[:100]}...")
    
    print(f"\n✅ Complete! Found {inspirational_count} inspirational plays")

if __name__ == "__main__":
    main()
