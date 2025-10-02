#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
from typing import Any, Dict, List
import psycopg
from psycopg.rows import dict_row

PG_URL = os.getenv("PG_URL")

LAST_N_PLAYS = 5

def is_snap(r: Dict[str,Any]) -> bool:
    if (r.get("isnoplay") or "").strip() == "1": 
        return False
    pt = (r.get("playtype") or "").upper()
    return pt in {"RUSH","PASS","SCRAMBLE","SACK"}

def build_query(target_play: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    """Build query string from situation + history"""
    
    situation = {
        "yardline": target_play.get("yardlinefixed", ""),
        "down": target_play.get("down", ""),
        "togo": target_play.get("togo", ""),
        "quarter": target_play.get("quarter", ""),
        "offense": target_play.get("offenseteam", ""),
        "defense": target_play.get("defenseteam", ""),
        "season": target_play.get("seasonyear", "")
    }
    
    history_plays = []
    for h in history[:LAST_N_PLAYS]:
        history_plays.append({
            "playtype": h.get("playtype", ""),
            "direction": h.get("rushdirection") or h.get("passtype", ""),
            "yards": h.get("yards", "")
        })
    
    query = {
        "situation": situation,
        "history": history_plays
    }
    
    return json.dumps(query, ensure_ascii=False)

def build_document(play: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    """Build document string from play details + context when it was called"""
    
    # The play itself
    play_details = {
        "formation": play.get("formation", ""),
        "playtype": play.get("playtype", ""),
        "direction": play.get("rushdirection") or play.get("passtype", ""),
        "yards": play.get("yards", ""),
        "description": play.get("description", "")
    }
    
    # The situation when this play was called
    situation = {
        "yardline": play.get("yardlinefixed", ""),
        "down": play.get("down", ""),
        "togo": play.get("togo", ""),
        "quarter": play.get("quarter", ""),
        "offense": play.get("offenseteam", ""),
        "defense": play.get("defenseteam", ""),
    }
    
    # Recent history before this play
    history_plays = []
    for h in history[:LAST_N_PLAYS]:
        history_plays.append({
            "playtype": h.get("playtype", ""),
            "direction": h.get("rushdirection") or h.get("passtype", ""),
            "yards": h.get("yards", "")
        })
    
    doc = {
        "play_id": str(play.get("id", "")),
        "play": play_details,
        "context": {
            "situation": situation,
            "history": history_plays
        }
    }
    
    return json.dumps(doc, ensure_ascii=False)

def main():
    from pathlib import Path
    output_file = os.getenv("TRAINING_DATA_OUT") or str((Path(__file__).parent / "training_data.csv").resolve())
    
    if not PG_URL:
        raise RuntimeError("PG_URL not set in environment")
    with psycopg.connect(PG_URL, row_factory=dict_row) as conn:
        print("Fetching all inspirations...")
        with conn.cursor() as cur:
            cur.execute("SELECT play_id, inspiration_play_id, reasoning FROM nfl_inspirations ORDER BY play_id;")
            inspirations = cur.fetchall()
        
        print(f"Found {len(inspirations)} inspiration pairs")
        
        all_play_ids = set()
        for insp in inspirations:
            all_play_ids.add(str(insp["play_id"]))
            all_play_ids.add(str(insp["inspiration_play_id"]))
        
        print(f"Fetching {len(all_play_ids)} unique plays...")
        play_cache = {}
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    p.id, p.gameid, p.gamedate, p.quarter, p.minute, p.second,
                    p.offenseteam, p.defenseteam, p.down, p.togo, p.yardlinefixed,
                    p.yardlinedirection, p.formation, p.playtype, p.rushdirection,
                    p.passtype, p.yards, p.description, p.seasonyear, p.isnoplay
                FROM nfl_plays p
                WHERE p.id = ANY(%s);
            """, (list(all_play_ids),))
            
            for row in cur:
                play_cache[str(row["id"])] = dict(row)
        
        print(f"Loaded {len(play_cache)} plays into cache")
        
        game_ids = set(p["gameid"] for p in play_cache.values() if p.get("gameid"))
        print(f"Fetching game histories for {len(game_ids)} games...")
        
        game_cache = {}
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    p.id, p.gameid, p.quarter, p.minute, p.second, p.down, p.togo,
                    p.yardlinefixed, p.playtype, p.rushdirection, p.passtype, p.yards,
                    p.formation, p.description, p.isnoplay
                FROM nfl_plays p
                WHERE p.gameid = ANY(%s)
                  AND CAST(NULLIF(p.seasonyear,'') AS INT) IN (2024, 2025)
                ORDER BY 
                    p.gameid,
                    CAST(NULLIF(p.quarter,'') AS INT) ASC,
                    (COALESCE(CAST(NULLIF(p.minute,'') AS INT),0)*60 + COALESCE(CAST(NULLIF(p.second,'') AS INT),0)) DESC;
            """, (list(game_ids),))
            
            for row in cur:
                gid = row["gameid"]
                if gid not in game_cache:
                    game_cache[gid] = []
                game_cache[gid].append(dict(row))
        
        print(f"Loaded histories for {len(game_cache)} games")
        
        print("Building training data...")
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                'query',
                'document',
                'reasoning',
                'target_play_id',
                'inspiration_play_id'
            ])
            writer.writeheader()
            
            for idx, insp in enumerate(inspirations):
                if idx % 1000 == 0:
                    print(f"Processing {idx}/{len(inspirations)}...")
                
                target_play_id = str(insp["play_id"])
                inspiration_play_id = str(insp["inspiration_play_id"])
                reasoning = insp.get("reasoning", "")
                
                if target_play_id not in play_cache:
                    continue
                
                target_play = play_cache[target_play_id]
                game_id = target_play.get("gameid")
                
                if game_id not in game_cache:
                    continue
                
                game_plays = game_cache[game_id]
                
                target_idx = None
                for i, p in enumerate(game_plays):
                    if str(p.get("id")) == target_play_id:
                        target_idx = i
                        break
                
                if target_idx is None:
                    continue
                
                history = []
                for i in range(target_idx - 1, -1, -1):
                    if is_snap(game_plays[i]):
                        history.append(game_plays[i])
                    if len(history) >= LAST_N_PLAYS:
                        break
                
                if inspiration_play_id not in play_cache:
                    continue
                
                inspiration_play = play_cache[inspiration_play_id]
                
                # Get history for the inspiration play too
                insp_game_id = inspiration_play.get("gameid")
                insp_history = []
                if insp_game_id and insp_game_id in game_cache:
                    insp_game_plays = game_cache[insp_game_id]
                    insp_idx = None
                    for i, p in enumerate(insp_game_plays):
                        if str(p.get("id")) == inspiration_play_id:
                            insp_idx = i
                            break
                    
                    if insp_idx is not None:
                        for i in range(insp_idx - 1, -1, -1):
                            if is_snap(insp_game_plays[i]):
                                insp_history.append(insp_game_plays[i])
                            if len(insp_history) >= LAST_N_PLAYS:
                                break
                
                query = build_query(target_play, history)
                document = build_document(inspiration_play, insp_history)
                
                writer.writerow({
                    'query': query,
                    'document': document,
                    'reasoning': reasoning,
                    'target_play_id': target_play_id,
                    'inspiration_play_id': inspiration_play_id
                })
    
    print(f"\nTraining data written to: {output_file}")

if __name__ == "__main__":
    main()
