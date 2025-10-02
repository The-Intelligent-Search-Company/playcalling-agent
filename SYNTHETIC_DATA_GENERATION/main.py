#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, re
from typing import Any, Dict, List, Tuple
import psycopg
from psycopg.rows import dict_row
from openai import OpenAI

# ---------------- env / constants ----------------
PG_URL = os.getenv("PG_URL")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
LAST_N_PLAYS = int(os.getenv("LAST_N_PLAYS", "5"))
MAX_TOOL_STEPS = int(os.getenv("MAX_TOOL_STEPS", "10"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")
client = OpenAI(api_key=OPENAI_API_KEY)
TABLE = "nfl_plays"

# ---------------- helpers ----------------
KEEP = ["id","gameid","gamedate","quarter","minute","second","offenseteam","defenseteam",
        "down","togo","yardlinefixed","yardlinedirection","formation","playtype",
        "rushdirection","passtype","yards","description","seasonyear"]

def key_fields(r: Dict[str,Any]) -> Dict[str,Any]:
    out = {k: r.get(k) for k in KEEP}
    if out.get("id") is not None:
        out["id"] = str(out["id"])
    return out

def is_snap(r: Dict[str,Any]) -> bool:
    if (r.get("isnoplay") or "").strip() == "1": return False
    pt = (r.get("playtype") or "").upper()
    return pt in {"RUSH","PASS","SCRAMBLE","SACK"} or any(
        (r.get("isrush")=="1", r.get("ispass")=="1", r.get("issack")=="1")
    )

# ---------------- target selection ----------------
SQL_PLAYS_TARGET = f"""
SELECT p.*,
       CAST(NULLIF(p.quarter,'') AS INT) AS q_int,
       CAST(NULLIF(p.minute,'')  AS INT) AS m_int,
       CAST(NULLIF(p.second,'')  AS INT) AS s_int
FROM {TABLE} p
WHERE CAST(NULLIF(p.seasonyear,'') AS INT) IN (2024, 2025)
  AND COALESCE(NULLIF(p.isnoplay,''),'0') = '0'
  AND UPPER(COALESCE(p.playtype,'')) IN ('RUSH','PASS','SCRAMBLE','SACK')
  AND CAST(NULLIF(p.yardlinefixed,'') AS INT) <= 20
  AND p.gameid IS NOT NULL AND p.gameid <> ''
ORDER BY
  p.gameid ASC,
  CAST(NULLIF(p.quarter,'') AS INT) ASC,
  (COALESCE(CAST(NULLIF(p.minute,'') AS INT),0)*60 + COALESCE(CAST(NULLIF(p.second,'') AS INT),0)) DESC,
  CAST(NULLIF(p.down,'') AS INT) NULLS FIRST,
  CAST(NULLIF(p.togo,'') AS INT) NULLS FIRST;
"""

# ---------------- tool ----------------
def is_read_only_sql(sql: str) -> bool:
    s = sql.strip().lower()
    return s.startswith("select") or s.startswith("with")

def run_sql(conn: psycopg.Connection, sql: str) -> List[Dict[str,Any]]:
    """
    Execute SQL with automatic transaction cleanup.
    Returns results on success, raises exception on error.
    ALWAYS commits or rolls back to clear transaction state.
    """
    if not is_read_only_sql(sql):
        raise ValueError("Only read-only SELECT/CTE statements are allowed.")
    if re.search(r"\blimit\b", sql, re.IGNORECASE) is None:
        sql = f"{sql.rstrip().rstrip(';')} LIMIT 100"
    
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            results = cur.fetchall()
        conn.commit()  # Clear transaction state after successful query
        return results
    except Exception as e:
        conn.rollback()  # CRITICAL: Clear aborted transaction state
        raise e

tools = [{
    "type": "function",
    "name": "execute_sql",
    "description": "Execute a read-only SQL query against the nfl_plays table to retrieve candidate plays.",
    "strict": True,
    "parameters": {
        "type":"object",
        "properties":{"query":{"type":"string","description":"Complete SQL SELECT or WITH statement."}},
        "required":["query"],
        "additionalProperties":False
    },
}]

# ---------------- prompt ----------------
SYSTEM = """You are an NFL play retrieval assistant.

**WORKFLOW:**
1. Receive current red-zone play (yardlinefixed ≤ 20)
2. Call execute_sql to retrieve 30-50 candidates from ANY field position that would match well.
3. Call execute_sql multiple times if needed to refine candidates or try multiple theses.
3. Read candidate descriptions, select top 5 with the most LATENT similarity across all retrievals.
4. Return JSON: { "inspirations": [ { "inspiration_play_id": "<uuid>", "why": "<reasoning>" }, ... ] }

**DATABASE SCHEMA:**
Table: nfl_plays
- id (UUID, PRIMARY KEY - use as inspiration_play_id)
- gameid, gamedate, quarter, minute, second
- offenseteam, defenseteam
- down (TEXT - must cast to INT), togo (TEXT - must cast to INT), yardlinefixed (TEXT - must cast to INT)
- playtype (RUSH/PASS/SCRAMBLE/SACK), rushdirection, passtype, yards
- description (FREE TEXT: player names, actions, outcomes)
- seasonyear (TEXT - must cast to INT)

**CRITICAL SQL CASTING RULES:**
ALL numeric columns are stored as TEXT. You MUST cast them:
- CAST(down AS INT) or NULLIF(down, '') for nullable
- CAST(togo AS INT) or NULLIF(togo, '') for nullable
- CAST(yardlinefixed AS INT)
- CAST(seasonyear AS INT)

**Example descriptions:**
- "(3:47) (Shotgun) J.Allen scrambles right end to BUF 3 for 5 yards."
- "(2:14) P.Mahomes pass short middle to T.Kelce to KC 8 for 12 yards (D.James)."
- "(2:55) (Shotgun) P.Mahomes pass short left to I.Pacheco ran ob at KC 33 for 23 yards (B.Stephens) [K.Van Noy]."

**LATENT FEATURES (extract from description):**
1. QB Archetype: Mobile (Allen, Jackson, Hurts, Mahomes when scrambling) vs. Pocket (Brady, Stafford)
2. Target Archetype: Contested-catch (Kelce, Evans) vs. Separation (Hill, Waddle)
3. RB Style: Power (Henry) vs. Zone (McCaffrey) vs. Receiving (Pacheco, Kamara)
4. Coverage: Single defender = man. Multiple = zone. Blitz mentioned (e.g., [K.Van Noy]) = pressure.
5. Play Concept: Bootleg, screen, draw, power run, zone run, play-action, quick game, RPO
6. Team Scheme: Shanahan (SF, ATL), Air Raid (KC, BUF), Power (BAL, TEN)

**COMPOSITIONAL MATCHING:**
Match on MULTIPLE dimensions: situational pressure + play family + player archetype + coverage context + team scheme + strategic intent

**SQL STRATEGY - CORRECT CASTING:**
```sql
SELECT id, description, offenseteam, 
       CAST(down AS INT) as down_int,
       CAST(togo AS INT) as togo_int,
       CAST(yardlinefixed AS INT) as yardline_int,
       playtype, yards, 
       CAST(seasonyear AS INT) as season_int
FROM nfl_plays
WHERE playtype = 'PASS'
  AND down ~ '^[0-9]+$' 
  AND CAST(down AS INT) BETWEEN 0 AND 2
  AND togo ~ '^[0-9]+$'
  AND CAST(togo AS INT) BETWEEN 7 AND 13
  AND seasonyear ~ '^[0-9]+$'
  AND CAST(seasonyear AS INT) BETWEEN 2023 AND 2025
  AND description IS NOT NULL
ORDER BY CAST(seasonyear AS INT) DESC
LIMIT 50;
```

**GOOD "WHY" EXAMPLE:**
"Mobile QB (Mahomes, similar scramble threat) targeting receiving back (Pacheco) on screen/flat concept vs. blitz pressure (Van Noy mentioned in brackets). Archetype match + play concept (quick release to RB to beat pressure) + coverage context (man blitz) all align. Strategic intent: exploit aggressive pass rush with quick outlet to space."

**RETURN EXACT JSON FORMAT:**
{ "inspirations": [ { "inspiration_play_id": "uuid", "why": "..." } ] }
"""

def run_gpt5_with_full_sql(conn, current, history) -> Tuple[Dict[str,Any], List[Dict[str,Any]]]:
    reasoning_trail = []
    
    # Build initial input_messages with conversation history
    input_messages = [{
        "role": "user",
        "content": json.dumps({
            "current_play": key_fields(current),
            "history_most_recent_first": [key_fields(h) for h in history]
        }, ensure_ascii=False, default=str)
    }]
    
    step_num = 0
    for iteration in range(MAX_TOOL_STEPS):
        step_num += 1
        reasoning_trail.append({
            "step": step_num,
            "type": "iteration_start",
            "iteration": iteration + 1,
            "input_messages_count": len(input_messages)
        })
        
        try:
            # Create response with CURRENT state of input_messages
            resp = client.responses.create(
                model=OPENAI_MODEL,
                instructions=SYSTEM,
                input=input_messages,
                tools=tools,
            )
        except Exception as e:
            reasoning_trail.append({
                "step": step_num,
                "type": "api_error",
                "error": str(e)
            })
            return {"inspirations": []}, reasoning_trail
        
        # Process response output
        if not resp.output:
            reasoning_trail.append({
                "step": step_num,
                "type": "no_output",
                "warning": "Response has no output"
            })
            return {"inspirations": []}, reasoning_trail
        
        # Track tool calls and text output
        tool_calls_found = []
        response_text = ""
        needs_continuation = False
        
        for item in resp.output:
            item_type = getattr(item, "type", None)
            
            # Capture reasoning (if present)
            if item_type == "message":
                role = getattr(item, "role", None)
                content = getattr(item, "content", [])
                if role == "assistant" and content:
                    for c in content:
                        if hasattr(c, "text"):
                            response_text += c.text
            
            # Capture tool calls
            elif item_type == "function_call":
                call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
                name = getattr(item, "name", None)
                arguments = getattr(item, "arguments", None)
                
                if name == "execute_sql" and call_id and arguments:
                    tool_calls_found.append({
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments
                    })
        
        # If we got tool calls, execute them and continue
        if tool_calls_found:
            reasoning_trail.append({
                "step": step_num,
                "type": "tool_calls_found",
                "count": len(tool_calls_found)
            })
            
            for tc in tool_calls_found:
                call_id = tc["call_id"]
                name = tc["name"]
                raw_args = tc["arguments"]
                
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {}
                
                query = args.get("query", "")
                
                reasoning_trail.append({
                    "step": step_num,
                    "type": "sql_query",
                    "query": query
                })
                
                try:
                    rows = run_sql(conn, query)  # This now handles commit/rollback internally
                    output = json.dumps(rows, ensure_ascii=False, default=str)
                    
                    reasoning_trail.append({
                        "step": step_num,
                        "type": "sql_result",
                        "row_count": len(rows),
                        "result_preview": rows[:2] if len(rows) > 2 else rows
                    })
                    
                    # Append BOTH function_call AND function_call_output to input_messages
                    input_messages.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": raw_args
                    })
                    input_messages.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output
                    })
                    
                    needs_continuation = True
                    
                except Exception as e:
                    error_output = json.dumps({"error": str(e)})
                    
                    reasoning_trail.append({
                        "step": step_num,
                        "type": "sql_error",
                        "error": str(e)
                    })
                    
                    # Still append to input_messages so model knows about the error
                    input_messages.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": raw_args
                    })
                    input_messages.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": error_output
                    })
                    
                    needs_continuation = True
            
            if needs_continuation:
                reasoning_trail.append({
                    "step": step_num,
                    "type": "continuing_with_tool_results",
                    "new_input_messages_count": len(input_messages)
                })
                continue  # Go to next iteration with updated input_messages
        
        # If no tool calls, try to parse final response
        if response_text:
            reasoning_trail.append({
                "step": step_num,
                "type": "final_response_attempt",
                "text_length": len(response_text)
            })
            
            try:
                result = json.loads(response_text)
                reasoning_trail.append({
                    "step": step_num,
                    "type": "json_parse_success",
                    "inspirations_count": len(result.get("inspirations", []))
                })
                return result, reasoning_trail
            except Exception as e:
                reasoning_trail.append({
                    "step": step_num,
                    "type": "json_parse_error",
                    "error": str(e),
                    "raw_text": response_text[:500]
                })
        
        # If we get here with no tool calls and no valid response, something's wrong
        reasoning_trail.append({
            "step": step_num,
            "type": "no_tool_calls_and_no_response",
            "warning": "Model returned neither tool calls nor valid JSON"
        })
        return {"inspirations": []}, reasoning_trail
    
    reasoning_trail.append({
        "step": step_num,
        "type": "max_iterations_reached"
    })
    return {"inspirations": []}, reasoning_trail

# ---------------- upsert ----------------
SQL_UPSERT = """
INSERT INTO nfl_inspirations (play_id, inspiration_play_id, reasoning)
VALUES (%(play_id)s, %(inspiration_play_id)s, %(reasoning)s)
ON CONFLICT (play_id, inspiration_play_id) DO UPDATE
SET reasoning = EXCLUDED.reasoning;
"""

def main():
    if not PG_URL:
        raise RuntimeError("PG_URL not set in environment")
    with psycopg.connect(PG_URL, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(SQL_PLAYS_TARGET)
            plays = cur.fetchall()

        prev: Dict[str,List[Dict[str,Any]]] = {}
        current_gid = None

        print(len(plays))

        for r in plays:
            gid = r["gameid"]
            if gid != current_gid:
                current_gid = gid
                prev[gid] = []

            history = prev[gid][:LAST_N_PLAYS]
            out, reasoning_trail = run_gpt5_with_full_sql(conn, r, history)
            items = out.get("inspirations", [])

            if items:
                with conn.cursor() as cur:
                    for it in items[:5]:
                        insp_id = it.get("inspiration_play_id")
                        why = (it.get("why") or "")[:2000]
                        if not insp_id:
                            continue
                        cur.execute(SQL_UPSERT, {
                            "play_id": r["id"],
                            "inspiration_play_id": insp_id,
                            "reasoning": why
                        })
                conn.commit()

            n_insp = len(items)
            log_data = {
                "play_id": str(r["id"]),
                "n_inspirations": n_insp,
                "description": r.get("description", "")[:150],
                "down": r.get("down"),
                "togo": r.get("togo"),
                "yardlinefixed": r.get("yardlinefixed"),
                "playtype": r.get("playtype"),
                "offenseteam": r.get("offenseteam"),
                "reasoning_trail": reasoning_trail,
                "inspirations": items
            }
            
            if n_insp < 5:
                log_data["warning"] = "fewer_than_5_inspirations"
            if n_insp == 0:
                log_data["warning"] = "no_inspirations_found"
            
            print(json.dumps(log_data, ensure_ascii=False, default=str, indent=2))
            print("=" * 80)

            if is_snap(r):
                prev[gid].insert(0, r)
                if len(prev[gid]) > LAST_N_PLAYS*3:
                    prev[gid] = prev[gid][:LAST_N_PLAYS*3]

if __name__ == "__main__":
    main()
