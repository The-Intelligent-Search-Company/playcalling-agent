#!/usr/bin/env python3

import json
from openai import OpenAI
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

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

test_play = {
    "play": {
        "formation": "SHOTGUN",
        "playtype": "PASS",
        "direction": "SHORT MIDDLE",
        "yards": "12",
        "description": "D.Prescott pass short middle to C.Lamb for 12 yards"
    },
    "situation": {
        "yardline": "35",
        "down": "2",
        "togo": "7",
        "quarter": "2",
        "offense": "DAL",
        "defense": "NYG"
    },
    "history": [
        {"playtype": "RUSH", "direction": "LEFT GUARD", "yards": "3"},
        {"playtype": "PASS", "direction": "SHORT RIGHT", "yards": "5"}
    ]
}

prompt = "**PLAY TO EVALUATE:**\n\n"
prompt += json.dumps(test_play, indent=2)
prompt += "\n\nCould this play inspire a red-zone situation? Generate synthetic scenario if yes."

print("=" * 80)
print("PROMPT:")
print("=" * 80)
print(prompt)
print("\n" + "=" * 80)
print("CALLING GPT-5...")
print("=" * 80 + "\n")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")
client = OpenAI(api_key=OPENAI_API_KEY)

response = client.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ],
    response_format={"type": "json_object"}
)

result = json.loads(response.choices[0].message.content)

print("=" * 80)
print("GPT-5 RESPONSE:")
print("=" * 80)
print(json.dumps(result, indent=2))
print("\n" + "=" * 80)
print(f"is_inspirational: {result.get('is_inspirational')}")
print("=" * 80)
