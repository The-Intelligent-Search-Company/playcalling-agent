# NFL Hybrid Play Retrieval System

**Real-Time AI Play Recommendations with Hybrid Retrieval + LLM Generation**

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-key-here"

# Run a single query
python PIPELINE/llm_agent.py "3rd and 7 at 30 yard line, need first down"

# Run full evaluation
python EVALUATION/evaluate_final_hybrid.py
```

**Expected Output** (~900ms):
```
🎯 Recommended Play Call:
   SHOTGUN - PASS SHORT MIDDLE to SLOT RECEIVER

📊 Similar Historical Play (for reference):
   Situation: 3 and 7 at 7, Q3
   Teams: TB vs MIN
   Formation: SHOTGUN
   Play: PASS SHORT MIDDLE → 7 yards

⚡ Performance:
   Retrieval: 105.37ms
   LLM: 1204.21ms
   Total: 1309.58ms
```

---

## Environment Variables

Set these in your shell before running components that require them.

Required (core pipeline):
- `OPENAI_API_KEY` — OpenAI API key for LLM calls.

Optional (core pipeline):
- `OPENAI_MODEL` — Model name (default: `gpt-4o-mini` for pipeline; synthetic tools default to `gpt-4o`).

Required (synthetic data + SQL agent tools):
- `PG_URL` — Postgres connection string for the inspirations database (used by scripts in `SYNTHETIC_DATA_GENERATION/`).

Optional (synthetic data + SQL agent tools):
- `TRAINING_DATA_OUT` — Output path for generated `training_data.csv` (defaults to `SYNTHETIC_DATA_GENERATION/training_data.csv`).
- `LAST_N_PLAYS` — Number of prior snaps to include in context (default: `5`).
- `MAX_TOOL_STEPS` — Max tool iterations for the SQL agent (default: `10`).

Example:
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"
export PG_URL="postgresql://user:pass@host/db?sslmode=require"
export TRAINING_DATA_OUT="$(pwd)/SYNTHETIC_DATA_GENERATION/training_data.csv"
export LAST_N_PLAYS=5
export MAX_TOOL_STEPS=10
```

Security note:
- Do not hardcode secrets in source files or commit them to version control. Use environment variables or a secrets manager.

---

## Project Structure

```
HYBRID_PLAY_RETRIEVAL_SYSTEM/
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── PIPELINE/
│   ├── prepare_play_corpus.py             # Extract plays from training_data.csv
│   ├── build_play_index.py                # Build FAISS semantic index
│   ├── build_bm25_index.py                # Build BM25 keyword index
│   ├── query_pipeline.py                  # Hybrid retrieval implementation
│   ├── llm_agent.py                       # LLM play call generation
│   ├── test_hybrid.py                     # Test hybrid vs semantic-only
│   └── demo_inference.py                  # Demo with example queries
│
├── DATA/
│   ├── play_corpus.json                   # 7,381 plays with metadata
│   ├── play_embeddings.npy                # E5-base-v2 embeddings (768-dim)
│   ├── play_index.faiss                   # FAISS HNSW index
│   ├── play_index_metadata.pkl            # Index metadata
│   ├── bm25_index.pkl                     # BM25 index
│   └── corpus_stats.json                  # Corpus statistics
│
├── EVALUATION/
│   ├── evaluate_final_hybrid.py           # Comprehensive evaluation script
│   └── final_evaluation_results.json      # Evaluation results
│
└── DOCS/
    └── METHODOLOGY.md                     # Full methodology & evaluation report
```

---

## System Overview

### What It Does

Input: **Game situation** (down, distance, field position, teams, recent plays)  
Output: **Detailed play call** (formation + play type + direction + target)

**Example**:
- Input: "3rd and 7 at 30 yard line, need first down"
- Output: "SHOTGUN - PASS SHORT MIDDLE to SLOT RECEIVER"

### How It Works

1. **Hybrid Retrieval** (40ms):
   - Semantic search: E5-base-v2 embeddings + FAISS HNSW
   - Keyword search: BM25Okapi
   - Fusion: 0.7 * semantic + 0.3 * BM25
   - Output: Top-20 candidate plays from 7,381-play corpus

2. **LLM Reasoning** (800ms):
   - GPT-4o-mini analyzes top-20 candidates
   - Synthesizes optimal play call
   - Output: Detailed play call string

3. **Matching** (<1ms):
   - Fuzzy match play call to retrieved candidates
   - Return similar historical play for reference

**Total Latency**: ~900ms (p50)

---

## Key Features

### 1. Hybrid Retrieval (Semantic + Keyword)

**Why Hybrid?**
- **Semantic search** captures conceptual similarity (e.g., "3rd and short" situations)
- **BM25 keyword search** captures exact matches (e.g., "SHOTGUN" formation)
- **Fusion** combines both for optimal relevance

**Performance**:
- **41% Hits@10** (finds correct play in top-10 out of 7,381 plays)
- **293x better than random** (random: 0.14%)
- **+7.9% improvement** over semantic-only

### 2. Contextual Embeddings

Each play is embedded with:
- **Teams**: Offense vs defense matchup
- **Situation**: Down, distance, field position, quarter
- **Play History**: Last 5 plays (drive context)
- **Play Details**: Formation, type, direction, yards
- **Description**: Full play-by-play text

**Impact**: Adding teams + history improved Hits@10 from 6% → 38% (6.3x improvement)

### 3. LLM Play Call Generation

**Output Format**: `FORMATION - PLAYTYPE DIRECTION to TARGET`

**Examples**:
- `SHOTGUN - PASS SHORT MIDDLE to SLOT RECEIVER`
- `UNDER CENTER - RUSH LEFT TACKLE through A-GAP`
- `I-FORMATION - RUSH RIGHT END outside`

**Benefits**:
- Detailed, actionable play calls
- Not limited to exact historical plays
- Synthesizes patterns from multiple candidates

### 4. Real-Time Performance

**Latency**:
- p50: 866ms (median)
- p95: 1,311ms
- Sub-2-second response time suitable for in-game decisions

**Bottleneck**: 92% of latency is LLM reasoning (GPT-4o-mini)

---

## Evaluation Results

### Retrieval Quality (100 queries, 7,381-play corpus)

| Metric | Value |
|--------|-------|
| **Hits@5** | 41.0% |
| **Hits@10** | 41.0% |
| **Hits@20** | 41.0% |
| **MRR** | 0.3950 |
| **Retrieval p50** | 38.47ms |

**Interpretation**: 41% of queries find the ground-truth play in top-10 (293x better than random).

### LLM Play Call Quality (50 queries)

| Metric | Value |
|--------|-------|
| **Play Calls Generated** | 100% (50/50) |
| **Top-K Coverage** | 38.0% |
| **Play Type Match** | 36.0% |
| **End-to-End p50** | 865.91ms |

**Interpretation**: System generates detailed play calls with 100% success rate and sub-1-second latency.

### Latency Breakdown

| Component | p50 | p90 | p95 | p99 |
|-----------|-----|-----|-----|-----|
| **Retrieval** | 67.81ms | - | - | - |
| **LLM** | 802.51ms | - | - | - |
| **Total** | 865.91ms | 1096.21ms | 1310.61ms | 1697.77ms |

---

## Technical Details

### Embedding Model

- **Model**: E5-base-v2 (intfloat/e5-base-v2)
- **Dimensions**: 768
- **Optimization**: Retrieval-optimized contrastive learning
- **Prefix**: "query:" for searches, "passage:" for documents

### Vector Index

- **Type**: FAISS HNSW (Hierarchical Navigable Small World)
- **Parameters**: M=32, efConstruction=200, efSearch=64
- **Size**: 23.54 MB
- **Search Latency**: <1ms (approximate nearest neighbor)

### Keyword Index

- **Algorithm**: BM25Okapi
- **Tokenization**: Lowercase + regex word boundaries
- **Size**: 3.09 MB
- **Search Latency**: ~10-15ms

### LLM Configuration

- **Model**: GPT-4o-mini (gpt-4o-mini)
- **Temperature**: 0.3
- **Max Tokens**: 50
- **Typical Output**: 20-40 tokens

---

## Usage Examples

### Example 1: Third Down Conversion

```bash
python PIPELINE/llm_agent.py "3rd and 7 at 30 yard line, need first down"
```

**Output**:
```
🎯 Recommended Play Call:
   SHOTGUN - PASS SHORT MIDDLE to SLOT RECEIVER
```

### Example 2: Goal Line Situation

```bash
python PIPELINE/llm_agent.py "2nd and goal at 2 yard line, looking for touchdown"
```

**Output**:
```
🎯 Recommended Play Call:
   SHOTGUN - RUSH LEFT TACKLE through A-GAP
```

### Example 3: Establishing Run Game

```bash
python PIPELINE/llm_agent.py "1st and 10 at midfield, establish run game"
```

**Output**:
```
🎯 Recommended Play Call:
   I-FORMATION - RUSH RIGHT GUARD
```

---

## Development & Regeneration

If you want to regenerate the system from scratch:

### Step 1: Prepare Corpus

```bash
python PIPELINE/prepare_play_corpus.py
```

Extracts 7,381 unique plays from `training_data.csv` and creates `DATA/play_corpus.json`.

### Step 2: Build FAISS Index

```bash
python PIPELINE/build_play_index.py
```

Generates E5-base-v2 embeddings and builds FAISS HNSW index (~40 seconds).

### Step 3: Build BM25 Index

```bash
python PIPELINE/build_bm25_index.py
```

Tokenizes corpus and builds BM25Okapi index (~0.2 seconds).

### Step 4: Test Hybrid Retrieval

```bash
python PIPELINE/test_hybrid.py
```

Compares semantic-only vs hybrid retrieval on sample queries.

### Step 5: Run Evaluation

```bash
export OPENAI_API_KEY="your-key-here"
python EVALUATION/evaluate_final_hybrid.py
```

Runs comprehensive evaluation (100 retrieval queries + 50 LLM queries, ~5 minutes).

---

## Dependencies

```txt
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
torch>=2.0.0
openai>=1.0.0
numpy>=1.24.0
pandas>=2.0.0
rank-bm25>=0.2.2
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Methodology & Interpretation

For detailed methodology, evaluation design, and metric interpretation, see:

**[DOCS/METHODOLOGY.md](DOCS/METHODOLOGY.md)**

Covers:
- System architecture
- Data & corpus details
- Retrieval pipeline design
- LLM play call generation
- Evaluation methodology
- Results interpretation
- Strengths & limitations

---

## Performance Benchmarks

### Retrieval Evolution

| Phase | Approach | Hits@10 | Improvement |
|-------|----------|---------|-------------|
| Baseline | Semantic-only, no context | 6% | - |
| Phase 1 | + Teams + History | 38% | +32pp (6.3x) |
| Phase 2 | + BM25 Hybrid | **41%** | **+3pp (7.9%)** |

### Comparison to Baselines

| Method | Hits@10 | Latency p50 |
|--------|---------|-------------|
| **Random Selection** | 0.14% | N/A |
| **Semantic-only** | 38% | 10ms |
| **BM25-only** | ~25% (est.) | 15ms |
| **Hybrid (Ours)** | **41%** | **38ms** |

---

## Limitations & Future Work

### Current Limitations

1. **LLM Latency**: 800ms bottleneck (92% of total time)
2. **No EPA Data**: Cannot evaluate expected point value
3. **Retrieval Ceiling**: Hits@K plateaus at 41%
4. **Play Type Match**: Only 36% (LLM trades precision for creativity)

### Potential Improvements

1. **Fine-tune Small LLM**: Llama-3-1B → <200ms LLM latency (866ms → 300ms end-to-end)
2. **Query Expansion**: Multiple query variations → 41% → 45-48% Hits@10
3. **Cross-Encoder Reranking**: Rerank top-20 → improve MRR from 0.395 → 0.45+
4. **EPA Integration**: Filter by success probability

---

## Citation

If you use this system in your research or application:

```bibtex
@software{nfl_hybrid_retrieval_2025,
  title={NFL Hybrid Play Retrieval System: Real-Time AI Play Recommendations},
  author={MIT-2 Research Team},
  year={2025},
  note={Hybrid retrieval (semantic + BM25) + LLM play call generation},
  url={https://github.com/your-repo}
}
```

---

## License

Research use only.

---

## Contact

**MIT-2 Research Team**  
**Date**: 2025-10-01  
**Version**: 1.0.0

For questions, issues, or collaboration:
- See [DOCS/METHODOLOGY.md](DOCS/METHODOLOGY.md) for technical details
- Review [EVALUATION/final_evaluation_results.json](EVALUATION/final_evaluation_results.json) for full metrics

---

**⚡ Sub-1-second AI play calling. 🏈 7,381 plays. 🎯 41% top-10 accuracy.**
