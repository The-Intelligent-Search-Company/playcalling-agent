# Real-Time Hybrid Retrieval for NFL Play Inspiration

Introduction — Coaches build call sheets during Film Study: families of plays keyed to situations, annotated with leverage, coverage, motion effects, quarterback traits, and defensive structure. In live game contexts, the status quo is human recall: coordinators rely on memory—not database filters—to adapt calls within a 15–40‑second clock. In this paper, “retrieval” means an automated, corpus‑scale search that stands in for that recall. Similarity is fluid (e.g., archetypes, motion + hash + formation vs pressure), and traditional analytics are slow and offline, so most relevant precedents are invisible at call time. We ask: can real‑time retrieval replicate this amorphous recall at corpus scale and support in‑game play selection?

Methods — We target offensive red‑zone decision‑making (NFL 2023–2024). Each play is represented as contextualized text and indexed in a 7,381‑play corpus. Retrieval fuses semantic search (E5‑base‑v2 embeddings with FAISS HNSW) and keyword matching (BM25) with a 0.7/0.3 weighting. For a given state, we retrieve top‑K candidates in ~40 ms; an LLM synthesizes a play‑family call, and a fuzzy match provides a historical analogue. Evaluation uses two supervision sources: (1) Synthetic inspirations—starting from successful non–red‑zone plays, an LLM proposes plausible red‑zone scenarios with structured situations and short rationales; and (2) Agentive inspirations—per real red‑zone snap, an LLM “analyst” composes/refines SQL over normalized play/event, per‑player stat, and game‑context tables to retrieve, score, and justify inspirations. We report Hits@K, MRR, latency percentiles, top‑K coverage, and play‑type match.

Results — The system achieves 41.0% Hits@10 and 0.3950 MRR with 38.47 ms median retrieval time on the 7,381‑play corpus. With generation (50 queries), we observe 100% play‑call outputs, 38.0% top‑K coverage, 36.0% play‑type match (PASS/RUSH), and 865.91 ms median end‑to‑end latency (retrieval ≈ 8% of total). Practically, the system surfaces 3–5 inspirations instantly and expresses them as consistent play families.

| Metric | Retrieval (100q) | End‑to‑End (50q) |
|---|---:|---:|
| Hits@10 | 41.0% | — |
| MRR | 0.3950 | — |
| Latency p50 | 38.47 ms | 865.91 ms |
| Top‑K Coverage | — | 38.0% |
| Play‑Type Match | — | 36.0% |

Figure — Example Query and Output
```
Query: 3rd-and-7 at own 30, need first down
LLM: SHOTGUN - PASS SHORT MIDDLE to SLOT RECEIVER
Similar (reference): SHOTGUN, PASS SHORT MIDDLE → 7yd (Q3, TB vs MIN)
Latency: Retrieval 68ms, LLM 802ms, Total 866ms
```

Conclusion — Real‑time, corpus‑scale retrieval can replicate coaching recall with machine‑level breadth and speed, enabling software‑assisted in‑game decisions—novel for this domain. This study primarily benchmarks retrieval: off‑the‑shelf embeddings already enable accurate, low‑latency inspiration. Forward‑looking, we will (i) train football‑specific embeddings that better encode personnel, formation, motion, and coverage cues, and (ii) build a generative retrieval model—a neural index that maps state directly to inspiration identifiers, decoding IDs instead of searching/reranking—to capture latent criteria and achieve near‑constant‑time inference. Together, these advances move play calling from offline analytics to an always‑on memory for the modern sideline.
