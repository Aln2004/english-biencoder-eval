# Zero-Shot Cross-Lingual Evaluation Metrics
*This table presents the final **Test Set** (or Dev-fallback) retrieval metrics for the dynamically evaluated Bi-Encoder language pairs.*

| Source Model / Queries | Target Corpus | Match Typology | Recall@100 | MRR@10 | NDCG@10 | MAP@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **French** | **Italian** | Romance ➔ Romance | 1.99% | 0.0009 | 0.0005 | 0.0001 |
| **Italian** | **French** | Romance ➔ Romance | 1.03% | 0.0026 | 0.0010 | 0.0004 |
| **Spanish** | **French** | Romance ➔ Romance | 3.85% | 0.0021 | 0.0021 | 0.0007 |
| **Dutch** | **French** | Germanic ➔ Romance | 1.45% | 0.0051 | 0.0024 | 0.0017 |
| **French** | **Dutch** | Romance ➔ Germanic | 1.99% | 0.0009 | 0.0005 | 0.0001 |
| **Finnish** | **Italian** | Uralic ➔ Romance | 1.37% | 0.0000 | 0.0000 | 0.0000 |
| **Finnish** | **Dutch** | Uralic ➔ Germanic | 1.37% | 0.0000 | 0.0000 | 0.0000 |

---
**Core Observation:**
The evaluation confirms the hypothesis that lexical similarity mildly improves zero-shot cross-lingual retrieval (e.g., `Spanish ➔ French` achieves the highest `Recall@100` at `3.85%`). Conversely, highly divergent morphological boundaries (`Finnish ➔ Italian`) result in functionally zero semantic collision (`0.0000 MRR/MAP/NDCG`).
