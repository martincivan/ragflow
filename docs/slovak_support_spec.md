# Slovak (and Czech) language support — specification & handoff

Status: DRAFT for review. This branch carries a proof-of-concept patch
(`rag/nlp/rag_tokenizer.py`, opt-in via env var) plus this spec. The goal is
full first-class Slovak support upstreamed to RAGFlow/infinity, developed
against a production deployment indexing ~105k Slovak architecture-office
documents (Elasticsearch engine, BGE-M3 embeddings).

## 1. Problem statement (measured, v0.26.4)

The `rag_tokenizer` (implemented in the `infinity-sdk` package,
`infinity/rag_tokenizer.py`, and re-exported by `rag/nlp/rag_tokenizer.py`)
**fragments every word containing diacritics** before it reaches the
full-text index:

```
tokenize('škola')                -> 'š kola'
tokenize('účet')                 -> 'úč et'
tokenize('daňové')               -> 'da ň ov é'
tokenize('požiarna bezpečnosť')  -> 'po ž iarna bezp č no ť'
tokenize('poziarna bezpecnost')  -> 'poziarna bezpecnost'   # ASCII passes
```

Root causes, all in the infinity SDK tokenizer:

1. `SPLIT_CHAR` regex captures only ASCII letter/digit runs
   (`[a-zA-Z0-9,\.-]+`), so any accented character breaks a word apart.
2. The stemmer gate `re.match(r"[a-zA-Z_-]+$", ...)` and the synonym lookup
   (`[a-z]+`) skip accented tokens entirely.
3. NLTK Snowball (used since ragflow PR #14140 / infinity #3356 for the
   dataset `language` parameter) has **no Slovak or Czech stemmer**;
   unsupported languages silently fall back to English.
4. `set_language()` is called only in the task executor (index side) —
   `rag/nlp/query.py` on the API server never sets it, so even supported
   languages have an index/query stemming asymmetry.
5. Elasticsearch mapping (`conf/mapping.json`) uses the `whitespace`
   analyzer on `*_tks`/`*_ltks` — no server-side folding can compensate,
   and users routinely type Slovak queries **without diacritics**.

Consequence: the BM25/keyword leg of hybrid search (weight 0.6–0.7 in
common configs) contributes almost nothing for accented Slovak text, and
keyword-dependent features (`auto_keywords` → `important_kwd^30`,
`question_tks^20`, synonyms, keyword analysis) are silently muted.

Measured on our corpus (103-question eval, document-level hit@10, identical
documents, ES engine): symmetric diacritics folding improved the
keyword-dominant configuration from 25% → 32% and the balanced one from
40% → 50%; *asymmetric* folding (index folded, queries not) **regressed**
accented queries — the fix must cover both sides atomically.

## 2. Proof-of-concept in this branch

`rag/nlp/rag_tokenizer.py`: NFD-decompose + strip combining marks before
tokenization, gated by `RAG_TOKENIZER_FOLD_DIACRITICS=1` (default off, zero
behavior change otherwise). Because both the task executor and the API
server import this module singleton, the fold is automatically symmetric
for index and query in the Python path. Notes:

- Only token fields (`*_tks`) change; chunk display text is untouched.
- Existing datasets need a re-parse to benefit.
- Under `DOC_ENGINE=infinity` the Python tokenizer is a passthrough
  (tokenization happens server-side in C++) — the env var has no effect;
  the equivalent fix belongs in infinity's `rag-coarse`/`rag-fine`
  analyzers.
- The Go ingestor path (`API_PROXY_SCHEME=go/hybrid`) is a separate code
  path and is not covered by the PoC.

## 3. Workplan for full support (for the implementing agent)

**Phase 1 — folding (this PoC, productized).** Move the fold into the
infinity SDK `RagTokenizer.tokenize()` behind either (a) the existing
`set_language()` mechanism (fold for a configurable language set:
`slovak`, `czech`, and arguably any latin-script language when enabled) or
(b) a tokenizer option plumbed from the dataset `language` parameter.
Upstream precedent for exactly this kind of change: ragflow PR #14140 +
infinity PR #3356 (Dutch stemmer). Include unit tests with the examples in
§1 (word-integrity + ASCII-idempotence + index/query symmetry).

**Phase 2 — query-side language symmetry.** Call `set_language()` (or the
folding option) in the query path too — `rag/nlp/query.py` /
`FulltextQueryer` — driven by the KB's language. Without this, Phase 1 for
stemmed languages remains asymmetric (see §1 point 4).

**Phase 3 — Slovak/Czech stemming (optional, measure first).** NLTK has no
Slovak Snowball. Options: (a) skip — folding alone recovers most of the
value; (b) light suffix-stripper (Slovak inflectional endings; keep it
conservative); (c) external stemmer dependency. Gate any choice on an eval
harness A/B, not intuition — our data shows vector-heavy hybrid configs
shrink the marginal value of stemming.

**Phase 4 — UI/plumbing.** Add Slovak/Czech to the dataset language
dropdown (`web/src/constants/common.ts`) once they do something; document
the re-parse requirement; consider an ES `asciifolding` filter in
`conf/mapping.json` as belt-and-braces for residual accented tokens
(index-creation-time only; cannot substitute for Phase 1 because tokens
arrive pre-fragmented).

**Phase 5 — Slovak synonym resource.** `rag/nlp/synonym.py` consults
`rag/res/synonym.json` (plus Redis `kevin_synonyms` hot-reload); WordNet
fallback fires only for `[a-z]+` tokens. Provide a Slovak
abbreviation/synonym seed (žb/železobetón, PD/projektová dokumentácia,
norm aliases) keyed in *folded lowercase* form to match post-Phase-1
tokens.

**Testing/eval.** Unit tests per phase; end-to-end: build two small KBs
from identical Slovak documents (folded vs stock), run retrieval A/B at
keyword-dominant weight — the effect must show there first. Recall the
`vector_similarity_weight` inversion bug fixed in v0.26 (PR #15108): do
not trust pre-0.26 tuning data.

## 4. Upstream strategy

1. Open an issue on infiniflow/ragflow referencing #7419 (delegate
   analysis to ES) describing §1 with the fragmentation examples —
   maintainers may prefer the mapping.json route long-term; the folding
   fix is compatible with both worlds.
2. PR the tokenizer change to **infiniflow/infinity**
   (`python/infinity_sdk/infinity/rag_tokenizer.py`) with tests, then a
   companion ragflow PR if any plumbing is needed (language dropdown,
   query-side call).

## Appendix A — separate upstream bug: Qwen models on OpenAI-compatible providers

RAGFlow sends `extra_body`/argument `{"enable_thinking": false}` for
Qwen-family chat models. OpenAI-compatible endpoints that don't implement
the switch (observed: OVH AI Endpoints, model `Qwen3.6-27B`) reject every
request with:

```
Error code: 400 - {'message': 'feature 'extra arguments:
{"enable_thinking":false}' is not currently supported'}
```

which makes any Qwen chat assistant configured against such a provider
fail on 100% of messages, silently from the user's perspective (the error
string is rendered as the answer). Suggested fix: only pass
`enable_thinking` to providers/factories known to support it (native
Qwen/DashScope, vLLM), or catch the 400 and retry without extra arguments.
Worth filing as its own ragflow issue with the repro above.
