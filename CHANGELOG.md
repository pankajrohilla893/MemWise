# Changelog

## 0.1.0

First release.

- `MemoryManager` with `ingest`, `add`, `recall`, `build_context`, `current`, `history`,
  `forget`, `forget_all`, `maintain`, `maintain_all` and `reembed`
- A write gate with consent for sensitive data; ADD / CONFIRM / SUPERSEDE / RETRACT / IGNORE reconciliation
- Validity windows, expiry and staleness checks; history is kept when facts change
- Ranked retrieval, and the USE / MENTION / VERIFY / HOLD decision
- SQLite store with an audit log, thread-safe, with collision-free ids across processes
- `MemoryConfig` for every threshold; pluggable `Embedder` and `Extractor`
- Works with any LLM: `LLMExtractor` takes any `(system, messages) -> str` function, and
  `OpenAICompatible` is a zero-dependency client for any `/chat/completions` API
- Real embedding models: `OpenAICompatibleEmbedder` for any `/embeddings` API (cached,
  batched), `MemoryConfig.for_embedding_model()` thresholds, and `examples/tune_embedder.py`
  to measure your own model. Mixing vectors from two embedders raises a clear error
- `ingest()` takes the whole conversation (`{"role", "content"}` turns). `LLMExtractor` uses the
  assistant's turns and the user's stored memories as context; memories still only come from the
  user, sensitive memories are never shown, and repeated-back memories don't count as confirmed
- "Please forget that…" in a conversation deletes the memory (FORGET), the same as `forget()`
- Scenario evals (`evals/scenarios.py`): 23 multi-session conversations to test with your own model
- Examples: quickstart, guided demo, interactive chat
