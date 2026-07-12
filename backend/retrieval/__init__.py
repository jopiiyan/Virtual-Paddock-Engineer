"""Config-driven retrieval pipeline.

Every retrieval component (dense, BM25, fusion, multi-query, rerank) is an
independently toggleable stage so that any subset can be evaluated from a single
code path. The naive baseline and the full pipeline differ only by YAML config —
never by which script runs them (CLAUDE.md R5).
"""
