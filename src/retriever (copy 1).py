"""Retriever implementation with reranking."""

from langchain.schema import BaseRetriever, Document
from langchain.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from typing import List, Sequence, Tuple
from .config import RERANKER_MODEL, INITIAL_K, FINAL_K, USE_HYBRID, HYBRID_WEIGHTS


def _parse_weights(weights: str) -> Tuple[float, float]:
    """Parse dense/sparse weights from env string like "0.6,0.4"."""
    try:
        dense_w, sparse_w = [float(x.strip()) for x in weights.split(",")[:2]]
        total = dense_w + sparse_w
        if total == 0:
            return 0.5, 0.5
        return dense_w / total, sparse_w / total
    except Exception:
        return 0.5, 0.5


class RerankerRetriever(BaseRetriever):
    """
    Two-stage retriever with reranker.

    Stage 1: FAISS fast retrieval (coarse).
    Stage 2: CrossEncoder reranking (fine).
    """

    vectorstore: FAISS
    reranker: CrossEncoder
    initial_k: int = INITIAL_K
    final_k: int = FINAL_K
    score_threshold: float = 0.0  # Optional score threshold

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents.

        Args:
            query: Search query

        Returns:
            List of relevant documents
        """
        # Stage 1: FAISS coarse retrieval
        initial_docs = self.vectorstore.similarity_search(query, k=self.initial_k)

        if not initial_docs:
            return []

        # Stage 2: Reranker fine scoring
        # Build pairs: [(query, doc1), (query, doc2), ...]
        pairs = [[query, doc.page_content] for doc in initial_docs]

        # Compute scores
        scores = self.reranker.predict(pairs)

        # Pair docs with scores
        doc_score_pairs = list(zip(initial_docs, scores))

        # Optional: filter low scores
        if self.score_threshold > 0:
            doc_score_pairs = [
                (doc, score)
                for doc, score in doc_score_pairs
                if score >= self.score_threshold
            ]

        # Sort by score descending
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return top-K
        reranked_docs = [doc for doc, score in doc_score_pairs[: self.final_k]]

        return reranked_docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version (delegates to sync)."""
        return self._get_relevant_documents(query)


def get_retriever(
    vectorstore: FAISS, initial_k: int = None, final_k: int = None
) -> RerankerRetriever:
    """Create a retriever instance (hybrid if enabled)."""
    print(f"  • Reranker model: {RERANKER_MODEL}")
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

    init_k = initial_k or INITIAL_K
    fin_k = final_k or FINAL_K

    if not USE_HYBRID:
        return RerankerRetriever(
            vectorstore=vectorstore, reranker=reranker, initial_k=init_k, final_k=fin_k
        )

    dense = vectorstore.as_retriever(search_kwargs={"k": init_k})
    # Build BM25 from existing documents in vectorstore
    docs: Sequence[Document] = list(vectorstore.docstore._dict.values())
    sparse = BM25Retriever.from_documents(docs)
    sparse.k = init_k

    dense_w, sparse_w = _parse_weights(HYBRID_WEIGHTS)
    print(f"  • Hybrid mode enabled (dense={dense_w:.2f}, sparse={sparse_w:.2f})")

    return HybridRerankerRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        reranker=reranker,
        initial_k=init_k,
        final_k=fin_k,
        dense_weight=dense_w,
        sparse_weight=sparse_w,
    )


class HybridRerankerRetriever(BaseRetriever):
    """Hybrid sparse+dense retriever with CrossEncoder reranking."""

    dense_retriever: BaseRetriever
    sparse_retriever: BM25Retriever
    reranker: CrossEncoder
    initial_k: int = INITIAL_K
    final_k: int = FINAL_K
    dense_weight: float = 0.5
    sparse_weight: float = 0.5

    class Config:
        arbitrary_types_allowed = True

    # ---------- 辅助：简单 tokenizer，用于 entity / keyword bonus ----------

    @staticmethod
    def _simple_tokenize(text: str) -> list[str]:
        import re

        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    def _entity_bonus(self, query: str, doc_text: str) -> float:
        """
        对问句里的"重要 token"做加权，返回 0.0 ~ 1.0 的 bonus。
        """
        # 扩展的停用词列表
        stopwords = {
            # Articles
            "the",
            "a",
            "an",
            # Conjunctions
            "and",
            "or",
            "but",
            "nor",
            "yet",
            "so",
            # Prepositions
            "of",
            "to",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            # Common verbs
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "may",
            "might",
            # Pronouns
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "this",
            "that",
            "these",
            "those",
            "them",
        }

        # 提取查询中的重要词（长度 > 3 且非停用词）
        q_tokens = [
            t for t in self._simple_tokenize(query) if len(t) > 3 and t not in stopwords
        ]
        d_tokens = self._simple_tokenize(doc_text)

        if not q_tokens or not d_tokens:
            return 0.0

        q_set = set(q_tokens)
        d_set = set(d_tokens)
        inter = q_set & d_set

        if not inter:
            return 0.0

        # 交集占比，最多 1.0
        return min(1.0, len(inter) / len(q_set))

    # ---------- 核心：同步检索接口（给 langchain 用） ----------

    def _get_relevant_documents(self, query: str) -> List[Document]:
        def _retrieve(r: BaseRetriever) -> List[Document]:
            """优先用 invoke，兼容新版本 langchain；否则 fallback 到 get_relevant_documents。"""
            invoke_fn = getattr(r, "invoke", None)
            if callable(invoke_fn):
                return invoke_fn(query)
            return r.get_relevant_documents(query)

        # 1) Stage 1：从 dense / sparse 两个 retriever 拿到候选
        dense_docs = _retrieve(self.dense_retriever) or []
        sparse_docs = _retrieve(self.sparse_retriever) or []

        # 2) 收集分数，使用 dict 去重（按 source + page + chunk_id）
        scored_map: dict[str, dict] = {}

        # ---- Dense 部分：用排名衰减近似相似度 ----
        for rank, doc in enumerate(dense_docs):
            # 尽量用 chunk_id / id，如果没有就退回 source+page
            chunk_id = doc.metadata.get("chunk_id") or doc.metadata.get("id") or id(doc)
            key = f"{doc.metadata.get('source', '')}-{doc.metadata.get('page', '')}-{chunk_id}"

            if key not in scored_map:
                scored_map[key] = {
                    "doc": doc,
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                }

            # 排名衰减：top-1 ≈ 1.0，往后逐渐降低，最低保留 0.1
            rank_score = max(0.1, 1.0 - rank * 0.025)
            scored_map[key]["dense_score"] = max(
                scored_map[key]["dense_score"],
                rank_score * self.dense_weight,
            )

        # ---- Sparse 部分：BM25 分数（存在 metadata["score"] 里） ----
        for doc in sparse_docs:
            chunk_id = doc.metadata.get("chunk_id") or doc.metadata.get("id") or id(doc)
            key = f"{doc.metadata.get('source', '')}-{doc.metadata.get('page', '')}-{chunk_id}"

            if key not in scored_map:
                scored_map[key] = {
                    "doc": doc,
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                }

            bm_score = doc.metadata.get("score", 1.0)
            scored_map[key]["sparse_score"] = max(
                scored_map[key]["sparse_score"],
                bm_score * self.sparse_weight,
            )

        if not scored_map:
            return []

        # 3) 归一化 sparse 分数（0~1），避免某些 query 上 BM25 尺度差异太大
        sparse_vals = [v["sparse_score"] for v in scored_map.values()]
        sparse_min = min(sparse_vals)
        sparse_max = max(sparse_vals)

        def _norm_sparse(v: float) -> float:
            if sparse_max <= sparse_min:
                return 0.0
            return (v - sparse_min) / (sparse_max - sparse_min)

        merged: List[Document] = []

        # 短问题更依赖关键词 → 提高 bonus 比例
        query_len = len(query.split())
        bonus_weight = 0.15 if query_len < 10 else 0.05

        # 4) 计算最终 _pre_score = dense + normalized_sparse + keyword/entity bonus
        for key, info in scored_map.items():
            doc = info["doc"]
            dense_s = info["dense_score"]
            sparse_s = _norm_sparse(info["sparse_score"])

            # keywords / entities overlap bonus
            bonus = self._entity_bonus(query, doc.page_content)

            pre_score = dense_s + sparse_s + bonus_weight * bonus

            # 写回 metadata，后面排序用
            doc.metadata["_pre_score"] = pre_score
            merged.append(doc)

        # 5) 按 _pre_score 排序，取前 initial_k 进入 CrossEncoder rerank
        merged.sort(key=lambda d: d.metadata.get("_pre_score", 0.0), reverse=True)
        merged = merged[: self.initial_k]
        if not merged:
            return []

        # 6) Stage 2：CrossEncoder 重新打分
        pairs = [[query, doc.page_content] for doc in merged]
        scores = self.reranker.predict(pairs)

        doc_score_pairs = list(zip(merged, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # 返回最终 top-k 文档
        return [doc for doc, _ in doc_score_pairs[: self.final_k]]

    # ---------- 异步接口（langchain 需要） ----------

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # 简单地调用同步版本（如果以后需要，可以改成真正异步的）
        return self._get_relevant_documents(query)


class HybridRerankerRetriever_1(BaseRetriever):
    """Hybrid sparse+dense retriever with CrossEncoder reranking."""

    dense_retriever: BaseRetriever
    sparse_retriever: BM25Retriever
    reranker: CrossEncoder
    initial_k: int = INITIAL_K
    final_k: int = FINAL_K
    dense_weight: float = 0.5
    sparse_weight: float = 0.5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        def _retrieve(r):
            """Call retriever using invoke when available to avoid deprecation warnings."""
            invoke_fn = getattr(r, "invoke", None)
            if callable(invoke_fn):
                return invoke_fn(query)
            return r.get_relevant_documents(query)

        # Stage 1: get candidates from both retrievers
        dense_docs = _retrieve(self.dense_retriever) or []
        sparse_docs = _retrieve(self.sparse_retriever) or []

        # Weighted scores: BM25 retriever returns scores in metadata
        scored = []
        for doc in dense_docs:
            scored.append((doc, self.dense_weight))
        for doc in sparse_docs:
            bm_score = doc.metadata.get("score", 1.0)
            scored.append((doc, bm_score * self.sparse_weight))

        # Deduplicate by source+page
        seen = set()
        merged = []
        for doc, sc in scored:
            key = f"{doc.metadata.get('source', '')}-{doc.metadata.get('page', '')}"
            if key in seen:
                continue
            seen.add(key)
            doc.metadata["_pre_score"] = sc
            merged.append(doc)

        # Keep up to initial_k before reranking
        merged = merged[: self.initial_k]
        if not merged:
            return []

        # Stage 2: CrossEncoder reranking
        pairs = [[query, doc.page_content] for doc in merged]
        scores = self.reranker.predict(pairs)

        doc_score_pairs = list(zip(merged, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in doc_score_pairs[: self.final_k]]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)
