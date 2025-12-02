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
        initial_docs = self.vectorstore.similarity_search(
            query,
            k=self.initial_k
        )

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
                (doc, score) for doc, score in doc_score_pairs
                if score >= self.score_threshold
            ]

        # Sort by score descending
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return top-K
        reranked_docs = [doc for doc, score in doc_score_pairs[:self.final_k]]

        return reranked_docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version (delegates to sync)."""
        return self._get_relevant_documents(query)

def get_retriever(vectorstore: FAISS, initial_k: int = None, final_k: int = None) -> RerankerRetriever:
    """Create a retriever instance (hybrid if enabled)."""
    print(f"  • Reranker model: {RERANKER_MODEL}")
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

    init_k = initial_k or INITIAL_K
    fin_k = final_k or FINAL_K

    if not USE_HYBRID:
        return RerankerRetriever(
            vectorstore=vectorstore,
            reranker=reranker,
            initial_k=init_k,
            final_k=fin_k
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
            key = f"{doc.metadata.get('source','')}-{doc.metadata.get('page', '')}"
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
