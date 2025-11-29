"""Retriever implementation with reranking."""
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from typing import List
from .config import RERANKER_MODEL, INITIAL_K, FINAL_K

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
    """
    Create a retriever instance.

    Args:
        vectorstore: FAISS vector store
        initial_k: Initial coarse results (defaults to config)
        final_k: Final reranked results (defaults to config)

    Returns:
        RerankerRetriever instance
    """
    print(f"  • Reranker model: {RERANKER_MODEL}")
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

    return RerankerRetriever(
        vectorstore=vectorstore,
        reranker=reranker,
        initial_k=initial_k or INITIAL_K,
        final_k=final_k or FINAL_K
    )
