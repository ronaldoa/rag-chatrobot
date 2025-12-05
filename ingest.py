"""
Document ingestion script - build the vector database.

Usage:
    python ingest.py

Steps:
    1. Load all supported documents from the data/ directory
    2. Split into smaller chunks
    3. Generate vector embeddings
    4. Persist to the FAISS vector store
"""

import sys
from pathlib import Path
from tqdm import tqdm
import re

from langchain.document_loaders import (
    DirectoryLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from src.embeddings import get_embeddings
from src.config import (
    DATA_PATH,
    VECTOR_STORE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SUPPORTED_EXTENSIONS,
    SEMANTIC_CHUNKING,
    SEMANTIC_BREAKPOINT_PERCENTILE,
    SEMANTIC_MIN_CHUNK_SIZE,
    validate_environment,
)
from src.utils import setup_logger, ensure_directories, get_file_size

# Set up logging
logger = setup_logger()

# Keep loader map aligned with SUPPORTED_EXTENSIONS
LOADERS_MAP = {
    ".txt": TextLoader,
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
}


def validate_loaders():
    """Validate that every declared extension has a loader."""
    unsupported = [ext for ext in SUPPORTED_EXTENSIONS.keys() if ext not in LOADERS_MAP]

    if unsupported:
        logger.warning(
            "Extensions declared in SUPPORTED_EXTENSIONS without loaders: "
            + ", ".join(unsupported)
        )
        logger.warning("These file types will be ignored.")
        return False
    return True


def load_documents_by_type(data_path: str):
    """
    Load documents grouped by file type.

    Args:
        data_path: Directory containing documents.

    Returns:
        List of loaded documents.
    """
    all_documents = []
    data_dir = Path(data_path)

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_path}")
        return []

    # Count only extensions we can load
    file_stats = {}
    for ext in LOADERS_MAP.keys():
        files = list(data_dir.rglob(f"*{ext}"))
        if files:
            file_stats[ext] = len(files)

    if not file_stats:
        logger.warning(f"No supported documents found in {data_path}")
        logger.info(f"Supported formats: {', '.join(LOADERS_MAP.keys())}")
        return []

    logger.info("Document stats:")
    for ext, count in file_stats.items():
        logger.info(f"  • {ext}: {count} files")

    # Load supported formats
    for ext, loader_cls in LOADERS_MAP.items():
        if ext in file_stats:
            logger.info(f"Loading {ext} files...")
            try:
                # TextLoader needs encoding detection
                loader_kwargs = {}
                if loader_cls == TextLoader:
                    loader_kwargs = {"autodetect_encoding": True}

                loader = DirectoryLoader(
                    data_path,
                    glob=f"**/*{ext}",
                    loader_cls=loader_cls,
                    show_progress=True,
                    loader_kwargs=loader_kwargs,
                )
                docs = loader.load()
                # Basic cleanup to strip common headers/footers on PDFs
                if loader_cls == PyMuPDFLoader:
                    for d in docs:
                        content = d.page_content
                        # Remove runs of whitespace and common header patterns (page numbers, title fragments)
                        content = re.sub(r"\s+", " ", content).strip()
                        # Example: remove isolated page numbers or short header tokens at start
                        content = re.sub(
                            r"^(?:\d+\s+)?Reminiscence of a Stock Operator\s*",
                            "",
                            content,
                            flags=re.IGNORECASE,
                        )
                        d.page_content = content
                all_documents.extend(docs)
                logger.success(f"✓ Loaded {len(docs)} {ext} documents")
            except Exception as e:
                logger.error(f"✗ Failed to load {ext} files: {e}")
                logger.debug("Details:", exc_info=True)

    return all_documents


def build_text_splitter(chunk_size: int, chunk_overlap: int, embeddings=None):
    """
    Choose between semantic and recursive splitters.
    """
    if SEMANTIC_CHUNKING:
        try:
            from langchain_experimental.text_splitter import SemanticChunker
        except ImportError:
            logger.error(
                "Semantic chunking is enabled but langchain-experimental is not installed. "
                "Run: pip install langchain-experimental"
            )
            sys.exit(1)

        if embeddings is None:
            embeddings = get_embeddings()

        logger.info(
            "Using semantic chunking "
            f"(percentile={SEMANTIC_BREAKPOINT_PERCENTILE}, "
            f"min_chunk_size={SEMANTIC_MIN_CHUNK_SIZE}, enforced post-split)"
        )
        semantic_kwargs = {
            "breakpoint_threshold_type": "percentile",
            "breakpoint_threshold_amount": SEMANTIC_BREAKPOINT_PERCENTILE,
        }
        logger.info(f"SemanticChunker init kwargs: {semantic_kwargs}")

        return SemanticChunker(embeddings, **semantic_kwargs)

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )


def split_documents(documents, chunk_size: int, chunk_overlap: int, embeddings=None):
    """
    Split documents into chunks.

    Args:
        documents: List of documents.
        chunk_size: Chunk size.
        chunk_overlap: Chunk overlap.
        embeddings: Embeddings instance (used for semantic chunking).

    Returns:
        List of document chunks (each with chunk_id, source, page in metadata).
    """
    logger.info(
        f"Splitting documents (chunk size: {chunk_size}, overlap: {chunk_overlap})..."
    )

    text_splitter = build_text_splitter(chunk_size, chunk_overlap, embeddings)

    texts = []
    for doc in tqdm(documents, desc="Split progress"):
        try:
            splits = text_splitter.split_documents([doc])

            # Manually merge tiny semantic chunks with the previous one to honor min size.
            if SEMANTIC_CHUNKING and SEMANTIC_MIN_CHUNK_SIZE > 0:
                merged_splits = []
                for chunk in splits:
                    if merged_splits and len(chunk.page_content) < SEMANTIC_MIN_CHUNK_SIZE:
                        merged_splits[-1].page_content = (
                            merged_splits[-1].page_content + "\n\n" + chunk.page_content
                        )
                    else:
                        merged_splits.append(chunk)
                splits = merged_splits

            # 🔹 确保每个 chunk 继承 source / page 等关键信息
            for chunk in splits:
                # 一般情况下 langchain 会继承 metadata，这里做兜底
                if "source" not in chunk.metadata and "source" in doc.metadata:
                    chunk.metadata["source"] = doc.metadata["source"]
                if "page" not in chunk.metadata and "page" in doc.metadata:
                    chunk.metadata["page"] = doc.metadata["page"]

            texts.extend(splits)
        except Exception as e:
            logger.warning(
                f"Failed to split document ({doc.metadata.get('source', 'Unknown')}): {e}"
            )
            continue

    # 🔹 统一给所有 chunk 加上全局唯一的 chunk_id，方便 Hybrid retriever 对齐 dense/sparse
    for idx, t in enumerate(texts):
        t.metadata["chunk_id"] = idx

    logger.success(f"✓ Split complete, {len(texts)} chunks produced")

    if texts:
        avg_length = sum(len(t.page_content) for t in texts) / len(texts)
        logger.info(f"  • Average chunk size: {avg_length:.0f} characters")

    # 可选调试：检查是否包含某些关键短语（比如 absolutely current）
    # debug_phrase = "absolutely current"
    # hit_count = sum(
    #     1 for t in texts if debug_phrase in t.page_content.lower()
    # )
    # logger.info(f"DEBUG: chunks containing '{debug_phrase}': {hit_count}")

    return texts


def create_vectorstore(texts, embeddings):
    """
    Create a FAISS vector store.

    Args:
        texts: List of document chunks.
        embeddings: Embedding model.

    Returns:
        FAISS vector store.
    """
    logger.info(
        "Creating vector store (may take a few minutes depending on corpus size)..."
    )

    # Batch to avoid high memory usage
    batch_size = 100
    vectorstore = None

    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Vectorization progress"):
            batch = texts[i : i + batch_size]

            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                batch_store = FAISS.from_documents(batch, embeddings)
                vectorstore.merge_from(batch_store)

        logger.success(f"✓ Vector store built with {len(texts)} vectors")
        return vectorstore

    except Exception as e:
        logger.error(f"Vector store creation failed: {e}")
        raise


def main():
    """Run the ingestion pipeline."""
    print("\n" + "=" * 60)
    print("📚 Document ingestion and vector store build")
    print("=" * 60 + "\n")

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    # Validate loader configuration
    if not validate_loaders():
        logger.error(
            "Loader configuration incomplete; some formats cannot be processed."
        )
        response = input("Continue anyway? (y/N): ")
        if response.lower() != "y":
            sys.exit(1)

    # Ensure directories exist
    ensure_directories(DATA_PATH, VECTOR_STORE_PATH)

    try:
        # Step 1: Load documents
        logger.info(f"Loading documents from {DATA_PATH}...")
        documents = load_documents_by_type(DATA_PATH)

        if not documents:
            logger.error("No documents found, exiting")
            logger.info(f"Place documents in: {DATA_PATH}")
            logger.info(f"Supported formats: {', '.join(LOADERS_MAP.keys())}")
            sys.exit(1)

        logger.success(f"✓ Loaded {len(documents)} documents")

        embeddings = None
        if SEMANTIC_CHUNKING:
            logger.info("Semantic chunking enabled; loading embedding model early...")
            embeddings = get_embeddings()
            logger.success("✓ Embedding model loaded")

        # Step 2: Split documents
        texts = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP, embeddings)

        if not texts:
            logger.error("Document splitting failed, exiting")
            sys.exit(1)

        # Step 3: Load embedding model (reuse if already loaded for semantic splitting)
        if embeddings is None:
            logger.info("Loading embedding model...")
            embeddings = get_embeddings()
            logger.success("✓ Embedding model loaded")

        # Step 4: Create vector store
        vectorstore = create_vectorstore(texts, embeddings)

        # Step 5: Save vector store
        logger.info(f"Saving vector store to {VECTOR_STORE_PATH}...")
        vectorstore.save_local(VECTOR_STORE_PATH)

        # Display summary
        print("\n" + "=" * 60)
        print("🎉 Vector store build complete!")
        print("=" * 60)
        print("Statistics:")
        print(f"  • Documents:       {len(documents)}")
        print(f"  • Document chunks: {len(texts)}")
        print(f"  • Vector dim:      {len(embeddings.embed_query('test'))}")
        print(f"  • Saved at:        {VECTOR_STORE_PATH}")

        faiss_file = Path(VECTOR_STORE_PATH) / "index.faiss"
        if faiss_file.exists():
            print(f"  • Index size:      {get_file_size(str(faiss_file))}")

        print("\n✓ You can now run: streamlit run ui/ui.py")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Error occurred: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
