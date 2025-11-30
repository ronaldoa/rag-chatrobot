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

from langchain.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from src.embeddings import get_embeddings
from src.config import (
    DATA_PATH,
    VECTOR_STORE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SUPPORTED_EXTENSIONS,
    validate_environment
)
from src.utils import setup_logger, ensure_directories, get_file_size

# Set up logging
logger = setup_logger()

# Keep loader map aligned with SUPPORTED_EXTENSIONS
LOADERS_MAP = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
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
                    loader_kwargs = {'autodetect_encoding': True}

                loader = DirectoryLoader(
                    data_path,
                    glob=f"**/*{ext}",
                    loader_cls=loader_cls,
                    show_progress=True,
                    loader_kwargs=loader_kwargs
                )
                docs = loader.load()
                all_documents.extend(docs)
                logger.success(f"✓ Loaded {len(docs)} {ext} documents")
            except Exception as e:
                logger.error(f"✗ Failed to load {ext} files: {e}")
                logger.debug("Details:", exc_info=True)

    return all_documents


def split_documents(documents, chunk_size: int, chunk_overlap: int):
    """
    Split documents into chunks.

    Args:
        documents: List of documents.
        chunk_size: Chunk size.
        chunk_overlap: Chunk overlap.

    Returns:
        List of document chunks.
    """
    logger.info(f"Splitting documents (chunk size: {chunk_size}, overlap: {chunk_overlap})...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    texts = []
    for doc in tqdm(documents, desc="Split progress"):
        try:
            splits = text_splitter.split_documents([doc])
            texts.extend(splits)
        except Exception as e:
            logger.warning(f"Failed to split document ({doc.metadata.get('source', 'Unknown')}): {e}")
            continue

    logger.success(f"✓ Split complete, {len(texts)} chunks produced")

    if texts:
        avg_length = sum(len(t.page_content) for t in texts) / len(texts)
        logger.info(f"  • Average chunk size: {avg_length:.0f} characters")

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
    logger.info("Creating vector store (may take a few minutes depending on corpus size)...")

    # Batch to avoid high memory usage
    batch_size = 100
    vectorstore = None

    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Vectorization progress"):
            batch = texts[i:i + batch_size]

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
    print("\n" + "="*60)
    print("📚 Document ingestion and vector store build")
    print("="*60 + "\n")

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    # Validate loader configuration
    if not validate_loaders():
        logger.error("Loader configuration incomplete; some formats cannot be processed.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
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

        # Step 2: Split documents
        texts = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)

        if not texts:
            logger.error("Document splitting failed, exiting")
            sys.exit(1)

        # Step 3: Load embedding model
        logger.info("Loading embedding model...")
        embeddings = get_embeddings()
        logger.success("✓ Embedding model loaded")

        # Step 4: Create vector store
        vectorstore = create_vectorstore(texts, embeddings)

        # Step 5: Save vector store
        logger.info(f"Saving vector store to {VECTOR_STORE_PATH}...")
        vectorstore.save_local(VECTOR_STORE_PATH)

        # Display summary
        print("\n" + "="*60)
        print("🎉 Vector store build complete!")
        print("="*60)
        print("Statistics:")
        print(f"  • Documents:       {len(documents)}")
        print(f"  • Document chunks: {len(texts)}")
        print(f"  • Vector dim:      {len(embeddings.embed_query('test'))}")
        print(f"  • Saved at:        {VECTOR_STORE_PATH}")

        faiss_file = Path(VECTOR_STORE_PATH) / "index.faiss"
        if faiss_file.exists():
            print(f"  • Index size:      {get_file_size(str(faiss_file))}")

        print("\n✓ You can now run: streamlit run ui/ui.py")
        print("="*60 + "\n")

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
