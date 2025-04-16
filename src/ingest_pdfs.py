import os
import shutil
import logging
from dotenv import load_dotenv
import chromadb

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

logging.basicConfig(level=logging.INFO)

def load_and_process_pdfs(data_dir: str, chunk_size: int, chunk_overlap: int):
    """
    Load PDF files from the specified directory and split them into smaller chunks.
    Returns a list of chunked documents.
    
    Args:
        data_dir: Path to the directory containing PDF files.
        chunk_size: The size of each text chunk.
        chunk_overlap: Overlap size between consecutive chunks.
        
    Returns:
        A list of chunked documents.
    """
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    
    try:
        # Load all documents at once
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} PDF documents.")
    except Exception as e:
        logging.error(f"Failed to load documents: {str(e)}")
        return []

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks, persist_directory: str):
    """
    Create and persist a Chroma vector store from the provided text chunks.
    """

    if os.path.exists(persist_directory):
        logging.info(f"Clearing existing vector store at {persist_directory}")
        shutil.rmtree(persist_directory)

    # Initialize Azure OpenAI embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_EMBEDDINGS_MODEL"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        model=os.getenv("AZURE_EMBEDDINGS_MODEL")
    )

    client = chromadb.Client(chromadb.config.Settings(
        is_persistent=True,
        persist_directory=persist_directory,
        anonymized_telemetry=False
    ))

    logging.info("Creating new vector store...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        client=client
    )
    return vectordb

def main():
    """
    Main entry point for ingesting PDFs and creating a vector store.
    """
    # Load directories from environment or defaults
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "data")
    db_dir = os.path.join(script_dir, "chroma_db")

    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))

    logging.info("Loading and processing PDFs...")
    chunks = load_and_process_pdfs(data_dir, chunk_size, chunk_overlap)
    logging.info(f"Created {len(chunks)} chunks from PDFs.")

    logging.info("Creating vector store...")
    create_vector_store(chunks, db_dir)
    logging.info("Vector store created successfully.")

if __name__ == "__main__":
    main()
