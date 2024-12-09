from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from text_processing import extract_text_from_pdf, clean_text

def chunk_text(text, chunk_size=1000, chunk_overlap=150):
    """
    Split text into overlapping chunks.

    Args:
        text (str): Input text.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "]
    )
    return text_splitter.split_text(text)

def generate_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    """
    Generate embeddings for text chunks.

    Args:
        chunks (list): List of text chunks.
        model_name (str): SentenceTransformer model name.

    Returns:
        np.ndarray: Array of embeddings.
    """
    model = SentenceTransformer(model_name)
    return model.encode(chunks, convert_to_numpy=True)

def process_pdf_for_rag(pdf_path, chunk_size=500):
    """
    Process a PDF for RAG by extracting, cleaning, and chunking.

    Args:
        pdf_path (str): Path to the PDF file.
        chunk_size (int): Size of each chunk.

    Returns:
        list: List of text chunks.
    """
    
    print("Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)
    print("Cleaning text...")
    clean_text_content = clean_text(raw_text)
    print("Chunking text...")
    chunks = chunk_text(clean_text_content, chunk_size)
    print("Processing complete!")

    return chunks
