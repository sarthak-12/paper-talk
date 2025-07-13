import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import torch
import pickle

# Ensure we're using the GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text content from a PDF file.

    Args:
        pdf_path: The file path to the PDF.

    Returns:
        A single string containing all the text from the PDF.
    """
    print(f"Reading text from {pdf_path}...")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    print("Finished reading PDF.")
    return full_text

def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    """
    Splits a long text into smaller chunks with overlap.

    Args:
        text: The input text string.
        chunk_size: The desired character length of each chunk.
        chunk_overlap: The number of characters to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    print("Splitting text into chunks...")
    # We will use a simple character-based splitter for now.
    # More advanced methods (like splitting by paragraphs or sentences) can also be used.
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    print(f"Created {len(chunks)} chunks.")
    return chunks

def generate_embeddings(chunks: list[str], model_name: str) -> torch.Tensor:
    """
    Generates embeddings for a list of text chunks using a sentence-transformer model.

    Args:
        chunks: A list of text chunks.
        model_name: The name of the sentence-transformer model to use.

    Returns:
        A PyTorch tensor containing the embeddings for each chunk.
    """
    print(f"Loading embedding model: {model_name}...")
    # We load our chosen model. The first time you run this, it will be downloaded.
    model = SentenceTransformer(model_name, device=device)
    
    print("Generating embeddings for all chunks...")
    # The .encode() method handles the embedding generation for us.
    embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
    print("Embeddings generated successfully.")
    return embeddings

# --- Main Execution ---
if __name__ == "__main__":
    PDF_FILE_PATH = "attention-is-all-you-need.pdf"
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

    # 1. Extract text from the PDF
    paper_text = extract_text_from_pdf(PDF_FILE_PATH)
    
    # 2. Split the text into manageable chunks
    text_chunks = chunk_text(paper_text)
    
    # 3. Generate an embedding for each chunk
    # These embeddings are the numerical representation of our document's content.
    embeddings_tensor = generate_embeddings(text_chunks, model_name=EMBEDDING_MODEL)

    # For now, we'll just print the shape of the output tensor.
    # It should be [number_of_chunks, embedding_dimension]
    # For bge-small-en-v1.5, the dimension is 384.
    print(f"\nShape of our embeddings tensor: {embeddings_tensor.shape}")

    # In the next steps, we will save these chunks and embeddings
    # into a vector database for efficient searching.
    # --- Add this to the end of your process_paper.py script ---

# 4. Save the chunks and embeddings for our API to use
print("\nSaving chunks and embeddings...")
torch.save(embeddings_tensor, "embeddings.pt")

with open("text_chunks.pkl", "wb") as f:
    pickle.dump(text_chunks, f)

print("Saved successfully. Ready for the API!")