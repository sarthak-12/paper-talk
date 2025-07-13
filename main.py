import os
import torch
import pickle
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq  # Import the Groq client

# --- 1. Global Objects & Configuration ---

# Load environment variables from the .env file
load_dotenv()

# We load the model and data once when the app starts.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"API is using device: {device}")

# Load the sentence-transformer model for embedding
print("Loading embedding model...")
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
print("Embedding model loaded.")

# Load the pre-computed document chunks and embeddings
print("Loading document chunks and embeddings...")
with open("text_chunks.pkl", "rb") as f:
    text_chunks = pickle.load(f)
embeddings_tensor = torch.load("embeddings.pt", map_location=device)
print("Chunks and embeddings loaded.")

# NEW: Initialize the Groq client
try:
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    print("Groq client initialized successfully.")
except KeyError:
    print("ERROR: GROQ_API_KEY environment variable not found.")
    groq_client = None

# --- 2. FastAPI App Initialization ---

app = FastAPI(
    title="Paper-Chat API",
    description="An API to ask questions about a research paper using RAG.",
    version="0.2.0",
)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

# --- 3. API Endpoint ---

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Accepts a question, finds relevant context, and generates a natural language answer.
    """
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized. Check API key.")

    try:
        # 1. Embed the user's question
        print(f"Received question: {request.question}")
        question_embedding = embedding_model.encode(
            request.question, convert_to_tensor=True, device=device
        )
        
        # 2. Find the top-k most similar chunks (our retrieval step)
        cosine_scores = util.cos_sim(question_embedding, embeddings_tensor)[0]
        top_k = 3 # Let's use 3 chunks for a more focused context
        top_results = torch.topk(cosine_scores, k=top_k)
        
        # 3. NEW: Build the context for the LLM
        context = ""
        for score, idx in zip(top_results[0], top_results[1]):
            context += text_chunks[idx] + "\n\n"
            
        # 4. NEW: Create the prompt and call the Groq API (our generation step)
        print("Generating answer with Groq...")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Q&A assistant. Your goal is to answer the user's question based *only* on the provided context.\n"
                        "Do not use any outside knowledge. If the answer is not in the context, say 'The answer is not available in the provided text.'"
                    )
                },
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{request.question}"
                }
            ],
            model="llama3-8b-8192", # A powerful and fast model on Groq
            temperature=0.2, # Lower temperature for more factual answers
        )
        
        generated_answer = chat_completion.choices[0].message.content
        print("Answer generated successfully.")
        
        return {"answer": generated_answer}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    """A simple endpoint to check if the API is running."""
    return {"status": "ok"}