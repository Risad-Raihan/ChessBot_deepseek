import os
import pickle
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss


class DocumentChunker:
    """Handles document chunking with overlap for better context preservation."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into overlapping chunks."""
        # Split into sentences for better semantic chunks
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Add sentence to current chunk
            if len(current_chunk) + len(sentence) + 2 <= self.chunk_size:
                current_chunk += sentence + ". "
            else:
                # Save current chunk and start new one
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'source': source,
                        'length': len(current_chunk)
                    })
                
                # Start new chunk with overlap
                current_chunk = sentence + ". "
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'source': source,
                'length': len(current_chunk)
            })
        
        return chunks


class KnowledgeBase:
    """Manages the chess knowledge base with FAISS vector storage."""
    
    def __init__(self, data_dir: str = "data", vector_store_dir: str = "vector_store"):
        self.data_dir = data_dir
        self.vector_store_dir = vector_store_dir
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunker = DocumentChunker()
        self.index = None
        self.document_store = []
        
        # Create vector store directory if it doesn't exist
        os.makedirs(vector_store_dir, exist_ok=True)
    
    def load_documents(self) -> List[Dict]:
        """Load all text documents from the data directory."""
        documents = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.data_dir, filename)
                print(f"Loading {filename}...")
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Chunk the document
                    chunks = self.chunker.chunk_text(content, filename)
                    documents.extend(chunks)
                    print(f"  - Created {len(chunks)} chunks from {filename}")
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        print(f"Total documents loaded: {len(documents)}")
        return documents
    
    def create_embeddings(self, documents: List[Dict]) -> np.ndarray:
        """Generate embeddings for all document chunks."""
        print("Generating embeddings...")
        texts = [doc['text'] for doc in documents]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_vector_store(self):
        """Build and save the FAISS vector store."""
        print("Building vector store...")
        
        # Load documents
        documents = self.load_documents()
        self.document_store = documents
        
        # Generate embeddings
        embeddings = self.create_embeddings(documents)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        
        # Save the vector store
        self.save_vector_store()
        print(f"Vector store built with {len(documents)} documents")
    
    def save_vector_store(self):
        """Save the FAISS index and document store."""
        index_path = os.path.join(self.vector_store_dir, "chess_knowledge.index")
        docs_path = os.path.join(self.vector_store_dir, "documents.pkl")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save document store
        with open(docs_path, 'wb') as f:
            pickle.dump(self.document_store, f)
        
        print(f"Vector store saved to {self.vector_store_dir}")
    
    def load_vector_store(self):
        """Load existing FAISS index and document store."""
        index_path = os.path.join(self.vector_store_dir, "chess_knowledge.index")
        docs_path = os.path.join(self.vector_store_dir, "documents.pkl")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            print("Loading existing vector store...")
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load document store
            with open(docs_path, 'rb') as f:
                self.document_store = pickle.load(f)
            
            print(f"Vector store loaded with {len(self.document_store)} documents")
            return True
        return False
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents given a query."""
        if self.index is None:
            raise ValueError("Vector store not loaded. Call load_vector_store() or build_vector_store() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Return relevant documents with scores
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.document_store):
                doc = self.document_store[idx].copy()
                doc['relevance_score'] = float(score)
                doc['rank'] = i + 1
                results.append(doc)
        
        return results


def initialize_knowledge_base():
    """Initialize the knowledge base - build or load vector store."""
    kb = KnowledgeBase()
    
    # Try to load existing vector store
    if not kb.load_vector_store():
        print("No existing vector store found. Building new one...")
        kb.build_vector_store()
    
    return kb


if __name__ == "__main__":
    # Test the knowledge base
    kb = initialize_knowledge_base()
    
    # Test search
    test_query = "What is the English opening in chess?"
    results = kb.search(test_query)
    
    print(f"\nTest search for: '{test_query}'")
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {result['relevance_score']:.3f}) ---")
        print(f"Source: {result['source']}")
        print(f"Text: {result['text'][:200]}...") 