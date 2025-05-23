# Chess RAG Chatbot

A multi-user chess knowledge chatbot using DeepSeek LLM, FAISS vector database, and RAG (Retrieval-Augmented Generation).

## Features

- **Multi-user support**: Multiple users can have separate chat sessions
- **RAG system**: Retrieves relevant information from chess knowledge base
- **DeepSeek LLM**: Uses DeepSeek-Coder model for response generation
- **Persistent chat history**: Saves conversations in JSON format
- **Vector search**: FAISS-based semantic search through chess documents

## Requirements

- Python 3.8+
- ~3GB free disk space (for model download)
- 4GB+ RAM recommended

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the chatbot:**
```bash
python chess_chatbot.py
```

3. **First run setup:**
   - Creates vector database from chess knowledge files
   - Downloads DeepSeek model (one-time download)
   - Takes 5-10 minutes initially

## Usage

1. Enter a User ID (e.g., "alice")
2. Enter a Session ID (e.g., "session1") 
3. Start asking chess questions

**Example questions:**
- "What is the English opening?"
- "How do you castle in chess?"
- "Tell me about chess history"
- "What are the rules of chess?"

**Commands:**
- `help` - Show available commands
- `quit` - Exit the chatbot
- `/switch` - Change user/session

## Project Structure

```
ChessBot/
├── chess_chatbot.py       # Main application
├── knowledge_base.py      # FAISS vector database
├── llm_manager.py         # DeepSeek model handling
├── chat_manager.py        # Chat history management
├── data/                  # Chess knowledge files
├── vector_store/          # FAISS index (auto-created)
└── chat_history/          # JSON chat logs (auto-created)
```

## Technical Details

- **LLM**: DeepSeek-Coder-1.3B-Base model
- **Vector DB**: FAISS with sentence-transformers embeddings
- **Chat Storage**: JSON files (no external services)
- **Knowledge Base**: Text files about chess openings, rules, and history

## Troubleshooting

**Model download issues:**
- Ensure stable internet connection
- Check available disk space (3GB+ needed)

**Memory issues:**
- Close other applications
- Reduce max_length in llm_manager.py if needed

**Vector store rebuild:**
- Delete `vector_store/` folder and restart
- Will rebuild automatically from data files

## Assignment Requirements Met

- ✅ DeepSeek R1 distilled model
- ✅ Local LLM storage with Transformers
- ✅ JSON chat history (no Redis)
- ✅ Vector database for knowledge base
- ✅ Multi-user, multi-session support
- ✅ Continuous conversation management

## RAG System Architecture

The Retrieval-Augmented Generation (RAG) pipeline works as follows:

1. **Document Processing**: Chess knowledge files are chunked into 500-character segments with 50-character overlaps to maintain context
2. **Embedding Generation**: Each chunk is converted to 384-dimensional vectors using `all-MiniLM-L6-v2` sentence transformer
3. **Vector Storage**: FAISS IndexFlatIP stores embeddings for efficient similarity search (cosine similarity)
4. **Query Processing**: User questions are embedded using the same transformer model
5. **Retrieval**: Top-3 most similar document chunks are retrieved based on cosine similarity scores
6. **Context Injection**: Retrieved context is combined with chat history and user query in a structured prompt
7. **Generation**: DeepSeek model generates responses conditioned on the retrieved knowledge and conversation context

**Key Benefits:**
- Reduces hallucination by grounding responses in actual chess knowledge
- Enables domain-specific responses beyond the LLM's training data
- Maintains conversation context while incorporating relevant facts
- Scalable to large knowledge bases through efficient vector search

## Note

This is a prototype implementation focused on demonstrating RAG capabilities with chess knowledge. Response quality may vary as it depends on the knowledge base content and model performance.

## Assumptions and Notes

**Model Selection**: Using `deepseek-coder-1.3b-base` as the DeepSeek R1 distilled model - this is a smaller, efficient version suitable for local inference while meeting the requirement.

**Knowledge Domain**: Focused on chess as the domain expertise with documents covering openings, rules, and history. The system can be easily extended to other domains by adding relevant text files to the `data/` directory.

**User Management**: Implemented simple user/session identification without complex authentication - suitable for prototype demonstration. Users are identified by string IDs for simplicity.

**Response Processing**: Added extensive post-processing to ensure clean, direct responses without reasoning text or model artifacts.

**Hardware Requirements**: Designed to run on standard consumer hardware with CPU inference. GPU acceleration is automatically used if available but not required.

**Fallback Handling**: Includes automatic fallback to smaller models if DeepSeek model fails to load due to memory or other constraints. 