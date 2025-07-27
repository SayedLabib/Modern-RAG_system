# Multilingual RAG System

A simple and efficient Retrieval-Augmented Generation (RAG) system optimized for multilingual content, especially Bangla and English text processing.

## Architecture

This RAG system follows the recommended tech stack:

- **Chunking**: LangChain's RecursiveCharacterTextSplitter
- **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2 (local, multilingual)
- **Vector DB**: Qdrant (fast, hybrid-ready)
- **LLM**: meta-llama models via Groq Cloud API
- **Backend**: FastAPI + Docker
- **Orchestration**: Custom pipeline with LangChain components

## Project Structure

```
app/
├── main.py                    # FastAPI application entry point
├── process_data.py           # Data processing script
├── api/
│   └── routes.py             # API endpoints (/query, /health)
├── core/
│   └── config.py             # Configuration and settings
├── models/
│   ├── request.py            # Pydantic request models
│   └── response.py           # Pydantic response models
├── embeddings/
│   └── embeddings.py         # Embedding service (MiniLM)
├── chunking/
│   └── chunker.py            # Text chunking with LangChain
├── vectorstore/
│   └── qdrant_client.py      # Qdrant vector store client
├── retrieval/
│   └── retriever.py          # Hybrid retrieval logic
├── llm/
│   └── generator.py          # LLM generation via Groq
├── services/
│   └── rag_service.py        # Main RAG orchestration
└── data/                     # Raw and processed data
    └── Bangla_cleaned_book.txt
```

## Features

- **Simple API**: Only 2 endpoints - `/query` and `/health`
- **Multilingual Support**: Optimized for Bangla and English
- **Local Embeddings**: No external API calls for embeddings
- **Fast Vector Search**: Qdrant for efficient similarity search
- **Hybrid Retrieval**: Dense vector search with optional reranking
- **Health Monitoring**: Comprehensive health checks for all components
- **Docker Ready**: Easy deployment with Docker

## Quick Start

### 1. Prerequisites

- Python 3.8+
- Docker (optional, for Qdrant)
- Groq API key (for LLM generation)

### 2. Installation

```bash
# Clone and navigate to the project
cd "Modern RAG system"

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Qdrant Vector Database

**Option A: Using Docker (Recommended)**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Option B: Local Installation**
Follow [Qdrant installation guide](https://qdrant.tech/documentation/quick_start/)

### 4. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your Groq API key
# Get API key from: https://console.groq.com/keys
```

### 5. Process Your Data

```bash
# Place your text files in app/data/
# Then run the data processing script
cd app
python process_data.py
```

### 6. Start the Server

```bash
cd app
python main.py
```

The API will be available at `http://localhost:8000`

## API Usage

### Query Endpoint

**POST** `/api/v1/query`

```json
{
  "query": "বাংলা সাহিত্যের ইতিহাস সম্পর্কে বলুন",
  "top_k": 5
}
```

**Response:**
```json
{
  "response": "বাংলা সাহিত্যের ইতিহাস অত্যন্ত সমৃদ্ধ...",
  "sources": ["Document chunk_0001: ...", "Document chunk_0023: ..."],
  "confidence_score": 0.85,
  "retrieval_time": 0.1,
  "generation_time": 0.5
}
```

### Health Check Endpoint

**GET** `/api/v1/health`

```json
{
  "status": "healthy",
  "embedding_service": true,
  "vector_store": true,
  "llm_service": true
}
```

## Configuration Options

Key settings in `.env`:

```env
# Environment Configuration for Multilingual RAG System

# Application Settings
APP_NAME=Multilingual RAG System
APP_VERSION=1.0.0
DEBUG=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=multilingual_knowledge_base
QDRANT_VECTOR_SIZE=384

# Embedding Model Configuration
EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DEVICE=cpu

# LLM Configuration (Groq)
# Get your API key from: https://console.groq.com/keys
GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
GROQ_TEMPERATURE=0.2
GROQ_MAX_TOKENS=2048

# Retrieval Configuration
DEFAULT_TOP_K=5
SIMILARITY_THRESHOLD=0.3
MAX_CHUNK_LENGTH=800
CHUNK_OVERLAP=100

# Data Paths
DATA_DIR=app/data
PROCESSED_CHUNKS_FILE=processed_chunks.txt
BANGLA_BOOK_FILE=Bangla_cleaned_book_improved.txt
```

## Development

### Running in Development Mode

```bash
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Adding New Data

1. Place text files in `app/data/`
2. Run `python process_data.py`
3. Files will be automatically chunked and indexed

### Testing the API

```bash
# Test query endpoint
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "আপনার প্রশ্ন লিখুন", "top_k": 5}'

# Test health endpoint
curl "http://localhost:8000/api/v1/health"
```

## Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t multilingual-rag .

# Run with environment file
docker run -p 8000:8000 --env-file .env multilingual-rag
```

### Docker Compose (with Qdrant)

```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
  
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - QDRANT_HOST=qdrant
    depends_on:
      - qdrant
```

## Performance Tips

1. **Use GPU**: Set `EMBEDDING_DEVICE=cuda` if you have a GPU
2. **Optimize Chunk Size**: Adjust `MAX_CHUNK_LENGTH` based on your data
3. **Tune Retrieval**: Modify `SIMILARITY_THRESHOLD` for precision/recall balance
4. **Model Selection**: Choose appropriate Groq model for your use case

## Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**
   - Ensure Qdrant is running on the specified host/port
   - Check firewall settings

2. **Import Errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python path configuration

3. **Groq API Errors**
   - Verify API key in `.env` file
   - Check API rate limits

4. **Empty Responses**
   - Ensure data has been processed and indexed
   - Lower `SIMILARITY_THRESHOLD` if no results found

### Logs

Check logs for detailed error information:
```bash
# Application logs are printed to stdout
# For production, configure proper logging
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the logs for error details
- Open an issue on the repository

---

**Note**: This is a simplified RAG system designed for easy deployment and maintenance. For production use, consider additional features like authentication, rate limiting, and monitoring.
