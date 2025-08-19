import os
import logging
from typing import List, Dict, Any, Tuple
import faiss
import pickle
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# NLP and ML libraries
from sentence_transformers import SentenceTransformer, CrossEncoder
import cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document


logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        # Model configurations
        self.embedding_model_name = "all-MiniLM-L6-v2"  
        self.reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        # Initialize models (will be loaded during initialize())
        self.embedding_model = None
        self.reranker_model = None
        
        # Vector store
        self.faiss_index = None
        self.documents = []
        self.document_metadata = []
        
        # Configuration
        self.chunk_size = 512
        self.chunk_overlap = 64
        self.top_k_retrieve = 20  
        self.top_k_rerank = 5     # Final number after reranking
        
        # Cohere client

        # Load environment variables
        load_dotenv()
        
        # Get the API key from environment
        cohere_api_key = os.getenv("COHERE_API_KEY")
        
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")
        
        # Initialize Cohere client
        self.cohere_client = cohere.Client(api_key=cohere_api_key)
        
        # Paths
        self.data_dir = Path("data")
        self.index_dir = Path("indices")
        self.index_dir.mkdir(exist_ok=True)
        
    def initialize(self):
        """Initialize all models and load/create the document index"""
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        logger.info("Loading reranker model...")
        self.reranker_model = CrossEncoder(self.reranker_model_name)
        
        # Load or create document index
        if self._index_exists():
            logger.info("Loading existing document index...")
            self._load_index()
        else:
            logger.info("Creating new document index...")
            self._create_index()
            
    def _index_exists(self) -> bool:
        """Check if pre-built index exists"""
        index_path = self.index_dir / "faiss_index.bin"
        metadata_path = self.index_dir / "metadata.pkl"
        return index_path.exists() and metadata_path.exists()
        
    def _load_index(self):
        """Load pre-built FAISS index and metadata"""
        try:
            # Load FAISS index
            index_path = self.index_dir / "faiss_index.bin"
            self.faiss_index = faiss.read_index(str(index_path))
            
            # Load metadata
            metadata_path = self.index_dir / "metadata.pkl"
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.document_metadata = data['metadata']
                
            logger.info(f"Loaded index with {len(self.documents)} document chunks")
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            logger.info("Falling back to creating new index...")
            self._create_index()
            
    def _create_index(self):
        """Create FAISS index from documents"""
        # Load documents
        documents = self._load_documents()
        
        if not documents:
            logger.warning("No documents found")
        
        # Chunk documents
        chunks = self._chunk_documents(documents)
        self.documents = chunks

        print(f"Documents loaded: {len(documents)}")
        print(f"Chunks created: {len(chunks)}")

        if not chunks:
            logger.error("No document chunks found! Cannot create embeddings.")
            return
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [doc.page_content for doc in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Create metadata
        self.document_metadata = [
            {
                "chunk_id": i,
                "source": doc.metadata.get("source", "unknown"),
                "title": doc.metadata.get("title", "Untitled"),
                "page": doc.metadata.get("page", 0),
                "chunk_text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            for i, doc in enumerate(chunks)
        ]
        
        # Save index and metadata
        self._save_index()
        
        logger.info(f"Created index with {len(chunks)} document chunks")
        
    def _load_documents(self) -> List[Document]:
        """Load documents from the data directory"""
        documents = []
        
        if not self.data_dir.exists():
            self.data_dir.mkdir(exist_ok=True)
            return documents
            
        # Load text files
        for txt_file in self.data_dir.glob("*.txt"):
            try:
                loader = TextLoader(str(txt_file))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = txt_file.name
                    doc.metadata["title"] = txt_file.stem.replace("_", " ").title()
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {txt_file}: {e}")
                
        # Load PDF files
        for pdf_file in self.data_dir.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = pdf_file.name
                    doc.metadata["title"] = pdf_file.stem.replace("_", " ").title()
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")
                
        return documents
        
    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents using intelligent text splitting"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = []
        for doc in documents:
            doc_chunks = text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
            
        return chunks
        
    def _save_index(self):
        """Save FAISS index and metadata"""
        # Save FAISS index
        index_path = self.index_dir / "faiss_index.bin"
        faiss.write_index(self.faiss_index, str(index_path))
        
        # Save metadata
        metadata_path = self.index_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.document_metadata,
                'created_at': datetime.now().isoformat()
            }, f)
            
    def process_query(
            self, 
            question: str, 
            conversation_history: List[Dict] = None
        ) -> Dict[str, Any]:
            """Process a query through the full RAG pipeline"""
            
            # Step 1: Generate query embedding
            query_embedding = self.embedding_model.encode([question])
            faiss.normalize_L2(query_embedding)
            
            # Step 2: Retrieve top-k similar chunks
            scores, indices = self.faiss_index.search(
                query_embedding.astype('float32'), 
                self.top_k_retrieve
            )
            
            # Step 3: Prepare candidates for reranking
            candidates = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):  # Valid index
                    doc = self.documents[idx]
                    candidates.append({
                        'text': doc.page_content,
                        'metadata': self.document_metadata[idx],
                        'initial_score': float(score),
                        'index': idx
                    })
            
            # Step 4: Rerank 
            top_candidates = []
            if candidates:
                try:
                    pairs = [[question, candidate['text']] for candidate in candidates]
                    rerank_scores = self.reranker_model.predict(pairs)
                    
                    # Add rerank scores to candidates
                    for candidate, rerank_score in zip(candidates, rerank_scores):
                        candidate['rerank_score'] = float(rerank_score)
                    
                    # Sort by rerank score and take top candidates
                    candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
                    top_candidates = candidates[:self.top_k_rerank]
                    
                except Exception as e:
                    logger.error(f"Error in reranking: {e}")
                    # Fallback: use initial scores
                    candidates.sort(key=lambda x: x['initial_score'], reverse=True)
                    top_candidates = candidates[:self.top_k_rerank]
                    # Add rerank_score as fallback
                    for candidate in top_candidates:
                        candidate['rerank_score'] = candidate['initial_score']
            
            # Step 5: Generate answer using LLM
            answer, confidence = self._generate_answer(
                question, 
                top_candidates, 
                conversation_history
            )
            
            # Step 6: Format sources (COMMENTED OUT - can be enabled later)
            sources = [
                {
                    "title": candidate['metadata']['title'],
                    "source": candidate['metadata']['source'],
                    "relevance_score": candidate.get('rerank_score', candidate.get('initial_score', 0.0)),
                    "snippet": candidate['text'][:300] + "..." if len(candidate['text']) > 300 else candidate['text']
                }
                for candidate in top_candidates
            ]
            
            return {
                "answer": answer,
                "sources": sources,  # COMMENTED OUT - can be enabled later
                "confidence_score": confidence # COMMENTED OUT - can be enabled later
            }

    def _generate_answer(
        self, 
        question: str, 
        context_chunks: List[Dict], 
        conversation_history: List[Dict] = None
    ) -> Tuple[str, float]:
        """Generate answer using Cohere's Command model"""
        
        # Prepare context
        context = "\n\n".join([
            f"Source: {chunk['metadata']['title']}\n{chunk['text']}"
            for chunk in context_chunks
        ])
        
        # Prepare conversation context
        conversation_context = ""
        if conversation_history:
            recent_history = conversation_history[-3:]  # Last 3 exchanges
            history_parts = []
            for exchange in recent_history:
                if exchange.get('user'):
                    history_parts.append(f"User: {exchange['user']}")
                if exchange.get('assistant'):
                    history_parts.append(f"Assistant: {exchange['assistant']}")
            if history_parts:
                conversation_context = f"\nRecent conversation:\n{chr(10).join(history_parts)}\n"
        
        # Create prompt
        prompt = f"""You are an expert assistant specializing in natural disasters especially in forest fires. Answer the following question using the provided context. Be accurate, informative and short with your answers. Do not ask questions in the end.

Context from knowledge base:
{context}
{conversation_context}
Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to fully answer the question, say so and provide what information you can."""

        try:
            # Use Cohere's Chat API
            response = self.cohere_client.chat(
                message=prompt,
                model="command",  # or "command-light" for faster/cheaper responses
                temperature=0.1,
                max_tokens=500,
                preamble="You are an expert in natural disasters especially in forest fires. Provide accurate, helpful answers based on the given context."
            )
            
            answer = response.text.strip()
            
            # Simple confidence estimation based on context relevance
            confidence = min(len(context_chunks) / self.top_k_rerank, 1.0) * 0.8 + 0.2
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Error generating answer with Cohere: {e}")
            return "I'm sorry, I encountered an error while generating an answer. Please try again.", 0.0
    
    def get_document_metadata(self) -> List[Dict]:
        """Return metadata about indexed documents"""
        return self.document_metadata
        
    def reload_documents(self):
        """Reload documents and rebuild index"""
        self._create_index()