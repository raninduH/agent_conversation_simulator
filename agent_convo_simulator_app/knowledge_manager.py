import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import time
from datetime import datetime
import PyPDF2
import json
import uuid
import io

# Load environment variables from .env file
load_dotenv()

def setup_embedding_model():
    """
    Loads the embedding model and sets it globally.
    This function should be called once when the application starts.
    """
    # Use a pre-trained model from HuggingFace
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    print(f"🔧 EMBEDDING MODEL SETUP: Starting setup for embedding model '{model_name}'...")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        start_time = time.time()
        print(f"⏳ Loading SentenceTransformer model '{model_name}'...")
        
        # Load the model directly with SentenceTransformers
        model = SentenceTransformer(model_name)
        
        load_time = time.time() - start_time
        print(f"✅ EMBEDDING MODEL SETUP COMPLETE!")
        print(f"   📊 Model: {model_name}")
        print(f"   ⏱️  Load time: {load_time:.2f} seconds")
        print(f"   🎯 Ready for document embedding tasks")
        print("-" * 60)
        
        return model
        
    except Exception as e:
        print(f"❌ EMBEDDING MODEL SETUP FAILED!")
        print(f"   🚨 Error: {e}")
        print(f"   💡 Please ensure 'sentence-transformers' and 'torch' are installed")
        print(f"   💻 Run: pip install sentence-transformers torch")
        print("-" * 60)
        raise

def load_document(file_path):
    """Load document content from various file types."""
    print(f"   📄 Loading: {os.path.basename(file_path)}")
    
    try:
        if file_path.lower().endswith('.pdf'):
            # Load PDF file
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += page_text + "\n"
                print(f"      📄 PDF pages processed: {len(pdf_reader.pages)}")
                return text.strip()
        else:
            # Load text file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                print(f"      📝 Text file loaded successfully")
                return content.strip()
    except Exception as e:
        print(f"      ❌ Failed to load document: {e}")
        return ""

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks

def ingest_agent_documents(agent_id: str):
    """
    Chunks, embeds, and stores documents for a given agent in a unique Pinecone index.
    
    Args:
        agent_id: The unique identifier for the agent.
    """
    print("=" * 80)
    print(f"🚀 STARTING DOCUMENT INGESTION FOR AGENT: {agent_id}")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    agent_docs_path = os.path.join("knowledge_base", agent_id)
    
    # Check if documents exist
    print(f"📂 DOCUMENT DISCOVERY:")
    print(f"   📁 Checking path: {os.path.abspath(agent_docs_path)}")
    
    if not os.path.exists(agent_docs_path):
        print(f"   ❌ Directory does not exist!")
        print(f"   ℹ️  No documents found for agent {agent_id}. Skipping ingestion.")
        print("=" * 80)
        return
    
    files_in_directory = os.listdir(agent_docs_path)
    if not files_in_directory:
        print(f"   ❌ Directory is empty!")
        print(f"   ℹ️  No documents found for agent {agent_id}. Skipping ingestion.")
        print("=" * 80)
        return
    
    print(f"   ✅ Found {len(files_in_directory)} file(s):")
    for i, file in enumerate(files_in_directory, 1):
        file_path = os.path.join(agent_docs_path, file)
        file_size = os.path.getsize(file_path)
        print(f"      {i}. {file} ({file_size:,} bytes)")

    # Check environment variables
    print(f"\n🔐 ENVIRONMENT VALIDATION:")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENV")
    
    if pinecone_api_key:
        print(f"   ✅ PINECONE_API_KEY found (length: {len(pinecone_api_key)} chars)")
    else:
        print(f"   ❌ PINECONE_API_KEY not found!")
        
    if pinecone_env:
        print(f"   ✅ PINECONE_ENV found: {pinecone_env}")
    else:
        print(f"   ❌ PINECONE_ENV not found!")

    if not pinecone_api_key or not pinecone_env:
        print(f"\n❌ CONFIGURATION ERROR!")
        print(f"   🚨 PINECONE_API_KEY and PINECONE_ENV must be set in the .env file.")
        print(f"   💡 Please check your .env file configuration")
        print("=" * 80)
        return

    try:
        # Initialize Pinecone
        print(f"\n🌲 PINECONE CONNECTION:")
        print(f"   ⏳ Initializing Pinecone connection...")
        start_time = time.time()
        
        # Use new Pinecone API
        pc = Pinecone(api_key=pinecone_api_key)
        
        connection_time = time.time() - start_time
        print(f"   ✅ Pinecone initialized successfully!")
        print(f"   ⏱️  Connection time: {connection_time:.2f} seconds")
        print(f"   🌍 Environment: {pinecone_env}")
        
        # Create index name
        index_name = f"agent-kb-{agent_id.lower().replace('_', '-')}"
        print(f"   📋 Index name: {index_name}")
        
        # Check existing indexes
        print(f"\n📋 INDEX MANAGEMENT:")
        print(f"   ⏳ Checking existing indexes...")
        existing_indexes = pc.list_indexes().names()
        print(f"   📊 Found {len(existing_indexes)} existing index(es):")
        for idx in existing_indexes:
            print(f"      - {idx}")
        
        # The dimension for all-MiniLM-L6-v2 is 384
        dimension = 384
        
        if index_name not in existing_indexes:
            print(f"   🆕 Creating new index '{index_name}'...")
            print(f"      📐 Dimension: {dimension}")
            print(f"      📏 Metric: cosine")
            
            create_start = time.time()
            pc.create_index(
                name=index_name, 
                dimension=dimension, 
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            create_time = time.time() - create_start
            
            print(f"   ✅ Index created successfully!")
            print(f"   ⏱️  Creation time: {create_time:.2f} seconds")
        else:
            print(f"   ♻️  Using existing index '{index_name}'")

        # Connect to index
        print(f"\n🔗 INDEX CONNECTION:")
        print(f"   ⏳ Connecting to index '{index_name}'...")
        
        pinecone_index = pc.Index(index_name)
        
        # Get index stats
        try:
            stats = pinecone_index.describe_index_stats()
            print(f"   ✅ Connected to index successfully!")
            print(f"   📊 Index stats:")
            print(f"      🔢 Total vectors: {stats.get('total_vector_count', 'Unknown')}")
            print(f"      📏 Dimension: {stats.get('dimension', 'Unknown')}")
        except Exception as e:
            print(f"   ⚠️  Connected but couldn't get stats: {e}")

        # Setup embedding model
        print(f"\n🤖 EMBEDDING MODEL SETUP:")
        model = setup_embedding_model()

        # Load and process documents
        print(f"\n📚 DOCUMENT PROCESSING:")
        all_chunks = []
        total_chars = 0
        
        for i, file in enumerate(files_in_directory, 1):
            file_path = os.path.join(agent_docs_path, file)
            print(f"   📄 Processing file {i}/{len(files_in_directory)}: {file}")
            
            # Load document content
            content = load_document(file_path)
            if not content:
                print(f"      ⚠️  Skipping empty or unreadable file")
                continue
                
            char_count = len(content)
            total_chars += char_count
            print(f"      📊 Content: {char_count:,} characters")
            
            # Chunk the document
            print(f"      � Chunking document...")
            chunks = chunk_text(content, chunk_size=1000, overlap=200)
            print(f"      📦 Created {len(chunks)} chunks")
            
            # Add metadata to chunks
            for j, chunk in enumerate(chunks):
                all_chunks.append({
                    'id': f"{agent_id}_{file}_{j}",
                    'text': chunk,
                    'source': file,
                    'chunk_index': j
                })
        
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   📄 Files processed: {len(files_in_directory)}")
        print(f"   📝 Total characters: {total_chars:,}")
        print(f"   📦 Total chunks: {len(all_chunks)}")

        # Generate embeddings and upload to Pinecone
        print(f"\n🔄 EMBEDDING & UPLOAD:")
        print(f"   ⏳ Generating embeddings for {len(all_chunks)} chunks...")
        
        batch_size = 100
        uploaded_count = 0
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            print(f"   📦 Processing batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
            
            # Generate embeddings for this batch
            texts = [chunk['text'] for chunk in batch]
            embeddings = model.encode(texts)
            
            # Prepare vectors for upsert
            vectors = []
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                vectors.append({
                    'id': chunk['id'],
                    'values': embedding.tolist(),
                    'metadata': {
                        'text': chunk['text'],
                        'source': chunk['source'],
                        'chunk_index': chunk['chunk_index']
                    }
                })
            
            # Upload to Pinecone
            pinecone_index.upsert(vectors)
            uploaded_count += len(vectors)
            print(f"      ✅ Uploaded {len(vectors)} vectors (total: {uploaded_count})")
        
        print(f"\n✅ INGESTION COMPLETED SUCCESSFULLY!")
        print(f"   📊 Final results:")
        print(f"      🎯 Agent: {agent_id}")
        print(f"      📋 Index: {index_name}")
        print(f"      📄 Documents processed: {len(files_in_directory)}")
        print(f"      📝 Total characters: {total_chars:,}")
        print(f"      📦 Chunks created: {len(all_chunks)}")
        print(f"      🔢 Vectors uploaded: {uploaded_count}")
        
        # Final index stats
        try:
            final_stats = pinecone_index.describe_index_stats()
            print(f"      � Final index vectors: {final_stats.get('total_vector_count', 'Unknown')}")
        except Exception as e:
            print(f"      ⚠️  Couldn't get final stats: {e}")
            
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ INGESTION FAILED!")
        print(f"   🚨 Error: {e}")
        print(f"   📋 Agent: {agent_id}")
        print(f"   📁 Path: {agent_docs_path}")
        print(f"   💡 Please check your configuration and try again")
        print("=" * 80)
        import traceback
        traceback.print_exc()


def query_pinecone(index_name: str, query: str, top_k: int = 3):
    """
    Query a Pinecone index for relevant documents.
    
    Args:
        index_name: The name of the Pinecone index
        query: The search query
        top_k: Number of results to return
        
    Returns:
        List of dictionaries containing relevant document chunks
    """
    print(f"\n🔍 KNOWLEDGE BASE QUERY:")
    print(f"   📋 Index: {index_name}")
    print(f"   🔎 Query: '{query}'")
    print(f"   🔢 Requesting top {top_k} results")
    print(f"   📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Check environment variables
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENV")
        
        if not pinecone_api_key or not pinecone_env:
            print(f"   ❌ Missing Pinecone credentials!")
            return []
        
        print(f"   ✅ Pinecone credentials found")
        
        # Initialize Pinecone
        print(f"   ⏳ Connecting to Pinecone...")
        start_time = time.time()
        
        # Use new Pinecone API
        pc = Pinecone(api_key=pinecone_api_key)
        
        connection_time = time.time() - start_time
        print(f"   ✅ Connected to Pinecone ({connection_time:.2f}s)")
        
        # Check if index exists
        existing_indexes = pc.list_indexes().names()
        if index_name not in existing_indexes:
            print(f"   ❌ Index '{index_name}' not found!")
            print(f"   📋 Available indexes: {existing_indexes}")
            return []
        
        print(f"   ✅ Index '{index_name}' found")
        
        # Connect to index
        pinecone_index = pc.Index(index_name)
        
        # Check index stats
        try:
            stats = pinecone_index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            print(f"   📊 Index contains {vector_count} vectors")
            
            if vector_count == 0:
                print(f"   ⚠️  Index is empty - no documents to search!")
                return []
                
        except Exception as e:
            print(f"   ⚠️  Couldn't get index stats: {e}")
        
        # Setup embedding model and encode query
        print(f"   ⏳ Encoding query...")
        model = setup_embedding_model()
        query_embedding = model.encode([query])[0].tolist()
        
        # Perform query
        print(f"   🔍 Executing search...")
        query_start = time.time()
        
        search_results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        query_time = time.time() - query_start
        print(f"   ⏱️  Query completed in {query_time:.2f} seconds")
        
        # Process results
        results = []
        if search_results.matches:
            print(f"   ✅ Found {len(search_results.matches)} relevant result(s):")
            
            for i, match in enumerate(search_results.matches, 1):
                score = match.score
                content = match.metadata.get('text', 'No content available')
                source = match.metadata.get('source', 'Unknown source')
                
                print(f"      {i}. Score: {score:.4f} | Source: {source} | Content: {content[:100]}...")
                
                results.append({
                    "content": content,
                    "score": score,
                    "source": source,
                    "rank": i
                })
        else:
            print(f"   ❌ No relevant results found")
            
        print(f"   🎯 Returning {len(results)} result(s)")
        return results
        
    except Exception as e:
        print(f"   ❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
        return []


class KnowledgeManager:
    """Main class for managing knowledge base operations."""
    
    def __init__(self):
        print(f"🧠 KNOWLEDGE MANAGER INITIALIZED")
        print(f"   📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            setup_embedding_model()
        except Exception as e:
            print(f"   ⚠️  Warning: Embedding model setup failed: {e}")
    
    def ingest_agent_documents(self, agent_id: str):
        """Wrapper method for document ingestion."""
        return ingest_agent_documents(agent_id)
    
    def query_pinecone(self, index_name: str, query: str, top_k: int = 3):
        """Wrapper method for querying Pinecone."""
        return query_pinecone(index_name, query, top_k)

