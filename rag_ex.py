"""
RAG Chatbot Demo
- Uses ChromaDB (free vector database)
- Uses sentence-transformers (free embeddings)
- Uses Groq/HuggingFace (free LLM)
"""
 
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")
# ============================================
# CONFIGURATION
# ============================================
 
# Free embedding model (runs locally, no API key needed!)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Small, fast, free
 
# ============================================
# STEP 1: CREATE KNOWLEDGE BASE (Documents)
# ============================================
 
# Sample company documents
documents = [
    {
        "id": "doc1",
        "text": "Our company offers 20 days of paid time off per year for all full-time employees. Part-time employees receive 10 days of PTO.",
        "metadata": {"source": "HR Policy", "section": "Leave"}
    },
    {
        "id": "doc2",
        "text": "Health insurance coverage includes medical, dental, and vision. The company pays 80% of premiums for employees and 50% for dependents.",
        "metadata": {"source": "Benefits Guide", "section": "Insurance"}
    },
    {
        "id": "doc3",
        "text": "Remote work is available 3 days per week. Employees must be in office on Tuesdays and Thursdays for team meetings.",
        "metadata": {"source": "Work Policy", "section": "Remote Work"}
    },
    {
        "id": "doc4",
        "text": "Annual performance reviews occur in December. Salary increases range from 3-8% based on performance ratings.",
        "metadata": {"source": "HR Policy", "section": "Performance"}
    },
    {
        "id": "doc5",
        "text": "New employees receive a laptop, monitor, keyboard, and mouse. Additional equipment can be requested through IT portal.",
        "metadata": {"source": "IT Policy", "section": "Equipment"}
    },
    {
        "id": "doc6",
        "text": "Maternity leave is 12 weeks of paid leave for full-time employees. Part-time employees receive 6 weeks.",
        "metadata": {"source": "HR Policy", "section": "Maternity Leave"}
    }
]
 
print("="*70)
print("RAG CHATBOT DEMO - Building Knowledge Base")
print("="*70)
 
# ============================================
# STEP 2: INITIALIZE VECTOR DATABASE (ChromaDB)
# ============================================
 
print("\n[STEP 1] Initializing ChromaDB (Vector Database)...")
 
# Create ChromaDB client (stores in memory)
client = chromadb.Client()
 
# Create collection (like a table in database)
collection = client.create_collection(
    name="company_docs",
    metadata={"description": "Company policy documents"}
)
 
print("✅ Vector database initialized")
 
# ============================================
# STEP 3: LOAD EMBEDDING MODEL
# ============================================
 
print("\n[STEP 2] Loading embedding model...")
print(f"   Model: {EMBEDDING_MODEL}")
 
# Load free sentence transformer model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
 
print("✅ Embedding model loaded")
 
# ============================================
# STEP 4: CREATE EMBEDDINGS & STORE IN DB
# ============================================
 
print("\n[STEP 3] Creating embeddings for documents...")
 
for doc in documents:
    # Convert text to vector (embedding)
    embedding = embedding_model.encode(doc["text"]).tolist()
    
    # Store in vector database
    collection.add(
        embeddings=[embedding],
        documents=[doc["text"]],
        metadatas=[doc["metadata"]],
        ids=[doc["id"]]
    )
    
    print(f"   ✅ Embedded: {doc['id']} - {doc['metadata']['section']}")
 
print(f"\n✅ Knowledge base created with {len(documents)} documents")
 
# ============================================
# STEP 5: SEARCH FUNCTION (Retrieval)
# ============================================
 
def search_knowledge_base(query, n_results=2):
    """
    Search for relevant documents using semantic similarity
    """
    print(f"\n[SEARCH] Query: '{query}'")
    
    # Convert query to embedding
    query_embedding = embedding_model.encode(query).tolist()
    
    # Search vector database
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    # Extract relevant documents
    documents_found = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    print(f"[SEARCH] Found {len(documents_found)} relevant documents:")
    for i, (doc, meta, dist) in enumerate(zip(documents_found, metadatas, distances)):
        print(f"   {i+1}. {meta['section']} (similarity: {1-dist:.2f})")
        print(f"      Preview: {doc[:80]}...")
    
    return documents_found
 
# ============================================
# STEP 6: LLM QUERY FUNCTION (Generation)
# ============================================
 
def ask_llm(question, context_docs):
    """
    Send question + context to LLM for answer generation
    """
    
    # Combine retrieved documents as context
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context_docs)])
    
    # Create prompt with context
    prompt = f"""You are a helpful HR assistant. Answer the question based ONLY on the context provided.
 
Context:
{context}
 
Question: {question}
 
Answer (be concise and specific):"""
 
    print(f"\n[LLM] Generating answer...")
    
    # Check if API key is configured
    if GROQ_API_KEY:
        print("⚠️  API key not configured. Using mock response.")
        return f"Based on the company policy, {context_docs[0][:100]}..."
    
    try:
        # Call Groq API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,  # Low temp for factual answers
            "max_tokens": 150
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            return answer
        else:
            return f"Error: API returned {response.status_code}"
            
    except Exception as e:
        return f"Error calling LLM: {e}"
 
# ============================================
# STEP 7: RAG CHATBOT FUNCTION
# ============================================
 
def rag_chatbot(question):
    """
    Complete RAG pipeline:
    1. Retrieve relevant documents
    2. Augment LLM prompt with context
    3. Generate answer
    """
    print("\n" + "="*70)
    print(f"QUESTION: {question}")
    print("="*70)
    
    # Step 1: Retrieve (search vector DB)
    relevant_docs = search_knowledge_base(question, n_results=2)
    
    # Step 2: Augment + Generate (ask LLM with context)
    answer = ask_llm(question, relevant_docs)
    
    print("\n[ANSWER]")
    print(answer)
    print("="*70)
    
    return answer
 
# ============================================
# DEMO: TEST THE RAG CHATBOT
# ============================================
 
if __name__ == "__main__":
    
    print("\n\n" + "="*70)
    print("🤖 RAG CHATBOT IS READY!")
    print("="*70)
    
    # Test questions
    test_questions = [
        "How many maternity leave days do I get?",
        "How many days of vacation do I get?",
        "What is the remote work policy?",
        "Does the company provide health insurance?",
        "When are performance reviews conducted?",
        "What equipment will I receive as a new employee?"
    ]
    
    print("\n📝 Testing with sample questions...\n")
    
    for question in test_questions[:3]:  # Test first 3 questions
        rag_chatbot(question)
        print("\n")
    
    # ============================================
    # COMPARISON: With vs Without RAG
    # ============================================
    
    print("\n" + "="*70)
    print("🔍 RAG vs NON-RAG COMPARISON")
    print("="*70)
    
    print("\n❌ WITHOUT RAG (No context):")
    print("User: 'How many vacation days do I get?'")
    print("LLM:  'I don't have information about your company's vacation policy.'")
    
    print("\n✅ WITH RAG (Context provided):")
    print("User: 'How many vacation days do I get?'")
    print("LLM:  'Full-time employees receive 20 days of PTO per year.'")
    
    # ============================================
    # STATISTICS
    # ============================================
    
    print("\n" + "="*70)
    print("📊 KNOWLEDGE BASE STATISTICS")
    print("="*70)
    
    print(f"\nTotal Documents: {collection.count()}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Embedding Dimensions: 384")
    print(f"Vector Database: ChromaDB (in-memory)")
    
    # Show sample embedding
    sample_embedding = embedding_model.encode("test").tolist()
    print(f"\nSample Embedding (first 5 values): {sample_embedding[:5]}")
    
    print("\n" + "="*70)
    print("💡 RAG BENEFITS")
    print("="*70)
    print("✅ Answers based on YOUR documents")
    print("✅ No hallucinations (LLM can't make up facts)")
    print("✅ Always up-to-date (just update documents)")
    print("✅ Cites sources (you know where answer came from)")
    print("✅ Works with private/proprietary data")
    print("="*70)
