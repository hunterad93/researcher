import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
from typing import Dict, Any, List
from datetime import datetime, timedelta
import hashlib

pc = Pinecone(api_key=st.secrets.PINECONE_API_KEY)
cache_index = pc.Index('researcher-cache')

openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)


def generate_embedding(text: str) -> list[float]:
    """Generate an embedding for the given text."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=[text],
        encoding_format="float",
        dimensions=256
    )
    return response.data[0].embedding

def query_pinecone(query_embedding: List[float], top_k: int = 1, presearch_filter: Dict[str, Any] = {}) -> Dict[str, Any]:
    """Query Pinecone index with the given embedding."""
    results = cache_index.query(
        vector=query_embedding,
        filter=presearch_filter,
        top_k=top_k,
        include_metadata=True
    )
    return results

def generate_id(text: str) -> str:
    """Generate a hash ID from the given text."""
    return hashlib.sha256(text.encode()).hexdigest()

def cache_summary(domain: str, data_type: str, initial_prompt: str, summary: str):
    """Cache the summary in Pinecone with a timestamp."""
    embedding = generate_embedding(initial_prompt)
    id = generate_id(initial_prompt)
    metadata = {
        "domain": domain,
        "data_type": data_type,
        "summary": summary,
        "initial_prompt": initial_prompt,
        "timestamp": int(datetime.now().timestamp())
    }
    cache_index.upsert(vectors=[(id, embedding, metadata)])

def get_cached_summary(initial_prompt: str):
    """Retrieve a cached summary from Pinecone, filtering for recent entries."""
    embedding = generate_embedding(initial_prompt)
        
    # Query Pinecone with the embedding and timestamp filter
    results = query_pinecone(
        query_embedding=embedding,
        top_k=1,
        presearch_filter={
            "timestamp": {"$gte": int((datetime.now() - timedelta(days=30)).timestamp())}
        }
    )

    if results['matches'] and results['matches'][0]['score'] > 0.95:
        return results['matches'][0]['metadata']
    return None