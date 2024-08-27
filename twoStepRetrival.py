import os
import pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Pinecone
try:
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))
    logger.info("Pinecone initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Pinecone: {e}")
    raise

# Load the sentence transformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence transformer model loaded successfully")
except Exception as e:
    logger.error(f"Error loading sentence transformer model: {e}")
    raise

def generate_embedding(query):
    try:
        query_embedding = model.encode(query).tolist()
        logger.info(f"Generated embedding for query: {query}")
        return query_embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise

query = "What camera is best for underwater photography?"
query_embedding = generate_embedding(query)

def query_pinecone(embedding, top_k=50):
    try:
        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
        logger.info(f"Retrieved {len(results['matches'])} results from Pinecone")
        return results
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        raise

search_results = query_pinecone(query_embedding)

def prepare_documents(results):
    try:
        documents = [
            {
                "id": match["id"],
                "text": match["metadata"].get("description", "No description available"),
                "score": match["score"]
            }
            for match in results["matches"]
        ]
        logger.info(f"Prepared {len(documents)} documents for reranking")
        return documents
    except Exception as e:
        logger.error(f"Error preparing documents: {e}")
        raise

documents = prepare_documents(search_results)

def rerank_results(query, documents, top_n=10):
    try:
        reranked = pinecone.rerank(
            model="bge-reranker-v2-m3",
            query=query,
            documents=documents,
            top_n=top_n
        )
        logger.info(f"Reranked {len(reranked)} documents")
        return reranked
    except Exception as e:
        logger.error(f"Error reranking results: {e}")
        raise

reranked_results = rerank_results(query, documents)

def format_results(reranked):
    formatted_results = []
    for rank, result in enumerate(reranked, 1):
        formatted_results.append({
            "rank": rank,
            "score": result["score"],
            "id": result["document"]["id"],
            "description": result["document"]["text"][:200] + "..."  # Truncate long descriptions
        })
    return formatted_results

def display_results(formatted_results):
    print("\nTop Recommended Products:")
    for result in formatted_results:
        print(f"\nRank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Product ID: {result['id']}")
        print(f"Description: {result['description']}")

formatted_output = format_results(reranked_results)
display_results(formatted_output)

# Optionally, save results to a file
import json
with open("recommendations.json", "w") as f:
    json.dump(formatted_output, f, indent=2)
logger.info("Results saved to recommendations.json")