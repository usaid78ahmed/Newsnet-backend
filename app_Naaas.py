from flask import Flask, request, jsonify, Response, stream_with_context
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import json
import requests
from flask_cors import CORS
import networkx as nx
import psycopg2
from psycopg2.extras import RealDictCursor

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

DB_CONFIG = {
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',  # Adjust if your database is hosted elsewhere
    'port': 5432,
    'database': 'naas'
}

def get_db_connection():
    """Create and return a database connection."""
    return psycopg2.connect(**DB_CONFIG)
# Explicitly handle OPTIONS requests (preflight requests)
@app.route('/query', methods=['OPTIONS'])
def handle_options():
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Custom JSON Encoder to handle NumPy data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj,set):
            return list(obj)
        return super(NumpyEncoder, self).default(obj)

# Load the SentenceTransformer model.
model1 = SentenceTransformer("all-mpnet-base-v2")

def compute_cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return np.dot(a, b) / (a_norm * b_norm)

def embed_query_fn(query, index):
    """Embed the query using the SentenceTransformer model."""
    embedding = model.encode(query, convert_to_numpy=True)
    return embedding.reshape(1, -1)

# Load the FAISS index.
index = faiss.read_index("embeddings_index.faiss")

# Connect to MongoDB.
client = MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]
neighbors_collection = db["multihop_neighbors"]
articles_collection = db["id_to_article"]


def format_titles_query_prompt(seed_article, query, titles, neighbor_articles):
    """
    Build a system prompt that includes:
      - The user query.
      - The list of selected titles (Title Neighbors).
      - The seed article details (Query Neighbors).
      - Neighbor articles information.
    """
    system_prompt = (
        "You are an AI that answers questions based on two sets of evidence: the selected titles (Title Neighbors) and related articles (Query Neighbors).\n"
        "Answer the query using insights from both sets, and ensure your response is well-structured and informative.\n\n"
        f"User Query: {query}\n\n"
        "Title Neighbors:\n"
    )
    for title in titles:
        system_prompt += f"- {title}\n"
    system_prompt += "\nQuery Neighbor Seed Article:\n"
    system_prompt += f"**Title:** {seed_article.get('title', 'No title')}\n"
    system_prompt += f"**Published Date:** {seed_article.get('date', 'Unknown')}\n"
    system_prompt += f"**Source:** {seed_article.get('link', 'Unknown')}\n"
    system_prompt += f"**Content:** {seed_article.get('content', 'No content')[:500]}...\n\n"
    system_prompt += "Neighbor Articles:\n"
    for neighbor in neighbor_articles:
        system_prompt += f"- {neighbor.get('title', 'No title')} (Published: {neighbor.get('date', 'Unknown')})\n"
    system_prompt += "\nBased on the above evidence, please answer the query."
    return system_prompt

# Initialize OpenAI client with OpenRouter
from openai import OpenAI
client = OpenAI(
    api_key='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',
    base_url="https://openrouter.ai/api/v1"
)

# Set the model
model = "google/gemini-2.0-flash-exp:free"

@app.route('/query_with_titles', methods=['POST'])
def query_with_titles():
    """
    This route expects a JSON payload with:
      - "query": the user's query.
      - "titles": an array of article titles (strings) selected on the frontend.
      
    For each title, the backend:
      1. Appends the query (i.e., "Article Title" + " " + query).
      2. Computes the embedding for that appended text.
      3. Uses FAISS to search for one seed article.
      4. Retrieves multihop neighbor information for that seed article (limited to 1 neighbor).
      
    Then, it builds a combined system prompt that includes:
      • The user query.
      • The original selected titles (Title Neighbors).
      • For each title: the associated seed article (Query Neighbor) and one neighbor article.
      
    This prompt is then streamed to the LLM (via OpenRouter) and the response is streamed back to the client.
    """
    data = request.get_json()
    print("DEBUG: Received data:", data)
    query_text = data.get("query", "")
    titles = data.get("titles", [])  # Expecting an array of title strings
    links = data.get("links", [])
    print("DEBUG: Received links:", links)
    print("DEBUG: Received titles:", titles)
    print("DEBUG: Received query:", query_text)
    db_articles = []
    if links:
        try:
            # Connect to the database
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Create placeholders for the SQL query
            placeholders = ','.join(['%s'] * len(links))
            
            # Query to fetch details and summary for the provided links
            sql = f"SELECT id, header, summary, details, link FROM news_dawn WHERE link IN ({placeholders})"
            cursor.execute(sql, links)
            
            # Fetch all matching articles
            db_articles = cursor.fetchall()
            
            # Close cursor and connection
            cursor.close()
            conn.close()
            
            print(f"DEBUG: Found {len(db_articles)} articles in database")
        except Exception as e:
            print(f"ERROR: Database access failed: {str(e)}")
    
    #print("DEBUG: Found articles:", db_articles)

    results_db = []
    for article in db_articles:
        results_db.append(
        article["header"] + " " + 
        (article["details"] if article["details"] else "") + " " + 
        (article["summary"] if article["summary"] else "")
    )

    
    print("DEBUG: Found articles:", results_db)


    if not query_text:
        return jsonify({"error": "No query provided."}), 400
    if not titles:
        return jsonify({"error": "No titles provided."}), 400

    # Process each title independently:
    results_db.append(query_text)  # Append the query to the titles list for processing
    results = []
    for result in results_db:
        appended_text = f"{result}"
        # Compute embedding for this appended text
        embedding = model1.encode(appended_text, convert_to_numpy=True).reshape(1, -1)
        # Search FAISS for one seed article
        distances, seed_ids = index.search(embedding, 3)
        seed_id = str(seed_ids[0][0])
        
        # Retrieve seed article's neighbor information from MongoDB
        seed_doc = neighbors_collection.find_one(
            {"node_id": {"$in": [seed_id]}},
            {"_id": 0, "node_id": 1, "neighbors": 1}
        )
        if seed_doc:
            neighbors_list = seed_doc.get("neighbors", [])
        else:
            neighbors_list = []
        # Limit to only 1 neighbor if available
        limited_neighbors = neighbors_list[:5] if neighbors_list else []
        
        
        # Query MongoDB for full article details for seed article and neighbor(s)
        all_ids = [seed_id] + limited_neighbors
        articles_cursor = articles_collection.find(
            {"article_id": {"$in": all_ids}}, {"_id": 0}
        )
        article_details = {doc["article_id"]: doc for doc in articles_cursor}
        
        seed_article = article_details.get(seed_id, {})
        neighbor_article = article_details.get(limited_neighbors[0], {}) if limited_neighbors else {}
        
        results.append({
            "org_art": result,
            "seed_article": seed_article,
            "neighbor_article": neighbor_article
        })
    
    # Build a combined system prompt based on the per-title results.
    combined_prompt = ("You are an AI that summarizes multiple articles into a single, comprehensive response. "
        "Your response should include:\n"
        "1. A well-structured summary of key information from all articles. More than one line\n"
        "2. A timeline of events (if applicable) in chronological order.\n"
        "3. Sources (URLs) for credibility. Keep sources at the end dont mention at every point\n"
        "4. Keep the response well-formatted and informative.\n"
        "Here are the articles:\n\n")
    for result in results:
        combined_prompt += f"Original Article: {result['org_art']}\n"
        combined_prompt += "Seed Article:\n"
        combined_prompt += f"   Title: {result['seed_article'].get('title', 'No title')}\n"
        combined_prompt += f"   Published Date: {result['seed_article'].get('date', 'Unknown')}\n"
        combined_prompt += f"   Source: {result['seed_article'].get('link', 'Unknown')}\n"
        combined_prompt += f"   Content: {result['seed_article'].get('content', 'No content')[:500]}...\n"
        combined_prompt += "Neighbor Article:\n"
        if result['neighbor_article']:
            combined_prompt += f"   Title: {result['neighbor_article'].get('title', 'No title')}\n"
            combined_prompt += f"   Published Date: {result['neighbor_article'].get('date', 'Unknown')}\n"
            combined_prompt += f"   Source: {result['neighbor_article'].get('link', 'Unknown')}\n"
            combined_prompt += f"   Content: {result['neighbor_article'].get('content', 'No content')[:500]}...\n"
        else:
            combined_prompt += "   None\n"
        combined_prompt += "\n"
    combined_prompt += f"User Query: {query_text}\n\n"
    combined_prompt += "Based on the above evidence for each title, please answer the query accordingly."

    # Print the combined prompt for debugging
    print("DEBUG: Combined prompt:", combined_prompt)

    def generate():
        # Yield the initial computed results (optional debug step)
        yield json.dumps({"results": results}, cls=NumpyEncoder) + "\n"
        
        try:
            # Create a streaming response from OpenRouter instead of Ollama
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided evidence."},
                    {"role": "user", "content": combined_prompt}
                ],
                stream=True  # Enable streaming
            )
            
            response_text = ""
            
            # Process the streaming response
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text_fragment = chunk.choices[0].delta.content
                    response_text += text_fragment
                    
                    # Send the updated text to the client
                    yield json.dumps({
                        "response": text_fragment,
                        "done": False
                    }) + "\n"
            
            # Signal that we're done
            yield json.dumps({
                "done": True,
                "final_response": response_text
            }) + "\n"
                
        except Exception as e:
            # Handle errors
            yield json.dumps({
                "error": str(e),
                "done": True
            }) + "\n"
    
    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')


# ---------------------------- END NEW ROUTE ----------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
