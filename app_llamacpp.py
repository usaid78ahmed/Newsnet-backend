from flask import Flask, request, jsonify, Response, stream_with_context
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import json
from flask_cors import CORS
import networkx as nx
from llama_cpp import Llama  # New import for llama-cpp-python
import threading
import queue
import time

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Path to your GGUF model file
MODEL_PATH = "models/phi3-3.8b-q4_k_m.gguf"

# Initialize the Llama model (done at startup)
print("Loading Llama model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,          # Context window size
    n_threads=8,         # CPU threads to use
    n_gpu_layers=-1,     # -1 means offload all layers to GPU if available
    verbose=False
)
print("Model loaded successfully!")

# Custom JSON Encoder to handle NumPy data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Explicitly handle OPTIONS requests (preflight requests)
@app.route('/query', methods=['OPTIONS'])
def handle_options():
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Load the SentenceTransformer model
model = SentenceTransformer("all-mpnet-base-v2")

# Load the FAISS index
index = faiss.read_index("embeddings_index.faiss")

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]
neighbors_collection = db["multihop_neighbors"]
articles_collection = db["id_to_article"]
entity_relationships_collection = db["entity_relationships"]

# Helper function for llama.cpp streaming
def stream_llama_response(prompt, response_queue):
    """
    Stream the generation from llama-cpp model into a queue
    """
    for output in llm.generate(
        prompt,
        max_tokens=2048,
        stop=["</s>"],
        stream=True
    ):
        token = output["choices"][0]["text"]
        response_queue.put(token)
    
    # Signal completion
    response_queue.put(None)

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

def format_articles_for_llama(seed_articles):
    """
    Format retrieved articles as input for Llama with a system prompt.
    """
    system_prompt = (
        "You are an AI that summarizes multiple articles into a single, comprehensive response. "
        "Your response should include:\n"
        "1. A well-structured summary of key information from all articles. More than one line\n"
        "2. A timeline of events (if applicable) in chronological order.\n"
        "3. Sources (URLs) for credibility. Keep sources at the end dont mention at every point\n"
        "4. Keep the response well-formatted and informative.\n"
        "Here are the articles:\n\n"
    )

    formatted_text = system_prompt

    # Add seed articles
    formatted_text += "**Main Articles:**\n\n"
    for article in seed_articles:
        formatted_text += f"**Title:** {article['title']}\n"
        formatted_text += f"**Published Date:** {article['date']}\n"
        formatted_text += f"**Source:** {article['link']}\n"
        formatted_text += f"**Content:** {article['content'][:500]}...\n\n"  # Truncated for brevity
    
    formatted_text += "**Relevant Articles (Neighbors):**\n\n"
    
    return formatted_text

def get_entity_relationships_for_articles(article_ids):
    # Convert article_ids to integers
    try:
        article_ids_int = [int(aid) for aid in article_ids]
    except ValueError:
        # If conversion fails, just use the original list
        article_ids_int = article_ids

    # Query documents where id is in the provided list
    relationships_cursor = entity_relationships_collection.find(
        {"id": {"$in": article_ids_int}}
    )
    
    # Aggregate all relationships
    relationships = []
    for doc in relationships_cursor:
        relationships.extend(doc.get("relationships", []))
    
    return relationships

def build_entity_graph(relationships):
    """
    Build a graph from entity relationships.
    """
    G = nx.Graph()
    
    # Add nodes and edges to the graph
    for triplet in relationships:
        if len(triplet) >= 3:
            subject, predicate, obj = triplet[:3]
            
            # Add nodes
            G.add_node(subject)
            G.add_node(obj)
            
            # Add edge with predicate as attribute
            G.add_edge(subject, obj, relation=predicate)
    
    return G

def get_most_connected_entities(G, top_n=10):
    """
    Get the most connected entities from the graph.
    """
    # Get node degrees (number of connections)
    degrees = dict(G.degree())
    
    # Sort nodes by degree in descending order
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N nodes
    return sorted_nodes[:top_n]

def extract_important_triplets(G, important_entities, max_triplets=20):
    """
    Extract important triplets involving the key entities.
    """
    important_entities_set = set([entity for entity, _ in important_entities])
    important_triplets = []
    
    for u, v, data in G.edges(data=True):
        if u in important_entities_set or v in important_entities_set:
            triplet = [u, data.get('relation', 'related_to'), v]
            important_triplets.append(triplet)
            
            if len(important_triplets) >= max_triplets:
                break
    
    return important_triplets

@app.route('/query', methods=['POST'])
def query_api():
    data = request.get_json()
    query = data.get("query", "")
    n_seed = data.get("n_seed", 3)  # Number of seed articles to retrieve
    threshold = 0.5  # Cosine similarity threshold for neighbors only

    if not query:
        return jsonify({"error": "No query provided."}), 400

    # 1. Embed the query and retrieve seed article IDs from FAISS
    q_emb = embed_query_fn(query, index).reshape(1, -1)
    q_vec = q_emb.flatten()
    distances, seed_ids = index.search(q_emb, n_seed)
    seed_ids = [str(x) for x in seed_ids[0]]  # Convert IDs to strings

    # 2. Query MongoDB for the seed articles' neighbors
    seed_docs = list(neighbors_collection.find(
        {"node_id": {"$in": seed_ids}}, {"_id": 0, "node_id": 1, "neighbors": 1}
    ))

    # 3. Extract neighbors for only the retrieved seeds
    neighbor_ids = set()
    for doc in seed_docs:
        for neighbor in doc.get("neighbors", []):
            neighbor_ids.add(neighbor)
    neighbor_ids = list(neighbor_ids)

    # 4. Query MongoDB for full article details (both seed and neighbor articles)
    all_ids = seed_ids + neighbor_ids
    article_details = {doc["article_id"]: doc for doc in articles_collection.find(
        {"article_id": {"$in": all_ids}}, {"_id": 0}
    )}

    # 5. Compute cosine similarity for neighbor articles only
    enriched_neighbors = []
    for neighbor_id in neighbor_ids:
        if neighbor_id in article_details:
            article_emb = index.index.reconstruct(int(neighbor_id))  # Get embedding from FAISS
            sim = compute_cosine_similarity(q_vec, article_emb)
            if sim >= threshold:
                article_details[neighbor_id]["similarity"] = float(sim)  # Convert NumPy float to Python float
                enriched_neighbors.append(article_details[neighbor_id])

    # 6. Prepare articles for the model
    seed_articles_data = [article_details[seed_id] for seed_id in seed_ids if seed_id in article_details]
    formatted_articles = format_articles_for_llama(seed_articles_data)

    # 7. Get entity relationships for all articles
    entity_relationships = get_entity_relationships_for_articles(all_ids)
    
    # 8. Build graph from entity relationships
    entity_graph = build_entity_graph(entity_relationships)
    
    # 9. Extract important entities and triplets
    important_entities = get_most_connected_entities(entity_graph, top_n=8)
    important_triplets = extract_important_triplets(entity_graph, important_entities, max_triplets=50)

    def generate():
        # First yield the seed articles information and entity relationships
        seed_info = {
            "seed_articles": seed_articles_data,
            "neighbor_count": len(enriched_neighbors),
            "entity_triplets": important_triplets,
            "important_entities": [entity for entity, _ in important_entities]
        }
        yield json.dumps(seed_info, cls=NumpyEncoder) + '\n'
        
        # Create a queue for streaming responses from the llama model
        response_queue = queue.Queue()
        
        # Start a thread to run the model inference (non-blocking)
        thread = threading.Thread(
            target=stream_llama_response,
            args=(formatted_articles, response_queue)
        )
        thread.start()
        
        # Accumulate complete response
        response_text = ""
        
        # Stream results back to client as they become available
        while True:
            try:
                # Wait for the queue to have data (with timeout)
                token = response_queue.get(timeout=30)
                
                # Check if generation is complete
                if token is None:
                    break
                
                response_text += token
                
                # Send the token to the client
                yield json.dumps({
                    "response": token,
                    "done": False
                }) + '\n'
                
            except queue.Empty:
                # If no output for 30 seconds, assume something went wrong
                yield json.dumps({
                    "error": "Model generation timeout",
                    "done": True
                }) + '\n'
                break
        
        # Send the final response
        yield json.dumps({
            "done": True,
            "final_response": response_text
        }) + '\n'

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

@app.route('/query_with_titles', methods=['POST'])
def query_with_titles():
    data = request.get_json()
    query_text = data.get("query", "")
    titles = data.get("titles", [])  # Expecting an array of title strings

    if not query_text:
        return jsonify({"error": "No query provided."}), 400
    if not titles:
        return jsonify({"error": "No titles provided."}), 400

    # Process each title independently:
    titles.append(query_text)  # Append the query to the titles list for processing
    results = []
    for title in titles:
        appended_text = f"{title}"
        # Compute embedding for this appended text
        embedding = model.encode(appended_text, convert_to_numpy=True).reshape(1, -1)
        # Search FAISS for one seed article
        distances, seed_ids = index.search(embedding, 1)
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
        limited_neighbors = neighbors_list[:1] if neighbors_list else []
        
        # Query MongoDB for full article details for seed article and neighbor(s)
        all_ids = [seed_id] + limited_neighbors
        articles_cursor = articles_collection.find(
            {"article_id": {"$in": all_ids}}, {"_id": 0}
        )
        article_details = {doc["article_id"]: doc for doc in articles_cursor}
        
        seed_article = article_details.get(seed_id, {})
        neighbor_article = article_details.get(limited_neighbors[0], {}) if limited_neighbors else {}

        results.append({
            "title": title,
            "seed_article": seed_article,
            "neighbor_article": neighbor_article
        })
    
    # Build a combined system prompt based on the per-title results
    combined_prompt = "You are an AI that answers queries based on per-title evidence.\n\n"
    for result in results:
        combined_prompt += f"Title: {result['title']}\n"
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

    def generate():
        # Yield the initial computed results (optional debug step)
        yield json.dumps({"results": results}, cls=NumpyEncoder) + "\n"
        
        # Create a queue for streaming responses from the llama model
        response_queue = queue.Queue()
        
        # Start a thread to run the model inference (non-blocking)
        thread = threading.Thread(
            target=stream_llama_response,
            args=(combined_prompt, response_queue)
        )
        thread.start()
        
        # Accumulate complete response
        response_text = ""
        
        # Stream results back to client as they become available
        while True:
            try:
                # Wait for the queue to have data (with timeout)
                token = response_queue.get(timeout=30)
                
                # Check if generation is complete
                if token is None:
                    break
                
                response_text += token
                
                # Send the token to the client
                yield json.dumps({
                    "response": token,
                    "done": False
                }) + '\n'
                
            except queue.Empty:
                # If no output for 30 seconds, assume something went wrong
                yield json.dumps({
                    "error": "Model generation timeout",
                    "done": True
                }) + '\n'
                break
        
        # Send the final response
        yield json.dumps({
            "done": True,
            "final_response": response_text
        }) + '\n'

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

@app.route('/hello_model', methods=['GET'])
def hello_model():
    prompt = "Hello, please greet me back!"
    
    def generate():
        try:
            # Create a response queue
            response_queue = queue.Queue()
            
            # Start thread for model inference
            thread = threading.Thread(
                target=stream_llama_response,
                args=(prompt, response_queue)
            )
            thread.start()
            
            # Stream tokens back as they're generated
            response_text = ""
            while True:
                try:
                    token = response_queue.get(timeout=10)
                    if token is None:  # End of generation
                        break
                    
                    response_text += token
                    yield json.dumps({
                        "response": token,
                        "done": False
                    }) + "\n"
                    
                except queue.Empty:
                    yield json.dumps({
                        "error": "Generation timeout",
                        "done": True
                    }) + "\n"
                    break
            
            # Final complete response
            yield json.dumps({
                "done": True, 
                "final_response": response_text
            }) + "\n"
                
        except Exception as e:
            yield json.dumps({"error": str(e)})
    
    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)