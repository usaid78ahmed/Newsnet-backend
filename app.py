from flask import Flask, request, jsonify, Response, stream_with_context
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import json
import requests
from flask_cors import CORS
import networkx as nx
from openai import OpenAI

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

openrouter_client = OpenAI(
    api_key= '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',
    base_url="https://openrouter.ai/api/v1"
)

#gemini-2.5-pro-exp-03-25:free
#'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'
# Set the model
openrouter_model = "google/gemini-2.0-flash-exp:free"

# Explicitly handle OPTIONS requests (preflight requests)
@app.route('/query', methods=['OPTIONS'])
@app.route('/get_seed_articles', methods=['OPTIONS'])
@app.route('/get_neighbor_articles', methods=['OPTIONS'])
@app.route('/generate_summary', methods=['OPTIONS'])
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
        return super(NumpyEncoder, self).default(obj)

# Load the SentenceTransformer model.
model = SentenceTransformer("all-mpnet-base-v2")

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
entity_relationships_collection = db["entity_relationships"]

def format_articles_for_llama(seed_articles, neighboring):
    """
    Format retrieved articles as input for Llama3 with a system prompt.
    """
    system_prompt = (
        "You are an AI that summarizes multiple articles into a single, comprehensive response. "
        "Your response should include:\n"
        "1. A well-structured summary of key information from all articles.\n"
        "2. A timeline of events (if applicable) in chronological order.\n"
        "3. Sources (URLs) for credibility. Keep sources at the end.\n"
        "4. Keep the response well-formatted and informative.\n\n"
        "IMPORTANT: FORMAT YOUR RESPONSE WITH MARKDOWN:\n"
        "- Use # for main headings\n"
        "- Use ## for subheadings\n"
        "- Use **bold text** for emphasis\n"
        "- Use bullet points (- or *) for lists\n"
        "- Use numbered lists where appropriate\n"
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
    
    formatted_text += "**Relevant Articles (Neighbors)\n\n"
    for article in neighboring:
        formatted_text += f"**Title:** {article['title']}\n"
        formatted_text += f"**Published Date:** {article['date']}\n"
        formatted_text += f"**Source:** {article['link']}\n"
        formatted_text += f"**Content:** {article['content'][:500]}...\n\n"  # Truncated for brevity

    return formatted_text

def get_entity_relationships_for_articles(article_ids):
    # Convert article_ids to integers.
    try:
        article_ids_int = [int(aid) for aid in article_ids]
    except ValueError:
        # If conversion fails, just use the original list.
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

@app.route('/get_seed_articles', methods=['POST'])
def get_seed_articles():
    data = request.get_json()
    query = data.get("query", "")
    n_seed = data.get("n_seed", 5)  # Number of seed articles to retrieve.

    if not query:
        return jsonify({"error": "No query provided."}), 400

    # 1. Embed the query and retrieve seed article IDs from FAISS.
    q_emb = embed_query_fn(query, index).reshape(1, -1)
    distances, seed_ids = index.search(q_emb, n_seed)
    seed_ids = [str(x) for x in seed_ids[0]]  # Convert IDs to strings

    # 2. Query MongoDB for the full article details.
    seed_articles_data = []
    for doc in articles_collection.find({"article_id": {"$in": seed_ids}}, {"_id": 0}):
        # Add the article_id to each document for reference
        doc["id"] = doc["article_id"]
        seed_articles_data.append(doc)

    return jsonify({
        "seed_articles": seed_articles_data
    })


@app.route('/get_neighbor_articles', methods=['POST'])
def get_neighbor_articles():
    data = request.get_json()
    seed_id = data.get("seed_id")
    query = data.get("query", "")  # Original query to compute similarity
    threshold = data.get("threshold", 0.5)  # Similarity threshold

    if not seed_id:
        return jsonify({"error": "No seed article ID provided."}), 400

    # 1. Get neighbors for this seed article
    seed_doc = neighbors_collection.find_one(
        {"node_id": seed_id}, {"_id": 0, "neighbors": 1}
    )
    
    if not seed_doc:
        return jsonify({"error": f"No neighbors found for seed article with ID {seed_id}"}), 404
        
    neighbor_ids = seed_doc.get("neighbors", [])
    
    # 2. If we have a query, compute similarity for filtering
    if query:
        q_emb = embed_query_fn(query, index).reshape(1, -1)
        q_vec = q_emb.flatten()
    
    # 3. Get neighbor article details
    neighbor_articles = []
    for doc in articles_collection.find({"article_id": {"$in": neighbor_ids}}, {"_id": 0}):
        # If we have a query, compute similarity
        if query:
            article_id = doc["article_id"]
            try:
                article_emb = index.index.reconstruct(int(article_id))
                sim = compute_cosine_similarity(q_vec, article_emb)
                doc["similarity"] = float(sim)
                
                # Only add if above threshold
                if sim >= threshold:
                    doc["id"] = article_id  # Add ID for frontend reference
                    neighbor_articles.append(doc)
            except:
                # Skip if error in reconstruction
                continue
        else:
            # If no query, just add all neighbors
            doc["id"] = doc["article_id"]
            neighbor_articles.append(doc)
    
    # Sort by similarity if available
    if query:
        neighbor_articles.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
    return jsonify({
        "neighbor_articles": neighbor_articles
    })

@app.route('/query', methods=['POST'])
def query_api():
    data = request.get_json()
    query = data.get("query", "")
    n_seed = data.get("n_seed", 5)  # Number of seed articles to retrieve.
    threshold = 0.5  # Cosine similarity threshold for neighbors only.

    if not query:
        return jsonify({"error": "No query provided."}), 400

    # 1. Embed the query and retrieve seed article IDs from FAISS.
    q_emb = embed_query_fn(query, index).reshape(1, -1)
    q_vec = q_emb.flatten()
    distances, seed_ids = index.search(q_emb, n_seed)
    seed_ids = [str(x) for x in seed_ids[0]]  # Convert IDs to strings

    # 2. Query MongoDB for the seed articles' neighbors.
    seed_docs = list(neighbors_collection.find(
        {"node_id": {"$in": seed_ids}}, {"_id": 0, "node_id": 1, "neighbors": 1}
    ))

    # 3. Extract neighbors for only the retrieved seeds.
    neighbor_ids = set()
    for doc in seed_docs:
        for neighbor in doc.get("neighbors", []):
            neighbor_ids.add(neighbor)
    neighbor_ids = list(neighbor_ids)

    # 4. Query MongoDB for full article details (both seed and neighbor articles).
    all_ids = seed_ids + neighbor_ids
    article_details = {doc["article_id"]: doc for doc in articles_collection.find(
        {"article_id": {"$in": all_ids}}, {"_id": 0}
    )}
    seed_id_links = {doc["article_id"]: doc["link"] for doc in articles_collection.find(
        {"article_id": {"$in": seed_ids}}, {"_id": 0,"article_id": 1, "link": 1}
    )}

    # 5. Compute cosine similarity for neighbor articles only.
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
    formatted_articles = format_articles_for_llama(
        seed_articles_data,
        enriched_neighbors
    )

    # 7. Get entity relationships for all articles
    entity_relationships = get_entity_relationships_for_articles(all_ids)
    
    # 8. Build graph from entity relationships
    entity_graph = build_entity_graph(entity_relationships)
    
    # 9. Extract important entities and triplets
    important_entities = get_most_connected_entities(entity_graph, top_n=8)
    important_triplets = extract_important_triplets(entity_graph, important_entities, max_triplets=50)
    #print("DEBUG: Final entity_graph:", entity_graph)
    #print("DEBUG: Final entity_relationships:", entity_relationships)
    def generate():
        # First yield the seed articles information and entity relationships
        seed_info = {
            "seed_articles": seed_articles_data,
            "seed_id_links": seed_id_links,
            "neighbor_count": len(enriched_neighbors),
            "entity_triplets": important_triplets,
            "important_entities": [entity for entity, _ in important_entities]
        }
        yield json.dumps(seed_info, cls=NumpyEncoder) + '\n'
        
        
        try:
            # Create a streaming response from OpenRouter
            stream = openrouter_client.chat.completions.create(
                model=openrouter_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes news articles. Format your response with Markdown, using # for main headings, ## for subheadings, **bold text** for emphasis, bullet points with * or -, and numbered lists where appropriate. Make your response visually structured and easy to read."},
                    {"role": "user", "content": formatted_articles}
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

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    data = request.get_json()
    seed_article_id = data.get("seed_article_id")
    neighbor_article_ids = data.get("neighbor_article_ids", [])
    
    if not seed_article_id:
        return jsonify({"error": "No seed article ID provided."}), 400
    
    # 1. Get article details
    all_ids = [seed_article_id] + neighbor_article_ids
    
    # Get all article details
    article_details = {}
    for doc in articles_collection.find({"article_id": {"$in": all_ids}}, {"_id": 0}):
        article_details[doc["article_id"]] = doc
    
    # 2. Organize articles
    seed_article = article_details.get(seed_article_id)
    if not seed_article:
        return jsonify({"error": f"Seed article with ID {seed_article_id} not found"}), 404
    
    seed_articles = [seed_article]
    neighbor_articles = [article_details.get(nid) for nid in neighbor_article_ids if nid in article_details]
    
    # 3. Format articles for the LLM
    formatted_articles = format_articles_for_llama(seed_articles, neighbor_articles)
    
    # 4. Get entity relationships
    entity_relationships = get_entity_relationships_for_articles(all_ids)
    
    # 5. Build entity graph
    entity_graph = build_entity_graph(entity_relationships)
    
    # 6. Extract important entities and triplets
    important_entities = get_most_connected_entities(entity_graph, top_n=8)
    important_triplets = extract_important_triplets(entity_graph, important_entities, max_triplets=50)
    
    def generate():
        # First yield the initial metadata
        seed_info = {
            "seed_articles": seed_articles,
            "neighbor_articles": neighbor_articles,
            "entity_triplets": important_triplets,
            "important_entities": [entity for entity, _ in important_entities]
        }
        yield json.dumps(seed_info, cls=NumpyEncoder) + '\n'
        
        try:
            # Create a streaming response from OpenRouter
            stream = openrouter_client.chat.completions.create(
                model=openrouter_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes news articles. Format your response with Markdown, using # for main headings, ## for subheadings, **bold text** for emphasis, bullet points with * or -, and numbered lists where appropriate. Make your response visually structured and easy to read."},
                    {"role": "user", "content": formatted_articles}
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

@app.route('/dummy_stream', methods=['GET'])
def dummy_stream():
    def generate():
        for i in range(10000):
            yield f"Line {i}\n"
    return Response(stream_with_context(generate()), mimetype="text/plain")

@app.route('/hello_model', methods=['GET'])
def hello_model():
    prompt = "Hello, please greet me back!"
    max_tokens = 20  # Adjust as needed.
    ollama_url = "http://0.0.0.0:11434/api/generate"  # Update if necessary.

    def generate():
        try:
            # Call the Ollama API with streaming enabled
            with requests.post(
                ollama_url,
                json={
                    "model": "llama3.1:8b",
                    "prompt": prompt,
                    "stream": True,
                    "max_tokens": max_tokens
                },
                stream=True
            ) as response:
                response.raise_for_status()
                
                # Stream the response back
                for line in response.iter_lines():
                    if line:
                        yield line.decode('utf-8') + '\n'
                        
        except requests.exceptions.RequestException as e:
            yield json.dumps({"error": str(e)})
    
    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify, Response, stream_with_context
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from pymongo import MongoClient
# import json
# import requests
# from flask_cors import CORS 
# app = Flask(__name__)
# CORS(app, resources={
#     r"/*": {
#         "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
#         "methods": ["GET", "POST", "OPTIONS"],
#         "allow_headers": ["Content-Type", "Authorization"]
#     }
# })

# # Explicitly handle OPTIONS requests (preflight requests)
# @app.route('/query', methods=['OPTIONS'])
# def handle_options():
#     response = jsonify({'status': 'ok'})
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
#     return response

# # Custom JSON Encoder to handle NumPy data types
# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super(NumpyEncoder, self).default(obj)

# # Load the SentenceTransformer model.
# model = SentenceTransformer("all-mpnet-base-v2")

# def compute_cosine_similarity(a, b):
#     """Compute cosine similarity between two vectors."""
#     a_norm = np.linalg.norm(a)
#     b_norm = np.linalg.norm(b)
#     if a_norm == 0 or b_norm == 0:
#         return 0.0
#     return np.dot(a, b) / (a_norm * b_norm)

# def embed_query_fn(query, index):
#     """Embed the query using the SentenceTransformer model."""
#     embedding = model.encode(query, convert_to_numpy=True)
#     return embedding.reshape(1, -1)

# # Load the FAISS index.
# index = faiss.read_index("embeddings_index.faiss")

# # Connect to MongoDB.
# client = MongoClient("mongodb://localhost:27017/")
# db = client["mydatabase"]
# neighbors_collection = db["multihop_neighbors"]
# articles_collection = db["id_to_article"]

# def format_articles_for_llama(seed_articles, neighboring):
#     """
#     Format retrieved articles as input for Llama3 with a system prompt.
#     """
#     system_prompt = (
#         "You are an AI that summarizes multiple articles into a single, comprehensive response. "
#         "Your response should include:\n"
#         "1. A well-structured summary of key information from all articles. More than one line\n"
#         "2. A timeline of events (if applicable) in chronological order.\n"
#         "3. Sources (URLs) for credibility. Keep sources at the end dont mention at every point\n"
#         "4. Keep the response well-formatted and informative.\n"
#         "Here are the articles:\n\n"
#     )

#     formatted_text = system_prompt

#     # Add seed articles
#     formatted_text += "**Main Articles:**\n\n"
#     for article in seed_articles:
#         formatted_text += f"**Title:** {article['title']}\n"
#         formatted_text += f"**Published Date:** {article['date']}\n"
#         formatted_text += f"**Source:** {article['link']}\n"
#         formatted_text += f"**Content:** {article['content'][:500]}...\n\n"  # Truncated for brevity
    
#     formatted_text += "**Relevant Articles (Neighbors)\n\n"
#     for article in neighboring:
#         formatted_text += f"**Title:** {article['title']}\n"
#         formatted_text += f"**Published Date:** {article['date']}\n"
#         formatted_text += f"**Source:** {article['link']}\n"
#         formatted_text += f"**Content:** {article['content'][:500]}...\n\n"  # Truncated for brevity

#     return formatted_text

# @app.route('/query', methods=['POST'])
# def query_api():
#     data = request.get_json()
#     query = data.get("query", "")
#     n_seed = data.get("n_seed", 3)  # Number of seed articles to retrieve.
#     threshold = 0.5  # Cosine similarity threshold for neighbors only.

#     if not query:
#         return jsonify({"error": "No query provided."}), 400

#     # 1. Embed the query and retrieve seed article IDs from FAISS.
#     q_emb = embed_query_fn(query, index).reshape(1, -1)
#     q_vec = q_emb.flatten()
#     distances, seed_ids = index.search(q_emb, n_seed)
#     seed_ids = [str(x) for x in seed_ids[0]]  # Convert IDs to strings

#     # 2. Query MongoDB for the seed articles' neighbors.
#     seed_docs = list(neighbors_collection.find(
#         {"node_id": {"$in": seed_ids}}, {"_id": 0, "node_id": 1, "neighbors": 1}
#     ))

#     # 3. Extract neighbors for only the retrieved seeds.
#     neighbor_ids = set()
#     for doc in seed_docs:
#         for neighbor in doc.get("neighbors", []):
#             neighbor_ids.add(neighbor)
#     neighbor_ids = list(neighbor_ids)

#     # 4. Query MongoDB for full article details (both seed and neighbor articles).
#     all_ids = seed_ids + neighbor_ids
#     article_details = {doc["article_id"]: doc for doc in articles_collection.find(
#         {"article_id": {"$in": all_ids}}, {"_id": 0}
#     )}

#     # 5. Compute cosine similarity for neighbor articles only.
#     enriched_neighbors = []
#     for neighbor_id in neighbor_ids:
#         if neighbor_id in article_details:
#             article_emb = index.index.reconstruct(int(neighbor_id))  # Get embedding from FAISS
#             sim = compute_cosine_similarity(q_vec, article_emb)
#             if sim >= threshold:
#                 article_details[neighbor_id]["similarity"] = float(sim)  # Convert NumPy float to Python float
#                 enriched_neighbors.append(article_details[neighbor_id])

#     # 6. Prepare articles for the model
#     formatted_articles = format_articles_for_llama(
#         [article_details[seed_id] for seed_id in seed_ids if seed_id in article_details],
#         enriched_neighbors[:2]
#     )

#     # 7. Get seed articles for the response
#     seed_articles_data = [article_details[seed_id] for seed_id in seed_ids if seed_id in article_details]
    
#     # First, send seed articles information to the client
#     def generate():
#         # First yield the seed articles information
#         seed_info = {
#             "seed_articles": seed_articles_data,
#             "neighbor_count": len(enriched_neighbors)
#         }
#         yield json.dumps(seed_info, cls=NumpyEncoder) + '\n'
        
#         # Set up streaming with Ollama
#         ollama_url = "http://0.0.0.0:11434/api/generate"
#         payload = {
#             "model": "phi3:3.8b",
#             "prompt": formatted_articles,
#             "stream": True  # Enable streaming from Ollama
#         }
        
#         try:
#             # Use stream=True to get response in chunks
#             with requests.post(ollama_url, json=payload, stream=True) as r:
#                 r.raise_for_status()
                
#                 response_text = ""
                
#                 # Process the streaming response from Ollama
#                 for line in r.iter_lines():
#                     if line:
#                         line_json = json.loads(line)
                        
#                         # Get the text fragment
#                         if 'response' in line_json:
#                             text_fragment = line_json.get('response', '')
#                             response_text += text_fragment
                            
#                             # Send the updated text to the client
#                             yield json.dumps({
#                                 "response": text_fragment,
#                                 "done": False
#                             }) + '\n'
                
#                 # Signal that we're done
#                 yield json.dumps({
#                     "done": True,
#                     "final_response": response_text
#                 }) + '\n'
                
#         except Exception as e:
#             # Handle errors
#             yield json.dumps({
#                 "error": str(e),
#                 "done": True
#             }) + '\n'

#     return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

# @app.route('/dummy_stream', methods=['GET'])
# def dummy_stream():
#     def generate():
#         for i in range(10000):
#             yield f"Line {i}\n"
#     return Response(stream_with_context(generate()), mimetype="text/plain")

# @app.route('/hello_model', methods=['GET'])
# def hello_model():
#     prompt = "Hello, please greet me back!"
#     max_tokens = 20  # Adjust as needed.
#     ollama_url = "http://0.0.0.0:11434/api/generate"  # Update if necessary.

#     def generate():
#         try:
#             # Call the Ollama API with streaming enabled
#             with requests.post(
#                 ollama_url,
#                 json={
#                     "model": "llama3.1:8b",
#                     "prompt": prompt,
#                     "stream": True,
#                     "max_tokens": max_tokens
#                 },
#                 stream=True
#             ) as response:
#                 response.raise_for_status()
                
#                 # Stream the response back
#                 for line in response.iter_lines():
#                     if line:
#                         yield line.decode('utf-8') + '\n'
                        
#         except requests.exceptions.RequestException as e:
#             yield json.dumps({"error": str(e)})
    
#     return Response(stream_with_context(generate()), mimetype="application/x-ndjson")

# if __name__ == '__main__':
#     app.run(debug=True)