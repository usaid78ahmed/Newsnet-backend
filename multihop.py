import json
from pymongo import MongoClient

# Connect to MongoDB (adjust the connection string if needed).
client = MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]
collection = db["multihop_neighbors"]

# Optional: Clear the collection before inserting.


# Load your multi-hop neighbors dictionary from the JSON file.
with open("multi_hop_neighbors_dict.json", "r") as f:
    multi_hop_neighbors = json.load(f)

# Prepare documents for insertion.
# Each document will have the format: {"node_id": "1", "neighbors": ["2", "3", ...]}
documents = [{"node_id": node, "neighbors": neighbors} 
             for node, neighbors in multi_hop_neighbors.items()]

# Insert the documents into the MongoDB collection.
result = collection.insert_many(documents)
print("Inserted document IDs:")
